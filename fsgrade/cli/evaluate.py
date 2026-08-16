"""
Train and evaluate every method on every fold, writing everything to disk.

    python -m fsgrade.cli.evaluate --config configs/experiment/main_loso.yaml

Produces ``results/<experiment>/<run>/`` containing:

    per_query.csv.gz   every prediction -- the source of truth
    per_episode.csv    per-episode metrics
    metrics.json       aggregated metrics with self-describing intervals
    comparisons.json   paired tests on the shared episode bank
    config.yaml        fully resolved config, including the resolved data root
    env.json           versions, GPU, git commit, determinism report
    status.json        running | ok | failed, with traceback on failure
    run.log            complete log

Because every method is evaluated on the *same* episode bank, the paired
statistics in ``comparisons.json`` are valid -- which the original
``ttest_rel`` on independently resampled episodes was not.
"""

from __future__ import annotations

import traceback
from typing import Any

from fsgrade.cli.common import banks_dir, base_parser, build_index, setup
from fsgrade.config import get_in
from fsgrade.data.episodes import EpisodeBank, build_folds
from fsgrade.data.index import split_seen_species
from fsgrade.evaluation.aggregate import MetricAggregator
from fsgrade.evaluation.runner import (
    PER_EPISODE_COLUMNS,
    PER_QUERY_COLUMNS,
    evaluate_method,
)
from fsgrade.evaluation.stats import (
    EpisodeSeries,
    compare_against_reference,
    comparisons_to_payload,
)
from fsgrade.io.results import RunDirectory
from fsgrade.logging_utils import log_table
from fsgrade.methods.base import FitContext, MethodUnavailable
from fsgrade.methods.registry import build_method, resolve_method_list


def _load_or_build_banks(index, fold, cfg, logger) -> dict[str, EpisodeBank]:
    from fsgrade.cli.build_banks import build_banks_for_fold

    directory = banks_dir(cfg)
    banks = build_banks_for_fold(index, fold, cfg, logger=None)
    for split, bank in banks.items():
        path = directory / f"{bank.bank_id}.jsonl"
        if path.exists():
            loaded = EpisodeBank.load(path)
            if loaded.content_hash() == bank.content_hash():
                banks[split] = loaded
                continue
        bank.save(directory)
    return banks


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Train and evaluate the method ladder under a protocol.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Override the method list (default: from config)")
    parser.add_argument("--folds", nargs="*", default=None,
                        help="Restrict to specific fold ids")
    parser.add_argument("--shots", nargs="*", type=int, default=None,
                        help="Evaluate at these K values (default: protocol n_shot)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Evaluate only the first N episodes of each test bank")
    args = parser.parse_args(argv)

    ctx = setup(args)
    cfg, device, logger = ctx["cfg"], ctx["device"], ctx["logger"]
    experiment = cfg.get("experiment", "experiment")

    run = RunDirectory.create(get_in(cfg, "paths.results_root", "./results"), experiment, cfg)
    file_logger = __import__("fsgrade.logging_utils", fromlist=["get_logger"]).get_logger(
        "fsgrade", log_file=run.path(get_in(cfg, "logging.file", "run.log")),
        level=get_in(cfg, "logging.level", "INFO"),
    )
    logger = file_logger
    run.write_config()
    run.write_env()

    try:
        index = build_index(cfg)
        folds = build_folds(cfg, get_in(cfg, "data.species", []))
        if args.folds:
            folds = [f for f in folds if f.fold_id in set(args.folds)]

        methods = args.methods or resolve_method_list(cfg.get("methods", "core"))
        shots = args.shots or [int(get_in(cfg, "protocol.episodes.n_shot", 5))]
        headline = get_in(cfg, "eval.headline_metric", "accuracy")
        progress_mode = get_in(cfg, "logging.progress", "plain")
        workers = int(get_in(cfg, "num_workers", 0))

        logger.info("=" * 72)
        logger.info("experiment=%s | run=%s", experiment, run.run_id)
        logger.info("protocol=%s | folds=%s", get_in(cfg, "protocol.name"),
                    [f.fold_id for f in folds])
        logger.info("methods=%s | shots=%s | device=%s", methods, shots, device)
        logger.info("=" * 72)

        from fsgrade.data.loaders import build_transforms

        transforms = build_transforms(cfg)

        aggregators: dict[str, MetricAggregator] = {}
        method_state: dict[str, dict[str, Any]] = {}
        unavailable: dict[str, str] = {}

        pq_writer = run.open_csv("per_query.csv.gz", PER_QUERY_COLUMNS,
                                 gzip_output=True, flush_every=200)
        pe_writer = run.open_csv("per_episode.csv", PER_EPISODE_COLUMNS, flush_every=25)

        with pq_writer, pe_writer:
            for fold in folds:
                logger.info("-" * 72)
                logger.info("FOLD %s | train=%s | test=%s",
                            fold.fold_id, list(fold.train_species), list(fold.test_species))
                banks = _load_or_build_banks(index, fold, cfg, logger)
                test_bank = banks["test"]
                if args.max_episodes:
                    test_bank = test_bank.head(args.max_episodes)

                assignment = split_seen_species(
                    index, fold.train_species,
                    val_ratio=float(get_in(cfg, "data.val_ratio", 0.15)),
                    seed=int(get_in(cfg, "data.split_seed", 42)),
                )
                fit_ctx = FitContext(
                    fold=fold, cfg=cfg, device=device,
                    data_root=get_in(cfg, "paths.data_root"), index=index,
                    train_pools=assignment.train, val_pools=assignment.val,
                    train_bank=banks["train"], val_bank=banks["val"],
                    run=run, logger=logger,
                )

                for method_name in methods:
                    if method_name in unavailable:
                        continue
                    try:
                        logger.info("[%s / %s] preparing", fold.fold_id, method_name)
                        method = build_method(method_name)
                        method.prepare(fit_ctx)
                    except (MethodUnavailable, ImportError) as exc:
                        unavailable[method_name] = f"{type(exc).__name__}: {exc}"
                        logger.warning("skipping %s -- %s", method_name, exc)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        unavailable[method_name] = f"{type(exc).__name__}: {exc}"
                        logger.error("FAILED to prepare %s: %s", method_name, exc)
                        logger.debug(traceback.format_exc())
                        continue

                    method_state[method_name] = method.state_summary()

                    for k in shots:
                        key = method_name if len(shots) == 1 else f"{method_name}@{k}shot"
                        agg = aggregators.setdefault(key, MetricAggregator(
                            class_names=index.classes,
                            ece_bins=int(get_in(cfg, "eval.ece_bins", 15)),
                            bootstrap_n=int(get_in(cfg, "eval.bootstrap_n", 10000)),
                            bootstrap_seed=int(get_in(cfg, "eval.bootstrap_seed", 7)),
                            ci_level=float(get_in(cfg, "eval.ci_level", 0.95)),
                        ))
                        evaluate_method(
                            method, test_bank,
                            data_root=get_in(cfg, "paths.data_root"),
                            transforms=transforms, device=device, k_shot=k,
                            num_workers=workers,
                            per_query_writer=pq_writer, per_episode_writer=pe_writer,
                            run_id=run.run_id, logger=logger,
                            progress_mode=progress_mode, aggregator=agg,
                        )

        # ---------------- aggregate ---------------- #
        logger.info("=" * 72)
        logger.info("AGGREGATING")
        payload: dict[str, Any] = {
            "run_id": run.run_id,
            "experiment": experiment,
            "protocol": {
                "name": get_in(cfg, "protocol.name"),
                "folds": [f.fold_id for f in folds],
            },
            "episode_bank": {
                "n_shot_evaluated": shots,
                "n_query_per_class": get_in(cfg, "protocol.episodes.n_query_per_class"),
                "bank_seed": get_in(cfg, "protocol.episodes.bank_seed"),
                "manifest_hash": index.manifest_hash,
            },
            "chance_level": 1.0 / float(get_in(cfg, "protocol.episodes.n_way", 2)),
            "methods": {},
            "methods_unavailable": unavailable,
        }
        for name, agg in aggregators.items():
            block = agg.summary(headline_metric=headline)
            block["state"] = method_state.get(name.split("@")[0], {})
            payload["methods"][name] = block

        run.write_json("metrics.json", payload, schema_version="1.0.0")

        # ---------------- paired statistics ---------------- #
        reference = get_in(cfg, "stats.reference_method", "ours")
        metric = get_in(cfg, "stats.metric", "accuracy")
        if reference in aggregators and len(aggregators) > 1:
            series = {
                name: EpisodeSeries.from_aggregator(agg, name, metric)
                for name, agg in aggregators.items()
            }
            comparisons = compare_against_reference(
                series, reference,
                alpha=float(get_in(cfg, "stats.alpha", 0.05)),
                alternative=get_in(cfg, "stats.alternative", "two-sided"),
                family=get_in(cfg, "stats.family", "main_comparison"),
                bootstrap_n=int(get_in(cfg, "eval.bootstrap_n", 10000)),
                seed=int(get_in(cfg, "eval.bootstrap_seed", 7)),
            )
            run.write_json(
                "comparisons.json",
                comparisons_to_payload(
                    comparisons, reference=reference, metric=metric,
                    alpha=float(get_in(cfg, "stats.alpha", 0.05)),
                    source_runs=[str(run.root)],
                ),
            )
        else:
            logger.warning("reference method %r not evaluated; skipping paired tests", reference)

        # ---------------- summary table ---------------- #
        rows = []
        for name, block in payload["methods"].items():
            ci = block["episode_mean"].get(headline, {})
            pooled = block.get("pooled", {})
            rows.append([
                name,
                f"{100 * ci.get('mean', float('nan')):.1f}",
                f"{100 * (ci.get('ci_hi', 0) - ci.get('ci_lo', 0)) / 2:.1f}",
                ci.get("n", 0),
                f"{pooled.get('roc_auc_pooled', float('nan')):.3f}",
                f"{pooled.get('mcc', float('nan')):.3f}",
            ])
        rows.sort(key=lambda r: float(r[1]), reverse=True)
        log_table(
            logger, rows,
            [f"method", f"{headline} %", "+/- CI", "n episodes", "ROC-AUC", "MCC"],
            title=f"RESULTS (chance = {100 * payload['chance_level']:.0f}%)",
        )

        run.finalize(n_methods=len(aggregators), n_folds=len(folds))
        logger.info("results written -> %s", run.root)
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        logger.error("run FAILED: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
