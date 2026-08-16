"""
The evaluation loop: one method, one bank, streamed to disk.

Per-query predictions are written as they are produced. ``per_query.csv.gz`` is
the source of truth for the whole thesis: every table in ``metrics.json`` is
recomputable from it on a CPU in seconds. A new metric requested at the viva
costs one command, not a retraining run.

It is also the crash-tolerance story. A run that dies partway still leaves every
completed episode on disk, which is precisely what did not happen when three of
the original eight experiments were lost to a Jupyter output-rate limit.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsgrade.data.episodes import EpisodeBank
from fsgrade.data.loaders import EpisodeDataset, episode_collate
from fsgrade.evaluation.aggregate import EpisodeRecord, MetricAggregator
from fsgrade.methods.base import FewShotMethod

PER_QUERY_COLUMNS = [
    "run_id", "method", "fold_id", "split", "bank_id", "episode_id", "species",
    "n_shot", "query_index", "y_true", "y_pred", "logit_0", "logit_1",
    "score_positive", "query_relpath",
]

PER_EPISODE_COLUMNS = [
    "run_id", "method", "fold_id", "split", "bank_id", "episode_id", "species",
    "n_shot", "n_query", "seed", "adapt_seconds",
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision_fresh", "recall_fresh", "f1_fresh",
    "precision_rotten", "recall_rotten", "f1_rotten",
    "precision_macro", "recall_macro", "f1_macro",
    "tn", "fp", "fn", "tp", "roc_auc", "pr_auc", "cohen_kappa", "mcc", "brier", "nll",
]


def evaluate_method(
    method: FewShotMethod,
    bank: EpisodeBank,
    *,
    data_root: str,
    transforms: dict[str, Any],
    device: torch.device,
    k_shot: int | None = None,
    num_workers: int = 0,
    per_query_writer: Any = None,
    per_episode_writer: Any = None,
    run_id: str = "",
    logger: Any = None,
    progress_mode: str = "plain",
    aggregator: MetricAggregator | None = None,
) -> MetricAggregator:
    """Evaluate ``method`` on every episode of ``bank``."""
    from fsgrade.logging_utils import progress

    dataset = EpisodeDataset(
        bank, data_root,
        support_transform=transforms["support"],
        query_transform=transforms["eval"],
        k_shot=k_shot,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=episode_collate
    )

    # `is not None`, never truthiness: MetricAggregator defines __len__, so an
    # empty aggregator is falsy and `aggregator or MetricAggregator()` would
    # silently discard the caller's object and collect into a throwaway one.
    agg = aggregator if aggregator is not None else MetricAggregator()
    started = time.time()

    for batch in progress(
        loader, desc=f"eval {method.name}", total=len(dataset),
        mode=progress_mode, logger=logger, every=100,
    ):
        spec = batch.spec
        output = method.predict_episode(batch)

        y_true = batch.query_y.cpu().numpy().astype(int)
        y_pred = output.predictions().astype(int)
        probs = output.probabilities()
        y_score = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        record = EpisodeRecord(
            episode_id=spec.episode_id,
            method=method.name,
            fold_id=spec.fold_id,
            species=spec.species,
            split=spec.split,
            n_shot=spec.n_shot_max,
            n_query_per_class=spec.n_query_per_class,
            seed=spec.seed,
            y_true=y_true,
            y_pred=y_pred,
            y_score_pos=y_score,
            logits=output.logits,
            query_paths=spec.query_paths(),
            adapt_seconds=output.adapt_seconds,
        )
        agg.add(record)

        if per_query_writer is not None:
            paths = spec.query_paths()
            for i in range(len(y_true)):
                per_query_writer.write_row({
                    "run_id": run_id,
                    "method": method.name,
                    "fold_id": spec.fold_id,
                    "split": spec.split,
                    "bank_id": spec.bank_id,
                    "episode_id": spec.episode_id,
                    "species": spec.species,
                    "n_shot": spec.n_shot_max,
                    "query_index": i,
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "logit_0": float(output.logits[i, 0]),
                    "logit_1": float(output.logits[i, 1]) if output.logits.shape[1] > 1 else 0.0,
                    "score_positive": float(y_score[i]),
                    "query_relpath": paths[i] if i < len(paths) else "",
                })

        if per_episode_writer is not None:
            m = record.metrics()
            row = {
                "run_id": run_id,
                "method": method.name,
                "fold_id": spec.fold_id,
                "split": spec.split,
                "bank_id": spec.bank_id,
                "episode_id": spec.episode_id,
                "species": spec.species,
                "n_shot": spec.n_shot_max,
                "n_query": len(y_true),
                "seed": spec.seed,
                "adapt_seconds": output.adapt_seconds,
            }
            row.update({k: m.get(k) for k in PER_EPISODE_COLUMNS if k in m})
            per_episode_writer.write_row(row)

    if logger:
        elapsed = time.time() - started
        acc = float(np.nanmean(agg.series("accuracy"))) if len(agg) else float("nan")
        logger.info(
            "%s: %d episodes in %.1fs | mean episode accuracy %.4f",
            method.name, len(agg), elapsed, acc,
        )
    return agg


def measure_inference_cost(
    method: FewShotMethod,
    batch: Any,
    *,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Inference latency per episode and per image, for the model-complexity table."""
    for _ in range(n_warmup):
        method.predict_episode(batch)

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        method.predict_episode(batch)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times) * 1000.0
    n_query = int(batch.query_y.numel())
    return {
        "latency_ms_per_episode_mean": float(arr.mean()),
        "latency_ms_per_episode_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "latency_ms_per_image_mean": float(arr.mean() / max(n_query, 1)),
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "n_query_per_episode": n_query,
    }


def count_parameters(model: torch.nn.Module | None) -> dict[str, int]:
    if model is None:
        return {"total": 0, "trainable": 0, "frozen": 0}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
