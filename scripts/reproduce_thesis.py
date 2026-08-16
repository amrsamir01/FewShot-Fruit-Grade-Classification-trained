#!/usr/bin/env python
"""
Regenerate every number that appears in the thesis, from one committed script.

Why this exists
---------------
Before this script, the reported results could not be reproduced by anything in
the repository:

  * no code in src/ or the notebook ever called json.dump, so results/*.json had
    no traceable producer (they came from gitignored helpers that were deleted);
  * notebook outputs were injected rather than executed (execution_count 12, 13
    and 14 each appear twice, which cannot happen in a linear run);
  * episode sampling used the global `random` module, so numbers depended on
    cell execution order;
  * figures and text in the same notebook cell came from different runs.

Everything the thesis quotes must come out of one invocation of this file.

Usage
-----
    # one-off sanity check that the dataset is present and intact
    python scripts/reproduce_thesis.py --verify-data

    # a smoke run (minutes, not hours) to prove the pipeline is wired
    python scripts/reproduce_thesis.py --seed 42 --smoke

    # the real thing, one seed
    python scripts/reproduce_thesis.py --seed 42

    # the three seeds the thesis reports
    for s in 42 1337 2024; do python scripts/reproduce_thesis.py --seed $s; done
    python scripts/reproduce_thesis.py --aggregate

Set FRUITVISION_ROOT to point at the image tree if it is not beside the repo.

Outputs, per run, under results/run_<timestamp>_seed<N>/
    metrics.json   every number, keyed so the thesis can cite it
    env.json       package versions, torch/CUDA, git commit, host
    summary.csv    flat one-row-per-measurement view for eyeballing
    figures/*.png  every figure, written here and nowhere else
    checkpoints/   trained weights, in the layout fsgrade/demo expects
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Must precede pyplot import: the visualization module calls plt.show(), which
# would block forever under a headless/batch run.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config, ensure_dirs  # noqa: E402
from src.dataset import FruitQualityDataset, EpisodicDataLoader  # noqa: E402
from src.losses import PrototypicalLoss  # noqa: E402
from src.models import PrototypicalNetwork  # noqa: E402
from src.train import train_protonet  # noqa: E402
from src.transforms import build_transforms  # noqa: E402
from src import experiments as exp  # noqa: E402
from src import visualization as viz  # noqa: E402


# ====================================================================== #
#  Determinism
# ====================================================================== #

def set_seed(seed: int) -> None:
    """Seed every RNG this pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ====================================================================== #
#  Provenance
# ====================================================================== #

def capture_env(seed: int) -> dict:
    """Record everything needed to explain why a rerun might differ."""
    def _pkg(name):
        try:
            import importlib.metadata as md
            return md.version(name)
        except Exception:
            return None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip())
    except Exception:
        commit, dirty = None, None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {
            name: _pkg(name)
            for name in ("numpy", "scipy", "torchvision", "scikit-learn",
                         "matplotlib", "seaborn", "pandas", "Pillow", "tqdm")
        },
    }


def dataset_manifest(data_root: str, fruits, classes) -> dict:
    """
    Content hash + per-class counts for the image tree.

    REPRODUCIBILITY.md commits sha256:9e14ddad...4fa9f39 over the sorted relative
    paths; recomputing it here is what makes "same dataset" a checkable claim
    rather than an assertion.
    """
    entries, counts = [], {}
    for fruit in sorted(fruits):
        counts[fruit] = {}
        for quality in classes:
            folder = Path(data_root) / fruit / quality
            files = sorted(
                f.name for f in folder.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ) if folder.is_dir() else []
            counts[fruit][quality] = len(files)
            entries.extend(f"{fruit}/{quality}/{n}" for n in files)

    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    total = sum(v for f in counts.values() for v in f.values())
    return {"manifest_sha256": digest, "counts": counts, "total_images": total}


def verify_data(config) -> int:
    """Check the dataset is present and report its manifest. Returns exit code."""
    root = Path(config.DATA_ROOT)
    print(f"Dataset root: {root}")
    if not root.is_dir():
        print("  MISSING. Set FRUITVISION_ROOT to the image tree, e.g.")
        print("    export FRUITVISION_ROOT=/path/to/FruitVision")
        return 1

    all_fruits = list(config.TRAIN_FRUITS) + list(config.TEST_FRUITS)
    man = dataset_manifest(config.DATA_ROOT, all_fruits, config.CLASSES)

    for fruit, qualities in man["counts"].items():
        line = "  ".join(f"{q}={n:4d}" for q, n in qualities.items())
        print(f"  {fruit:<8} {line}")
    print(f"  TOTAL {man['total_images']}")
    print(f"  manifest sha256: {man['manifest_sha256']}")

    expected_total = 6978
    if man["total_images"] != expected_total:
        print(f"  WARNING: expected {expected_total} images "
              f"(REPRODUCIBILITY.md), found {man['total_images']}.")
        print("  The thesis dataset section must be updated to match, or the "
              "wrong image tree is mounted.")
        return 1

    print("  OK - counts match REPRODUCIBILITY.md")
    return 0


# ====================================================================== #
#  Run directory
# ====================================================================== #

def make_run_dir(seed: int, smoke: bool, variant: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "smoke_" if smoke else ""
    suffix = f"_{variant}" if variant else ""
    run_dir = REPO_ROOT / "results" / f"run_{tag}{stamp}_seed{seed}{suffix}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def flatten_for_csv(metrics: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Depth-first flatten so every scalar gets one CSV row."""
    rows = []
    for key, value in metrics.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_for_csv(value, path))
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(v, (int, float)) for v in value):
                rows.append((path + ".n", str(len(value))))
                rows.append((path + ".mean", str(float(np.mean(value)))))
            # Long raw arrays stay in metrics.json only; the CSV is for reading.
        else:
            rows.append((path, str(value)))
    return rows


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


# ====================================================================== #
#  Main pipeline
# ====================================================================== #

def run(args) -> int:
    config = Config()

    if args.smoke:
        # Enough to prove every code path executes; not enough to report.
        # Shots and queries shrink too, so the smoke run works against a small
        # stand-in dataset without needing the full FruitVision tree.
        config.EPOCHS = 2
        config.WARMUP_EPOCHS = 1
        config.N_EPISODES_TRAIN = 8
        config.N_EPISODES_VAL = 4
        config.N_EPISODES_TEST = 8
        config.N_SHOT = 2
        config.N_QUERY = 3
        config.ABLATION_EPISODES = 4
        config.ABLATION_SHOTS = [1, 2]

    if not Path(config.DATA_ROOT).is_dir():
        print(f"Dataset not found at {config.DATA_ROOT}")
        print("Set FRUITVISION_ROOT, or run --verify-data for details.")
        return 1

    set_seed(args.seed)
    # Encode non-default choices in the directory name so variant runs are
    # distinguishable at a glance and cannot silently overwrite the baseline.
    variant = "-".join(
        part for part in (
            args.val_protocol if args.val_protocol != "seen_holdout" else "",
            args.norm_layer if args.norm_layer != "batchnorm" else "",
            args.freeze_mode if args.freeze_mode != "stem_layer1" else "",
            "bnfrozen" if args.freeze_bn_stats else "",
        ) if part
    )
    run_dir = make_run_dir(args.seed, args.smoke, variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Figures and checkpoints belong to THIS run, not to a shared directory.
    config.RESULTS_DIR = str(run_dir / "figures")
    config.THESIS_FIGS_DIR = str(run_dir / "figures")
    config.CHECKPOINT_DIR = str(run_dir / "checkpoints")
    ensure_dirs(config)

    print("=" * 72)
    print(f"  REPRODUCE THESIS   seed={args.seed}   device={device}"
          f"{'   [SMOKE]' if args.smoke else ''}")
    print(f"  run dir: {run_dir}")
    print("=" * 72)

    started = time.time()
    all_fruits = list(config.TRAIN_FRUITS) + list(config.TEST_FRUITS)
    metrics: dict = {
        "seed": args.seed,
        "smoke": args.smoke,
        "config": {
            k: getattr(config, k) for k in dir(config)
            if k.isupper() and not k.startswith("_")
        },
        "dataset": dataset_manifest(config.DATA_ROOT, all_fruits, config.CLASSES),
    }

    tf = build_transforms(config)

    # ---------------------------------------------------------------- #
    # 1. Data
    # ---------------------------------------------------------------- #
    #
    # Two validation protocols, because model selection currently optimises a
    # different objective than the thesis claims to care about.
    #
    #   seen_holdout (original): validate on held-out IMAGES of the same three
    #       species that are trained on. Early stopping and best-checkpoint
    #       selection therefore maximise SEEN-species accuracy (~0.94), while the
    #       thesis is about UNSEEN-species accuracy (~0.85). Those two are only
    #       weakly related, so the selected checkpoint is not the one that
    #       transfers best — it is the one that fits the training species best.
    #
    #   loso: hold out one training species entirely. Train on two, validate on
    #       episodes from the third, which the model has never seen. Validation
    #       then measures cross-species transfer directly, which is the quantity
    #       being optimised for. Costs one species of training data.
    #
    # The held-out test species (mango, orange) are untouched by both. Run both
    # and report the comparison: "does selecting on a transfer proxy improve
    # transfer?" is a methodological finding in its own right.
    if args.val_protocol == "loso":
        val_species = args.loso_species or config.TRAIN_FRUITS[-1]
        if val_species not in config.TRAIN_FRUITS:
            print(f"--loso-species must be one of {config.TRAIN_FRUITS}")
            return 1
        fit_species = [f for f in config.TRAIN_FRUITS if f != val_species]
        print(f"  validation protocol: LOSO — train on {fit_species}, "
              f"validate on unseen '{val_species}'")
        train_ds = FruitQualityDataset(
            config.DATA_ROOT, fit_species, config.CLASSES,
            support_transform=tf["support"], query_transform=tf["train"],
            split="all", seed=args.seed,
        )
        val_ds = FruitQualityDataset(
            config.DATA_ROOT, [val_species], config.CLASSES,
            transform=tf["eval"], split="all", seed=args.seed,
        )
        metrics_val_meta = {"protocol": "loso", "fit_species": fit_species,
                            "val_species": val_species}
    else:
        print("  validation protocol: seen_holdout — validating on held-out "
              "images of the training species")
        train_ds = FruitQualityDataset(
            config.DATA_ROOT, config.TRAIN_FRUITS, config.CLASSES,
            support_transform=tf["support"], query_transform=tf["train"],
            split="train", val_ratio=config.VAL_SPLIT_RATIO, seed=args.seed,
        )
        val_ds = FruitQualityDataset(
            config.DATA_ROOT, config.TRAIN_FRUITS, config.CLASSES,
            transform=tf["eval"], split="val",
            val_ratio=config.VAL_SPLIT_RATIO, seed=args.seed,
        )
        metrics_val_meta = {"protocol": "seen_holdout",
                            "fit_species": list(config.TRAIN_FRUITS),
                            "val_species": list(config.TRAIN_FRUITS)}
    test_ds = FruitQualityDataset(
        config.DATA_ROOT, config.TEST_FRUITS, config.CLASSES,
        transform=tf["eval"], split="all", seed=args.seed,
    )

    train_loader = EpisodicDataLoader(
        train_ds, config.N_SHOT, config.N_QUERY, config.N_EPISODES_TRAIN)
    val_loader = EpisodicDataLoader(
        val_ds, config.N_SHOT, config.N_QUERY, config.N_EPISODES_VAL)

    # ---------------------------------------------------------------- #
    # 2. Train the proposed model
    # ---------------------------------------------------------------- #
    print("\n[1/9] Training proposed model")
    model = PrototypicalNetwork(
        backbone=config.BACKBONE, embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED, dropout_rate=config.DROPOUT_RATE,
        temperature=config.TEMPERATURE,
        freeze_mode=args.freeze_mode, norm_layer=args.norm_layer,
        freeze_bn_stats=args.freeze_bn_stats,
    ).to(device)
    criterion = PrototypicalLoss(
        contrastive_weight=config.CONTRASTIVE_WEIGHT,
        label_smoothing=config.LABEL_SMOOTHING,
    )
    history = train_protonet(model, train_loader, val_loader, criterion, config, device)
    metrics["training_history"] = history
    metrics["validation"] = metrics_val_meta
    metrics["model"] = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "learned_temperature": float(model.temperature.detach().cpu()),
        "freeze_mode": args.freeze_mode,
        "norm_layer": args.norm_layer,
        "freeze_bn_stats": args.freeze_bn_stats,
        "hue_jitter": config.HUE_JITTER,
    }

    # ---------------------------------------------------------------- #
    # 3. Headline: unseen-species evaluation
    # ---------------------------------------------------------------- #
    print("\n[2/9] Cross-species evaluation on held-out species")
    metrics["cross_species"] = exp.test_on_unseen_fruits(
        model, test_ds, device, config, n_trials=3 if args.smoke else 5)

    # ---------------------------------------------------------------- #
    # 4. Confusion matrices + classification report (same predictions)
    # ---------------------------------------------------------------- #
    print("\n[3/9] Confusion matrices and classification report")
    metrics["confusion"] = viz.plot_confusion_matrices(model, test_ds, device, config)

    # ---------------------------------------------------------------- #
    # 5. Baselines + significance
    # ---------------------------------------------------------------- #
    print("\n[4/9] Few-shot baselines")
    baseline_results = exp.run_all_fsl_baselines(
        model, train_ds, val_ds, test_ds, config, device)

    # The two controls that decide whether episodic training is needed at all.
    # Folded into the same dict so they are covered by the paired significance
    # tests rather than sitting in a footnote.
    if not args.skip_transfer_controls:
        controls = exp.transfer_controls(
            train_ds, test_ds, tf, config, device,
            epochs=2 if args.smoke else 15)
        metrics["transfer_controls"] = controls
        baseline_results.update(controls)

    metrics["baselines"] = baseline_results

    print("\n[5/9] Paired significance tests (two-sided)")
    metrics["significance"] = exp.statistical_significance_tests(baseline_results)

    # ---------------------------------------------------------------- #
    # 6. Ablations
    # ---------------------------------------------------------------- #
    print("\n[6/9] Component ablation")
    metrics["component_ablation"] = exp.component_ablation(
        model, train_ds, val_ds, test_ds, config, device)

    print("\n[7/9] N-shot ablation")
    metrics["nshot_ablation"] = exp.ablation_n_shot(model, test_ds, device, config)
    viz.plot_ablation_nshot(metrics["nshot_ablation"], config)

    if not args.skip_expensive:
        print("\n[8/9] Backbone ablation")
        metrics["backbone_ablation"] = exp.backbone_ablation(
            train_ds, val_ds, test_ds, config, device)

        print("\n[9/9] Species-split cross-validation")
        metrics["species_cv"] = exp.species_split_cross_validation(
            tf, config, device, epochs=2 if args.smoke else 20)

        print("\n[+] Single-species baselines")
        metrics["single_species"] = exp.run_single_species_baselines(
            model, test_ds, tf, config, device)
    else:
        print("\n[8/9,9/9] Skipped (--skip-expensive)")

    # ---------------------------------------------------------------- #
    # 7. Remaining figures
    # ---------------------------------------------------------------- #
    viz.plot_training_history(history, config)
    viz.visualize_embeddings(model, test_ds, tf["eval"], device, config)

    # ---------------------------------------------------------------- #
    # 8. Persist
    # ---------------------------------------------------------------- #
    metrics["wall_clock_seconds"] = round(time.time() - started, 1)

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, cls=NumpyEncoder), encoding="utf-8")
    (run_dir / "env.json").write_text(
        json.dumps(capture_env(args.seed), indent=2), encoding="utf-8")

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(flatten_for_csv(metrics))

    write_demo_layout(run_dir, model, config, metrics)

    print("\n" + "=" * 72)
    print(f"  DONE in {metrics['wall_clock_seconds']}s -> {run_dir}")
    o = metrics["cross_species"]["overall"]
    print(f"  Cross-species accuracy: {o['mean']*100:.2f}% "
          f"+/- {o['ci_95_episode']*100:.2f}% (episode-level, n={o['n_episodes']})")
    print("=" * 72)
    return 0


def write_demo_layout(run_dir: Path, model, config, metrics: dict) -> None:
    """
    Save a self-describing checkpoint bundle for `scripts/demo_app.py`.

    Note on the fsgrade demo: it cannot consume this. `fsgrade/demo/discovery.py`
    scans a two-level `results/<experiment>/<run>/` tree, and
    `fsgrade/demo/checkpoints.py:load_for_inference` rebuilds an *fsgrade*
    architecture and calls `load_state_dict(..., strict=True)`. A `src/`
    PrototypicalNetwork has different parameter names and shapes, so that call
    fails by construction — matching the directory layout would not be enough,
    it would need an adapter inside fsgrade. Since fsgrade is scoped as an
    engineering deliverable rather than a results source, the thesis model gets
    its own small demo instead. See scripts/demo_app.py.
    """
    bundle = {
        "model_state_dict": model.state_dict(),
        "arch": {
            "backbone": config.BACKBONE,
            "embedding_dim": config.EMBEDDING_DIM,
            "dropout_rate": config.DROPOUT_RATE,
            "freeze_mode": model.encoder.freeze_mode,
            "norm_layer": model.encoder.norm_layer,
        },
        "temperature": float(model.temperature.detach().cpu()),
        "train_fruits": list(config.TRAIN_FRUITS),
        "test_fruits": list(config.TEST_FRUITS),
        "n_shot": config.N_SHOT,
        "n_query": config.N_QUERY,
        "classes": list(config.CLASSES),
        "seed": metrics["seed"],
        "cross_species_accuracy": metrics.get("cross_species", {}).get("overall", {}).get("mean"),
    }
    torch.save(bundle, run_dir / "model_bundle.pt")


# ====================================================================== #
#  Multi-seed aggregation
# ====================================================================== #

def aggregate(variant: str = "") -> int:
    """
    Combine per-seed runs into the seed-level spread the thesis must report.

    Episode-level CIs describe sampling noise for ONE trained model. They say
    nothing about how much the result moves when training is repeated. With a
    single seed there was no basis for any stability claim at all.

    Only runs sharing the same variant are combined — averaging a layernorm run
    with a batchnorm one would report a configuration difference as seed noise.
    """
    runs = sorted((REPO_ROOT / "results").glob("run_2*_seed*"))
    runs = [r for r in runs if (r / "metrics.json").is_file() and "smoke" not in r.name]

    def variant_of(run_dir: Path) -> str:
        # "run_<stamp>_seed<N>[-<variant>]" -> the part after the seed token
        tail = run_dir.name.split("_seed", 1)[1]
        return tail.split("-", 1)[1] if "-" in tail else ""

    available = sorted({variant_of(r) for r in runs})
    runs = [r for r in runs if variant_of(r) == variant]
    if len(runs) < 2:
        print(f"Need at least 2 completed non-smoke runs for variant "
              f"{variant or '<baseline>'}, found {len(runs)}.")
        if available:
            print("Variants present: " +
                  ", ".join(v or "<baseline>" for v in available))
            print("Select one with --aggregate-variant <name>.")
        return 1

    print(f"Variant: {variant or '<baseline>'}")
    per_seed = {}
    for run_dir in runs:
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        per_seed[m["seed"]] = m

    def collect(path):
        out = {}
        for seed, m in per_seed.items():
            node = m
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                out[seed] = node
        return out

    targets = {
        "cross_species_overall": ["cross_species", "overall", "mean"],
        **{f"cross_species_{f}": ["cross_species", "per_fruit", f, "mean"]
           for f in ("mango", "orange")},
    }

    summary = {"seeds": sorted(per_seed), "n_seeds": len(per_seed), "metrics": {}}
    print(f"Aggregating {len(per_seed)} seeds: {sorted(per_seed)}\n")
    for label, path in targets.items():
        values = collect(path)
        if len(values) < 2:
            continue
        arr = np.array(list(values.values()))
        entry = {
            "per_seed": values,
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        summary["metrics"][label] = entry
        print(f"  {label:<28} {entry['mean']*100:6.2f}% "
              f"+/- {entry['sd']*100:.2f} (sd over {len(values)} seeds)  "
              f"[{entry['min']*100:.2f}, {entry['max']*100:.2f}]")

    out = REPO_ROOT / "results" / (
        f"seed_aggregate{'_' + variant if variant else ''}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("Quote these as 'mean +/- sd over N seeds' — do not mix them with the "
          "episode-level CIs, which measure a different thing.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-protocol", choices=("seen_holdout", "loso"),
                   default="seen_holdout",
                   help="seen_holdout: validate on held-out images of the "
                        "training species (original). loso: hold out one "
                        "training species entirely and validate on it, so model "
                        "selection tracks cross-species transfer.")
    p.add_argument("--loso-species", default=None,
                   help="Which training species to hold out for --val-protocol "
                        "loso. Defaults to the last of TRAIN_FRUITS.")
    p.add_argument("--freeze-mode", choices=("stem_layer1", "legacy_substring"),
                   default="stem_layer1",
                   help="legacy_substring reproduces the over-broad freezing the "
                        "committed checkpoint was trained with.")
    p.add_argument("--norm-layer", choices=("batchnorm", "layernorm"),
                   default="batchnorm",
                   help="Projection-head normalisation. batchnorm matches the "
                        "committed checkpoint; layernorm is batch-size "
                        "independent and better suited to episodic training.")
    p.add_argument("--freeze-bn-stats", action="store_true",
                   help="Keep BatchNorm on running statistics during training. "
                        "Episodes are 32-40 images, far too few for reliable "
                        "batch statistics. Most likely single accuracy win.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny budget: proves the pipeline runs. Not reportable.")
    p.add_argument("--skip-expensive", action="store_true",
                   help="Skip backbone ablation, species CV and single-species runs.")
    p.add_argument("--skip-transfer-controls", action="store_true",
                   help="Skip the fine-tune and zero-shot transfer controls. "
                        "Not recommended: they are the thesis's most exposed "
                        "methodological omission.")
    p.add_argument("--verify-data", action="store_true",
                   help="Check the dataset and print its manifest hash, then exit.")
    p.add_argument("--aggregate", action="store_true",
                   help="Combine completed per-seed runs into seed-level spread.")
    p.add_argument("--aggregate-variant", default="",
                   help="Which variant to aggregate (e.g. 'loso', "
                        "'layernorm'). Default is the baseline configuration.")
    args = p.parse_args()

    if args.verify_data:
        return verify_data(Config())
    if args.aggregate:
        return aggregate(args.aggregate_variant)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
