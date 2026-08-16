"""
Find trained runs and the methods inside them.

Scans ``results/`` for run directories, reads each one's ``metrics.json``,
``config.yaml`` and ``status.json``, and reports which methods can actually be
loaded for inference.

Three traps this module exists to absorb:

1. Checkpoint paths inside ``metrics.json`` were written on Windows, so they
   contain backslashes and are relative to whatever the CWD was at training
   time. They must be normalised and re-anchored to the run directory.
2. The supervised checkpoint is stored under the method name ``"supervised"``
   (hardcoded in ``fsgrade/methods/trained.py``), and is *shared* by
   ``zeroshot_supervised``, ``ncc_supervised`` and ``finetune_supervised``.
3. Several methods need no checkpoint at all -- ``sap``, ``clip_text_zeroshot``,
   ``nc_pixel``, ``chance`` are fully determined by frozen weights plus config,
   so they are available even before any experiment has been run.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Methods that require a trained checkpoint to run at all.
NEEDS_CHECKPOINT: frozenset[str] = frozenset({
    "ours", "protonet", "protonet_temp", "siamese", "matching",
    "zeroshot_supervised", "ncc_supervised", "finetune_supervised",
})

# Methods whose weights are frozen or absent -- runnable with no training.
NO_CHECKPOINT: frozenset[str] = frozenset({
    "chance", "nc_pixel", "sap", "clip_text_zeroshot",
    "dinov2_ncc", "dinov2_probe", "clip_ncc", "clip_probe",
})

# The three transfer methods all load the same file.
SHARED_SUPERVISED_KEY = "supervised"
SUPERVISED_FAMILY: frozenset[str] = frozenset({
    "zeroshot_supervised", "ncc_supervised", "finetune_supervised",
})

# Which encoder family a method consumes, for preprocessing.
METHOD_ENCODER: dict[str, str] = {
    "sap": "clip", "clip_text_zeroshot": "clip", "clip_ncc": "clip", "clip_probe": "clip",
    "dinov2_ncc": "dinov2", "dinov2_probe": "dinov2",
}


class DiscoveryError(RuntimeError):
    """Raised when a run directory cannot be interpreted."""


@dataclass
class MethodRef:
    """One method inside one run, and whether it can be loaded."""

    name: str
    family: str = "unknown"
    variant: str | None = None
    needs_checkpoint: bool = False
    checkpoint: pathlib.Path | None = None
    checkpoint_ok: bool = False
    available: bool = False
    reason: str = ""
    n_episodes: int = 0
    best_val: float | None = None
    kappa: float | None = None
    kappa_source: str = ""
    encoder_family: str = "imagenet"
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "variant": self.variant,
            "needs_checkpoint": self.needs_checkpoint,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_ok": self.checkpoint_ok,
            "available": self.available,
            "reason": self.reason,
            "n_episodes": self.n_episodes,
            "best_val": self.best_val,
            "kappa": self.kappa,
            "kappa_source": self.kappa_source,
            "encoder_family": self.encoder_family,
        }


@dataclass
class RunRef:
    """A single run directory."""

    run_id: str
    run_dir: pathlib.Path
    experiment: str = ""
    status: str = "unknown"
    created_utc: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    folds: list[dict[str, Any]] = field(default_factory=list)
    methods: dict[str, MethodRef] = field(default_factory=dict)
    chance_level: float = 0.5

    @property
    def classes(self) -> list[str]:
        from fsgrade.config import get_in

        return list(get_in(self.config, "data.classes", ["fresh", "rotten"]))

    def available_methods(self) -> list[MethodRef]:
        return [m for m in self.methods.values() if m.available]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "experiment": self.experiment,
            "status": self.status,
            "created_utc": self.created_utc,
            "config_hash": self.config_hash,
            "chance_level": self.chance_level,
            "protocol": {"folds": self.folds},
            "methods": [m.to_dict() for m in self.methods.values()],
        }


# --------------------------------------------------------------------------- #
#  Path normalisation
# --------------------------------------------------------------------------- #

_WINDOWS_SEP = re.compile(r"\\+")


def normalise_checkpoint_path(raw: str | None, run_dir: pathlib.Path) -> pathlib.Path | None:
    """Turn a stored checkpoint path into one that resolves on this machine.

    Stored paths look like ``results\\smoke\\<run>\\checkpoints\\<fold>\\<method>\\<hash>.pt``
    -- Windows separators, relative to the training-time CWD. We keep only the
    part from ``checkpoints/`` onward and re-anchor it to ``run_dir``, so a run
    directory that has been moved or copied still resolves.
    """
    if not raw:
        return None

    posix = _WINDOWS_SEP.sub("/", str(raw))
    parts = posix.split("/")

    if "checkpoints" in parts:
        tail = parts[parts.index("checkpoints"):]
        candidate = run_dir.joinpath(*tail)
        if candidate.exists():
            return candidate

    # Fall back to the literal path, in case it was absolute and is still valid.
    literal = pathlib.Path(posix)
    if literal.exists():
        return literal
    return run_dir.joinpath(*tail) if "checkpoints" in parts else literal


def find_checkpoint(
    run_dir: pathlib.Path, fold_id: str, method: str, spec_hash: str | None = None
) -> pathlib.Path | None:
    """Locate a checkpoint by convention when metrics.json does not name one.

    Handles the shared supervised checkpoint: the three transfer methods all
    read ``checkpoints/<fold>/supervised/<hash>.pt``.
    """
    key = SHARED_SUPERVISED_KEY if method in SUPERVISED_FAMILY else method
    directory = run_dir / "checkpoints" / fold_id / key
    if not directory.is_dir():
        return None
    if spec_hash:
        exact = directory / f"{spec_hash}.pt"
        if exact.exists():
            return exact
    candidates = sorted(directory.glob("*.pt"))
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
#  Scanning
# --------------------------------------------------------------------------- #

def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_config(run_dir: pathlib.Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001 - a malformed config must not kill discovery
        return {}


def load_run(run_dir: str | pathlib.Path) -> RunRef:
    """Read one run directory into a ``RunRef``."""
    run_dir = pathlib.Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise DiscoveryError(f"Not a directory: {run_dir}")

    metrics = _read_json(run_dir / "metrics.json")
    status = _read_json(run_dir / "status.json")
    config = _read_config(run_dir)

    run = RunRef(
        run_id=metrics.get("run_id") or run_dir.name,
        run_dir=run_dir,
        experiment=metrics.get("experiment") or status.get("experiment", ""),
        status=status.get("status", "unknown"),
        created_utc=status.get("started_utc", ""),
        config=config,
        config_hash=_read_json(run_dir / "config_hash.json").get("config_hash", ""),
        chance_level=float(metrics.get("chance_level", 0.5)),
    )

    protocol = metrics.get("protocol", {}) or {}
    fold_ids = protocol.get("folds", []) or []
    run.folds = [{"fold_id": f} for f in fold_ids]

    for name, block in (metrics.get("methods") or {}).items():
        run.methods[name] = _build_method_ref(name, block, run_dir, fold_ids)

    return run


def _build_method_ref(
    name: str, block: dict[str, Any], run_dir: pathlib.Path, fold_ids: list[str]
) -> MethodRef:
    state = (block or {}).get("state", {}) or {}
    base = name.split("@")[0]

    ref = MethodRef(
        name=name,
        family=state.get("family", "unknown"),
        variant=state.get("variant"),
        needs_checkpoint=base in NEEDS_CHECKPOINT,
        n_episodes=int((block or {}).get("n_episodes", 0)),
        best_val=state.get("best_val"),
        encoder_family=METHOD_ENCODER.get(base, "imagenet"),
        state=state,
    )

    kappa, source = _resolve_kappa(state)
    ref.kappa, ref.kappa_source = kappa, source

    if not ref.needs_checkpoint:
        ref.available = True
        ref.checkpoint_ok = True
        ref.reason = "no checkpoint required"
        return ref

    path = normalise_checkpoint_path(state.get("checkpoint"), run_dir)
    if path is None or not path.exists():
        spec_hash = state.get("train_spec_hash")
        for fold in fold_ids:
            candidate = find_checkpoint(run_dir, fold, base, spec_hash)
            if candidate is not None:
                path = candidate
                break

    ref.checkpoint = path
    if path is not None and path.exists():
        ref.checkpoint_ok = True
        ref.available = True
        ref.reason = ""
    else:
        ref.reason = (
            "checkpoint not found on disk"
            + (f" (recorded as {state['checkpoint']!r})" if state.get("checkpoint") else "")
        )
    return ref


def _resolve_kappa(state: dict[str, Any]) -> tuple[float | None, str]:
    """κ is read, never fitted.

    Calibrating κ on the demo species would leak target information and
    invalidate the cross-species claim, so the demo only ever reports the value
    a real run calibrated on *seen* species -- or says plainly that it is a
    default.
    """
    if "kappa" in state and state["kappa"] is not None:
        calib = state.get("calibration") or {}
        if calib:
            return float(state["kappa"]), (
                f"calibrated on seen species"
                + (f" ({calib.get('n_val_episodes')} val episodes)"
                   if calib.get("n_val_episodes") else "")
                + (f", val acc {calib['val_accuracy']:.3f}"
                   if isinstance(calib.get("val_accuracy"), (int, float)) else "")
            )
        return float(state["kappa"]), "from run state (calibration details absent)"

    calib = state.get("calibration") or {}
    if calib.get("parameter") == "kappa" and calib.get("value") is not None:
        return float(calib["value"]), "calibrated on seen species"

    return None, ""


def discover_runs(
    results_root: str | pathlib.Path,
    *,
    include_failed: bool = False,
) -> list[RunRef]:
    """Scan ``results/`` for run directories, newest first.

    A run directory is anything containing ``metrics.json``. Directories whose
    names begin with ``_`` (``_episodes``, ``_index``) are skipped.
    """
    root = pathlib.Path(results_root).expanduser()
    if not root.is_dir():
        return []

    runs: list[RunRef] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("_"):
            continue
        for candidate in sorted(experiment_dir.iterdir()):
            if not candidate.is_dir() or not (candidate / "metrics.json").exists():
                continue
            try:
                run = load_run(candidate)
            except DiscoveryError:
                continue
            if run.status == "failed" and not include_failed:
                continue
            runs.append(run)

    runs.sort(key=lambda r: r.created_utc or r.run_id, reverse=True)
    return runs


def resolve_latest(results_root: str | pathlib.Path, experiment: str) -> RunRef | None:
    """Follow an experiment's ``latest.json`` pointer, if present."""
    pointer = pathlib.Path(results_root) / experiment / "latest.json"
    if pointer.exists():
        run_dir = _read_json(pointer).get("run_dir")
        if run_dir:
            path = pathlib.Path(_WINDOWS_SEP.sub("/", run_dir))
            if path.is_dir():
                return load_run(path)
            # The pointer may have been written on another machine.
            fallback = pathlib.Path(results_root) / experiment / path.name
            if fallback.is_dir():
                return load_run(fallback)
    runs = [r for r in discover_runs(results_root) if r.experiment == experiment]
    return runs[0] if runs else None


def best_method_pair(runs: Iterable[RunRef]) -> tuple[str | None, str | None]:
    """Pick a sensible (K-shot arm, zero-shot arm) default for the UI.

    Prefers the thesis's own comparison -- ``ours`` against
    ``zeroshot_supervised`` -- then falls back to the CLIP pair, which needs no
    training at all, and finally to the pixel floor.
    """
    available: set[str] = set()
    for run in runs:
        available.update(m.name for m in run.available_methods())

    # Methods with no learned weights are always offerable, whether or not any
    # run mentions them -- SAP and its text control need no training at all.
    available |= NO_CHECKPOINT

    for kshot, zeroshot in (
        ("ours", "zeroshot_supervised"),          # the thesis's own comparison
        ("ncc_supervised", "zeroshot_supervised"),
        ("protonet", "zeroshot_supervised"),
        ("sap", "clip_text_zeroshot"),            # needs no checkpoint
        ("ncc_frozen", "clip_text_zeroshot"),
    ):
        if kshot in available and zeroshot in available:
            return kshot, zeroshot

    return "sap", "clip_text_zeroshot"
