"""
Run directories and crash-tolerant result serialization.

The governing rule of this module: **nothing is a number unless it is on disk**.
stdout is a view, never a result.

``StreamingCsvWriter`` flushes periodically, so a run that dies at episode 4,317
of 18,000 still leaves 4,317 usable rows behind plus a ``status.json`` saying
``running``. That is the structural fix for losing three experiments to an
IOPub rate limit -- the failure mode is now "partial results" rather than
"no results".
"""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import hashlib
import json
import os
import pathlib
import traceback
from typing import Any, Iterable, Sequence

RESULTS_SCHEMA_VERSION = "1.0.0"


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays and paths JSON-serializable."""
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", None) == 0:
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    return str(obj)


def write_json_atomic(path: str | pathlib.Path, obj: Any, *, indent: int = 2) -> pathlib.Path:
    """Write JSON via a temp file + atomic replace.

    A run interrupted mid-write leaves the previous good file intact rather than
    a truncated one.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of a config dict, insensitive to key ordering."""
    canonical = json.dumps(cfg, sort_keys=True, default=_json_default)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StreamingCsvWriter:
    """Append-only CSV writer that flushes every ``flush_every`` rows.

    Supports gzip for the large per-query table. Use as a context manager.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        columns: Sequence[str],
        *,
        gzip_output: bool = False,
        flush_every: int = 50,
    ) -> None:
        self.path = pathlib.Path(path)
        self.columns = list(columns)
        self.gzip_output = gzip_output
        self.flush_every = max(1, int(flush_every))
        self.n_rows = 0
        self._fh: Any = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.gzip_output:
            self._fh = gzip.open(self.path, "wt", newline="", encoding="utf-8")
        else:
            self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()
        return self

    def write_row(self, row: dict[str, Any]) -> None:
        if self._writer is None:  # pragma: no cover
            raise RuntimeError("StreamingCsvWriter used outside a context manager")
        self._writer.writerow(row)
        self.n_rows += 1
        if self.n_rows % self.flush_every == 0:
            self._fh.flush()

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.write_row(row)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None


class RunDirectory:
    """One directory per run: ``results/<experiment>/<stamp>_<git7>_s<seed>/``.

    Also maintains a ``latest.json`` pointer in the experiment directory so
    downstream scripts can say ``--run results/main_loso/latest`` without
    knowing timestamps.
    """

    def __init__(self, root: pathlib.Path, experiment: str, cfg: dict[str, Any]) -> None:
        self.root = root
        self.experiment = experiment
        self.cfg = cfg
        self.run_id = root.name
        self._started = utc_iso()

    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        results_root: str | pathlib.Path,
        experiment: str,
        cfg: dict[str, Any],
        *,
        seed: int | None = None,
        git_short: str | None = None,
    ) -> "RunDirectory":
        results_root = pathlib.Path(results_root)
        if git_short is None:
            from fsgrade.env_capture import git_info

            git_short = (git_info().get("short") or "nogit")
        if seed is None:
            seed = cfg.get("seed", 0)

        name = f"{utc_stamp()}_{git_short}_s{seed}"
        root = results_root / experiment / name
        root.mkdir(parents=True, exist_ok=True)
        run = cls(root, experiment, cfg)
        run.set_status("running")
        return run

    @classmethod
    def resolve(cls, path: str | pathlib.Path) -> pathlib.Path:
        """Resolve a run path, following a ``latest`` pointer if given one."""
        path = pathlib.Path(path)
        if path.name == "latest":
            pointer = path.parent / "latest.json"
            if not pointer.exists():
                raise FileNotFoundError(f"No latest.json in {path.parent}")
            with open(pointer, "r", encoding="utf-8") as fh:
                return pathlib.Path(json.load(fh)["run_dir"])
        return path

    # ------------------------------------------------------------------ #
    def path(self, *parts: str) -> pathlib.Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, obj: Any, *, schema_version: str | None = None) -> pathlib.Path:
        payload = obj
        if isinstance(obj, dict) and schema_version is not None:
            payload = {"schema_version": schema_version, **obj}
        return write_json_atomic(self.path(name), payload)

    def open_csv(
        self,
        name: str,
        columns: Sequence[str],
        *,
        gzip_output: bool = False,
        flush_every: int = 50,
    ) -> StreamingCsvWriter:
        return StreamingCsvWriter(
            self.path(name), columns, gzip_output=gzip_output, flush_every=flush_every
        )

    def write_config(self) -> pathlib.Path:
        from fsgrade.config import dump_config

        text = dump_config(self.cfg)
        path = self.path("config.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        write_json_atomic(self.path("config_hash.json"), {"config_hash": config_hash(self.cfg)})
        return path

    def write_env(self, *, include_pip_freeze: bool = True) -> pathlib.Path:
        from fsgrade.env_capture import capture_environment

        return self.write_json("env.json", capture_environment(include_pip_freeze=include_pip_freeze))

    def checkpoint_path(self, fold_id: str, method: str, spec_hash: str) -> pathlib.Path:
        """Checkpoints are keyed by (fold, method, train-spec).

        The original code always wrote ``checkpoints/best_model.pth``, so every
        experiment silently overwrote the previous one's weights.
        """
        p = self.root / "checkpoints" / fold_id / method / f"{spec_hash}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ #
    def set_status(self, status: str, **extra: Any) -> pathlib.Path:
        payload = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": status,
            "started_utc": self._started,
            "updated_utc": utc_iso(),
            **extra,
        }
        return write_json_atomic(self.path("status.json"), payload)

    def fail(self, exc: BaseException) -> pathlib.Path:
        return self.set_status(
            "failed",
            ended_utc=utc_iso(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    def finalize(self, **extra: Any) -> pathlib.Path:
        status_path = self.set_status("ok", ended_utc=utc_iso(), **extra)
        write_json_atomic(
            self.root.parent / "latest.json",
            {
                "run_dir": str(self.root),
                "run_id": self.run_id,
                "experiment": self.experiment,
                "updated_utc": utc_iso(),
            },
        )
        return status_path

    def __repr__(self) -> str:  # pragma: no cover
        return f"RunDirectory({self.root})"
