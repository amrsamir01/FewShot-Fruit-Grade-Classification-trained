"""
Configuration loading, composition and validation.

Design goals
------------
* One YAML per concern, composed via a ``defaults:`` list (base -> data ->
  protocol -> method -> experiment), merged left-to-right.
* Dotted CLI overrides (``--set model.freeze.stage=2``) applied last.
* ``data_root`` is *resolved* through an explicit, logged chain rather than
  hardcoded, and the winning source is recorded in the run artifacts.
* Validation fails loudly, on the CPU, before a single GPU hour is spent.

The old ``config.py::Config`` hardcoded ``~/Desktop/Amr Samir`` which does not
exist on every machine, and ``FruitQualityDataset`` merely *printed* a warning
for a missing folder and carried on with an empty image list -- which then blew
up deep inside a training loop. Both behaviours are replaced here.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

try:  # PyYAML is the only config dependency, and it is optional at import time
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover - exercised only on incomplete installs
    _HAVE_YAML = False


CONFIG_SCHEMA_VERSION = "1.0.0"

# Environment variable consulted during data-root resolution.
DATA_ROOT_ENV_VAR = "FRUITVISION_DATA_ROOT"

# Probed only if nothing more explicit was supplied. Ordered most- to
# least-likely; every probe is logged so resolution is never mysterious.
_CANDIDATE_DATA_ROOTS: tuple[str, ...] = (
    "D:/Datasets/FruitVision",
    "./data/FruitVision",
    "../FruitVision",
    "~/Documents/datasets/FruitVision",
    "~/Desktop/AmrSamir_MasterThesis/FruitVision",
    "~/Desktop/Amr Samir/FruitVision",
)


class ConfigError(RuntimeError):
    """Raised when a configuration file is malformed or internally inconsistent."""


class DataRootError(RuntimeError):
    """Raised when the dataset root cannot be resolved or fails validation."""


# --------------------------------------------------------------------------- #
#  YAML composition
# --------------------------------------------------------------------------- #

def _require_yaml() -> None:
    if not _HAVE_YAML:
        raise ConfigError(
            "PyYAML is required to read configuration files. "
            "Install it with:  pip install pyyaml"
        )


def _read_yaml(path: pathlib.Path) -> dict[str, Any]:
    _require_yaml()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping at top level: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings merge key-wise; every other type (including lists) is replaced
    wholesale. Replacing lists rather than concatenating them is deliberate --
    it means a method config can *narrow* ``protocol.episodes.shots_to_evaluate``
    instead of silently appending to it.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str | pathlib.Path,
    *,
    config_dir: str | pathlib.Path | None = None,
    _seen: set[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Load a YAML config, resolving its ``defaults:`` list depth-first.

    ``defaults`` entries are paths relative to ``config_dir`` (the directory
    holding the top-level config), with the ``.yaml`` suffix optional.
    """
    path = pathlib.Path(path).expanduser().resolve()
    config_dir = pathlib.Path(config_dir).expanduser().resolve() if config_dir else path.parent
    _seen = _seen if _seen is not None else set()

    if path in _seen:
        raise ConfigError(f"Circular 'defaults' reference detected at: {path}")
    _seen.add(path)

    raw = _read_yaml(path)
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ConfigError(f"'defaults' must be a list of paths in {path}")

    merged: dict[str, Any] = {}
    for entry in defaults:
        child = pathlib.Path(str(entry))
        if child.suffix not in (".yaml", ".yml"):
            child = child.with_suffix(".yaml")
        child_path = child if child.is_absolute() else (config_dir / child)
        merged = deep_merge(merged, load_config(child_path, config_dir=config_dir, _seen=_seen))

    return deep_merge(merged, raw)


# --------------------------------------------------------------------------- #
#  Dotted overrides
# --------------------------------------------------------------------------- #

def _coerce_scalar(text: str) -> Any:
    """Convert a CLI string to the most specific sensible Python type.

    JSON first (handles ints, floats, bools, null, lists, nested objects), then
    a comma-separated-list fallback, then the raw string.
    """
    lowered = text.strip().lower()
    if lowered in ("none", "null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    if "," in text:
        return [_coerce_scalar(part) for part in text.split(",")]
    return text


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides to a nested config dict.

    Unknown keys are an error, not a silent no-op: a typo in an override is one
    of the easiest ways to believe you ran a sweep that you did not run.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be of the form key.path=value, got: {item!r}")
        dotted, _, raw_value = item.partition("=")
        keys = dotted.strip().split(".")
        node: Any = out
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(
                    f"Override path {dotted!r} does not exist in the config "
                    f"(failed at segment {key!r})"
                )
            node = node[key]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(
                f"Override path {dotted!r} does not exist in the config "
                f"(failed at final segment {leaf!r})"
            )
        node[leaf] = _coerce_scalar(raw_value)
    return out


def get_in(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``cfg['a']['b']['c']`` from the dotted path ``'a.b.c'``."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
#  Data-root resolution and validation
# --------------------------------------------------------------------------- #

@dataclass
class ResolvedDataRoot:
    path: pathlib.Path
    source: str
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "source": self.source, "trace": list(self.trace)}


@dataclass
class DataRootReport:
    """Outcome of validating a dataset root against the expected layout."""

    root: pathlib.Path
    species: list[str]
    classes: list[str]
    counts: dict[str, dict[str, int]]
    missing_dirs: list[str]
    empty_dirs: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_dirs and not self.empty_dirs

    @property
    def total_images(self) -> int:
        return sum(sum(per_class.values()) for per_class in self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "species": list(self.species),
            "classes": list(self.classes),
            "counts": self.counts,
            "missing_dirs": list(self.missing_dirs),
            "empty_dirs": list(self.empty_dirs),
            "total_images": self.total_images,
            "ok": self.ok,
        }


def resolve_data_root(
    cli_value: str | None,
    cfg: dict[str, Any] | None = None,
    *,
    local_config: str | pathlib.Path | None = "configs/local.yaml",
    candidates: Sequence[str] = _CANDIDATE_DATA_ROOTS,
) -> ResolvedDataRoot:
    """Resolve the dataset root. First hit wins; every step is recorded.

    Precedence
    ----------
    1. explicit CLI value
    2. ``$FRUITVISION_DATA_ROOT``
    3. ``configs/local.yaml`` (git-ignored, per-machine)
    4. ``paths.data_root`` in the composed config
    5. probed candidate list
    """
    trace: list[str] = []

    def _try(raw: Any, source: str) -> ResolvedDataRoot | None:
        if raw in (None, ""):
            return None
        path = pathlib.Path(str(raw)).expanduser()
        if path.is_dir():
            trace.append(f"{source}: {path}  [OK]")
            return ResolvedDataRoot(path=path.resolve(), source=source, trace=trace)
        trace.append(f"{source}: {path}  [not a directory]")
        return None

    hit = _try(cli_value, "cli")
    if hit:
        return hit

    hit = _try(os.environ.get(DATA_ROOT_ENV_VAR), f"env:{DATA_ROOT_ENV_VAR}")
    if hit:
        return hit

    if local_config is not None:
        local_path = pathlib.Path(local_config).expanduser()
        if local_path.exists() and _HAVE_YAML:
            try:
                local_cfg = _read_yaml(local_path)
            except ConfigError:
                local_cfg = {}
            hit = _try(get_in(local_cfg, "paths.data_root"), f"local:{local_path}")
            if hit:
                return hit
        else:
            trace.append(f"local:{local_path}  [absent]")

    if cfg is not None:
        hit = _try(get_in(cfg, "paths.data_root"), "config:paths.data_root")
        if hit:
            return hit

    for candidate in candidates:
        hit = _try(candidate, f"candidate:{candidate}")
        if hit:
            return hit

    raise DataRootError(
        "Could not resolve the FruitVision dataset root.\n"
        "Resolution trace:\n  " + "\n  ".join(trace or ["(nothing tried)"]) + "\n\n"
        "Fix by any one of:\n"
        "  * pass --data-root <path>\n"
        f"  * set the {DATA_ROOT_ENV_VAR} environment variable\n"
        "  * copy configs/local.yaml.example to configs/local.yaml and edit it\n"
    )


def validate_data_root(
    root: str | pathlib.Path,
    species: Sequence[str],
    classes: Sequence[str],
    *,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    raise_on_error: bool = True,
) -> DataRootReport:
    """Check that every ``<root>/<species>/<class>`` directory exists and is non-empty.

    Counts use the *same* extension filter as the loader, so the reported number
    is exactly what training will see. (On Windows this matters: stray
    ``desktop.ini`` files inflate a naive ``len(os.listdir())`` by one.)
    """
    root = pathlib.Path(root).expanduser()
    exts = tuple(e.lower() for e in extensions)

    counts: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    empty: list[str] = []

    for sp in species:
        counts[sp] = {}
        for cls in classes:
            folder = root / sp / cls
            rel = f"{sp}/{cls}"
            if not folder.is_dir():
                missing.append(rel)
                counts[sp][cls] = 0
                continue
            n = sum(
                1
                for entry in folder.iterdir()
                if entry.is_file() and entry.suffix.lower() in exts
            )
            counts[sp][cls] = n
            if n == 0:
                empty.append(rel)

    report = DataRootReport(
        root=root,
        species=list(species),
        classes=list(classes),
        counts=counts,
        missing_dirs=missing,
        empty_dirs=empty,
    )

    if raise_on_error and not report.ok:
        lines = [f"Dataset layout validation failed for root: {root}"]
        if missing:
            lines.append("  Missing directories:")
            lines += [f"    - {root / m}" for m in missing]
        if empty:
            lines.append("  Directories with no matching images:")
            lines += [f"    - {root / e}  (extensions: {', '.join(exts)})" for e in empty]
        lines.append("")
        lines.append("Expected layout:  <data_root>/<species>/<class>/*.jpg")
        raise DataRootError("\n".join(lines))

    return report


# --------------------------------------------------------------------------- #
#  Top-level entry point
# --------------------------------------------------------------------------- #

def build_config(
    config_path: str | pathlib.Path,
    *,
    overrides: Sequence[str] = (),
    data_root: str | None = None,
    validate_data: bool = True,
) -> dict[str, Any]:
    """Compose a config, apply overrides, resolve and validate the data root.

    Returns the fully-resolved config dict. ``paths.data_root`` is rewritten to
    the absolute resolved path and ``paths.data_root_source`` records how it was
    found, so the serialized ``config.yaml`` in a run directory is sufficient to
    reproduce the run on another machine.
    """
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("schema_version", CONFIG_SCHEMA_VERSION)

    resolved = resolve_data_root(data_root, cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["data_root"] = str(resolved.path)
    cfg["paths"]["data_root_source"] = resolved.source
    cfg["paths"]["data_root_trace"] = resolved.trace

    if validate_data:
        species = get_in(cfg, "data.species") or []
        classes = get_in(cfg, "data.classes") or []
        extensions = get_in(cfg, "data.extensions") or (".jpg", ".jpeg", ".png", ".bmp")
        report = validate_data_root(resolved.path, species, classes, extensions=extensions)
        cfg["paths"]["data_root_report"] = report.to_dict()

    return cfg


def dump_config(cfg: dict[str, Any]) -> str:
    """Serialize a resolved config back to YAML (or JSON if PyYAML is absent)."""
    if _HAVE_YAML:
        return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    return json.dumps(cfg, indent=2, sort_keys=False, default=str)
