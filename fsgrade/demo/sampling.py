"""
Draw support and query images from the local dataset.

Dragging ten files in front of an examiner is slow and error-prone. One click
that fills the tray with 5 fresh + 5 rotten mangoes is the difference between a
demo that flows and one that stalls.

Support and query draws are kept **disjoint** — grading a query image that is
also in the support set would be a leak on stage, and someone in the room will
be looking for exactly that.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class DatasetView:
    """A light index over ``<root>/<species>/<class>/*``."""

    root: pathlib.Path
    classes: list[str]
    pools: dict[str, dict[str, list[str]]]

    @property
    def available(self) -> bool:
        return bool(self.pools)

    def species(self) -> list[str]:
        return sorted(self.pools)

    def counts(self, species: str) -> dict[str, int]:
        return {c: len(self.pools.get(species, {}).get(c, [])) for c in self.classes}

    def abs_path(self, relpath: str) -> pathlib.Path:
        return self.root / relpath

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.root),
            "available": self.available,
            "classes": list(self.classes),
            "species": [
                {"name": s, "counts": self.counts(s), "total": sum(self.counts(s).values())}
                for s in self.species()
            ],
        }


def build_dataset_view(
    data_root: str | pathlib.Path | None,
    species: Sequence[str],
    classes: Sequence[str],
    *,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> DatasetView:
    """Scan the dataset if it is present. Absence is not an error."""
    classes = list(classes)
    if not data_root:
        return DatasetView(pathlib.Path("."), classes, {})

    root = pathlib.Path(data_root).expanduser()
    if not root.is_dir():
        return DatasetView(root, classes, {})

    exts = tuple(e.lower() for e in extensions)
    pools: dict[str, dict[str, list[str]]] = {}
    for sp in species:
        per_class: dict[str, list[str]] = {}
        for cls in classes:
            folder = root / sp / cls
            if not folder.is_dir():
                continue
            files = sorted(
                p.name for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
            if files:
                per_class[cls] = [f"{sp}/{cls}/{name}" for name in files]
        if per_class:
            pools[sp] = per_class
    return DatasetView(root, classes, pools)


class SamplingError(RuntimeError):
    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation


def sample_relpaths(
    view: DatasetView,
    species: str,
    *,
    n_per_class: int = 5,
    exclude: Sequence[str] = (),
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Draw ``n_per_class`` images per class, avoiding ``exclude``.

    ``exclude`` carries whatever is already in the session, which is what keeps
    support and query disjoint.
    """
    if not view.available:
        raise SamplingError(
            "No local dataset is configured, so images cannot be sampled.",
            remediation="Start the demo with --data-root, or drag images in instead.",
        )
    if species not in view.pools:
        raise SamplingError(
            f"{species!r} is not in the local dataset "
            f"(have: {', '.join(view.species())})",
            remediation="Pick one of the listed species, or upload your own images.",
        )

    rng = np.random.default_rng(seed)
    blocked = set(exclude)
    picked: dict[str, list[str]] = {}

    for cls in view.classes:
        pool = [p for p in view.pools[species].get(cls, []) if p not in blocked]
        if len(pool) < n_per_class:
            raise SamplingError(
                f"only {len(pool)} unused {species}/{cls} images remain, "
                f"{n_per_class} requested",
                remediation="Ask for fewer images, or clear the current set.",
            )
        idx = rng.choice(len(pool), size=n_per_class, replace=False)
        chosen = [pool[int(i)] for i in idx]
        picked[cls] = chosen
        blocked.update(chosen)          # disjoint across classes too

    return picked


def label_of(relpath: str, classes: Sequence[str]) -> int:
    """Recover the class index from ``species/class/file``."""
    parts = relpath.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2] in classes:
        return list(classes).index(parts[-2])
    return -1
