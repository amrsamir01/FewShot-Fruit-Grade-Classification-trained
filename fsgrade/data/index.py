"""
Dataset scanning and manifest construction.

Produces a flat, hashable manifest of every image, keyed by a path *relative to*
``data_root``. Relative paths make episode banks portable: a bank built on one
machine can be verified and replayed on another with a different data root.

Integrity checks live here too (duplicate detection, corrupt-file scan), because
they are cheap on the CPU and expensive to discover halfway through a GPU sweep.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

MANIFEST_SCHEMA_VERSION = "1.0.0"

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


class InsufficientImagesError(RuntimeError):
    """Raised when a species/class pool is too small to build the requested episodes.

    Deliberately raised at *bank-build* time -- on the CPU, in seconds -- rather
    than discovered inside a training loop. The original ``get_episode`` instead
    fell back to ``random.choices(all_images, ...)``, sampling with replacement
    from the full pool including the support images it had just drawn, which
    silently leaked support into query.
    """


@dataclass(frozen=True)
class ImageRecord:
    relpath: str
    species: str
    quality: str
    label_index: int
    bytes: int

    def to_row(self) -> dict[str, Any]:
        return {
            "relpath": self.relpath,
            "species": self.species,
            "quality": self.quality,
            "label_index": self.label_index,
            "bytes": self.bytes,
        }


@dataclass
class ImageIndex:
    """All images grouped by (species, quality), with a content hash."""

    data_root: pathlib.Path
    species: list[str]
    classes: list[str]
    records: list[ImageRecord]
    pools: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    manifest_hash: str = ""

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        data_root: str | pathlib.Path,
        species: Sequence[str],
        classes: Sequence[str],
        *,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ) -> "ImageIndex":
        data_root = pathlib.Path(data_root).expanduser().resolve()
        exts = tuple(e.lower() for e in extensions)

        records: list[ImageRecord] = []
        pools: dict[str, dict[str, list[str]]] = {}

        for sp in species:
            pools[sp] = {}
            for label_index, cls_name in enumerate(classes):
                folder = data_root / sp / cls_name
                if not folder.is_dir():
                    pools[sp][cls_name] = []
                    continue
                # sorted() keeps the scan deterministic regardless of filesystem
                # ordering -- essential for reproducible splits.
                files = sorted(
                    (p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in exts),
                    key=lambda p: p.name,
                )
                rels: list[str] = []
                for p in files:
                    rel = f"{sp}/{cls_name}/{p.name}"
                    rels.append(rel)
                    records.append(
                        ImageRecord(
                            relpath=rel,
                            species=sp,
                            quality=cls_name,
                            label_index=label_index,
                            bytes=p.stat().st_size,
                        )
                    )
                pools[sp][cls_name] = rels

        index = cls(
            data_root=data_root,
            species=list(species),
            classes=list(classes),
            records=records,
            pools=pools,
        )
        index.manifest_hash = index.compute_hash()
        return index

    # ------------------------------------------------------------------ #
    def compute_hash(self) -> str:
        """Hash over the sorted relpath list -- identifies the exact image set."""
        h = hashlib.sha256()
        for rel in sorted(r.relpath for r in self.records):
            h.update(rel.encode("utf-8"))
            h.update(b"\n")
        return "sha256:" + h.hexdigest()

    def pool(self, species: str, quality: str) -> list[str]:
        return self.pools.get(species, {}).get(quality, [])

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            sp: {cls: len(self.pool(sp, cls)) for cls in self.classes}
            for sp in self.species
        }

    def abs_path(self, relpath: str) -> pathlib.Path:
        return self.data_root / relpath

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    def require_capacity(
        self,
        species: Sequence[str],
        *,
        n_shot: int,
        n_query_per_class: int,
    ) -> None:
        """Fail fast if any species/class pool is too small for disjoint episodes."""
        need = n_shot + n_query_per_class
        problems: list[str] = []
        for sp in species:
            for cls in self.classes:
                have = len(self.pool(sp, cls))
                if have < need:
                    problems.append(
                        f"  {sp}/{cls}: need {need} "
                        f"({n_shot} support + {n_query_per_class} query), have {have}"
                    )
        if problems:
            raise InsufficientImagesError(
                "Cannot build episodes with disjoint support/query sets:\n"
                + "\n".join(problems)
                + "\n\nReduce n_shot or n_query_per_class, or supply more images."
            )

    # ------------------------------------------------------------------ #
    def find_duplicates(self, *, chunk_bytes: int = 65536) -> dict[str, list[str]]:
        """Group images by exact content hash. Non-trivial groups are duplicates.

        Near-duplicates across the fresh/rotten boundary would inflate every
        reported number, so this is worth knowing before drawing conclusions.
        """
        by_hash: dict[str, list[str]] = defaultdict(list)
        for rec in self.records:
            h = hashlib.sha1()
            with open(self.abs_path(rec.relpath), "rb") as fh:
                while True:
                    block = fh.read(chunk_bytes)
                    if not block:
                        break
                    h.update(block)
            by_hash[h.hexdigest()].append(rec.relpath)
        return {k: sorted(v) for k, v in by_hash.items() if len(v) > 1}

    def find_corrupt(self) -> list[dict[str, str]]:
        """Return images PIL cannot open. Empty list is the happy path."""
        from PIL import Image

        bad: list[dict[str, str]] = []
        for rec in self.records:
            try:
                with Image.open(self.abs_path(rec.relpath)) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001 - we want every failure kind
                bad.append({"relpath": rec.relpath, "error": f"{type(exc).__name__}: {exc}"})
        return bad

    # ------------------------------------------------------------------ #
    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "data_root": str(self.data_root),
            "species": self.species,
            "classes": self.classes,
            "n_images": len(self.records),
            "counts": self.counts(),
            "manifest_hash": self.manifest_hash,
        }

    def write_manifest(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write ``dataset_manifest.csv`` plus a ``.meta.json`` sidecar."""
        import csv

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["relpath", "species", "quality", "label_index", "bytes"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_row())

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_meta(), fh, indent=2)
        return path


# --------------------------------------------------------------------------- #
#  Seen-species train/val split
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SplitAssignment:
    """Deterministic train/val partition of the *seen* species' images.

    Test species are never touched here -- they are held out whole by the
    protocol, which is what makes this cross-species rather than cross-class.
    """

    train: dict[str, dict[str, list[str]]]
    val: dict[str, dict[str, list[str]]]

    def pool(self, split: str, species: str, quality: str) -> list[str]:
        table = self.train if split == "train" else self.val
        return table.get(species, {}).get(quality, [])


def split_seen_species(
    index: ImageIndex,
    seen_species: Sequence[str],
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitAssignment:
    """Partition each seen species/class pool into train and val.

    Uses an independent ``default_rng`` per (species, class) keyed by the split
    seed, so adding a species does not perturb the split of existing ones --
    which matters when running leave-one-species-out folds that differ in which
    species are seen.
    """
    from fsgrade.seeding import rng_for

    train: dict[str, dict[str, list[str]]] = {}
    val: dict[str, dict[str, list[str]]] = {}

    for sp in seen_species:
        train[sp] = {}
        val[sp] = {}
        for cls in index.classes:
            pool = list(index.pool(sp, cls))  # already sorted -> deterministic
            if not pool:
                train[sp][cls] = []
                val[sp][cls] = []
                continue
            rng = rng_for("split", seed, sp, cls)
            perm = rng.permutation(len(pool))
            n_val = max(1, int(len(pool) * val_ratio)) if val_ratio > 0 else 0
            val_idx = set(perm[:n_val].tolist())
            val[sp][cls] = [pool[i] for i in sorted(val_idx)]
            train[sp][cls] = [pool[i] for i in range(len(pool)) if i not in val_idx]

    assignment = SplitAssignment(train=train, val=val)
    _assert_disjoint(assignment, seen_species, index.classes)
    return assignment


def _assert_disjoint(
    assignment: SplitAssignment, species: Iterable[str], classes: Iterable[str]
) -> None:
    """Hard guarantee of no train/val image overlap. Cheap; run every time."""
    for sp in species:
        for cls in classes:
            tr = set(assignment.pool("train", sp, cls))
            va = set(assignment.pool("val", sp, cls))
            overlap = tr & va
            if overlap:  # pragma: no cover - defensive
                raise AssertionError(
                    f"train/val leakage in {sp}/{cls}: {len(overlap)} shared images, "
                    f"e.g. {sorted(overlap)[:3]}"
                )
