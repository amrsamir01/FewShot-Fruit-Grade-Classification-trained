"""
Deterministic episode banks.

This is the keystone of the refactor. Episodes are **data**: generated once,
hashed, written to disk, and replayed identically by every method.

Why this matters
----------------
The original ``statistical_significance_tests`` ran ``scipy.stats.ttest_rel``
and ``wilcoxon`` on two lists of episode accuracies, truncated to a common
length. But each method was evaluated through its own freshly-constructed
``EpisodicDataLoader``, which resampled episodes from scratch. Episode *i* for
one method and episode *i* for another were different images of different
species. A paired test on unpaired data is not valid, and the same model
reported 86.3% / 86.0% / 86.1% in three places purely from resampling noise.

With a shared bank, all three become one number and the pairing is real.

Design notes
------------
* Support sets are **nested** across shot counts: a bank stores ``n_shot_max``
  support images per class, and ``with_shots(k)`` takes a per-class prefix. The
  1/3/5/10-shot ablation therefore runs on identical query sets with nested
  supports, making it a paired comparison and removing sampler noise from the
  trend line.
* Episodes are **balanced per species** by construction. ``random.choice(fruits)``
  in the original loader gave each species a random share of the 600 episodes,
  so per-species confidence intervals were computed over a random ``n``.
* Support and query are **disjoint by construction**, enforced at build time.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Sequence

import numpy as np

from fsgrade.data.index import ImageIndex
from fsgrade.seeding import derive_seed, rng_for

BANK_SCHEMA_VERSION = "1.0.0"

Split = Literal["train", "val", "test"]


class EpisodeAlignmentError(RuntimeError):
    """Raised when two result series cannot be paired episode-for-episode."""


# --------------------------------------------------------------------------- #
#  Fold specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FoldSpec:
    """Which species are seen during training, and which are held out."""

    fold_id: str
    protocol: str
    train_species: tuple[str, ...]
    test_species: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train_species) & set(self.test_species)
        if overlap:
            raise ValueError(
                f"Fold {self.fold_id!r} has species in both train and test: {sorted(overlap)}. "
                "Cross-species evaluation requires disjoint species sets."
            )
        if not self.train_species or not self.test_species:
            raise ValueError(f"Fold {self.fold_id!r} needs non-empty train and test species.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "protocol": self.protocol,
            "train_species": list(self.train_species),
            "test_species": list(self.test_species),
        }


def leave_one_species_out(species: Sequence[str]) -> list[FoldSpec]:
    """5 folds for 5 species: each species is the unseen one exactly once.

    The primary protocol. Yields a fold-level interval (n = n_species) which is
    the statistic that actually supports "generalises to an unseen species" --
    a much wider and more honest interval than one over episodes.
    """
    species = list(species)
    return [
        FoldSpec(
            fold_id=f"loso-{held}",
            protocol="leave_one_species_out",
            train_species=tuple(s for s in species if s != held),
            test_species=(held,),
        )
        for held in species
    ]


def species_combinations(species: Sequence[str], n_train: int = 3) -> list[FoldSpec]:
    """All C(n, n_train) train/test partitions. Secondary robustness analysis.

    Not comparable to LOSO: fewer training species and a different test
    composition, so the two must never share a results column.
    """
    from itertools import combinations

    species = sorted(species)
    folds: list[FoldSpec] = []
    for train in combinations(species, n_train):
        test = tuple(s for s in species if s not in train)
        folds.append(
            FoldSpec(
                fold_id="c{}-{}".format(n_train, "+".join(train)),
                protocol="species_combination",
                train_species=tuple(train),
                test_species=test,
            )
        )
    return folds


def fixed_split(train_species: Sequence[str], test_species: Sequence[str]) -> list[FoldSpec]:
    """The original Apple/Banana/Grape -> Mango/Orange split, kept as a case study."""
    return [
        FoldSpec(
            fold_id="fixed-" + "+".join(sorted(test_species)),
            protocol="fixed",
            train_species=tuple(train_species),
            test_species=tuple(test_species),
        )
    ]


def build_folds(cfg: dict[str, Any], species: Sequence[str]) -> list[FoldSpec]:
    """Construct folds from the ``protocol`` block of a config."""
    from fsgrade.config import get_in

    name = get_in(cfg, "protocol.name", "leave_one_species_out")
    if name == "leave_one_species_out":
        return leave_one_species_out(species)
    if name == "species_combination":
        return species_combinations(species, int(get_in(cfg, "protocol.combination.n_train", 3)))
    if name == "fixed":
        return fixed_split(
            get_in(cfg, "protocol.fixed.train", []),
            get_in(cfg, "protocol.fixed.test", []),
        )
    raise ValueError(f"Unknown protocol: {name!r}")


# --------------------------------------------------------------------------- #
#  Episode specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeSpec:
    """A fully-determined episode: exactly which files, in exactly which roles."""

    episode_id: str
    bank_id: str
    fold_id: str
    split: str
    species: str
    n_way: int
    class_names: tuple[str, ...]
    n_shot_max: int
    n_query_per_class: int
    support: tuple[tuple[int, str], ...]   # (label_index, relpath)
    query: tuple[tuple[int, str], ...]
    seed: int
    content_hash: str = ""

    # ------------------------------------------------------------------ #
    def compute_hash(self) -> str:
        h = hashlib.sha1()
        for label, rel in sorted(self.support):
            h.update(f"S{label}:{rel}\n".encode("utf-8"))
        for label, rel in sorted(self.query):
            h.update(f"Q{label}:{rel}\n".encode("utf-8"))
        return h.hexdigest()

    def with_shots(self, k: int) -> "EpisodeSpec":
        """Take the first ``k`` support images *per class*.

        Nested by construction, so k-shot results are paired across k.
        """
        if k > self.n_shot_max:
            raise ValueError(
                f"Episode {self.episode_id} stores {self.n_shot_max} shots; {k} requested."
            )
        if k == self.n_shot_max:
            return self
        kept: list[tuple[int, str]] = []
        for label in range(self.n_way):
            per_class = [(lbl, rel) for lbl, rel in self.support if lbl == label]
            kept.extend(per_class[:k])
        new = replace(self, support=tuple(kept), n_shot_max=k, content_hash="")
        return replace(new, content_hash=new.compute_hash())

    @property
    def n_shot(self) -> int:
        return self.n_shot_max

    def support_paths(self) -> list[str]:
        return [rel for _, rel in self.support]

    def query_paths(self) -> list[str]:
        return [rel for _, rel in self.query]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "bank_id": self.bank_id,
            "fold_id": self.fold_id,
            "split": self.split,
            "species": self.species,
            "n_way": self.n_way,
            "class_names": list(self.class_names),
            "n_shot_max": self.n_shot_max,
            "n_query_per_class": self.n_query_per_class,
            "support": [[int(l), r] for l, r in self.support],
            "query": [[int(l), r] for l, r in self.query],
            "seed": self.seed,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeSpec":
        return cls(
            episode_id=d["episode_id"],
            bank_id=d["bank_id"],
            fold_id=d["fold_id"],
            split=d["split"],
            species=d["species"],
            n_way=int(d["n_way"]),
            class_names=tuple(d["class_names"]),
            n_shot_max=int(d["n_shot_max"]),
            n_query_per_class=int(d["n_query_per_class"]),
            support=tuple((int(l), r) for l, r in d["support"]),
            query=tuple((int(l), r) for l, r in d["query"]),
            seed=int(d["seed"]),
            content_hash=d.get("content_hash", ""),
        )


# --------------------------------------------------------------------------- #
#  Episode bank
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeBank:
    meta: dict[str, Any]
    specs: list[EpisodeSpec] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        index: ImageIndex,
        *,
        fold: FoldSpec,
        split: str,
        species: Sequence[str],
        n_episodes: int,
        n_shot_max: int,
        n_query_per_class: int,
        seed: int,
        n_way: int | None = None,
        pools: dict[str, dict[str, list[str]]] | None = None,
        per_species_balanced: bool = True,
        query_prevalence: float | None = None,
    ) -> "EpisodeBank":
        """Sample ``n_episodes`` episodes, balanced across ``species``.

        ``pools`` overrides the index pools -- used to restrict training/validation
        episodes to their respective halves of the seen-species split.

        ``query_prevalence`` optionally skews the query set toward class 0. With
        the default balanced 15/15, accuracy and balanced accuracy are
        algebraically identical; under skew they separate, which is what makes
        MCC/PR-AUC/calibration informative rather than decorative.
        """
        species = list(species)
        classes = list(index.classes)
        n_way = n_way if n_way is not None else len(classes)

        if per_species_balanced and n_episodes % len(species) != 0:
            n_episodes = (n_episodes // len(species)) * len(species)

        # Query composition
        if query_prevalence is None:
            per_class_query = [n_query_per_class] * n_way
        else:
            total_q = n_query_per_class * n_way
            n_pos = max(1, int(round(total_q * query_prevalence)))
            n_pos = min(n_pos, total_q - 1)
            per_class_query = [total_q - n_pos, n_pos]

        # `is None`, never truthiness: an *empty* pools dict must be an error,
        # not a silent fallback to index.pools -- that fallback would draw from
        # every species, including the held-out one, and destroy the
        # cross-species guarantee.
        source_pools = index.pools if pools is None else pools

        # Capacity check before any sampling
        problems: list[str] = []
        for sp in species:
            for label, cls_name in enumerate(classes[:n_way]):
                pool = source_pools.get(sp, {}).get(cls_name, [])
                need = n_shot_max + per_class_query[label]
                if len(pool) < need:
                    problems.append(
                        f"  {sp}/{cls_name}: need {need} "
                        f"({n_shot_max} support + {per_class_query[label]} query), "
                        f"have {len(pool)}"
                    )
        if problems:
            from fsgrade.data.index import InsufficientImagesError

            raise InsufficientImagesError(
                f"Cannot build bank for fold {fold.fold_id!r} split {split!r}:\n"
                + "\n".join(problems)
            )

        bank_id = cls._make_bank_id(
            fold.fold_id, split, species, n_episodes, n_shot_max,
            n_query_per_class, seed, index.manifest_hash, query_prevalence,
        )

        per_species = n_episodes // len(species) if per_species_balanced else None
        specs: list[EpisodeSpec] = []
        counter = 0

        for sp_idx, sp in enumerate(species):
            count = per_species if per_species is not None else n_episodes
            for local in range(count):
                episode_id = f"{fold.fold_id}|{split}|{sp}|{local:06d}"
                ep_seed = derive_seed(bank_id, episode_id)
                rng = np.random.default_rng(ep_seed)

                support: list[tuple[int, str]] = []
                query: list[tuple[int, str]] = []
                for label, cls_name in enumerate(classes[:n_way]):
                    pool = source_pools[sp][cls_name]
                    need = n_shot_max + per_class_query[label]
                    # Disjoint by construction: one draw without replacement,
                    # then split. No fallback path that can reuse support images.
                    chosen = rng.choice(len(pool), size=need, replace=False)
                    for i in chosen[:n_shot_max]:
                        support.append((label, pool[int(i)]))
                    for i in chosen[n_shot_max:]:
                        query.append((label, pool[int(i)]))

                spec = EpisodeSpec(
                    episode_id=episode_id,
                    bank_id=bank_id,
                    fold_id=fold.fold_id,
                    split=split,
                    species=sp,
                    n_way=n_way,
                    class_names=tuple(classes[:n_way]),
                    n_shot_max=n_shot_max,
                    n_query_per_class=n_query_per_class,
                    support=tuple(support),
                    query=tuple(query),
                    seed=ep_seed,
                )
                specs.append(replace(spec, content_hash=spec.compute_hash()))
                counter += 1

        meta = {
            "schema_version": BANK_SCHEMA_VERSION,
            "bank_id": bank_id,
            "fold": fold.to_dict(),
            "split": split,
            "species": species,
            "n_episodes": len(specs),
            "n_way": n_way,
            "n_shot_max": n_shot_max,
            "n_query_per_class": n_query_per_class,
            "per_class_query": per_class_query,
            "query_prevalence": query_prevalence,
            "master_seed": seed,
            "per_species_balanced": per_species_balanced,
            "disjoint_support_query": True,
            "dataset_manifest_hash": index.manifest_hash,
            "class_names": classes[:n_way],
        }
        bank = cls(meta=meta, specs=specs)
        bank.verify_disjoint()
        return bank

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_bank_id(*components: Any) -> str:
        joined = "|".join(str(c) for c in components)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    @property
    def bank_id(self) -> str:
        return self.meta["bank_id"]

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[EpisodeSpec]:
        return iter(self.specs)

    def __getitem__(self, i: int) -> EpisodeSpec:
        return self.specs[i]

    # ------------------------------------------------------------------ #
    def verify_disjoint(self) -> None:
        """Assert support and query never share an image within an episode."""
        for spec in self.specs:
            s = set(spec.support_paths())
            q = set(spec.query_paths())
            overlap = s & q
            if overlap:  # pragma: no cover - defensive
                raise AssertionError(
                    f"Episode {spec.episode_id} has {len(overlap)} images in both "
                    f"support and query: {sorted(overlap)[:3]}"
                )

    def verify(self, index: ImageIndex) -> None:
        """Check the bank matches this dataset and every referenced file exists."""
        expected = self.meta.get("dataset_manifest_hash")
        if expected and expected != index.manifest_hash:
            raise ValueError(
                f"Bank {self.bank_id} was built against manifest {expected}, "
                f"but this dataset hashes to {index.manifest_hash}. "
                "The image set changed; rebuild the banks."
            )
        missing: list[str] = []
        for spec in self.specs:
            for rel in spec.support_paths() + spec.query_paths():
                if not index.abs_path(rel).exists():
                    missing.append(rel)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images referenced by bank {self.bank_id} are missing, "
                f"e.g. {missing[:3]}"
            )

    # ------------------------------------------------------------------ #
    def with_shots(self, k: int) -> "EpisodeBank":
        meta = dict(self.meta)
        meta["n_shot_max"] = k
        meta["derived_from"] = self.bank_id
        return EpisodeBank(meta=meta, specs=[s.with_shots(k) for s in self.specs])

    def filter(self, *, species: str | None = None) -> "EpisodeBank":
        specs = [s for s in self.specs if species is None or s.species == species]
        meta = dict(self.meta)
        meta["filtered_species"] = species
        meta["n_episodes"] = len(specs)
        return EpisodeBank(meta=meta, specs=specs)

    def head(self, n: int) -> "EpisodeBank":
        """Prefix subset, for methods too expensive to run on the full bank.

        Because it is a prefix of a shared bank, paired tests against it remain
        valid -- with reduced coverage, which the stats module records.
        """
        meta = dict(self.meta)
        meta["n_episodes"] = min(n, len(self.specs))
        meta["subset_of"] = self.bank_id
        return EpisodeBank(meta=meta, specs=self.specs[:n])

    def episode_ids(self) -> list[str]:
        return [s.episode_id for s in self.specs]

    def ids_hash(self) -> str:
        h = hashlib.sha256()
        for spec in self.specs:
            h.update(spec.episode_id.encode("utf-8"))
            h.update(b"\n")
        return "sha256:" + h.hexdigest()

    def content_hash(self) -> str:
        """Hash over episode *contents* -- proves two banks are truly identical."""
        h = hashlib.sha256()
        for spec in self.specs:
            h.update(spec.content_hash.encode("utf-8"))
            h.update(b"\n")
        return "sha256:" + h.hexdigest()

    def species_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for spec in self.specs:
            counts[spec.species] = counts.get(spec.species, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        jsonl = directory / f"{self.bank_id}.jsonl"
        with open(jsonl, "w", encoding="utf-8") as fh:
            for spec in self.specs:
                fh.write(json.dumps(spec.to_dict()) + "\n")
        meta = dict(self.meta)
        meta["ids_hash"] = self.ids_hash()
        meta["content_hash"] = self.content_hash()
        meta["species_counts"] = self.species_counts()
        with open(directory / f"{self.bank_id}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return jsonl

    @classmethod
    def load(cls, jsonl_path: str | pathlib.Path) -> "EpisodeBank":
        jsonl_path = pathlib.Path(jsonl_path)
        meta_path = jsonl_path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = jsonl_path.parent / (jsonl_path.stem + ".meta.json")
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        specs: list[EpisodeSpec] = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    specs.append(EpisodeSpec.from_dict(json.loads(line)))
        return cls(meta=meta, specs=specs)
