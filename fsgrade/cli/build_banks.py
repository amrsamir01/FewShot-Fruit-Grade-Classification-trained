"""
Build and persist episode banks for every fold of a protocol.

Runs on the CPU in seconds and is the step that makes every downstream
comparison paired. Also fails fast, before any GPU time is spent, if a
species/class pool is too small for disjoint support and query sets.

    python -m fsgrade.cli.build_banks --config configs/experiment/main_loso.yaml
"""

from __future__ import annotations

import json
from typing import Any

from fsgrade.cli.common import banks_dir, base_parser, build_index, setup
from fsgrade.config import get_in
from fsgrade.data.episodes import EpisodeBank, build_folds
from fsgrade.data.index import split_seen_species
from fsgrade.logging_utils import log_table


def build_banks_for_fold(
    index: Any, fold: Any, cfg: dict[str, Any], *, logger: Any = None
) -> dict[str, EpisodeBank]:
    """Train/val banks over seen species, test bank over the held-out species."""
    episodes = get_in(cfg, "protocol.episodes", {})
    n_shot_max = int(episodes.get("n_shot_max", 10))
    n_query = int(episodes.get("n_query_per_class", 15))
    seed = int(episodes.get("bank_seed", 20260803))
    counts = episodes.get("n_episodes", {})
    balanced = bool(episodes.get("per_species_balanced", True))
    n_way = int(episodes.get("n_way", 2))

    assignment = split_seen_species(
        index, fold.train_species,
        val_ratio=float(get_in(cfg, "data.val_ratio", 0.15)),
        seed=int(get_in(cfg, "data.split_seed", 42)),
    )

    banks: dict[str, EpisodeBank] = {}
    for split, pools, species in (
        ("train", assignment.train, fold.train_species),
        ("val", assignment.val, fold.train_species),
        ("test", None, fold.test_species),
    ):
        banks[split] = EpisodeBank.build(
            index,
            fold=fold,
            split=split,
            species=list(species),
            n_episodes=int(counts.get(split, 500)),
            n_shot_max=n_shot_max,
            n_query_per_class=n_query,
            seed=seed,
            n_way=n_way,
            pools=pools,
            per_species_balanced=balanced,
        )
        if logger:
            logger.info(
                "  %-5s bank %s | %4d episodes | species=%s",
                split, banks[split].bank_id, len(banks[split]), list(species),
            )
    return banks


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Build deterministic episode banks for a protocol.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if banks exist")
    args = parser.parse_args(argv)

    ctx = setup(args)
    cfg, logger = ctx["cfg"], ctx["logger"]

    index = build_index(cfg)
    folds = build_folds(cfg, get_in(cfg, "data.species", []))
    out_dir = banks_dir(cfg)

    logger.info("protocol=%s | %d folds | manifest=%s",
                get_in(cfg, "protocol.name"), len(folds), index.manifest_hash[:22])

    registry: dict[str, dict[str, str]] = {}
    rows = []
    for fold in folds:
        logger.info("fold %s | train=%s | test=%s",
                    fold.fold_id, list(fold.train_species), list(fold.test_species))
        banks = build_banks_for_fold(index, fold, cfg, logger=logger)
        registry[fold.fold_id] = {}
        for split, bank in banks.items():
            bank.save(out_dir)
            registry[fold.fold_id][split] = bank.bank_id
            if split == "test":
                rows.append([
                    fold.fold_id, ",".join(fold.test_species), len(bank),
                    bank.bank_id, bank.content_hash()[7:19],
                ])

    log_table(logger, rows,
              ["fold", "test species", "episodes", "bank id", "content hash"],
              title="Test banks")

    registry_path = out_dir / f"registry_{get_in(cfg, 'protocol.name')}.json"
    with open(registry_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "protocol": get_in(cfg, "protocol.name"),
                "manifest_hash": index.manifest_hash,
                "bank_seed": get_in(cfg, "protocol.episodes.bank_seed"),
                "folds": registry,
            },
            fh, indent=2,
        )
    logger.info("wrote bank registry -> %s", registry_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
