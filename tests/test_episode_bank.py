"""
Episode bank tests: determinism, disjointness, nesting, balance.

These properties are what make every downstream paired comparison valid.
"""

from __future__ import annotations

import pytest

from fsgrade.data.episodes import (
    EpisodeAlignmentError,
    EpisodeBank,
    FoldSpec,
    leave_one_species_out,
    species_combinations,
)
from fsgrade.data.index import InsufficientImagesError, split_seen_species


def _bank(index, fold, **kw):
    params = dict(
        fold=fold, split="test", species=list(fold.test_species),
        n_episodes=20, n_shot_max=5, n_query_per_class=5, seed=999,
    )
    params.update(kw)
    return EpisodeBank.build(index, **params)


def test_same_seed_gives_identical_episodes(index, fold):
    a, b = _bank(index, fold), _bank(index, fold)
    assert a.bank_id == b.bank_id
    assert a.content_hash() == b.content_hash()
    assert a.ids_hash() == b.ids_hash()
    for x, y in zip(a, b):
        assert x.support == y.support
        assert x.query == y.query


def test_different_seed_gives_different_episodes(index, fold):
    a, b = _bank(index, fold, seed=1), _bank(index, fold, seed=2)
    assert a.content_hash() != b.content_hash()


def test_support_and_query_are_disjoint(index, fold):
    for spec in _bank(index, fold):
        assert not (set(spec.support_paths()) & set(spec.query_paths()))


def test_no_duplicate_images_within_an_episode(index, fold):
    """The original fell back to random.choices (with replacement) on small pools."""
    for spec in _bank(index, fold):
        paths = spec.support_paths() + spec.query_paths()
        assert len(paths) == len(set(paths))


def test_episodes_balanced_across_species(index, fold):
    bank = _bank(index, fold, n_episodes=20)
    counts = bank.species_counts()
    assert set(counts) == set(fold.test_species)
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"


def test_shots_are_nested_prefixes(index, fold):
    bank = _bank(index, fold, n_shot_max=5)
    b5, b3, b1 = bank.with_shots(5), bank.with_shots(3), bank.with_shots(1)
    for s5, s3, s1 in zip(b5, b3, b1):
        assert set(s1.support_paths()) <= set(s3.support_paths())
        assert set(s3.support_paths()) <= set(s5.support_paths())
        # Query sets are identical across k -> the shot ablation is paired.
        assert s1.query_paths() == s3.query_paths() == s5.query_paths()


def test_with_shots_yields_k_per_class(index, fold):
    bank = _bank(index, fold, n_shot_max=5)
    for k in (1, 2, 5):
        spec = bank.with_shots(k)[0]
        assert len(spec.support) == k * spec.n_way
        for c in range(spec.n_way):
            assert sum(1 for lbl, _ in spec.support if lbl == c) == k


def test_requesting_too_many_shots_raises(index, fold):
    with pytest.raises(ValueError, match="stores"):
        _bank(index, fold, n_shot_max=5)[0].with_shots(9)


def test_insufficient_images_fails_at_build_time(index, fold):
    """Fail on the CPU in seconds, not inside a GPU training loop."""
    with pytest.raises(InsufficientImagesError):
        _bank(index, fold, n_shot_max=100, n_query_per_class=100)


def test_roundtrip_save_load(index, fold, tmp_path):
    bank = _bank(index, fold)
    bank.save(tmp_path)
    loaded = EpisodeBank.load(tmp_path / f"{bank.bank_id}.jsonl")
    assert loaded.content_hash() == bank.content_hash()
    assert loaded.episode_ids() == bank.episode_ids()


def test_bank_detects_dataset_change(index, fold):
    bank = _bank(index, fold)
    bank.meta["dataset_manifest_hash"] = "sha256:something-else"
    with pytest.raises(ValueError, match="rebuild the banks"):
        bank.verify(index)


def test_fold_rejects_overlapping_species():
    with pytest.raises(ValueError, match="both train and test"):
        FoldSpec("bad", "fixed", ("apple", "mango"), ("mango",))


def test_leave_one_species_out_structure():
    species = ["apple", "banana", "grape", "mango", "orange"]
    folds = leave_one_species_out(species)
    assert len(folds) == 5
    held = sorted(f.test_species[0] for f in folds)
    assert held == sorted(species)
    for f in folds:
        assert len(f.train_species) == 4
        assert not set(f.train_species) & set(f.test_species)


def test_species_combinations_count():
    folds = species_combinations(["a", "b", "c", "d", "e"], n_train=3)
    assert len(folds) == 10          # C(5,3)
    for f in folds:
        assert len(f.train_species) == 3 and len(f.test_species) == 2


def test_train_val_split_is_disjoint_and_deterministic(index):
    seen = ["apple", "banana", "grape"]
    a = split_seen_species(index, seen, val_ratio=0.2, seed=42)
    b = split_seen_species(index, seen, val_ratio=0.2, seed=42)
    for sp in seen:
        for cls in index.classes:
            assert not set(a.pool("train", sp, cls)) & set(a.pool("val", sp, cls))
            assert a.pool("train", sp, cls) == b.pool("train", sp, cls)


def test_split_partitions_the_whole_pool(index):
    seen = ["apple"]
    a = split_seen_species(index, seen, val_ratio=0.2, seed=42)
    for cls in index.classes:
        total = len(a.pool("train", "apple", cls)) + len(a.pool("val", "apple", cls))
        assert total == len(index.pool("apple", cls))


def test_empty_pools_dict_is_an_error_not_a_silent_fallback(index, fold):
    """An empty pools mapping must not fall back to the full index.

    That fallback would draw from every species -- including the held-out one --
    and silently destroy the cross-species guarantee.
    """
    with pytest.raises((InsufficientImagesError, KeyError)):
        EpisodeBank.build(
            index, fold=fold, split="train", species=list(fold.train_species),
            n_episodes=4, n_shot_max=5, n_query_per_class=5, seed=1, pools={},
        )
