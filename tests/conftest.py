"""Shared fixtures. Synthetic images so tests never need the real dataset."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

SPECIES = ["apple", "banana", "grape", "mango", "orange"]
CLASSES = ["fresh", "rotten"]
N_PER_CLASS = 40


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> str:
    """A miniature dataset with the real folder layout.

    Fresh images are biased green, rotten biased brown, so a classifier can
    actually learn something and end-to-end tests are meaningful.
    """
    root = tmp_path_factory.mktemp("FruitVisionSynthetic")
    rng = np.random.default_rng(0)

    for s_idx, species in enumerate(SPECIES):
        for c_idx, cls in enumerate(CLASSES):
            folder = root / species / cls
            folder.mkdir(parents=True, exist_ok=True)
            base = np.array([60, 160, 60]) if cls == "fresh" else np.array([120, 80, 40])
            base = base + s_idx * 6
            for i in range(N_PER_CLASS):
                noise = rng.normal(0, 18, size=(32, 32, 3))
                arr = np.clip(base.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(folder / f"img_{i:04d}.jpg", quality=95)
    return str(root)


@pytest.fixture(scope="session")
def index(synthetic_root):
    from fsgrade.data.index import ImageIndex

    return ImageIndex.build(synthetic_root, SPECIES, CLASSES)


@pytest.fixture
def base_cfg(synthetic_root) -> dict:
    return {
        "seed": 42,
        "deterministic": True,
        "device": "cpu",
        "num_workers": 0,
        "paths": {"data_root": synthetic_root, "results_root": "./results",
                  "cache_root": "./.cache"},
        "data": {
            "species": SPECIES, "classes": CLASSES,
            "extensions": [".jpg", ".jpeg", ".png", ".bmp"],
            "image_size": 32, "resize_size": 36, "val_ratio": 0.2, "split_seed": 42,
            "augmentation": {},
        },
        "protocol": {
            "name": "fixed",
            "fixed": {"train": ["apple", "banana", "grape"], "test": ["mango", "orange"]},
            "episodes": {
                "n_way": 2, "n_shot": 5, "n_shot_max": 5,
                "shots_to_evaluate": [1, 5], "n_query_per_class": 5,
                "n_episodes": {"train": 4, "val": 4, "test": 8},
                "bank_seed": 123, "per_species_balanced": True,
            },
        },
        "model": {
            "backbone": "resnet18", "pretrained": False, "embedding_dim": 32,
            "dropout": 0.0, "use_batchnorm": False, "transductive_forward": False,
            "freeze": {"strategy": "up_to_stage", "stage": 3, "freeze_bn_stats": True},
            "temperature": {"enabled": True, "init": 0.5},
        },
        "train": {
            "mode": "episodic", "epochs": 1, "supervised_epochs": 1, "batch_size": 8,
            "warmup_epochs": 0, "lr_backbone": 1e-5, "lr_head": 1e-4,
            "grad_clip": 1.0, "label_smoothing": 0.1, "contrastive_weight": 0.1,
            "early_stopping": {"monitor": "val_balanced_accuracy", "patience": 3},
        },
        "eval": {"ece_bins": 5, "bootstrap_n": 100, "bootstrap_seed": 7, "ci_level": 0.95,
                 "headline_metric": "accuracy"},
        "stats": {"reference_method": "ours", "alpha": 0.05, "metric": "accuracy"},
        "logging": {"level": "WARNING", "progress": "none"},
    }


@pytest.fixture
def fold():
    from fsgrade.data.episodes import FoldSpec

    return FoldSpec(
        fold_id="test-fold", protocol="fixed",
        train_species=("apple", "banana", "grape"),
        test_species=("mango", "orange"),
    )
