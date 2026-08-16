"""
Centralized configuration for all experiments.

Change _PROJECT_DIR or DATA_ROOT to match your local folder layout.
Everything else (hyper-parameters, ablation settings) lives here so that
notebooks and scripts stay thin.
"""

import os
import pathlib


class Config:
    # ── Portable paths ── change only these if your folder layout differs ──
    _PROJECT_DIR = pathlib.Path.home() / "Desktop" / "Amr Samir"
    DATA_ROOT = str(_PROJECT_DIR / "FruitVision")
    THESIS_FIGS_DIR = str(_PROJECT_DIR / "Final")

    # Species split
    TRAIN_FRUITS = ["apple", "banana", "grape"]
    TEST_FRUITS = ["mango", "orange"]

    # Quality classes
    CLASSES = ["fresh", "rotten"]
    N_CLASSES = 2

    # Few-shot episode settings
    N_SHOT = 5
    N_QUERY = 15
    N_EPISODES_TRAIN = 500
    N_EPISODES_VAL = 200
    N_EPISODES_TEST = 600

    # Model architecture
    BACKBONE = "resnet18"
    EMBEDDING_DIM = 256
    PRETRAINED = True

    # Training hyper-parameters
    EPOCHS = 30
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 1
    DROPOUT_RATE = 0.4
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 7
    CONTRASTIVE_WEIGHT = 0.1
    TEMPERATURE = 0.5
    WARMUP_EPOCHS = 3

    # Data split
    VAL_SPLIT_RATIO = 0.15

    # Image pre-processing
    IMAGE_SIZE = 224
    RESIZE_SIZE = 256

    # Output directories (relative to repo root)
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"

    EXPERIMENT_NAME = f"ProtoNet_{BACKBONE}_{N_SHOT}shot"

    # Ablation
    ABLATION_SHOTS = [1, 3, 5, 10]
    ABLATION_EPISODES = 300


def ensure_dirs(config: Config) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.THESIS_FIGS_DIR, exist_ok=True)
