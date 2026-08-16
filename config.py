"""
Centralized configuration for all experiments.

Change _PROJECT_DIR or DATA_ROOT to match your local folder layout.
Everything else (hyper-parameters, ablation settings) lives here so that
notebooks and scripts stay thin.
"""

import os
import pathlib


_REPO_ROOT = pathlib.Path(__file__).resolve().parent


def _resolve_data_root() -> str:
    """
    Locate the FruitVision image tree.

    Resolution order:
      1. $FRUITVISION_ROOT              – explicit override, use this on the GPU box
      2. <repo>/FruitVision             – dataset checked out beside the code
      3. ~/Desktop/Amr Samir/FruitVision – the original authoring layout

    The path is returned even when it does not exist so that import never fails;
    callers that actually need the images validate it via
    scripts/reproduce_thesis.py --verify-data.
    """
    env = os.environ.get("FRUITVISION_ROOT")
    if env:
        return str(pathlib.Path(env).expanduser())

    for candidate in (
        _REPO_ROOT / "FruitVision",
        pathlib.Path.home() / "Desktop" / "Amr Samir" / "FruitVision",
    ):
        if candidate.is_dir():
            return str(candidate)

    return str(_REPO_ROOT / "FruitVision")


class Config:
    # ── Portable paths ── override with $FRUITVISION_ROOT / $THESIS_FIGS_DIR ──
    DATA_ROOT = _resolve_data_root()
    THESIS_FIGS_DIR = os.environ.get(
        "THESIS_FIGS_DIR", str(_REPO_ROOT / "Imgs")
    )

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

    # Augmentation
    # Hue shift destroys the fresh/rotten signal (colour is the label). Matches
    # configs/base.yaml:41-43; the original run used 0.05 by accident.
    HUE_JITTER = 0.02

    # Training hyper-parameters
    EPOCHS = 30
    LEARNING_RATE = 5e-5

    # NOTE: with AdamW's decoupled weight decay the per-step shrink is lr*wd, so
    # at these learning rates this value is inert — over a full 22-epoch run it
    # shrinks encoder weights by 0.003% and projection weights by 0.03%. It is
    # left unchanged so the re-run stays comparable, but the thesis should not
    # describe weight decay as a regularisation mechanism: it is not acting as
    # one. Effective regularisation here comes from dropout and augmentation.
    WEIGHT_DECAY = 5e-4

    # Episodic training has no mini-batch: one episode is one optimiser step.
    # BATCH_SIZE = 1 used to live here and was never read by any code path,
    # while configs/base.yaml:90 declared 32 for the (separate) fsgrade
    # pipeline. Two contradictory values, neither wired to anything. Removed
    # rather than picked. The only real batch size is in the transfer controls
    # (src/experiments.py:transfer_controls), which trains conventionally and
    # takes its batch size as an explicit argument.

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

    # Output directories — absolute, so they do not depend on the current
    # working directory. Previously "./results" resolved relative to notebooks/,
    # which is why every figure lookup reported [Missing].
    CHECKPOINT_DIR = str(_REPO_ROOT / "checkpoints")
    RESULTS_DIR = str(_REPO_ROOT / "results")

    EXPERIMENT_NAME = f"ProtoNet_{BACKBONE}_{N_SHOT}shot"

    # Ablation
    ABLATION_SHOTS = [1, 3, 5, 10]
    ABLATION_EPISODES = 300


def ensure_dirs(config: Config) -> None:
    """Create output directories if they don't exist."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.THESIS_FIGS_DIR, exist_ok=True)
