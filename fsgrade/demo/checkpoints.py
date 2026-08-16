"""
Load a trained checkpoint back into a model, for inference.

This did not exist. ``torch.load`` appears nowhere in ``fsgrade/`` -- every
method's ``prepare()`` retrains from scratch, which is correct for experiments
and useless for a demo. The legacy ``src/train.py::load_checkpoint`` is not a
substitute: its parameter names differ (``encoder.encoder.*`` versus
``encoder.backbone.*``), so a strict load fails.

Checkpoints store weights and *training* hyperparameters only -- no architecture
metadata. The architecture therefore has to be rebuilt from the run's sibling
``config.yaml``, mirroring ``fsgrade/methods/trained.py``. Where the config and
the checkpoint disagree, the checkpoint wins: its tensor shapes are ground
truth, and a config edited after training would otherwise produce a confusing
strict-load failure.

Offline note
------------
``pretrained=False`` is hardcoded here, never read from config. The base config
ships ``model.pretrained: true``, which would make torchvision reach for
ImageNet weights over the network -- fatal in a viva room. The trained weights
come from the checkpoint anyway, so pretrained initialisation is pure waste.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Architecture facts recoverable from the state dict itself.
_BACKBONE_BY_FC_SHAPE = {512: ("resnet18", "resnet34"), 2048: ("resnet50",)}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be turned into a usable model."""

    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass
class ModelCard:
    """What was loaded, and how confident we are about it."""

    method: str
    module_class: str
    backbone: str
    embedding_dim: int
    n_classes: int
    checkpoint: pathlib.Path
    epoch: int | None = None
    spec_hash: str | None = None
    best_val: float | None = None
    params_total: int = 0
    params_trainable: int = 0
    encoder_family: str = "imagenet"
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "module_class": self.module_class,
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "n_classes": self.n_classes,
            "checkpoint": str(self.checkpoint),
            "epoch": self.epoch,
            "spec_hash": self.spec_hash,
            "best_val": self.best_val,
            "params": {"total": self.params_total, "trainable": self.params_trainable},
            "encoder_family": self.encoder_family,
            "conflicts": list(self.conflicts),
            "notes": list(self.notes),
        }


@dataclass
class LoadedModel:
    module: nn.Module
    card: ModelCard
    kind: str          # "episodic" | "supervised"


# --------------------------------------------------------------------------- #
#  Reading architecture out of a state dict
# --------------------------------------------------------------------------- #

def inspect_state_dict(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover architecture facts from tensor shapes alone.

    More trustworthy than the config, which may have been edited after training.
    """
    facts: dict[str, Any] = {
        "has_temperature": "temperature" in state,
        "has_classifier": any(k.startswith("classifier.") for k in state),
        "has_species_head": any(k.startswith("species_head.") for k in state),
        "has_relation": any(k.startswith("relation.") for k in state),
        "has_log_temperature": "log_temperature" in state,
        "uses_batchnorm_head": False,
        "embedding_dim": None,
        "backbone_features": None,
        "n_classes": None,
        "n_species": None,
    }

    # Projection head: Linear(in -> 2*emb) ... Linear(2*emb -> emb)
    first = state.get("encoder.projection.net.0.weight")
    last = state.get("encoder.projection.net.4.weight")
    if first is not None:
        facts["backbone_features"] = int(first.shape[1])
    if last is not None:
        facts["embedding_dim"] = int(last.shape[0])

    # BatchNorm1d in the head carries running statistics; LayerNorm does not.
    facts["uses_batchnorm_head"] = "encoder.projection.net.1.running_mean" in state

    clf = state.get("classifier.weight")
    if clf is not None:
        facts["n_classes"] = int(clf.shape[0])
    species = state.get("species_head.weight")
    if species is not None:
        facts["n_species"] = int(species.shape[0])

    return facts


def infer_backbone(facts: dict[str, Any], config_backbone: str | None) -> tuple[str, list[str]]:
    """Choose a backbone name consistent with the state dict."""
    conflicts: list[str] = []
    features = facts.get("backbone_features")

    if features is None:
        return config_backbone or "resnet18", conflicts

    compatible = _BACKBONE_BY_FC_SHAPE.get(features)
    if compatible is None:
        # EfficientNet-B0 emits 1280.
        compatible = ("efficientnet_b0",) if features == 1280 else None

    if compatible is None:
        return config_backbone or "resnet18", [
            f"unrecognised backbone feature width {features}; trusting config "
            f"value {config_backbone!r}"
        ]

    if config_backbone in compatible:
        return config_backbone, conflicts

    chosen = compatible[0]
    if config_backbone:
        conflicts.append(
            f"config says backbone={config_backbone!r} but the checkpoint's "
            f"{features}-d features imply {' or '.join(compatible)}; using {chosen!r}"
        )
    return chosen, conflicts


# --------------------------------------------------------------------------- #
#  Building modules
# --------------------------------------------------------------------------- #

def build_module(
    method: str,
    config: dict[str, Any],
    facts: dict[str, Any],
) -> tuple[nn.Module, str, ModelCard]:
    """Rebuild the architecture for ``method``, ready for a strict load."""
    from fsgrade.config import get_in
    from fsgrade.models.episodic import (
        MatchingNetwork,
        PrototypicalNetwork,
        SiameseNetwork,
        SupervisedClassifier,
    )
    from fsgrade.models.freezing import FreezeSpec

    base = method.split("@")[0]
    conflicts: list[str] = []
    notes: list[str] = []

    backbone, backbone_conflicts = infer_backbone(
        facts, get_in(config, "model.backbone", "resnet18")
    )
    conflicts.extend(backbone_conflicts)

    embedding_dim = facts.get("embedding_dim") or int(
        get_in(config, "model.embedding_dim", 256)
    )
    cfg_dim = int(get_in(config, "model.embedding_dim", 256))
    if facts.get("embedding_dim") and facts["embedding_dim"] != cfg_dim:
        conflicts.append(
            f"config says embedding_dim={cfg_dim} but the checkpoint has "
            f"{facts['embedding_dim']}; using the checkpoint"
        )

    # Dropout and freezing are inert at inference (model.eval(), no grad), but
    # they must be constructed identically or the module graph differs.
    common: dict[str, Any] = {
        "backbone": backbone,
        "embedding_dim": embedding_dim,
        "pretrained": False,          # hardcoded: see the module docstring
        "dropout": float(get_in(config, "model.dropout", 0.4)),
        "freeze": FreezeSpec.from_dict(get_in(config, "model.freeze", {})),
    }

    if base in SUPERVISED_METHODS:
        n_classes = facts.get("n_classes") or len(
            get_in(config, "data.classes", ["fresh", "rotten"])
        )
        module: nn.Module = SupervisedClassifier(
            n_classes=n_classes,
            n_species=facts.get("n_species") or 0,
            **common,
        )
        kind = "supervised"
        if facts.get("n_species"):
            notes.append(
                f"model carries a {facts['n_species']}-way species head "
                "(species-adversarial training was enabled)"
            )
    else:
        n_classes = int(get_in(config, "protocol.episodes.n_way", 2))
        use_bn = bool(facts.get("uses_batchnorm_head"))
        if use_bn != bool(get_in(config, "model.use_batchnorm", False)):
            conflicts.append(
                f"config says use_batchnorm={get_in(config, 'model.use_batchnorm', False)} "
                f"but the checkpoint's head {'has' if use_bn else 'lacks'} BatchNorm "
                f"running statistics; using the checkpoint"
            )
        episodic_common = dict(
            common,
            use_batchnorm=use_bn,
            transductive=bool(get_in(config, "model.transductive_forward", False)),
        )
        if base == "siamese":
            module = SiameseNetwork(**common)
        elif base == "matching":
            module = MatchingNetwork(**common)
        else:
            module = PrototypicalNetwork(
                temperature=float(get_in(config, "model.temperature.init", 0.5)),
                use_temperature=(base != "protonet"),
                **episodic_common,
            )
        kind = "episodic"

    total = sum(p.numel() for p in module.parameters())
    card = ModelCard(
        method=method,
        module_class=type(module).__name__,
        backbone=backbone,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
        checkpoint=pathlib.Path("<pending>"),
        params_total=total,
        params_trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
        conflicts=conflicts,
        notes=notes,
    )
    return module, kind, card


SUPERVISED_METHODS: frozenset[str] = frozenset({
    "zeroshot_supervised", "ncc_supervised", "finetune_supervised", "supervised",
})


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #

def load_checkpoint_file(path: str | pathlib.Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint, tolerating both raw and wrapped state dicts."""
    path = pathlib.Path(path)
    if not path.exists():
        raise CheckpointError(
            f"Checkpoint not found: {path}",
            remediation="Run the experiments first, or pick a different run in the demo.",
        )
    try:
        # weights_only=False because checkpoints carry a train_spec dict. The
        # files are produced by this project, not fetched from anywhere.
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(
            f"Could not read {path.name}: {type(exc).__name__}: {exc}",
            remediation="The file may be truncated -- re-run the experiment that produced it.",
        ) from exc

    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload
    if isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return {"model_state_dict": payload}
    raise CheckpointError(
        f"{path.name} does not look like an fsgrade checkpoint "
        f"(keys: {sorted(payload)[:6] if isinstance(payload, dict) else type(payload)})",
        remediation="Expected a dict containing 'model_state_dict'.",
    )


def load_for_inference(
    method: str,
    checkpoint: str | pathlib.Path,
    config: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> LoadedModel:
    """Rebuild ``method``'s architecture and load its trained weights.

    The returned module is in eval mode with gradients disabled.
    """
    device = torch.device(device)
    path = pathlib.Path(checkpoint)
    payload = load_checkpoint_file(path, device)
    state: dict[str, torch.Tensor] = payload["model_state_dict"]

    facts = inspect_state_dict(state)
    module, kind, card = build_module(method, config, facts)

    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            _explain_mismatch(method, path, module, state, exc),
            remediation=(
                "The run's config.yaml probably does not describe the weights in "
                "this checkpoint. Load the run directory that produced it."
            ),
        ) from exc

    module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)

    card.checkpoint = path
    card.epoch = payload.get("epoch")
    card.spec_hash = payload.get("spec_hash")
    metrics = payload.get("metrics") or {}
    card.best_val = metrics.get("val_balanced_accuracy") or metrics.get("val_accuracy")

    return LoadedModel(module=module, card=card, kind=kind)


def _explain_mismatch(
    method: str,
    path: pathlib.Path,
    module: nn.Module,
    state: dict[str, torch.Tensor],
    exc: RuntimeError,
) -> str:
    """Turn PyTorch's wall of text into something actionable."""
    expected = set(module.state_dict())
    found = set(state)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    lines = [f"Checkpoint {path.name} does not match the rebuilt {method!r} architecture."]
    if missing:
        lines.append(f"  missing {len(missing)} keys, e.g. {missing[:4]}")
    if unexpected:
        lines.append(f"  unexpected {len(unexpected)} keys, e.g. {unexpected[:4]}")

    shape_conflicts = [
        f"  {k}: checkpoint {tuple(state[k].shape)} vs model {tuple(module.state_dict()[k].shape)}"
        for k in sorted(expected & found)
        if state[k].shape != module.state_dict()[k].shape
    ]
    if shape_conflicts:
        lines.append("  shape conflicts:")
        lines.extend(shape_conflicts[:4])

    if any(k.startswith("encoder.encoder.") for k in found):
        lines.append(
            "  This looks like a LEGACY src/ checkpoint (encoder.encoder.*). "
            "Those are not loadable by fsgrade -- retrain, or load via src/train.py."
        )
    if not (missing or unexpected or shape_conflicts):
        lines.append(f"  torch said: {exc}")
    return "\n".join(lines)
