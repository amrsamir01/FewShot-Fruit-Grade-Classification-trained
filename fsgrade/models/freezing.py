"""
Explicit, stage-wise layer freezing.

The original implementation was::

    freeze_patterns = ("layer1", "conv1", "bn1")
    for name, param in self.encoder.named_parameters():
        if any(p in name for p in freeze_patterns):
            param.requires_grad = False

Substring matching means ``"conv1"`` also matches ``layer2.0.conv1``,
``layer3.1.conv1``, ``layer4.0.conv1`` and ``"bn1"`` matches every block's first
BatchNorm. The result froze **4,805,952** parameters rather than the intended
**157,504** -- a value that sits between "freeze through layer3" (2,782,784) and
"freeze the whole backbone" (11,176,512) and corresponds to no coherent
architectural choice. The published "- Frozen Layers" ablation row is therefore
uninterpretable and must be re-run.

Two fixes here:

1. Matching is by **module identity** (``get_submodule``), never by substring,
   and freeze depth becomes a first-class swept hyperparameter.
2. ``requires_grad = False`` does **not** stop BatchNorm running statistics from
   updating. Without ``freeze_bn_stats``, "frozen" stages still drift on fruit
   data and the freeze ablation measures a blend of two effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import torch.nn as nn

# Ordered stages, shallowest first. Freezing "up to stage k" freezes stages 0..k.
STAGE_MODULES: dict[str, list[list[str]]] = {
    "resnet18": [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]],
    "resnet34": [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]],
    "resnet50": [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]],
    "efficientnet_b0": [
        ["features.0"],
        ["features.1"],
        ["features.2", "features.3"],
        ["features.4", "features.5"],
        ["features.6", "features.7", "features.8"],
    ],
}

FreezeStrategy = Literal["none", "up_to_stage", "all"]


@dataclass(frozen=True)
class FreezeSpec:
    """How much of the backbone to hold fixed.

    stage = -1  nothing frozen (full fine-tune)
    stage =  0  stem only
    stage =  k  stages 0..k inclusive
    """

    strategy: FreezeStrategy = "up_to_stage"
    stage: int = 1
    freeze_bn_stats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "stage": self.stage,
            "freeze_bn_stats": self.freeze_bn_stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FreezeSpec":
        if not d:
            return cls(strategy="none", stage=-1)
        return cls(
            strategy=d.get("strategy", "up_to_stage"),
            stage=int(d.get("stage", 1)),
            freeze_bn_stats=bool(d.get("freeze_bn_stats", True)),
        )


@dataclass
class FreezeReport:
    """Exactly what was frozen. Serialized into metrics.json for every run."""

    backbone: str
    spec: dict[str, Any]
    frozen_params: int
    trainable_params: int
    total_params: int
    frozen_modules: list[str] = field(default_factory=list)
    per_stage: dict[str, int] = field(default_factory=dict)
    frozen_bn_modules: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "spec": self.spec,
            "frozen_params": self.frozen_params,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "frozen_fraction": (
                self.frozen_params / self.total_params if self.total_params else 0.0
            ),
            "frozen_modules": self.frozen_modules,
            "per_stage": self.per_stage,
            "frozen_bn_modules": self.frozen_bn_modules,
        }


def _resolve(module: nn.Module, dotted: str) -> nn.Module | None:
    try:
        return module.get_submodule(dotted)
    except AttributeError:
        return None


def stage_module_names(backbone: str, stage: int) -> list[str]:
    """Module names frozen when freezing up to ``stage`` (inclusive)."""
    if backbone not in STAGE_MODULES:
        raise KeyError(
            f"No freeze ladder defined for backbone {backbone!r}. "
            f"Known: {sorted(STAGE_MODULES)}"
        )
    ladder = STAGE_MODULES[backbone]
    if stage < 0:
        return []
    names: list[str] = []
    for s in range(min(stage + 1, len(ladder))):
        names.extend(ladder[s])
    return names


def count_stage_params(encoder: nn.Module, backbone: str) -> dict[str, int]:
    """Parameter count contributed by each stage. Used to document the ladder."""
    ladder = STAGE_MODULES.get(backbone, [])
    out: dict[str, int] = {}
    for i, group in enumerate(ladder):
        total = 0
        for name in group:
            mod = _resolve(encoder, name)
            if mod is not None:
                total += sum(p.numel() for p in mod.parameters())
        out[f"stage_{i}"] = total
    return out


def apply_freeze(encoder: nn.Module, backbone: str, spec: FreezeSpec) -> FreezeReport:
    """Freeze parameters according to ``spec`` and return an auditable report."""
    for p in encoder.parameters():
        p.requires_grad = True

    frozen_names: list[str] = []
    if spec.strategy == "none" or spec.stage < 0:
        target_names: list[str] = []
    elif spec.strategy == "all":
        target_names = ["<entire backbone>"]
        for p in encoder.parameters():
            p.requires_grad = False
        frozen_names = ["<entire backbone>"]
    else:
        target_names = stage_module_names(backbone, spec.stage)
        for name in target_names:
            mod = _resolve(encoder, name)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
            frozen_names.append(name)

    n_bn = 0
    if spec.freeze_bn_stats:
        n_bn = freeze_bn_running_stats(encoder, frozen_names, backbone)

    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    return FreezeReport(
        backbone=backbone,
        spec=spec.to_dict(),
        frozen_params=frozen,
        trainable_params=trainable,
        total_params=frozen + trainable,
        frozen_modules=frozen_names,
        per_stage=count_stage_params(encoder, backbone),
        frozen_bn_modules=n_bn,
    )


def freeze_bn_running_stats(
    encoder: nn.Module, frozen_module_names: Sequence[str], backbone: str
) -> int:
    """Tag frozen BatchNorm modules so the trainer keeps them in eval mode.

    ``requires_grad=False`` freezes the affine parameters but leaves
    ``running_mean``/``running_var`` updating on every forward pass in train
    mode. A "frozen" stage would therefore still adapt to fruit statistics.
    """
    if not frozen_module_names:
        return 0

    # Track identity: nn.Module.modules() yields the module itself first, so a
    # top-level BatchNorm (e.g. resnet's `bn1`) would otherwise be counted twice.
    seen: set[int] = set()

    def _tag(module: nn.Module) -> None:
        for sub in module.modules():          # includes `module` itself
            if isinstance(sub, nn.modules.batchnorm._BatchNorm) and id(sub) not in seen:
                sub._fsgrade_freeze_stats = True  # type: ignore[attr-defined]
                seen.add(id(sub))

    if frozen_module_names == ["<entire backbone>"]:
        _tag(encoder)
        return len(seen)

    for name in frozen_module_names:
        mod = _resolve(encoder, name)
        if mod is not None:
            _tag(mod)
    return len(seen)


def set_frozen_bn_eval(model: nn.Module) -> int:
    """Put every tagged BatchNorm back into eval mode.

    Called by the trainer at the start of each epoch, after ``model.train()``.
    """
    count = 0
    for mod in model.modules():
        if getattr(mod, "_fsgrade_freeze_stats", False):
            mod.eval()
            count += 1
    return count


def freeze_report(encoder: nn.Module, backbone: str, spec: FreezeSpec) -> FreezeReport:
    """Report current freeze state without modifying it."""
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_bn = sum(1 for m in encoder.modules() if getattr(m, "_fsgrade_freeze_stats", False))
    return FreezeReport(
        backbone=backbone,
        spec=spec.to_dict(),
        frozen_params=frozen,
        trainable_params=trainable,
        total_params=frozen + trainable,
        frozen_modules=stage_module_names(backbone, spec.stage) if spec.stage >= 0 else [],
        per_stage=count_stage_params(encoder, backbone),
        frozen_bn_modules=n_bn,
    )
