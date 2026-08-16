"""
Backbone construction and the projection head.

``BackboneBundle`` carries the normalization constants alongside the module,
because CLIP and DINOv2 expect different preprocessing from the ImageNet
statistics the original transforms hardcoded. Getting this wrong silently
degrades a frozen foundation model and is easy to miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class BackboneBundle:
    module: nn.Module
    out_dim: int
    name: str
    normalization: tuple[Sequence[float], Sequence[float]] = (IMAGENET_MEAN, IMAGENET_STD)
    input_size: int = 224


def build_backbone(name: str, *, pretrained: bool = True) -> BackboneBundle:
    """Build a torchvision backbone with its classifier removed."""
    from torchvision import models

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.resnet18(weights=weights)
        dim = enc.fc.in_features
        enc.fc = nn.Identity()
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.resnet34(weights=weights)
        dim = enc.fc.in_features
        enc.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.resnet50(weights=weights)
        dim = enc.fc.in_features
        enc.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.efficientnet_b0(weights=weights)
        dim = enc.classifier[1].in_features
        enc.classifier = nn.Identity()
    else:
        raise ValueError(
            f"Unknown backbone {name!r}. "
            "Known: resnet18, resnet34, resnet50, efficientnet_b0. "
            "For DINOv2/CLIP use fsgrade.models.foundation."
        )
    return BackboneBundle(module=enc, out_dim=dim, name=name)


class ProjectionHead(nn.Module):
    """MLP projection with optional BatchNorm.

    ``use_batchnorm`` defaults to False. The original head used
    ``BatchNorm1d`` and the model encoded support and query in a *single*
    concatenated forward pass, so in train mode the query batch statistics
    influenced the support embeddings -- transductive leakage. It was safe at
    eval only because ``model.eval()`` switches to running statistics, i.e. by
    accident rather than by design. LayerNorm has no such coupling.
    """

    def __init__(
        self,
        in_features: int,
        embedding_dim: int = 256,
        *,
        hidden_multiplier: int = 2,
        dropout: float = 0.4,
        use_batchnorm: bool = False,
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        hidden = embedding_dim * hidden_multiplier
        if use_batchnorm:
            norm_layer: nn.Module = nn.BatchNorm1d(hidden)
        elif norm == "layernorm":
            norm_layer = nn.LayerNorm(hidden)
        else:
            norm_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Backbone + projection head + L2 normalisation, with freeze control."""

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        *,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze: Any = None,
        use_batchnorm: bool = False,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        from fsgrade.models.freezing import FreezeSpec, apply_freeze

        bundle = build_backbone(backbone, pretrained=pretrained)
        self.backbone_name = backbone
        self.backbone = bundle.module
        self.normalization = bundle.normalization
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize

        self.projection = ProjectionHead(
            bundle.out_dim, embedding_dim, dropout=dropout, use_batchnorm=use_batchnorm
        )

        spec = freeze if isinstance(freeze, FreezeSpec) else FreezeSpec.from_dict(freeze)
        self.freeze_spec = spec
        self.freeze_report = apply_freeze(self.backbone, backbone, spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(self.backbone(x))
        return F.normalize(z, p=2, dim=1) if self.l2_normalize else z

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named groups so every method receives identical optimizer construction."""
        return {
            "backbone": [p for p in self.backbone.parameters() if p.requires_grad],
            "head": [p for p in self.projection.parameters() if p.requires_grad],
        }

    def n_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
