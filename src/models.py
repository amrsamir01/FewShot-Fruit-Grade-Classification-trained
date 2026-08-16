"""
All network architectures used in the project.

Shared helper
  _build_backbone         – build a torchvision encoder and return (encoder, in_features)

Our method
  EmbeddingNetwork        – backbone + projection head with frozen early layers
  PrototypicalNetwork     – episodic classifier with learnable temperature

Baselines
  SiameseNetwork          – Koch et al. 2015
  MatchingNetwork         – Vinyals et al. 2016
  StandardProtoNet        – Snell et al. 2017 (vanilla)
  ProtoNetWithTemp        – ProtoNet + learnable temperature only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ===================== shared helper ====================================== #

def _build_backbone(backbone: str, pretrained: bool = True):
    """Return (encoder, in_features) for the chosen backbone."""
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.resnet18(weights=weights)
        in_feat = enc.fc.in_features
        enc.fc = nn.Identity()
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.resnet50(weights=weights)
        in_feat = enc.fc.in_features
        enc.fc = nn.Identity()
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        enc = models.efficientnet_b0(weights=weights)
        in_feat = enc.classifier[1].in_features
        enc.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return enc, in_feat


# ===================== our method ========================================= #

class EmbeddingNetwork(nn.Module):
    """Feature extraction with dropout regularisation and optionally frozen early layers."""

    def __init__(self, backbone="resnet18", embedding_dim=256,
                 pretrained=True, dropout_rate=0.4, freeze_early=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.backbone_name = backbone
        self.encoder, in_features = _build_backbone(backbone, pretrained)

        if pretrained and freeze_early:
            self._freeze_early_layers()

        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def _freeze_early_layers(self):
        if self.backbone_name in ("resnet18", "resnet50"):
            freeze_patterns = ("layer1", "conv1", "bn1")
        elif self.backbone_name == "efficientnet_b0":
            freeze_patterns = ("features.0", "features.1")
        else:
            return
        for name, param in self.encoder.named_parameters():
            if any(p in name for p in freeze_patterns):
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network with learnable temperature."""

    def __init__(self, backbone="resnet18", embedding_dim=256,
                 pretrained=True, dropout_rate=0.4, temperature=0.5,
                 freeze_early=True, use_temperature=True):
        super().__init__()
        self.encoder = EmbeddingNetwork(
            backbone, embedding_dim, pretrained, dropout_rate, freeze_early
        )
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_temperature = use_temperature
        if not use_temperature:
            self.temperature.requires_grad = False

    def compute_prototypes(self, support_embeddings, support_labels, n_classes=2):
        prototypes = torch.zeros(
            n_classes, self.embedding_dim, device=support_embeddings.device
        )
        for c in range(n_classes):
            prototypes[c] = support_embeddings[support_labels == c].mean(dim=0)
        return prototypes

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        all_images = torch.cat([support_images, query_images], dim=0)
        all_embeddings = self.encoder(all_images)

        n_support = support_images.size(0)
        support_embeddings = all_embeddings[:n_support]
        query_embeddings = all_embeddings[n_support:]

        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_classes)
        distances = torch.cdist(query_embeddings, prototypes, p=2)

        if self.use_temperature:
            logits = -distances / self.temperature.clamp(min=0.1, max=2.0)
        else:
            logits = -distances

        return logits, query_embeddings, support_embeddings, prototypes


# ===================== baselines ========================================== #

class SiameseEmbedding(nn.Module):
    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=True):
        super().__init__()
        self.encoder, in_features = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return F.normalize(self.projection(self.encoder(x)), p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Siamese Network baseline (Koch et al., 2015)."""

    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=True):
        super().__init__()
        self.encoder = SiameseEmbedding(backbone, embedding_dim, pretrained)
        self.embedding_dim = embedding_dim
        self.relation = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature.requires_grad = False

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        support_emb = self.encoder(support_images)
        query_emb = self.encoder(query_images)
        n_query, n_support = query_emb.size(0), support_emb.size(0)

        query_exp = query_emb.unsqueeze(1).expand(-1, n_support, -1)
        support_exp = support_emb.unsqueeze(0).expand(n_query, -1, -1)
        scores = self.relation(torch.cat([query_exp, support_exp], dim=2)).squeeze(-1)

        logits = torch.zeros(n_query, n_classes, device=query_emb.device)
        for c in range(n_classes):
            mask = (support_labels == c).float()
            logits[:, c] = (scores * mask.unsqueeze(0)).sum(1) / mask.sum().clamp(min=1)

        prototypes = torch.zeros(n_classes, self.embedding_dim, device=support_emb.device)
        for c in range(n_classes):
            prototypes[c] = support_emb[support_labels == c].mean(0)
        return logits, query_emb, support_emb, prototypes


class MatchingNetwork(nn.Module):
    """Matching Network baseline (Vinyals et al., 2016)."""

    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=True):
        super().__init__()
        self.encoder_net, in_features = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature.requires_grad = False

    def encode(self, x):
        return F.normalize(self.projection(self.encoder_net(x)), p=2, dim=1)

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        support_emb = self.encode(support_images)
        query_emb = self.encode(query_images)

        attention = F.softmax(torch.mm(query_emb, support_emb.t()), dim=1)
        logits = torch.mm(attention, F.one_hot(support_labels, n_classes).float())

        prototypes = torch.zeros(n_classes, self.embedding_dim, device=support_emb.device)
        for c in range(n_classes):
            prototypes[c] = support_emb[support_labels == c].mean(0)
        return logits, query_emb, support_emb, prototypes


class StandardProtoNet(nn.Module):
    """Standard Prototypical Network (Snell et al., 2017) – no temperature, no contrastive."""

    def __init__(self, backbone="resnet18", embedding_dim=256, pretrained=True):
        super().__init__()
        self.encoder_net, in_features = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature.requires_grad = False

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        all_images = torch.cat([support_images, query_images], dim=0)
        all_emb = F.normalize(self.projection(self.encoder_net(all_images)), p=2, dim=1)

        n_support = support_images.size(0)
        support_emb, query_emb = all_emb[:n_support], all_emb[n_support:]

        prototypes = torch.zeros(n_classes, self.embedding_dim, device=support_emb.device)
        for c in range(n_classes):
            prototypes[c] = support_emb[support_labels == c].mean(0)

        return -torch.cdist(query_emb, prototypes, p=2), query_emb, support_emb, prototypes


class ProtoNetWithTemp(nn.Module):
    """ProtoNet + learnable temperature only (no contrastive, no frozen layers)."""

    def __init__(self, backbone="resnet18", embedding_dim=256,
                 pretrained=True, temperature=0.5):
        super().__init__()
        self.encoder_net, in_features = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        all_images = torch.cat([support_images, query_images], dim=0)
        all_emb = F.normalize(self.projection(self.encoder_net(all_images)), p=2, dim=1)

        n_support = support_images.size(0)
        support_emb, query_emb = all_emb[:n_support], all_emb[n_support:]

        prototypes = torch.zeros(n_classes, self.embedding_dim, device=support_emb.device)
        for c in range(n_classes):
            prototypes[c] = support_emb[support_labels == c].mean(0)

        distances = torch.cdist(query_emb, prototypes, p=2)
        logits = -distances / self.temperature.clamp(min=0.1, max=2.0)
        return logits, query_emb, support_emb, prototypes
