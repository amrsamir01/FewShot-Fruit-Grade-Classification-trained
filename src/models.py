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

    #: Reproduces the freezing behaviour the committed checkpoint was trained
    #: with. Kept so `notebooks/checkpoints/best_model.pth` stays loadable and
    #: so the two regimes can be compared as an ablation rather than silently
    #: swapped. See _freeze_early_layers for why it is not the default.
    FREEZE_MODES = ("stem_layer1", "legacy_substring")

    #: "batchnorm" is the architecture the committed checkpoint was trained with
    #: and remains the default so those weights load. "layernorm" is the better
    #: choice for episodic training — see below.
    NORM_LAYERS = ("batchnorm", "layernorm")

    def __init__(self, backbone="resnet18", embedding_dim=256,
                 pretrained=True, dropout_rate=0.4, freeze_early=True,
                 freeze_mode="stem_layer1", norm_layer="batchnorm"):
        super().__init__()
        if freeze_mode not in self.FREEZE_MODES:
            raise ValueError(
                f"freeze_mode must be one of {self.FREEZE_MODES}, got {freeze_mode!r}"
            )
        if norm_layer not in self.NORM_LAYERS:
            raise ValueError(
                f"norm_layer must be one of {self.NORM_LAYERS}, got {norm_layer!r}"
            )
        self.dropout_rate = dropout_rate
        self.backbone_name = backbone
        self.freeze_mode = freeze_mode
        self.norm_layer = norm_layer
        self.encoder, in_features = _build_backbone(backbone, pretrained)

        if pretrained and freeze_early:
            self._freeze_early_layers()

        # BatchNorm1d here is a poor fit for episodic training and gets worse now
        # that support and query are embedded separately (as they must be, to stay
        # inductive). The support pass has batch size n_shot * n_classes — 10 at
        # 5-shot, 2 at 1-shot. Measured divergence between train-mode output
        # (batch statistics) and eval-mode output (running statistics) is 1.97 at
        # batch 10 and 3.31 at batch 2, so the head sees a different normalisation
        # at training time than at test time, and the 1-shot case is degenerate.
        #
        # LayerNorm normalises per sample. It is batch-size independent, which
        # makes train and eval identical, makes 1-shot well-behaved, and makes
        # support/query independence structural rather than something a future
        # refactor could silently undo.
        norm = (nn.BatchNorm1d(embedding_dim * 2) if norm_layer == "batchnorm"
                else nn.LayerNorm(embedding_dim * 2))

        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim * 2),
            norm,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def _freeze_early_layers(self):
        """
        Freeze the stem and the first residual stage.

        "stem_layer1" selects whole modules by attribute. "legacy_substring"
        reproduces the original behaviour, which matched parameter NAMES against
        ("layer1", "conv1", "bn1"). That substring test also matches
        `layer2.0.conv1`, `layer2.0.bn1`, `layer3.0.conv1`, ... — the first
        convolution and its norm in *every block of every stage*. On ResNet-18
        it freezes 4,648,448 parameters beyond the stem and layer1, leaving
        6,765,569 trainable instead of 11,414,017: 40.7% of the intended
        trainable network was frozen without that being described anywhere.

        The legacy mode is retained because the committed checkpoint was trained
        under it, so it is needed to load those weights and to quantify the
        difference honestly rather than quietly restating the result.
        """
        if self.backbone_name in ("resnet18", "resnet50"):
            patterns = ("layer1", "conv1", "bn1")
            modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.layer1]
        elif self.backbone_name == "efficientnet_b0":
            patterns = ("features.0", "features.1")
            modules = [self.encoder.features[0], self.encoder.features[1]]
        else:
            return

        if self.freeze_mode == "legacy_substring":
            for name, param in self.encoder.named_parameters():
                if any(p in name for p in patterns):
                    param.requires_grad = False
            return

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network with learnable temperature."""

    def __init__(self, backbone="resnet18", embedding_dim=256,
                 pretrained=True, dropout_rate=0.4, temperature=0.5,
                 freeze_early=True, use_temperature=True,
                 freeze_mode="stem_layer1", norm_layer="batchnorm",
                 freeze_bn_stats=False):
        super().__init__()
        self.encoder = EmbeddingNetwork(
            backbone, embedding_dim, pretrained, dropout_rate, freeze_early,
            freeze_mode=freeze_mode, norm_layer=norm_layer,
        )
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_temperature = use_temperature
        self.freeze_bn_stats = freeze_bn_stats
        if not use_temperature:
            self.temperature.requires_grad = False

    def train(self, mode: bool = True):
        """
        Standard `.train()`, except BatchNorm keeps using its running statistics
        when `freeze_bn_stats` is set.

        Episodes are tiny batches — 40 images at 5-shot/15-query, 32 at 1-shot.
        BatchNorm estimated from 40 images is a far worse estimator than the
        ImageNet running statistics it would replace, and it makes every
        embedding depend on the rest of the episode. Measured on this model,
        changing a query's batch-mates shifts its embedding by 1.3e-01 with
        BatchNorm active and 1.0e-07 with it frozen.

        Freezing BN statistics during small-batch transfer is standard practice
        and is the single change here most likely to improve accuracy. It is
        overridden here rather than applied at the call site so that it survives
        every `.train()` call, including the ones inside `train_epoch`.
        """
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        return self

    def compute_prototypes(self, support_embeddings, support_labels, n_classes=2):
        prototypes = torch.zeros(
            n_classes, self.embedding_dim, device=support_embeddings.device
        )
        for c in range(n_classes):
            prototypes[c] = support_embeddings[support_labels == c].mean(dim=0)
        return prototypes

    def forward(self, support_images, support_labels, query_images, n_classes=2):
        # Support and query are embedded in SEPARATE forward passes.
        #
        # The projection head contains BatchNorm1d. Concatenating support and
        # query into one pass makes each query's embedding depend on the batch
        # statistics of the whole episode, including the other queries — i.e.
        # transductive leakage. The inductive few-shot protocol this thesis
        # claims to follow requires that a query be classified using only the
        # support set, so the two passes must stay separate.
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

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
