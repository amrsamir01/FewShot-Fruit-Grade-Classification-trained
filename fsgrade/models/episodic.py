"""
Episodic classifier heads.

**The logits contract.** Every module here returns *real logits*: unbounded,
shift-invariant scores suitable for ``nn.CrossEntropyLoss``. This is enforced by
``tests/test_methods_contract.py``.

Two baselines violated that contract in the original code and were consequently
handicapped in the very comparison they existed to provide:

``SiameseNetwork``
    ended in ``nn.Sigmoid()`` and averaged the per-pair probabilities per class,
    producing values in [0, 1].

``MatchingNetwork``
    returned ``softmax(QS^T) @ one_hot(labels)`` -- a probability vector summing
    to 1.

Both were then passed to ``nn.CrossEntropyLoss``, which applies ``log_softmax``
on top. A softmax over a range of width <= 1 is nearly uniform, so the gradients
were tiny and poorly scaled. Any conclusion drawn from "our method beats Siamese
and Matching" was confounded by this bug.

**Transductive leakage.** ``encode_split`` runs support and query through the
encoder in separate passes by default. The original concatenated them into one
batch; with BatchNorm in the projection head that let query statistics influence
support embeddings during training.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fsgrade.models.backbones import Encoder


class EpisodeForward(NamedTuple):
    logits: torch.Tensor              # [n_query, n_way] -- REAL LOGITS
    query_embeddings: torch.Tensor
    support_embeddings: torch.Tensor
    prototypes: torch.Tensor | None


def encode_split(
    encoder: nn.Module,
    support_x: torch.Tensor,
    query_x: torch.Tensor,
    *,
    transductive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode support and query.

    ``transductive=False`` (default) uses separate forward passes so no
    normalisation layer can carry information from the query set into the
    support embeddings.
    """
    if transductive:
        both = encoder(torch.cat([support_x, query_x], dim=0))
        return both[: support_x.size(0)], both[support_x.size(0):]
    return encoder(support_x), encoder(query_x)


def class_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, n_way: int
) -> torch.Tensor:
    """Mean embedding per class. Vectorised; identical semantics to the loop version."""
    dim = embeddings.size(1)
    protos = torch.zeros(n_way, dim, device=embeddings.device, dtype=embeddings.dtype)
    counts = torch.zeros(n_way, device=embeddings.device, dtype=embeddings.dtype)
    protos.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
    return protos / counts.clamp(min=1).unsqueeze(1)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network (Snell et al., 2017), optionally with learned temperature."""

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        *,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze: Any = None,
        use_temperature: bool = True,
        temperature: float = 0.5,
        temperature_clamp: tuple[float, float] = (0.1, 2.0),
        use_batchnorm: bool = False,
        transductive: bool = False,
        distance: str = "euclidean",
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            backbone, embedding_dim, pretrained=pretrained, dropout=dropout,
            freeze=freeze, use_batchnorm=use_batchnorm,
        )
        self.use_temperature = use_temperature
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.temperature.requires_grad = bool(use_temperature)
        self.temperature_clamp = temperature_clamp
        self.transductive = transductive
        self.distance = distance
        self.embedding_dim = embedding_dim

    def forward(
        self, support_x: torch.Tensor, support_y: torch.Tensor,
        query_x: torch.Tensor, n_way: int = 2,
    ) -> EpisodeForward:
        se, qe = encode_split(self.encoder, support_x, query_x, transductive=self.transductive)
        protos = class_prototypes(se, support_y, n_way)

        if self.distance == "cosine":
            scores = F.normalize(qe, dim=1) @ F.normalize(protos, dim=1).t()
        else:
            scores = -torch.cdist(qe, protos, p=2)

        if self.use_temperature:
            t = self.temperature.clamp(min=self.temperature_clamp[0], max=self.temperature_clamp[1])
            scores = scores / t

        return EpisodeForward(scores, qe, se, protos)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.encoder.param_groups()
        groups["temperature"] = [self.temperature] if self.temperature.requires_grad else []
        return groups


class SiameseNetwork(nn.Module):
    """Siamese/Relation baseline (Koch et al., 2015) -- corrected to emit logits.

    Per-pair relation score s(q, x_i) is now unbounded (no sigmoid). Class logit
    aggregates over that class's support via ``logsumexp`` (soft-max pooling) or
    ``mean``, then applies a learnable scale.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        *,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze: Any = None,
        aggregate: str = "logsumexp",
        transductive: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            backbone, embedding_dim, pretrained=pretrained, dropout=dropout, freeze=freeze
        )
        self.relation = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            # No Sigmoid: CrossEntropyLoss needs logits, not probabilities.
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.aggregate = aggregate
        self.transductive = transductive
        self.embedding_dim = embedding_dim

    def forward(
        self, support_x: torch.Tensor, support_y: torch.Tensor,
        query_x: torch.Tensor, n_way: int = 2,
    ) -> EpisodeForward:
        se, qe = encode_split(self.encoder, support_x, query_x, transductive=self.transductive)
        n_q, n_s = qe.size(0), se.size(0)

        pairs = torch.cat(
            [qe.unsqueeze(1).expand(-1, n_s, -1), se.unsqueeze(0).expand(n_q, -1, -1)],
            dim=2,
        )
        scores = self.relation(pairs).squeeze(-1)          # [n_query, n_support]

        logits = torch.empty(n_q, n_way, device=qe.device, dtype=scores.dtype)
        for c in range(n_way):
            mask = support_y == c
            if not bool(mask.any()):
                logits[:, c] = torch.full((n_q,), float("-inf"), device=qe.device)
                continue
            sel = scores[:, mask]
            logits[:, c] = (
                torch.logsumexp(sel, dim=1) - torch.log(torch.tensor(float(sel.size(1))))
                if self.aggregate == "logsumexp"
                else sel.mean(dim=1)
            )

        return EpisodeForward(
            logits * self.logit_scale, qe, se, class_prototypes(se, support_y, n_way)
        )

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.encoder.param_groups()
        groups["head"] = groups["head"] + list(self.relation.parameters()) + [self.logit_scale]
        return groups


class MatchingNetwork(nn.Module):
    """Matching Network baseline (Vinyals et al., 2016) -- corrected to emit logits.

    Class logit is the log of the attention mass on that class::

        att        = Q S^T / tau
        logit[:,c] = logsumexp_{i : y_i = c} att[:, i]

    ``log_softmax`` of this is exactly the negative log-likelihood Vinyals et al.
    optimise. The original returned ``softmax(att) @ one_hot`` -- a probability
    vector -- and fed it to ``CrossEntropyLoss``, double-softmaxing it.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        *,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze: Any = None,
        temperature: float = 0.1,
        transductive: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            backbone, embedding_dim, pretrained=pretrained, dropout=dropout, freeze=freeze
        )
        # Parameterised in log space so it stays positive under unconstrained SGD.
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        self.transductive = transductive
        self.embedding_dim = embedding_dim

    def forward(
        self, support_x: torch.Tensor, support_y: torch.Tensor,
        query_x: torch.Tensor, n_way: int = 2,
    ) -> EpisodeForward:
        se, qe = encode_split(self.encoder, support_x, query_x, transductive=self.transductive)
        tau = self.log_temperature.exp().clamp(min=1e-3, max=10.0)
        attention = (F.normalize(qe, dim=1) @ F.normalize(se, dim=1).t()) / tau

        neg_inf = torch.finfo(attention.dtype).min
        logits = torch.empty(qe.size(0), n_way, device=qe.device, dtype=attention.dtype)
        for c in range(n_way):
            mask = (support_y == c).unsqueeze(0).expand_as(attention)
            logits[:, c] = torch.logsumexp(attention.masked_fill(~mask, neg_inf), dim=1)

        return EpisodeForward(
            logits, qe, se, class_prototypes(se, support_y, n_way)
        )

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.encoder.param_groups()
        groups["temperature"] = [self.log_temperature]
        return groups


class SupervisedClassifier(nn.Module):
    """Plain binary fresh/rotten CNN -- the control the original work never ran.

    This is the single most important baseline in the thesis. The label set is
    fixed across species, so nothing about the task *requires* a support set:
    this model is trained on the seen species and applied directly to an unseen
    one with zero target labels. If it matches the episodic methods, the
    prototypical machinery is not earning its place and the thesis must say so.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        n_classes: int = 2,
        *,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze: Any = None,
        embedding_dim: int = 256,
        n_species: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            backbone, embedding_dim, pretrained=pretrained, dropout=dropout,
            freeze=freeze, l2_normalize=False,
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)
        # Optional species head for the adversarial regulariser (contribution C3).
        self.species_head = nn.Linear(embedding_dim, n_species) if n_species > 0 else None
        self.embedding_dim = embedding_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def forward_with_species(
        self, x: torch.Tensor, grl_lambda: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        from fsgrade.training.grl import grad_reverse

        z = self.encoder(x)
        quality = self.classifier(z)
        species = None
        if self.species_head is not None and grl_lambda > 0:
            species = self.species_head(grad_reverse(z, grl_lambda))
        return quality, species, z

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.encoder.param_groups()
        groups["head"] = groups["head"] + list(self.classifier.parameters())
        if self.species_head is not None:
            groups["head"] = groups["head"] + list(self.species_head.parameters())
        return groups
