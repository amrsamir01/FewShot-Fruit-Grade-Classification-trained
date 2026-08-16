"""
Loss functions.

``supervised_contrastive_loss`` is carried over from the original
``src/losses.py`` essentially unchanged -- the ``mask - eye`` self-exclusion is
correct SupCon (Khosla et al., 2020) and there was no bug to fix.

The one behavioural change: ``forward`` now returns a **dict** of named
components rather than a 3-tuple, so ``history.json`` can log each term by name
and the component ablation can attribute changes to a specific loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    """Cross-entropy (with label smoothing) + optional supervised contrastive term.

    Note on label smoothing: it systematically pushes the model toward
    underconfidence, which shows up directly in ECE and Brier score. Since
    calibration is now a headline metric, ``label_smoothing`` is a factor worth
    sweeping rather than a fixed 0.1.
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def supervised_contrastive_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        n = embeddings.size(0)

        sim = embeddings @ embeddings.t() / self.temperature
        labels = labels.view(-1, 1)
        eye = torch.eye(n, device=embeddings.device)
        positive_mask = torch.eq(labels, labels.t()).float() - eye

        # Numerical stability: subtract the row max before exponentiating.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        denom = positive_mask.sum(dim=1).clamp(min=1)
        return -((positive_mask * log_prob).sum(dim=1) / denom).mean()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        all_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = self.ce(logits, labels)
        components: dict[str, torch.Tensor] = {"ce": ce}
        total = ce

        if embeddings is not None and all_labels is not None and self.contrastive_weight > 0:
            supcon = self.supervised_contrastive_loss(embeddings, all_labels)
            components["supcon"] = supcon
            total = total + self.contrastive_weight * supcon

        components["total"] = total
        return components


class SupervisedLoss(nn.Module):
    """Cross-entropy for the plain binary classifier, with optional species adversary."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        adversarial_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.adversarial_weight = adversarial_weight

    def forward(
        self,
        quality_logits: torch.Tensor,
        quality_labels: torch.Tensor,
        species_logits: torch.Tensor | None = None,
        species_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = self.ce(quality_logits, quality_labels)
        components: dict[str, torch.Tensor] = {"ce": ce}
        total = ce

        if (
            species_logits is not None
            and species_labels is not None
            and self.adversarial_weight > 0
        ):
            species_ce = F.cross_entropy(species_logits, species_labels)
            components["species_ce"] = species_ce
            # The GRL already negates the gradient reaching the encoder, so the
            # term is *added* here, not subtracted.
            total = total + self.adversarial_weight * species_ce

        components["total"] = total
        return components
