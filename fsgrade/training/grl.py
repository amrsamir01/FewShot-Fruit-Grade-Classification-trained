"""
Gradient reversal for species-adversarial training (contribution C3).

The training set carries species labels that the original model never used: it
knows every training image is an apple, a banana or a grape, and discards that
information entirely. Under the Cross-Species Quality Grading task the species
is precisely the *nuisance* variable -- we want an embedding that separates
fresh from rotten while being uninformative about which fruit it is.

A gradient reversal layer (Ganin & Lempitsky, 2015) gives this for free: the
species head is trained to predict species, while the reversed gradient pushes
the encoder to make that prediction impossible.

``lambda`` is ramped from 0 following the DANN schedule. Starting at full
strength destabilises training, because early on the species head is random and
its gradients are noise.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Identity forward, negated-and-scaled gradient backward."""
    return _GradientReversal.apply(x, lambda_)


def dann_lambda(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """DANN schedule: 2/(1+exp(-gamma*p)) - 1, ramping 0 -> max_lambda.

    ``progress`` runs 0 -> 1 over training.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    return float(max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


class SpeciesAdversary(nn.Module):
    """Species classifier trained behind a gradient reversal layer."""

    def __init__(self, embedding_dim: int, n_species: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_species),
        )
        self.n_species = n_species

    def forward(self, embeddings: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(embeddings, lambda_))

    def loss(
        self, embeddings: torch.Tensor, species_labels: torch.Tensor, lambda_: float = 1.0
    ) -> torch.Tensor:
        return F.cross_entropy(self(embeddings, lambda_), species_labels)
