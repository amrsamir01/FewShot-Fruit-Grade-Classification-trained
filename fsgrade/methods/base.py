"""
The ``FewShotMethod`` interface.

Every adaptation strategy -- from a plain zero-shot CNN to SAP -- implements
this one protocol, so a single evaluation loop drives all of them on the same
episode bank. That is what makes the comparison unconfounded: the original code
had five near-duplicate training loops with different learning rates, epoch
budgets and regularisation, so no accuracy difference between methods was
attributable to the method itself.

**The logits contract.** ``predict_episode`` returns real logits: unbounded,
shift-invariant, suitable for ``log_softmax``. Enforced by
``tests/test_methods_contract.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from fsgrade.data.episodes import FoldSpec
from fsgrade.data.loaders import EpisodeBatch


class MethodUnavailable(RuntimeError):
    """Raised when a method's optional dependency is missing.

    Recorded in ``metrics.json`` with the reason rather than crashing a sweep --
    a missing ``open_clip`` should not cost you the other eleven methods.
    """


@dataclass
class FitContext:
    """Everything a method needs to prepare itself for one fold."""

    fold: FoldSpec
    cfg: dict[str, Any]
    device: torch.device
    data_root: str
    index: Any                       # ImageIndex
    train_pools: dict[str, dict[str, list[str]]]
    val_pools: dict[str, dict[str, list[str]]]
    train_bank: Any = None           # EpisodeBank | None
    val_bank: Any = None             # EpisodeBank | None
    run: Any = None                  # RunDirectory | None
    logger: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeOutput:
    """One episode's predictions."""

    logits: np.ndarray                       # [n_query, n_way], real logits
    query_embeddings: np.ndarray | None = None
    support_embeddings: np.ndarray | None = None
    prototypes: np.ndarray | None = None
    adapt_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def probabilities(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predictions(self) -> np.ndarray:
        return self.logits.argmax(axis=1)


@runtime_checkable
class FewShotMethod(Protocol):
    name: str
    family: str
    requires_support: bool
    trains_episodically: bool
    adapts_per_episode: bool

    def prepare(self, ctx: FitContext) -> None:
        """Called once per fold: build/restore the encoder, train if needed."""

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        """Predict the query labels of one episode."""

    def state_summary(self) -> dict[str, Any]:
        """Parameter counts, checkpoint path, training spec hash."""


class BaseMethod:
    """Convenience base with sensible defaults."""

    name: str = "base"
    family: str = "unknown"
    requires_support: bool = True
    trains_episodically: bool = False
    adapts_per_episode: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs
        self.model: torch.nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self._summary: dict[str, Any] = {}

    def prepare(self, ctx: FitContext) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:  # pragma: no cover
        raise NotImplementedError

    def state_summary(self) -> dict[str, Any]:
        out = dict(self._summary)
        out.setdefault("name", self.name)
        out.setdefault("family", self.family)
        out.setdefault("requires_support", self.requires_support)
        out.setdefault("adapts_per_episode", self.adapts_per_episode)
        if self.model is not None:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            out.setdefault(
                "params", {"total": total, "trainable": trainable, "frozen": total - trainable}
            )
        return out

    # -- helpers shared by concrete methods ----------------------------- #
    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()

    def _eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()
