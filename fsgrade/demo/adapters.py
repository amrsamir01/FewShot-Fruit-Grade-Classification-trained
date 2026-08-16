"""
One tensor-in interface over every arm.

The experiment harness drives methods through ``predict_episode(EpisodeBatch)``,
and ``EpisodeBatch`` reads ``n_way`` off an ``EpisodeSpec`` -- which means it
cannot be constructed without a dataset episode. The demo has no episodes, only
whatever the user dropped on the page, so it talks to models directly.

Every adapter returns an ``ArmResult`` carrying not just logits but the geometry
needed to draw the decision plane: prototypes (or a linear head's weights),
support embeddings and query embeddings, all in one consistent space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import torch

from fsgrade.demo.projection import (
    DegenerateFrameError,
    LogitForm,
    PlaneFrame,
    build_frame,
    build_frame_from_linear_head,
)

ArmKind = Literal["episodic", "supervised_linear", "supervised_centroid", "sap", "clip_text", "pixel"]


@dataclass
class ArmResult:
    """One arm's verdict on one query set, plus the geometry to draw it."""

    arm: str
    logits: np.ndarray                       # [n_query, n_way]
    query_embeddings: np.ndarray | None = None
    support_embeddings: np.ndarray | None = None
    prototypes: np.ndarray | None = None
    linear_head: tuple[np.ndarray, np.ndarray | None] | None = None
    logit_form: LogitForm = "cosine"
    scale: float = 1.0
    k_shot: int = 0
    uses_support: bool = True
    seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def probabilities(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predictions(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    def margins(self) -> np.ndarray:
        """Signed logit difference, positive = rotten."""
        if self.logits.shape[1] < 2:
            return np.zeros(len(self.logits))
        return self.logits[:, 1] - self.logits[:, 0]

    def accuracy(self, y_true: np.ndarray | None) -> float | None:
        if y_true is None or not len(self.logits):
            return None
        y_true = np.asarray(y_true)
        labelled = y_true >= 0
        if not labelled.any():
            return None
        return float((self.predictions()[labelled] == y_true[labelled]).mean())

    def build_frame(self) -> PlaneFrame | None:
        """The decision plane for this arm, or None when it has no geometry."""
        try:
            cloud = _stack(self.support_embeddings, self.query_embeddings)
            if self.prototypes is not None:
                return build_frame(
                    self.prototypes, cloud if len(cloud) else None,
                    logit_form=self.logit_form, scale=self.scale, built_at_k=self.k_shot,
                )
            if self.linear_head is not None and len(cloud):
                W, b = self.linear_head
                return build_frame_from_linear_head(W, b, cloud, built_at_k=self.k_shot)
        except DegenerateFrameError:
            return None
        return None


def _stack(*arrays: np.ndarray | None) -> np.ndarray:
    parts = [np.atleast_2d(a) for a in arrays if a is not None and len(a)]
    return np.vstack(parts) if parts else np.zeros((0, 1))


class Arm(Protocol):
    name: str
    kind: ArmKind
    uses_support: bool

    def predict(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        **kwargs: Any,
    ) -> ArmResult: ...


# --------------------------------------------------------------------------- #
#  Adapters
# --------------------------------------------------------------------------- #

class EpisodicArm:
    """ProtoNet / Siamese / Matching -- support and query in, logits out."""

    kind: ArmKind = "episodic"
    uses_support = True

    def __init__(self, name: str, module: torch.nn.Module, device: torch.device,
                 n_way: int = 2, distance: str = "euclidean") -> None:
        self.name = name
        self.module = module
        self.device = device
        self.n_way = n_way
        self.distance = distance

    @torch.no_grad()
    def predict(self, support, support_labels, query, **_: Any) -> ArmResult:
        t0 = time.perf_counter()
        if not len(support) or not len(query):
            raise ArmNeedsMore("episodic methods need at least one support image per class")

        out = self.module(
            support.to(self.device), support_labels.to(self.device),
            query.to(self.device), self.n_way,
        )
        protos = _np(out.prototypes)
        # ProtoNet uses negative Euclidean distance; the cosine variant and the
        # relation/attention heads are not a plane in prototype space, so we
        # only claim exactness for the Euclidean form.
        form: LogitForm = "euclidean" if self.distance == "euclidean" else "cosine"
        return ArmResult(
            arm=self.name,
            logits=_np(out.logits),
            query_embeddings=_np(out.query_embeddings),
            support_embeddings=_np(out.support_embeddings),
            prototypes=protos,
            logit_form=form,
            k_shot=int(len(support_labels) // max(self.n_way, 1)),
            uses_support=True,
            seconds=time.perf_counter() - t0,
            extras={"temperature": float(getattr(self.module, "temperature", torch.tensor(float("nan"))).item())
                    if hasattr(self.module, "temperature") else None},
        )


class SupervisedLinearArm:
    """The pivotal control: a plain classifier, no support set at all."""

    kind: ArmKind = "supervised_linear"
    uses_support = False

    def __init__(self, name: str, module: torch.nn.Module, device: torch.device) -> None:
        self.name = name
        self.module = module
        self.device = device

    @torch.no_grad()
    def predict(self, support, support_labels, query, **_: Any) -> ArmResult:
        t0 = time.perf_counter()
        if not len(query):
            return ArmResult(arm=self.name, logits=np.zeros((0, 2), np.float32), uses_support=False)

        q = query.to(self.device)
        logits = self.module(q)
        embeddings = self.module.embed(q)
        W = _np(self.module.classifier.weight)
        b = _np(self.module.classifier.bias) if self.module.classifier.bias is not None else None

        return ArmResult(
            arm=self.name,
            logits=_np(logits),
            query_embeddings=_np(embeddings),
            linear_head=(W, b),
            logit_form="linear",
            k_shot=0,
            uses_support=False,
            seconds=time.perf_counter() - t0,
            extras={"n_support_used": 0},
        )


class SupervisedCentroidArm:
    """Same encoder as the linear arm, but a centroid over the target support.

    Isolates exactly what the K labelled images contribute, because both arms
    read the same weights.
    """

    kind: ArmKind = "supervised_centroid"
    uses_support = True

    def __init__(self, name: str, module: torch.nn.Module, device: torch.device,
                 n_way: int = 2, scale: float = 10.0) -> None:
        self.name = name
        self.module = module
        self.device = device
        self.n_way = n_way
        self.scale = scale

    @torch.no_grad()
    def predict(self, support, support_labels, query, **_: Any) -> ArmResult:
        import torch.nn.functional as F

        t0 = time.perf_counter()
        if not len(support) or not len(query):
            raise ArmNeedsMore("nearest-centroid needs at least one support image per class")

        se = F.normalize(self.module.embed(support.to(self.device)), dim=1)
        qe = F.normalize(self.module.embed(query.to(self.device)), dim=1)
        sy = support_labels.to(self.device)

        protos = torch.stack([
            F.normalize(se[sy == c].mean(0), dim=0) if bool((sy == c).any())
            else torch.zeros_like(se[0])
            for c in range(self.n_way)
        ])
        return ArmResult(
            arm=self.name,
            logits=_np((qe @ protos.t()) * self.scale),
            query_embeddings=_np(qe),
            support_embeddings=_np(se),
            prototypes=_np(protos),
            logit_form="cosine",
            scale=self.scale,
            k_shot=int(len(support_labels) // max(self.n_way, 1)),
            uses_support=True,
            seconds=time.perf_counter() - t0,
        )


class SapArm:
    """Species-Anchored Prototypes on arbitrary tensors."""

    kind: ArmKind = "sap"
    uses_support = True

    def __init__(self, name: str, engine: Any, species: str, *, zero_shot: bool = False) -> None:
        self.name = name
        self.engine = engine
        self.species = species
        self.zero_shot = zero_shot
        self.uses_support = not zero_shot

    def predict(self, support, support_labels, query, *, kappa: float | None = None,
                species: str | None = None, **_: Any) -> ArmResult:
        t0 = time.perf_counter()
        target = species or self.species
        if not len(query):
            return ArmResult(arm=self.name, logits=np.zeros((0, 2), np.float32))

        qf = self.engine.encode(query)
        if self.zero_shot or not len(support):
            sf, sy = None, None
        else:
            sf, sy = self.engine.encode(support), support_labels.cpu().numpy()

        res = self.engine.predict(target, qf, sf, sy, kappa=kappa)
        return ArmResult(
            arm=self.name,
            logits=res.logits,
            query_embeddings=res.query_features,
            support_embeddings=res.support_features if len(res.support_features) else None,
            prototypes=res.prototypes,
            logit_form="cosine",
            scale=self.engine.scale,
            k_shot=res.k_shot,
            uses_support=self.uses_support,
            seconds=time.perf_counter() - t0,
            extras={
                "alpha": res.alpha,
                "kappa": res.kappa,
                "kappa_source": res.kappa_source,
                "species": res.species,
                "prompts": res.prompts,
                "text_prototypes": res.text_prototypes.tolist(),
            },
        )


class PixelArm:
    """Nearest centroid on raw pixels -- the training-free floor."""

    kind: ArmKind = "pixel"
    uses_support = True

    def __init__(self, name: str = "nc_pixel", n_way: int = 2) -> None:
        self.name = name
        self.n_way = n_way

    def predict(self, support, support_labels, query, **_: Any) -> ArmResult:
        t0 = time.perf_counter()
        if not len(support) or not len(query):
            raise ArmNeedsMore("the pixel baseline needs at least one support image per class")

        s = support.flatten(1)
        q = query.flatten(1)
        protos = torch.stack([
            s[support_labels == c].mean(0) if bool((support_labels == c).any())
            else torch.zeros_like(s[0])
            for c in range(self.n_way)
        ])
        return ArmResult(
            arm=self.name,
            logits=_np(-torch.cdist(q, protos, p=2)),
            query_embeddings=_np(q),
            support_embeddings=_np(s),
            prototypes=_np(protos),
            logit_form="euclidean",
            k_shot=int(len(support_labels) // max(self.n_way, 1)),
            seconds=time.perf_counter() - t0,
        )


class ChanceArm:
    """Uniform logits, so 86% is read against 50% and not against 0%."""

    kind: ArmKind = "pixel"
    uses_support = False

    def __init__(self, name: str = "chance", n_way: int = 2) -> None:
        self.name = name
        self.n_way = n_way

    def predict(self, support, support_labels, query, **_: Any) -> ArmResult:
        return ArmResult(
            arm=self.name,
            logits=np.zeros((len(query), self.n_way), dtype=np.float32),
            uses_support=False,
        )


class ArmNeedsMore(RuntimeError):
    """The arm cannot run yet -- usually an empty support class."""


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()
