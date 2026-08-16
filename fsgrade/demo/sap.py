"""
Species-Anchored Prototypes, on arbitrary images.

Why this module exists
----------------------
``fsgrade/methods/frozen.py`` cannot classify an uploaded image. Its
``_features()`` takes *dataset-relative paths* and looks them up in a
``FeatureCache`` keyed by the dataset manifest hash; anything not in that cache
raises ``KeyError``. That is correct for reproducible experiments and useless
for a demo where the examiner drops in a photo of their own lunch.

The escape hatch is that ``FoundationEncoder.encode_images`` accepts an
arbitrary tensor, and ``text_prototypes`` accepts a **free-form species string**.
So the demo can do something the experiment harness deliberately cannot: grade a
fruit the model has never seen, named by typing its name.

This module mirrors ``SpeciesAnchoredPrototypes._blend`` exactly -- same
shrinkage rule, same L2 normalisation, same scaled-cosine logits -- but
tensor-in. A parity test pins the two implementations together so they cannot
drift.

On kappa
--------
κ is **read, never fitted**. ``_calibrate`` grid-searches it on *seen-species*
validation episodes; doing that here on the demo species would leak target
information and invalidate the whole cross-species claim. The demo therefore
reports the calibrated value from a run, or says plainly that it is using the
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

# Matches SpeciesAnchoredPrototypes.__init__ defaults in fsgrade/methods/frozen.py.
DEFAULT_KAPPA = 5.0
DEFAULT_SCALE = 100.0
DEFAULT_TEMPLATE = "descriptive"
DEFAULT_USE_SYNONYMS = True


class SapUnavailable(RuntimeError):
    """CLIP is not installed or its weights are not present locally."""

    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mirror of ``fsgrade.methods.frozen._l2``."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def effective_alpha(k: int, kappa: float, *, mode: str = "shot_adaptive",
                    fixed_alpha: float = 0.5) -> float:
    """Weight on the visual prototype.

    A K-shot mean has variance ~1/K, so the visual prototype earns more trust as
    K grows: ``alpha = K / (K + kappa)``. At K = 0 this is 0, i.e. pure
    zero-shot text -- one model spans the whole shot range.
    """
    if mode != "shot_adaptive":
        return float(fixed_alpha)
    denom = float(k) + float(kappa)
    return float(k) / denom if denom > 0 else 0.0


def blend_prototypes(
    text_protos: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    n_way: int,
    *,
    kappa: float = DEFAULT_KAPPA,
    mode: str = "shot_adaptive",
    fixed_alpha: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Blend text and visual prototypes. Exact mirror of ``_blend``.

    ``support_features`` must already be L2-normalised, matching what
    ``CachedFeatureMethod._features`` returns.
    """
    text_protos = np.asarray(text_protos, dtype=np.float32)
    support_labels = np.asarray(support_labels, dtype=int)

    k = int(len(support_labels) // max(n_way, 1))
    alpha = effective_alpha(k, kappa, mode=mode, fixed_alpha=fixed_alpha)

    protos = []
    for c in range(n_way):
        mask = support_labels == c
        if mask.any() and len(support_features):
            visual = l2_normalize(support_features[mask].mean(axis=0))
        else:
            # No support for this class: fall back to pure text, which is
            # exactly what alpha=0 would give.
            visual = np.zeros_like(text_protos[c])
        protos.append(l2_normalize(alpha * visual + (1.0 - alpha) * text_protos[c]))
    return np.stack(protos).astype(np.float32), alpha


@dataclass
class SapResult:
    logits: np.ndarray
    prototypes: np.ndarray
    text_prototypes: np.ndarray
    query_features: np.ndarray
    support_features: np.ndarray
    alpha: float
    k_shot: int
    kappa: float
    kappa_source: str
    species: str
    prompts: dict[str, list[str]] = field(default_factory=dict)

    def probabilities(self) -> np.ndarray:
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class SapEngine:
    """Holds a frozen CLIP encoder and serves SAP on arbitrary tensors."""

    def __init__(
        self,
        encoder_name: str = "clip_vitb16",
        *,
        device: str | torch.device = "cpu",
        classes: Sequence[str] = ("fresh", "rotten"),
        template: str = DEFAULT_TEMPLATE,
        use_synonyms: bool = DEFAULT_USE_SYNONYMS,
        scale: float = DEFAULT_SCALE,
        kappa: float = DEFAULT_KAPPA,
        kappa_source: str = "default, not calibrated",
    ) -> None:
        self.encoder_name = encoder_name
        self.device = torch.device(device)
        self.classes = list(classes)
        self.template = template
        self.use_synonyms = use_synonyms
        self.scale = scale
        self.kappa = kappa
        self.kappa_source = kappa_source
        self._encoder: Any = None
        self._text_cache: dict[tuple[str, str, bool], np.ndarray] = {}

    # ------------------------------------------------------------------ #
    @property
    def encoder(self) -> Any:
        """Lazily build the CLIP encoder, with an actionable error if absent."""
        if self._encoder is None:
            try:
                from fsgrade.models.foundation import (
                    FoundationUnavailable,
                    build_foundation_encoder,
                )
            except ImportError as exc:  # pragma: no cover
                raise SapUnavailable(str(exc)) from exc

            try:
                self._encoder = build_foundation_encoder(self.encoder_name, device=self.device)
            except FoundationUnavailable as exc:
                raise SapUnavailable(
                    f"{self.encoder_name} is unavailable: {exc}",
                    remediation="pip install open_clip_torch, then run "
                                "python -m fsgrade.demo.prefetch",
                ) from exc
            except Exception as exc:  # noqa: BLE001 - offline weight lookup failure
                raise SapUnavailable(
                    f"Could not build {self.encoder_name}: {type(exc).__name__}: {exc}",
                    remediation="Run `python -m fsgrade.demo.prefetch` once while online "
                                "to cache the weights locally.",
                ) from exc
        return self._encoder

    def is_available(self) -> bool:
        try:
            _ = self.encoder
            return True
        except SapUnavailable:
            return False

    def transform_spec(self):
        from fsgrade.demo.preprocess import transform_spec

        return transform_spec("clip", encoder=self.encoder)

    # ------------------------------------------------------------------ #
    def text_prototypes(self, species: str) -> np.ndarray:
        """Species-conditioned text prototypes -- no labelled images required.

        This is the part that makes SAP interesting: an unseen species gets a
        usable prototype pair from its *name* alone.
        """
        key = (species.strip().lower(), self.template, self.use_synonyms)
        if key not in self._text_cache:
            from fsgrade.models.foundation import text_prototypes

            protos = text_prototypes(
                self.encoder, species, self.classes, self.device,
                template=self.template, use_synonyms=self.use_synonyms,
            )
            self._text_cache[key] = protos.detach().cpu().numpy().astype(np.float32)
        return self._text_cache[key]

    def prompts_for(self, species: str) -> dict[str, list[str]]:
        """The exact prompt strings used, so the UI can show them."""
        from fsgrade.models.foundation import build_prompts

        return build_prompts(
            species, self.classes,
            template=self.template, use_synonyms=self.use_synonyms,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> np.ndarray:
        """Embed a preprocessed image batch and L2-normalise.

        Matches ``CachedFeatureMethod._features``, which normalises by default.
        """
        if images.numel() == 0:
            return np.zeros((0, self.encoder.out_dim), dtype=np.float32)
        feats = self.encoder.encode_images(images.to(self.device))
        return l2_normalize(feats.detach().cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------------ #
    def predict(
        self,
        species: str,
        query_features: np.ndarray,
        support_features: np.ndarray | None = None,
        support_labels: np.ndarray | None = None,
        *,
        kappa: float | None = None,
        mode: str = "shot_adaptive",
        fixed_alpha: float = 0.5,
    ) -> SapResult:
        """Grade queries for ``species`` using K-shot support plus text anchors.

        With no support set this degenerates exactly to zero-shot CLIP, which is
        why the same engine serves both arms of the comparison.
        """
        n_way = len(self.classes)
        text = self.text_prototypes(species)
        kappa = self.kappa if kappa is None else float(kappa)

        if support_features is None or not len(support_features):
            support_features = np.zeros((0, text.shape[1]), dtype=np.float32)
            support_labels = np.zeros(0, dtype=int)
        support_labels = np.asarray(support_labels, dtype=int)

        protos, alpha = blend_prototypes(
            text, support_features, support_labels, n_way,
            kappa=kappa, mode=mode, fixed_alpha=fixed_alpha,
        )

        query_features = np.asarray(query_features, dtype=np.float32)
        logits = (
            (query_features @ protos.T) * self.scale
            if len(query_features)
            else np.zeros((0, n_way), dtype=np.float32)
        )

        return SapResult(
            logits=logits.astype(np.float32),
            prototypes=protos,
            text_prototypes=text,
            query_features=query_features,
            support_features=support_features,
            alpha=alpha,
            k_shot=int(len(support_labels) // max(n_way, 1)),
            kappa=kappa,
            kappa_source=self.kappa_source,
            species=species,
            prompts=self.prompts_for(species),
        )

    # ------------------------------------------------------------------ #
    def set_kappa(self, kappa: float, source: str) -> None:
        self.kappa = float(kappa)
        self.kappa_source = source

    def adopt_kappa_from(self, method_ref: Any) -> None:
        """Take κ from a discovered run, keeping its provenance string."""
        if getattr(method_ref, "kappa", None) is not None:
            self.set_kappa(method_ref.kappa, method_ref.kappa_source or "from run state")


def kappa_grid() -> list[float]:
    """The grid ``_calibrate`` searches -- used to tick the UI dial."""
    return [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
