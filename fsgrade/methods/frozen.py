"""
Methods over frozen features: nearest-centroid, linear probe, and SAP.

All of these consume cached features, so an entire fold evaluates in seconds of
CPU work once the cache exists.

``SpeciesAnchoredPrototypes`` (SAP) is the thesis's proposed method. The idea
follows from the structure of the task: under Cross-Species Quality Grading the
*label semantics* are invariant (fresh is fresh) but the *class-conditional
appearance* is species-specific (a rotten banana is brown-black; a rotten orange
is fuzzy white-green mould). A visual prototype built from K target images
cannot carry that species-conditional prior when K is small -- but CLIP's text
tower can supply it for free, with zero labelled target images.

    p_c = normalize( alpha_c * p_c^visual + (1 - alpha_c) * p_c^text )

At K=0 this degenerates to zero-shot CLIP; as K grows, alpha shifts weight
toward the visual prototype. The falsifiable claim is that the text anchor helps
most at low K.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from fsgrade.data.loaders import EpisodeBatch
from fsgrade.methods.base import BaseMethod, EpisodeOutput, FitContext, MethodUnavailable


def _l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


class CachedFeatureMethod(BaseMethod):
    """Shared machinery for methods that read from a feature cache."""

    family = "frozen"
    requires_support = True
    trains_episodically = False
    adapts_per_episode = True

    def __init__(self, encoder_name: str = "dinov2_vits14", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.cache = None
        self._normalize = bool(kwargs.get("normalize_features", True))

    def prepare(self, ctx: FitContext) -> None:
        import torch

        from fsgrade.config import get_in
        from fsgrade.models.foundation import get_or_build_cache

        self.device = ctx.device
        self.cache = get_or_build_cache(
            self.encoder_name,
            ctx.index,
            cache_root=get_in(ctx.cfg, "paths.cache_root", "./.cache"),
            device=ctx.device,
            batch_size=int(get_in(ctx.cfg, "eval.feature_batch_size", 64)),
            num_workers=int(get_in(ctx.cfg, "num_workers", 0)),
            logger=ctx.logger,
        )
        self._summary["encoder"] = self.encoder_name
        self._summary["cache_dir"] = str(self.cache.directory)
        self._summary["params"] = {"total": 0, "trainable": 0, "frozen": 0}
        self._summary["params_note"] = (
            "Encoder is frozen and shared; no task-specific parameters are trained."
        )

    def _features(self, relpaths: Sequence[str]) -> np.ndarray:
        assert self.cache is not None
        feats = self.cache.lookup(relpaths)
        return _l2(feats) if self._normalize else feats


class NearestCentroid(CachedFeatureMethod):
    """Class centroid of the support features; cosine similarity as the logit."""

    name = "ncc_frozen"

    def __init__(self, encoder_name: str = "dinov2_vits14", scale: float = 10.0, **kwargs: Any):
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.scale = scale

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        t0 = time.perf_counter()
        spec = batch.spec
        sf = self._features(spec.support_paths())
        qf = self._features(spec.query_paths())
        sy = np.asarray([lbl for lbl, _ in spec.support])

        protos = np.stack([_l2(sf[sy == c].mean(axis=0)) for c in range(spec.n_way)])
        # Scaled cosine similarity: real logits, and the scale keeps the softmax
        # from being near-uniform over a [-1, 1] range.
        logits = (qf @ protos.T) * self.scale
        return EpisodeOutput(
            logits=logits.astype(np.float32),
            query_embeddings=qf,
            support_embeddings=sf,
            prototypes=protos,
            adapt_seconds=time.perf_counter() - t0,
        )


class LinearProbe(CachedFeatureMethod):
    """Logistic regression fitted on the support features of each episode."""

    name = "probe_frozen"

    def __init__(self, encoder_name: str = "dinov2_vits14", C: float = 1.0, **kwargs: Any):
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.C = C

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        from sklearn.linear_model import LogisticRegression

        t0 = time.perf_counter()
        spec = batch.spec
        sf = self._features(spec.support_paths())
        qf = self._features(spec.query_paths())
        sy = np.asarray([lbl for lbl, _ in spec.support])

        if len(np.unique(sy)) < 2:  # pragma: no cover - guarded at bank build
            logits = np.zeros((len(qf), spec.n_way), dtype=np.float32)
            return EpisodeOutput(logits=logits, adapt_seconds=time.perf_counter() - t0)

        clf = LogisticRegression(C=self.C, max_iter=1000, random_state=int(spec.seed) % 2**31)
        clf.fit(sf, sy)
        logits = clf.decision_function(qf)
        if logits.ndim == 1:  # binary -> single margin; expand to two columns
            logits = np.stack([-logits, logits], axis=1)
        return EpisodeOutput(
            logits=logits.astype(np.float32),
            query_embeddings=qf,
            support_embeddings=sf,
            adapt_seconds=time.perf_counter() - t0,
        )


class ZeroShotText(CachedFeatureMethod):
    """Pure zero-shot CLIP: text prototypes only, no support images at all.

    ``requires_support = False`` -- this method is evaluated at K=0 and gives the
    reference point for how much the few labelled target images actually buy.
    """

    name = "clip_text_zeroshot"
    family = "zero_shot"
    requires_support = False
    adapts_per_episode = False

    def __init__(
        self,
        encoder_name: str = "clip_vitb16",
        template: str = "descriptive",
        use_synonyms: bool = True,
        scale: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.template = template
        self.use_synonyms = use_synonyms
        self.scale = scale
        self._text_cache: dict[str, np.ndarray] = {}
        self._encoder = None

    def prepare(self, ctx: FitContext) -> None:
        super().prepare(ctx)
        from fsgrade.models.foundation import build_foundation_encoder

        if not self.encoder_name.startswith("clip"):
            raise MethodUnavailable(
                f"{self.name} needs a CLIP-style encoder with a text tower, "
                f"got {self.encoder_name!r}"
            )
        self._encoder = build_foundation_encoder(self.encoder_name, device=ctx.device)
        self._classes = list(ctx.index.classes)
        self._summary.update({"template": self.template, "use_synonyms": self.use_synonyms})

    def _text_protos(self, species: str) -> np.ndarray:
        if species not in self._text_cache:
            from fsgrade.models.foundation import text_prototypes

            protos = text_prototypes(
                self._encoder, species, self._classes, self.device,
                template=self.template, use_synonyms=self.use_synonyms,
            )
            self._text_cache[species] = protos.cpu().numpy().astype(np.float32)
        return self._text_cache[species]

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        t0 = time.perf_counter()
        spec = batch.spec
        qf = self._features(spec.query_paths())
        protos = self._text_protos(spec.species)
        logits = (qf @ protos.T) * self.scale
        return EpisodeOutput(
            logits=logits.astype(np.float32),
            query_embeddings=qf,
            prototypes=protos,
            adapt_seconds=time.perf_counter() - t0,
            extras={"n_support_used": 0},
        )


class SpeciesAnchoredPrototypes(CachedFeatureMethod):
    """SAP -- the proposed method. Blends visual and text prototypes.

    ``alpha`` may be fixed, or calibrated on the *seen* species during
    ``prepare`` (never on the test species). ``alpha_mode='shot_adaptive'``
    implements the shrinkage intuition that a K-shot mean has variance ~1/K, so
    the weight on the visual prototype should grow with K:

        alpha(K) = K / (K + kappa)

    with kappa fitted on seen species. At K=0 this gives alpha=0, i.e. pure
    zero-shot text -- so a single model spans the whole shot range.
    """

    name = "sap"
    family = "vision_language"
    requires_support = True
    adapts_per_episode = True

    def __init__(
        self,
        encoder_name: str = "clip_vitb16",
        template: str = "descriptive",
        use_synonyms: bool = True,
        alpha: float = 0.5,
        alpha_mode: str = "shot_adaptive",
        kappa: float = 5.0,
        scale: float = 100.0,
        calibrate_on_seen: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.template = template
        self.use_synonyms = use_synonyms
        self.alpha = alpha
        self.alpha_mode = alpha_mode
        self.kappa = kappa
        self.scale = scale
        self.calibrate_on_seen = calibrate_on_seen
        self._text_cache: dict[str, np.ndarray] = {}
        self._encoder = None

    # ------------------------------------------------------------------ #
    def prepare(self, ctx: FitContext) -> None:
        super().prepare(ctx)
        from fsgrade.models.foundation import build_foundation_encoder

        if not self.encoder_name.startswith("clip"):
            raise MethodUnavailable(
                f"SAP requires a CLIP-style encoder with a text tower, got {self.encoder_name!r}"
            )
        self._encoder = build_foundation_encoder(self.encoder_name, device=ctx.device)
        self._classes = list(ctx.index.classes)

        if self.calibrate_on_seen and ctx.val_bank is not None:
            self._calibrate(ctx)

        self._summary.update({
            "template": self.template,
            "use_synonyms": self.use_synonyms,
            "alpha_mode": self.alpha_mode,
            "alpha": self.alpha,
            "kappa": self.kappa,
        })

    def _calibrate(self, ctx: FitContext) -> None:
        """Grid-search alpha/kappa on validation episodes from SEEN species only.

        Calibrating on the held-out species would leak target information and
        invalidate the whole cross-species claim.
        """
        bank = ctx.val_bank
        if bank is None or len(bank) == 0:
            return

        episodes = []
        for spec in list(bank)[:200]:
            sf = self._features(spec.support_paths())
            qf = self._features(spec.query_paths())
            sy = np.asarray([lbl for lbl, _ in spec.support])
            qy = np.asarray([lbl for lbl, _ in spec.query])
            episodes.append((spec.species, sf, sy, qf, qy, spec.n_way))

        if self.alpha_mode == "shot_adaptive":
            grid = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
            best, best_acc = self.kappa, -1.0
            for kappa in grid:
                acc = self._grid_accuracy(episodes, kappa=kappa)
                if acc > best_acc:
                    best, best_acc = kappa, acc
            self.kappa = best
            self._summary["calibration"] = {
                "parameter": "kappa", "value": best,
                "val_accuracy": best_acc, "grid": grid,
                "n_val_episodes": len(episodes),
                "calibrated_on": "seen species validation episodes",
            }
        else:
            grid = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
            best, best_acc = self.alpha, -1.0
            for alpha in grid:
                acc = self._grid_accuracy(episodes, alpha=alpha)
                if acc > best_acc:
                    best, best_acc = alpha, acc
            self.alpha = best
            self._summary["calibration"] = {
                "parameter": "alpha", "value": best,
                "val_accuracy": best_acc, "grid": grid,
                "n_val_episodes": len(episodes),
                "calibrated_on": "seen species validation episodes",
            }

        if ctx.logger:
            ctx.logger.info("SAP calibration on seen species: %s", self._summary["calibration"])

    def _grid_accuracy(
        self, episodes: list[tuple], *, alpha: float | None = None, kappa: float | None = None
    ) -> float:
        correct = total = 0
        for species, sf, sy, qf, qy, n_way in episodes:
            protos = self._blend(species, sf, sy, n_way, alpha=alpha, kappa=kappa)
            pred = (qf @ protos.T).argmax(axis=1)
            correct += int((pred == qy).sum())
            total += len(qy)
        return correct / total if total else 0.0

    # ------------------------------------------------------------------ #
    def _text_protos(self, species: str) -> np.ndarray:
        if species not in self._text_cache:
            from fsgrade.models.foundation import text_prototypes

            protos = text_prototypes(
                self._encoder, species, self._classes, self.device,
                template=self.template, use_synonyms=self.use_synonyms,
            )
            self._text_cache[species] = protos.cpu().numpy().astype(np.float32)
        return self._text_cache[species]

    def _effective_alpha(self, k: int, *, alpha: float | None, kappa: float | None) -> float:
        if self.alpha_mode == "shot_adaptive":
            kap = self.kappa if kappa is None else kappa
            return float(k) / (float(k) + kap) if (k + kap) > 0 else 0.0
        return self.alpha if alpha is None else alpha

    def _blend(
        self,
        species: str,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        n_way: int,
        *,
        alpha: float | None = None,
        kappa: float | None = None,
    ) -> np.ndarray:
        text = self._text_protos(species)
        k = int(len(support_labels) // max(n_way, 1))
        a = self._effective_alpha(k, alpha=alpha, kappa=kappa)

        protos = []
        for c in range(n_way):
            mask = support_labels == c
            visual = _l2(support_features[mask].mean(axis=0)) if mask.any() else np.zeros_like(text[c])
            protos.append(_l2(a * visual + (1.0 - a) * text[c]))
        return np.stack(protos)

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        t0 = time.perf_counter()
        spec = batch.spec
        sf = self._features(spec.support_paths()) if spec.support else np.zeros((0, 1), np.float32)
        qf = self._features(spec.query_paths())
        sy = np.asarray([lbl for lbl, _ in spec.support], dtype=int)

        protos = self._blend(spec.species, sf, sy, spec.n_way)
        logits = (qf @ protos.T) * self.scale
        k = int(len(sy) // max(spec.n_way, 1))
        return EpisodeOutput(
            logits=logits.astype(np.float32),
            query_embeddings=qf,
            support_embeddings=sf,
            prototypes=protos,
            adapt_seconds=time.perf_counter() - t0,
            extras={
                "alpha_effective": self._effective_alpha(k, alpha=None, kappa=None),
                "k_shot": k,
            },
        )


class PixelNearestCentroid(BaseMethod):
    """Nearest centroid on raw pixels -- a training-free sanity floor.

    Carried over from the original ``run_all_fsl_baselines``; it recorded 69.6%,
    which is well above the 50% chance level and worth keeping as context.
    """

    name = "nc_pixel"
    family = "metric"
    requires_support = True
    adapts_per_episode = True

    def prepare(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self._summary["params"] = {"total": 0, "trainable": 0, "frozen": 0}

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        import torch

        t0 = time.perf_counter()
        s = batch.support_x.flatten(1)
        q = batch.query_x.flatten(1)
        protos = torch.stack(
            [s[batch.support_y == c].mean(0) for c in range(batch.n_way)]
        )
        logits = -torch.cdist(q, protos, p=2)
        return EpisodeOutput(
            logits=self._to_numpy(logits),
            adapt_seconds=time.perf_counter() - t0,
        )
