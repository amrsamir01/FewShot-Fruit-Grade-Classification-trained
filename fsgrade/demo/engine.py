"""
The inference engine: model cache, arm construction, prediction orchestration.

Holds loaded models so a 137 MB checkpoint is read once rather than per request,
builds the right adapter for each arm, and runs the K-shot and zero-shot arms
against the *same* query set so the comparison on screen is honest.
"""

from __future__ import annotations

import pathlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fsgrade.demo.adapters import (
    ArmNeedsMore,
    ArmResult,
    ChanceArm,
    EpisodicArm,
    PixelArm,
    SapArm,
    SupervisedCentroidArm,
    SupervisedLinearArm,
)
from fsgrade.demo.checkpoints import CheckpointError, load_for_inference
from fsgrade.demo.discovery import RunRef, discover_runs
from fsgrade.demo.preprocess import TransformSpec, transform_spec
from fsgrade.demo.projection import PlaneFrame, project
from fsgrade.demo.sap import SapEngine, SapUnavailable

# Arms that read a supervised checkpoint (all three share one file).
_SUPERVISED = {"zeroshot_supervised", "ncc_supervised", "finetune_supervised"}
_EPISODIC = {"ours", "protonet", "protonet_temp", "siamese", "matching"}
_CLIP = {"sap", "clip_text_zeroshot", "clip_ncc", "clip_probe"}


@dataclass
class Verdict:
    """One arm's answer, ready to send to the browser."""

    arm: str
    label: str
    predictions: list[int]
    probabilities: list[float]
    margins: list[float]
    accuracy: float | None
    k_shot: int
    uses_support: bool
    seconds: float
    extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "label": self.label,
            "predictions": self.predictions, "probabilities": self.probabilities,
            "margins": self.margins, "accuracy": self.accuracy,
            "k_shot": self.k_shot, "uses_support": self.uses_support,
            "seconds": round(self.seconds, 4), "extras": self.extras, "error": self.error,
        }


class EngineError(RuntimeError):
    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation


class InferenceEngine:
    """Loads models on demand, caches them, and runs arms."""

    def __init__(
        self,
        results_root: str | pathlib.Path = "results",
        *,
        device: str | torch.device = "cpu",
        classes: tuple[str, ...] = ("fresh", "rotten"),
        cache_size: int = 4,
        clip_encoder: str = "clip_vitb16",
    ) -> None:
        self.results_root = pathlib.Path(results_root)
        self.device = torch.device(device)
        self.classes = list(classes)
        self.cache_size = cache_size
        self.clip_encoder_name = clip_encoder

        self._models: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._runs: list[RunRef] | None = None
        self._sap: SapEngine | None = None
        self._sap_error: str | None = None

    # ------------------------------------------------------------------ #
    def runs(self, *, refresh: bool = False) -> list[RunRef]:
        with self._lock:
            if self._runs is None or refresh:
                self._runs = discover_runs(self.results_root)
            return self._runs

    def find_run(self, run_id: str | None) -> RunRef | None:
        runs = self.runs()
        if run_id:
            for run in runs:
                if run.run_id == run_id:
                    return run
            return None
        return runs[0] if runs else None

    def default_config(self) -> dict[str, Any]:
        run = self.find_run(None)
        return run.config if run else {}

    # ------------------------------------------------------------------ #
    @property
    def sap(self) -> SapEngine:
        """The shared CLIP engine, built once."""
        with self._lock:
            if self._sap is None:
                self._sap = SapEngine(
                    self.clip_encoder_name, device=self.device, classes=self.classes
                )
                # Adopt a calibrated kappa if any run recorded one.
                for run in self.runs():
                    ref = run.methods.get("sap")
                    if ref is not None and ref.kappa is not None:
                        self._sap.adopt_kappa_from(ref)
                        break
            return self._sap

    def clip_available(self) -> bool:
        try:
            ok = self.sap.is_available()
            if not ok:
                self._sap_error = "CLIP weights are not available locally"
            return ok
        except Exception as exc:  # noqa: BLE001
            self._sap_error = str(exc)
            return False

    def capabilities(self) -> dict[str, Any]:
        runs = self.runs()
        available: set[str] = set()
        for run in runs:
            available.update(m.name for m in run.available_methods())
        clip_ok = self.clip_available()
        return {
            "device": str(self.device),
            "runs": len(runs),
            "clip": clip_ok,
            "clip_error": None if clip_ok else self._sap_error,
            "checkpoint_arms": sorted(a for a in available if a in _SUPERVISED | _EPISODIC),
            "training_free_arms": sorted(
                (["sap", "clip_text_zeroshot"] if clip_ok else []) + ["nc_pixel", "chance"]
            ),
            "kappa": self.sap.kappa if clip_ok else None,
            "kappa_source": self.sap.kappa_source if clip_ok else None,
        }

    # ------------------------------------------------------------------ #
    def transform_specs(self, arms: list[str]) -> dict[str, TransformSpec]:
        """Which preprocessing families the selected arms need.

        CLIP arms get the encoder's own constants; everything else gets
        ImageNet. Using the wrong ones silently degrades a frozen model.
        """
        specs: dict[str, TransformSpec] = {}
        config = self.default_config()
        from fsgrade.config import get_in

        if any(a in _CLIP for a in arms):
            try:
                specs["clip"] = self.sap.transform_spec()
            except SapUnavailable:
                specs["clip"] = transform_spec("clip")
        if any(a not in _CLIP for a in arms):
            specs["imagenet"] = transform_spec(
                "imagenet",
                image_size=int(get_in(config, "data.image_size", 224)),
                resize_size=int(get_in(config, "data.resize_size", 256)),
            )
        return specs or {"imagenet": transform_spec("imagenet")}

    @staticmethod
    def family_for(arm: str) -> str:
        return "clip" if arm in _CLIP else "imagenet"

    # ------------------------------------------------------------------ #
    def _load_model(self, run: RunRef, method: str) -> Any:
        key = f"{run.run_id}|{method}"
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]

            ref = run.methods.get(method)
            if ref is None or ref.checkpoint is None:
                raise EngineError(
                    f"{method!r} has no checkpoint in run {run.run_id}",
                    remediation="Pick a different run or method.",
                )
            try:
                loaded = load_for_inference(method, ref.checkpoint, run.config, device=self.device)
            except CheckpointError as exc:
                raise EngineError(str(exc), remediation=exc.remediation) from exc

            self._models[key] = loaded
            while len(self._models) > self.cache_size:
                self._models.popitem(last=False)
            return loaded

    def build_arm(self, arm: str, *, run: RunRef | None = None, species: str = "mango") -> Any:
        """Construct the adapter for one arm."""
        n_way = len(self.classes)

        if arm == "chance":
            return ChanceArm(n_way=n_way)
        if arm == "nc_pixel":
            return PixelArm(n_way=n_way)

        if arm in _CLIP:
            if not self.clip_available():
                raise EngineError(
                    f"{arm!r} needs CLIP, which is not available: {self._sap_error}",
                    remediation="pip install open_clip_torch && python -m fsgrade.demo.prefetch",
                )
            return SapArm(arm, self.sap, species, zero_shot=(arm == "clip_text_zeroshot"))

        if run is None:
            raise EngineError(
                f"{arm!r} needs a trained checkpoint but no run was selected",
                remediation="Run the experiments, or choose a training-free arm "
                            "such as sap / clip_text_zeroshot / nc_pixel.",
            )

        loaded = self._load_model(run, arm)
        if arm in _SUPERVISED:
            if arm == "zeroshot_supervised":
                return SupervisedLinearArm(arm, loaded.module, self.device)
            return SupervisedCentroidArm(arm, loaded.module, self.device, n_way=n_way)

        from fsgrade.config import get_in

        return EpisodicArm(
            arm, loaded.module, self.device, n_way=n_way,
            distance=get_in(run.config, "model.distance", "euclidean"),
        )

    # ------------------------------------------------------------------ #
    def predict(
        self,
        session: Any,
        *,
        arms: list[str] | None = None,
        kappa: float | None = None,
    ) -> dict[str, Any]:
        """Run every selected arm on the same query set and project the result."""
        arms = arms or [a for a in (session.arm_kshot, session.arm_zeroshot) if a]
        run = self.find_run(session.run_id)
        y_true = session.query_labels()

        verdicts: list[Verdict] = []
        results: dict[str, ArmResult] = {}

        for arm in arms:
            family = self.family_for(arm)
            sx, sy = session.tensors("support", family)
            qx, _ = session.tensors("query", family)

            try:
                adapter = self.build_arm(arm, run=run, species=session.species)
                result = adapter.predict(
                    sx, sy, qx,
                    kappa=kappa if kappa is not None else session.kappa,
                    species=session.species,
                )
            except (ArmNeedsMore, EngineError, SapUnavailable) as exc:
                verdicts.append(Verdict(
                    arm=arm, label=arm, predictions=[], probabilities=[], margins=[],
                    accuracy=None, k_shot=0, uses_support=True, seconds=0.0,
                    error=str(exc),
                ))
                continue

            results[arm] = result
            probs = result.probabilities()
            verdicts.append(Verdict(
                arm=arm,
                label=arm,
                predictions=result.predictions().tolist(),
                probabilities=probs[:, 1].tolist() if probs.shape[1] > 1 else probs[:, 0].tolist(),
                margins=result.margins().tolist(),
                accuracy=result.accuracy(y_true),
                k_shot=result.k_shot,
                uses_support=result.uses_support,
                seconds=result.seconds,
                extras=result.extras,
            ))

        projection = self._project(session, results, arms)
        return {
            "verdicts": [v.to_dict() for v in verdicts],
            "projection": projection,
            "n_query": len(session.query),
            "n_labelled": int((y_true >= 0).sum()),
            "revision": session.revision,
        }

    # ------------------------------------------------------------------ #
    def _project(
        self, session: Any, results: dict[str, ArmResult], arms: list[str]
    ) -> dict[str, Any] | None:
        """Project the primary arm's geometry onto its decision plane."""
        primary = next((a for a in arms if a in results and results[a].prototypes is not None
                        or a in results and results[a].linear_head is not None), None)
        if primary is None:
            return None

        result = results[primary]
        frame: PlaneFrame | None = session.locked_frame if session.frame_locked else None
        if frame is None:
            frame = result.build_frame()
            if frame is None:
                return None
            if session.frame_locked:
                session.locked_frame = frame

        support = result.support_embeddings
        query = result.query_embeddings
        if query is None:
            return None

        cloud = np.vstack([a for a in (support, query) if a is not None and len(a)])
        protos = (
            result.prototypes
            if result.prototypes is not None
            else np.zeros((2, cloud.shape[1]), dtype=np.float32)
        )

        proj = project(frame, cloud, protos, true_margins=None)
        # Fidelity is only meaningful on the points the model actually graded.
        if len(query):
            from fsgrade.demo.projection import _safe_corr, sign_agreement

            implied = frame.margin(query)
            proj.margin_fidelity = _safe_corr(result.margins(), implied)
            # For -d logits (ProtoNet, pixel) the correlation is below 1 while
            # every prediction still lands on the correct side of the line, so
            # report both rather than letting the weaker number stand alone.
            proj.sign_agreement = sign_agreement(result.margins(), implied)

        payload = proj.to_dict()
        payload["arm"] = primary
        payload["n_support"] = int(len(support)) if support is not None else 0
        payload["n_query"] = int(len(query))
        payload["locked"] = bool(session.frame_locked)
        return payload

    # ------------------------------------------------------------------ #
    def warm_up(self) -> dict[str, Any]:
        """Touch the slow paths once so the first click is not the slow one."""
        report: dict[str, Any] = {"runs": len(self.runs(refresh=True))}
        try:
            report["clip"] = self.clip_available()
        except Exception as exc:  # noqa: BLE001
            report["clip"] = False
            report["clip_error"] = str(exc)
        return report
