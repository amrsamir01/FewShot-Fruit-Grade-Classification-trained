"""
HTTP routes.

The shape mirrors the demo's story: discover what can run, open a session,
build a support set, then ask every selected arm the same question about the
same query images.
"""

# NOTE: this module deliberately does NOT use `from __future__ import
# annotations`. FastAPI resolves route annotations at definition time, and the
# fastapi symbols are imported inside build_router() so that importing this
# module stays cheap when fastapi is absent. Postponed annotations would turn
# `list[UploadFile]` into an unresolvable ForwardRef and every upload route
# would fail at request time.
import pathlib
from typing import Any, Optional

from fsgrade.demo.errors import (
    ArmUnavailable,
    CapacityReached,
    DatasetUnavailable,
    NotReady,
    SessionNotFound,
    UploadRejected,
)
from fsgrade.demo.settings import DemoSettings


def build_router(settings: DemoSettings) -> Any:
    from fastapi import APIRouter, Body, File, Form, UploadFile

    from fsgrade.demo.discovery import best_method_pair
    from fsgrade.demo.engine import EngineError, InferenceEngine
    from fsgrade.demo.preprocess import ImageRejected, ingest, load_path, thumbnail_data_uri, to_tensor
    from fsgrade.demo.sampling import SamplingError, build_dataset_view, label_of, sample_relpaths
    from fsgrade.demo.sap import SapUnavailable, kappa_grid
    from fsgrade.demo.sessions import ImageCard, SessionLimit, SessionStore, new_image_id

    router = APIRouter()

    device = settings.resolve_device()
    data_root = settings.resolve_data_root()

    engine = InferenceEngine(
        settings.results_root, device=device,
        classes=settings.classes, cache_size=settings.model_cache_size,
        clip_encoder=settings.clip_encoder,
    )
    store = SessionStore(ttl=settings.session_ttl, max_sessions=settings.max_sessions)
    dataset = build_dataset_view(data_root, settings.species, settings.classes)

    router.state = {"engine": engine, "store": store, "dataset": dataset}  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    def _session(session_id: str):
        try:
            return store.get(session_id)
        except KeyError:
            raise SessionNotFound(session_id) from None

    def _specs(session) -> dict[str, Any]:
        arms = [a for a in (session.arm_kshot, session.arm_zeroshot) if a]
        return engine.transform_specs(arms)

    def _card_from_upload(data: bytes, filename: str, label: int, specs) -> ImageCard:
        tensors, thumb, size = ingest(data, filename, specs)
        return ImageCard(
            image_id=new_image_id(), origin="upload", filename=filename,
            label=label, thumb=thumb, width=size[0], height=size[1], tensors=tensors,
        )

    def _card_from_dataset(relpath: str, label: int, specs) -> ImageCard:
        img = load_path(dataset.abs_path(relpath))
        tensors = {fam: to_tensor([img], spec)[0] for fam, spec in specs.items()}
        return ImageCard(
            image_id=new_image_id(), origin="dataset",
            filename=pathlib.Path(relpath).name, relpath=relpath, label=label,
            thumb=thumbnail_data_uri(img), width=img.size[0], height=img.size[1],
            tensors=tensors,
        )

    # ---------------------------------------------------- discovery ----- #
    @router.get("/health")
    async def health():
        caps = engine.capabilities()
        return {
            "status": "ok",
            "device": device,
            "settings": settings.to_dict(),
            "capabilities": {**caps, "dataset": dataset.available},
            "offline": {"enforced": settings.offline},
            "kappa_grid": kappa_grid(),
            "sessions": len(store),
            "warnings": (
                [] if caps["clip"] else
                ["CLIP is unavailable, so SAP and the zero-shot text arm are disabled. "
                 "Install with: pip install open_clip_torch, then run "
                 "python -m fsgrade.demo.prefetch"]
            ) + (
                ["Bound to a non-loopback address with no authentication."]
                if settings.is_public_bind else []
            ),
        }

    @router.get("/runs")
    async def runs(include_failed: bool = False):
        found = engine.runs(refresh=True)
        kshot, zeroshot = best_method_pair(found)
        return {
            "runs": [r.to_dict() for r in found],
            "default_arms": {"kshot": kshot, "zeroshot": zeroshot},
        }

    @router.get("/species")
    async def species():
        return dataset.to_dict() | {"free_form_allowed": True}

    @router.get("/methods")
    async def methods():
        caps = engine.capabilities()
        catalogue = []
        for name in ("chance", "nc_pixel", "clip_text_zeroshot", "sap",
                     "zeroshot_supervised", "ncc_supervised",
                     "protonet", "ours", "siamese", "matching"):
            needs_ckpt = name in caps["checkpoint_arms"] or name in (
                "zeroshot_supervised", "ncc_supervised", "protonet", "ours",
                "siamese", "matching",
            )
            available = (
                name in caps["checkpoint_arms"]
                if needs_ckpt
                else (caps["clip"] if name in ("sap", "clip_text_zeroshot") else True)
            )
            catalogue.append({
                "name": name,
                "needs_checkpoint": needs_ckpt,
                "available": available,
                "encoder": engine.family_for(name),
                "reason": "" if available else (
                    "no trained checkpoint found" if needs_ckpt else "CLIP unavailable"
                ),
            })
        return {"methods": catalogue, "kappa": caps["kappa"], "kappa_source": caps["kappa_source"]}

    # ------------------------------------------------------ session ----- #
    @router.post("/sessions")
    async def create_session(payload: dict = Body(default={})):
        kshot, zeroshot = best_method_pair(engine.runs())
        session = store.create(
            species=payload.get("species") or "mango",
            arm_kshot=payload.get("arm_kshot") or kshot or "sap",
            arm_zeroshot=payload.get("arm_zeroshot", zeroshot),
            run_id=payload.get("run_id"),
            fold_id=payload.get("fold_id"),
            kappa=payload.get("kappa"),
            classes=list(settings.classes),
        )
        return {"session": session.to_dict()}

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return {"session": _session(session_id).to_dict()}

    @router.patch("/sessions/{session_id}")
    async def patch_session(session_id: str, payload: dict = Body(default={})):
        session = _session(session_id)
        for key in ("species", "arm_kshot", "arm_zeroshot", "run_id", "fold_id",
                    "kappa", "frame_locked"):
            if key in payload:
                setattr(session, key, payload[key])
        if "frame_locked" in payload and not payload["frame_locked"]:
            session.locked_frame = None
        session.touch()
        return {"session": session.to_dict()}

    @router.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        store.drop(session_id)

    # -------------------------------------------------------- images ---- #
    @router.post("/sessions/{session_id}/{group}/upload")
    async def upload(session_id: str, group: str, label: str = Form(default=""),
                     files: list[UploadFile] = File(...)):
        session = _session(session_id)
        if group not in ("support", "query"):
            raise UploadRejected(f"unknown group {group!r}", remediation="Use 'support' or 'query'.")

        label_index = (
            session.classes.index(label) if label in session.classes
            else (-1 if group == "query" else None)
        )
        if label_index is None:
            raise UploadRejected(
                f"support images need a label, one of {session.classes}",
                remediation="Drop the image on the fresh or rotten tray.",
            )

        specs = _specs(session)
        cards, rejected = [], []
        for upload_file in files:
            data = await upload_file.read()
            try:
                cards.append(_card_from_upload(data, upload_file.filename or "upload", label_index, specs))
            except ImageRejected as exc:
                rejected.append({"filename": exc.filename, "reason": exc.reason})

        try:
            added = store.add_images(session, group, cards)
        except SessionLimit as exc:
            raise CapacityReached(str(exc), remediation=exc.remediation) from exc

        return {
            "added": [c.to_dict() for c in added],
            "rejected": rejected,
            "session": session.to_dict(),
        }

    @router.post("/sessions/{session_id}/{group}/sample")
    async def sample(session_id: str, group: str, payload: dict = Body(default={})):
        session = _session(session_id)
        if not dataset.available:
            raise DatasetUnavailable(
                "No local dataset is configured.",
                remediation="Start with --data-root, or drag images in instead.",
            )

        used = [c.relpath for c in session.support + session.query if c.relpath]
        try:
            picked = sample_relpaths(
                dataset, session.species,
                n_per_class=int(payload.get("n_per_class", 5)),
                exclude=used, seed=payload.get("seed"),
            )
        except SamplingError as exc:
            raise DatasetUnavailable(str(exc), remediation=exc.remediation) from exc

        specs = _specs(session)
        cards = [
            _card_from_dataset(rel, session.classes.index(cls) if group == "support" else session.classes.index(cls), specs)
            for cls, paths in picked.items() for rel in paths
        ]
        try:
            added = store.add_images(session, group, cards)
        except SessionLimit as exc:
            raise CapacityReached(str(exc), remediation=exc.remediation) from exc
        return {"added": [c.to_dict() for c in added], "session": session.to_dict()}

    @router.delete("/sessions/{session_id}/{group}/{image_id}")
    async def remove_image(session_id: str, group: str, image_id: str):
        session = _session(session_id)
        store.remove_image(session, group, image_id)
        return {"session": session.to_dict()}

    @router.post("/sessions/{session_id}/{group}/clear")
    async def clear_group(session_id: str, group: str):
        session = _session(session_id)
        store.clear(session, group)
        return {"session": session.to_dict()}

    @router.post("/sessions/{session_id}/support/balance")
    async def balance(session_id: str):
        session = _session(session_id)
        removed = store.balance_support(session)
        return {"removed": removed, "session": session.to_dict()}

    @router.patch("/sessions/{session_id}/query/{image_id}")
    async def label_query(session_id: str, image_id: str, payload: dict = Body(default={})):
        """Grade a query image after the fact, so accuracy becomes computable."""
        session = _session(session_id)
        label = payload.get("label")
        index = session.classes.index(label) if label in session.classes else -1
        for card in session.query:
            if card.image_id == image_id:
                card.label = index
                session.touch()
                break
        return {"session": session.to_dict()}

    # ----------------------------------------------------- inference ---- #
    @router.post("/sessions/{session_id}/predict")
    async def predict(session_id: str, payload: dict = Body(default={})):
        session = _session(session_id)
        if not session.query:
            raise NotReady(
                "There are no query images to grade.",
                remediation="Add query images, or click 'sample from dataset'.",
            )
        try:
            result = engine.predict(
                session,
                arms=payload.get("arms"),
                kappa=payload.get("kappa"),
            )
        except (EngineError, SapUnavailable) as exc:
            raise ArmUnavailable(
                str(exc), remediation=getattr(exc, "remediation", "")
            ) from exc
        return result | {"session": session.to_dict()}

    return router
