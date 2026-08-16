"""
Session state and dataset sampling.

A session is one examiner's working set: a species, a support set being built up
image by image, and a query set. It lives in memory only -- nothing a demo
session holds is worth persisting, and writing uploads to disk would turn a
viva-room convenience into a data-handling question nobody wants.

Every mutation bumps ``revision``. Responses carry it so a client that fires two
overlapping requests can discard the stale reply rather than render it.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

DEFAULT_TTL_SECONDS = 60 * 60
MAX_SUPPORT_PER_CLASS = 20
MAX_QUERY = 60
MAX_SESSIONS = 32

# Query images the user has not graded carry this label, so accuracy is
# computed over the labelled subset only rather than silently assuming.
UNLABELLED = -1


@dataclass
class ImageCard:
    """One image in a session, with its tensors already computed."""

    image_id: str
    origin: str                          # "upload" | "dataset"
    filename: str
    label: int = UNLABELLED
    thumb: str = ""
    width: int = 0
    height: int = 0
    relpath: str | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "origin": self.origin,
            "filename": self.filename,
            "label": self.label,
            "thumb": self.thumb,
            "width": self.width,
            "height": self.height,
            "relpath": self.relpath,
        }


@dataclass
class Session:
    session_id: str
    species: str = "mango"
    arm_kshot: str = "sap"
    arm_zeroshot: str | None = "clip_text_zeroshot"
    run_id: str | None = None
    fold_id: str | None = None
    kappa: float | None = None
    frame_locked: bool = False
    revision: int = 0
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    support: list[ImageCard] = field(default_factory=list)
    query: list[ImageCard] = field(default_factory=list)
    locked_frame: Any = None
    classes: list[str] = field(default_factory=lambda: ["fresh", "rotten"])

    # ------------------------------------------------------------------ #
    def touch(self) -> int:
        self.revision += 1
        self.touched = time.time()
        return self.revision

    def support_counts(self) -> dict[str, int]:
        return {
            name: sum(1 for c in self.support if c.label == i)
            for i, name in enumerate(self.classes)
        }

    def k_shot(self) -> int:
        """K is the *minimum* per class -- an unbalanced tray is not K-shot."""
        counts = self.support_counts()
        return min(counts.values()) if counts else 0

    def is_balanced(self) -> bool:
        return len(set(self.support_counts().values())) <= 1

    def can_run_kshot(self) -> bool:
        return all(v > 0 for v in self.support_counts().values())

    def tensors(self, group: str, family: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack one group's tensors for a given encoder family."""
        cards = self.support if group == "support" else self.query
        usable = [c for c in cards if family in c.tensors]
        if not usable:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)
        x = torch.stack([c.tensors[family] for c in usable])
        y = torch.tensor([c.label for c in usable], dtype=torch.long)
        return x, y

    def query_labels(self) -> np.ndarray:
        return np.asarray([c.label for c in self.query], dtype=int)

    def to_dict(self) -> dict[str, Any]:
        counts = self.support_counts()
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "species": self.species,
            "arm_kshot": self.arm_kshot,
            "arm_zeroshot": self.arm_zeroshot,
            "run_id": self.run_id,
            "fold_id": self.fold_id,
            "kappa": self.kappa,
            "frame_locked": self.frame_locked,
            "classes": self.classes,
            "support": [c.to_dict() for c in self.support],
            "query": [c.to_dict() for c in self.query],
            "counts": {"support": counts, "query": len(self.query)},
            "k_shot": self.k_shot(),
            "balanced": self.is_balanced(),
            "can_run_kshot": self.can_run_kshot(),
        }


class SessionLimit(RuntimeError):
    """A session capacity limit was hit."""

    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(message)
        self.remediation = remediation


class SessionStore:
    """In-memory sessions with a TTL sweep. Thread-safe."""

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, max_sessions: int = MAX_SESSIONS) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self, **kwargs: Any) -> Session:
        with self._lock:
            self._sweep()
            if len(self._sessions) >= self.max_sessions:
                # Evict the least recently touched rather than refusing service.
                oldest = min(self._sessions.values(), key=lambda s: s.touched)
                del self._sessions[oldest.session_id]
            session = Session(session_id=secrets.token_urlsafe(12), **kwargs)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.touched = time.time()
            return session

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _sweep(self) -> int:
        cutoff = time.time() - self.ttl
        stale = [sid for sid, s in self._sessions.items() if s.touched < cutoff]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- mutation helpers ------------------------------------------------ #
    def add_images(
        self, session: Session, group: str, cards: Sequence[ImageCard]
    ) -> list[ImageCard]:
        with self._lock:
            target = session.support if group == "support" else session.query
            limit = MAX_SUPPORT_PER_CLASS * len(session.classes) if group == "support" else MAX_QUERY
            room = limit - len(target)
            if room <= 0:
                raise SessionLimit(
                    f"the {group} set already holds {len(target)} images (limit {limit})",
                    remediation=f"Remove some {group} images first.",
                )
            accepted = list(cards)[:room]
            target.extend(accepted)
            session.touch()
            return accepted

    def remove_image(self, session: Session, group: str, image_id: str) -> bool:
        with self._lock:
            target = session.support if group == "support" else session.query
            before = len(target)
            target[:] = [c for c in target if c.image_id != image_id]
            if len(target) != before:
                session.touch()
                return True
            return False

    def clear(self, session: Session, group: str) -> None:
        with self._lock:
            (session.support if group == "support" else session.query).clear()
            session.locked_frame = None
            session.touch()

    def balance_support(self, session: Session) -> list[str]:
        """Trim the larger class so K is well defined."""
        with self._lock:
            k = session.k_shot()
            removed: list[str] = []
            if k == 0:
                return removed
            kept: list[ImageCard] = []
            seen = {i: 0 for i in range(len(session.classes))}
            for card in session.support:
                if seen.get(card.label, 0) < k:
                    seen[card.label] = seen.get(card.label, 0) + 1
                    kept.append(card)
                else:
                    removed.append(card.image_id)
            session.support[:] = kept
            if removed:
                session.touch()
            return removed


def new_image_id() -> str:
    return uuid.uuid4().hex[:12]
