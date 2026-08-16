"""
Typed errors that always carry a remediation.

An error a demo user cannot act on is a dead end in front of an audience, so
every error here answers "what do I do about it?" as well as "what happened".
"""

from __future__ import annotations

from typing import Any


class DemoError(Exception):
    """Base class. Serialises to a stable JSON shape."""

    code = "demo_error"
    status_code = 400

    def __init__(self, message: str, *, remediation: str = "", **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "remediation": self.remediation,
                "details": self.details,
            }
        }


class SessionNotFound(DemoError):
    code = "session_not_found"
    status_code = 404

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} has expired or never existed.",
            remediation="Reload the page to start a new session.",
        )


class ArmUnavailable(DemoError):
    code = "arm_unavailable"
    status_code = 409


class DatasetUnavailable(DemoError):
    code = "dataset_unavailable"
    status_code = 409


class UploadRejected(DemoError):
    code = "upload_rejected"
    status_code = 415


class CapacityReached(DemoError):
    code = "capacity_reached"
    status_code = 409


class NotReady(DemoError):
    code = "not_ready"
    status_code = 409
