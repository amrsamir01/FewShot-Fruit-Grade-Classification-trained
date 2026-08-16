"""
API and offline tests.

The offline test is the one that matters most: a viva room is exactly where a
silent weight download must not happen, so we block the socket layer outright
and assert the demo still starts, samples, predicts and projects.
"""

from __future__ import annotations

import io
import socket

import pytest

fastapi = pytest.importorskip("fastapi", reason="demo extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from fsgrade.demo.app import create_app  # noqa: E402
from fsgrade.demo.settings import DemoSettings  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _png(size=(48, 48), colour=(90, 160, 90)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(synthetic_root, tmp_path):
    settings = DemoSettings(
        results_root=tmp_path / "results",
        data_root=synthetic_root,
        device="cpu",
    )
    return TestClient(create_app(settings))


@pytest.fixture
def session(client):
    body = {"species": "mango", "arm_kshot": "nc_pixel", "arm_zeroshot": "chance"}
    return client.post("/api/sessions", json=body).json()["session"]


# ---------------------------------------------------------------- health --- #

def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    caps = body["capabilities"]
    assert caps["dataset"] is True
    # These need no training and must always be offerable.
    assert {"nc_pixel", "chance"} <= set(caps["training_free_arms"])


def test_health_warns_when_clip_is_missing(client):
    body = client.get("/api/health").json()
    if not body["capabilities"]["clip"]:
        assert any("CLIP" in w for w in body["warnings"])


def test_index_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    for path in ("/static/app.css", "/static/js/main.js", "/static/js/plot.js"):
        assert client.get(path).status_code == 200, path


def test_docs_are_disabled(client):
    """Swagger UI would fetch its assets from a CDN."""
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 200


# --------------------------------------------------------------- session --- #

def test_session_lifecycle(client):
    created = client.post("/api/sessions", json={"species": "mango"}).json()["session"]
    sid = created["session_id"]
    assert client.get(f"/api/sessions/{sid}").status_code == 200

    patched = client.patch(f"/api/sessions/{sid}", json={"species": "orange"}).json()["session"]
    assert patched["species"] == "orange"
    assert patched["revision"] > created["revision"]

    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_unknown_session_returns_actionable_error(client):
    body = client.get("/api/sessions/nope").json()
    assert body["error"]["code"] == "session_not_found"
    assert body["error"]["remediation"]


def test_revision_increases_on_every_mutation(client, session):
    sid = session["session_id"]
    revisions = [session["revision"]]
    for _ in range(2):
        body = client.post(f"/api/sessions/{sid}/support/sample",
                           json={"n_per_class": 2}).json()
        revisions.append(body["session"]["revision"])
    assert revisions == sorted(revisions) and len(set(revisions)) == len(revisions)


# --------------------------------------------------------------- images --- #

def test_sampling_keeps_support_and_query_disjoint(client, session):
    """Grading a query that is also a support image would be a leak on stage."""
    sid = session["session_id"]
    support = client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 5}).json()
    query = client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 5}).json()

    s = {c["relpath"] for c in support["added"]}
    q = {c["relpath"] for c in query["added"]}
    assert s and q and not (s & q)


def test_sampling_is_balanced_and_sets_k(client, session):
    sid = session["session_id"]
    body = client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 4}).json()
    assert body["session"]["counts"]["support"] == {"fresh": 4, "rotten": 4}
    assert body["session"]["k_shot"] == 4
    assert body["session"]["balanced"] is True


def test_upload_accepts_images_and_rejects_junk(client, session):
    sid = session["session_id"]
    body = client.post(
        f"/api/sessions/{sid}/support/upload",
        data={"label": "fresh"},
        files=[("files", ("ok.png", _png(), "image/png")),
               ("files", ("bad.txt", b"not an image", "text/plain"))],
    ).json()
    assert len(body["added"]) == 1
    assert body["rejected"][0]["filename"] == "bad.txt"
    assert "recognised" in body["rejected"][0]["reason"]


def test_upload_thumbnail_is_inline(client, session):
    """The UI must never make a second request for an image we already hold."""
    sid = session["session_id"]
    body = client.post(f"/api/sessions/{sid}/query/upload",
                       files=[("files", ("a.png", _png(), "image/png"))]).json()
    assert body["added"][0]["thumb"].startswith("data:image/jpeg;base64,")


def test_support_upload_requires_a_label(client, session):
    sid = session["session_id"]
    res = client.post(f"/api/sessions/{sid}/support/upload",
                      files=[("files", ("a.png", _png(), "image/png"))])
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "upload_rejected"


def test_remove_and_clear(client, session):
    sid = session["session_id"]
    added = client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 3}).json()
    image_id = added["added"][0]["image_id"]

    after = client.delete(f"/api/sessions/{sid}/support/{image_id}").json()["session"]
    assert len(after["support"]) == 5

    cleared = client.post(f"/api/sessions/{sid}/support/clear").json()["session"]
    assert cleared["support"] == []


def test_balance_trims_the_larger_class(client, session):
    sid = session["session_id"]
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 3})
    client.post(f"/api/sessions/{sid}/support/upload", data={"label": "fresh"},
                files=[("files", ("x.png", _png(), "image/png"))])

    unbalanced = client.get(f"/api/sessions/{sid}").json()["session"]
    assert unbalanced["balanced"] is False

    balanced = client.post(f"/api/sessions/{sid}/support/balance").json()["session"]
    assert balanced["balanced"] is True
    assert balanced["counts"]["support"] == {"fresh": 3, "rotten": 3}


# -------------------------------------------------------------- predict --- #

def test_predict_without_queries_is_refused_helpfully(client, session):
    sid = session["session_id"]
    res = client.post(f"/api/sessions/{sid}/predict", json={})
    assert res.status_code == 409
    assert res.json()["error"]["remediation"]


def test_predict_returns_verdicts_and_a_projection(client, session):
    sid = session["session_id"]
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 4})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 3})

    body = client.post(f"/api/sessions/{sid}/predict", json={}).json()
    arms = {v["arm"] for v in body["verdicts"]}
    assert {"nc_pixel", "chance"} <= arms

    proj = body["projection"]
    assert proj["boundary_u"] == 0.0
    assert len(proj["coords"]) == proj["n_support"] + proj["n_query"]
    assert len(proj["prototypes"]) == 2


def test_prototypes_sit_exactly_on_the_boundary_axis(client, session):
    """Both prototypes must land at v = 0 -- the picture's anchor."""
    sid = session["session_id"]
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 4})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 3})
    proj = client.post(f"/api/sessions/{sid}/predict", json={}).json()["projection"]
    for _, v in proj["prototypes"]:
        assert abs(v) < 1e-9


def test_drawn_boundary_agrees_with_every_prediction(client, session):
    """sign_agreement is the claim that is exact for distance-based logits."""
    sid = session["session_id"]
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 5})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 5})
    proj = client.post(f"/api/sessions/{sid}/predict", json={}).json()["projection"]
    assert proj["sign_agreement"] == pytest.approx(1.0)


def test_chance_arm_uses_no_support(client, session):
    sid = session["session_id"]
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 3})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 4})
    body = client.post(f"/api/sessions/{sid}/predict", json={}).json()
    chance = next(v for v in body["verdicts"] if v["arm"] == "chance")
    assert chance["uses_support"] is False and chance["k_shot"] == 0


def test_unavailable_arm_reports_instead_of_crashing(client, session):
    """A missing checkpoint must degrade the arm, not the request."""
    sid = session["session_id"]
    client.patch(f"/api/sessions/{sid}", json={"arm_kshot": "ours"})
    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 3})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 3})

    body = client.post(f"/api/sessions/{sid}/predict", json={}).json()
    ours = next(v for v in body["verdicts"] if v["arm"] == "ours")
    assert ours["error"] and "checkpoint" in ours["error"].lower()


# --------------------------------------------------------------- offline -- #

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _block_external(monkeypatch) -> list[str]:
    """Refuse every non-loopback connection, recording what was attempted.

    Loopback must stay open: asyncio builds its event loop from a local
    ``socketpair`` and TestClient talks to the app over it, so blocking all
    sockets would break the harness rather than the app. What matters is that
    nothing reaches the internet.
    """
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guard(address) -> bool:
        host = address[0] if isinstance(address, tuple) else str(address)
        if str(host) in LOOPBACK:
            return True
        attempts.append(str(host))
        return False

    def patched_connect(self, address, *args, **kwargs):
        if not guard(address):
            raise OSError(f"external network access blocked in this test: {address}")
        return real_connect(self, address, *args, **kwargs)

    def patched_create(address, *args, **kwargs):
        if not guard(address):
            raise OSError(f"external network access blocked in this test: {address}")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", patched_connect)
    monkeypatch.setattr(socket, "create_connection", patched_create)
    return attempts


def test_demo_works_with_the_network_blocked(monkeypatch, synthetic_root, tmp_path):
    """Start, sample, predict and project with outbound access refused."""
    attempts = _block_external(monkeypatch)

    settings = DemoSettings(
        results_root=tmp_path / "results", data_root=synthetic_root, device="cpu"
    )
    client = TestClient(create_app(settings))

    assert client.get("/api/health").json()["status"] == "ok"
    sid = client.post("/api/sessions", json={
        "species": "mango", "arm_kshot": "nc_pixel", "arm_zeroshot": "chance",
    }).json()["session"]["session_id"]

    client.post(f"/api/sessions/{sid}/support/sample", json={"n_per_class": 3})
    client.post(f"/api/sessions/{sid}/query/sample", json={"n_per_class": 3})
    body = client.post(f"/api/sessions/{sid}/predict", json={}).json()
    assert body["projection"]["boundary_u"] == 0.0
    assert attempts == [], f"the demo tried to reach {attempts}"


def test_offline_env_is_pinned_on_import():
    import os

    from fsgrade.demo import settings as demo_settings

    assert demo_settings.OFFLINE_ENV["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_static_assets_reference_nothing_external():
    """One CDN link would break the whole offline guarantee."""
    from fsgrade.demo.__main__ import _scan_external_refs
    from fsgrade.demo.app import STATIC_DIR

    assert _scan_external_refs(STATIC_DIR) == []
