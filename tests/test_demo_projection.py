"""
Decision-plane tests.

The claim being defended: the boundary drawn on screen is *exactly* where the
model puts it, for every logit form, with no approximation. These tests are the
evidence for that claim, so they check identities rather than tolerances
wherever the maths permits.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsgrade.demo.projection import (
    DegenerateFrameError,
    build_frame,
    build_frame_from_linear_head,
    project,
    project_episode,
)

D = 64


def _unit(rng, n=1):
    x = rng.normal(size=(n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def protos_unit(rng):
    """Two unit-norm prototypes, as produced by every cosine-based method."""
    return np.vstack(_unit(rng, 2))


@pytest.fixture
def cloud(rng):
    return rng.normal(size=(40, D))


# --------------------------------------------------------------- frame ----- #

def test_frame_is_orthonormal(protos_unit, cloud):
    f = build_frame(protos_unit, cloud)
    assert abs(f.e1 @ f.e2) < 1e-12, "e2 must be orthogonal to e1 by construction"
    assert abs(np.linalg.norm(f.e1) - 1.0) < 1e-12
    assert abs(np.linalg.norm(f.e2) - 1.0) < 1e-12


def test_e1_points_from_fresh_to_rotten(protos_unit, cloud):
    f = build_frame(protos_unit, cloud)
    w = protos_unit[1] - protos_unit[0]
    assert np.allclose(f.e1, w / np.linalg.norm(w))


def test_prototypes_land_symmetrically_on_the_boundary(protos_unit, cloud):
    """Both prototypes must sit at (-h, 0) and (+h, 0) exactly."""
    f = build_frame(protos_unit, cloud)
    p2d = f.project(protos_unit)
    h = f.prototype_half_gap
    assert p2d[0, 0] == pytest.approx(-h, abs=1e-12)
    assert p2d[1, 0] == pytest.approx(+h, abs=1e-12)
    assert p2d[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert p2d[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_origin_is_the_prototype_midpoint(protos_unit, cloud):
    """Anchoring the prototypes beats centring the cloud.

    Shifting the origin by the residual centroid would centre the cloud
    vertically and leave u untouched, but it would also move the prototypes off
    v = 0, because mean(R) is orthogonal to e1 and not in general to e2.
    """
    f = build_frame(protos_unit, cloud)
    m = (protos_unit[0] + protos_unit[1]) / 2.0
    assert np.allclose(f.origin, m, atol=1e-12)


def test_linear_head_centres_the_cloud_vertically(rng, cloud):
    """With no prototypes to anchor, centring is free and useful."""
    W = rng.normal(size=(2, D))
    f = build_frame_from_linear_head(W, rng.normal(size=2), cloud)
    assert f.project(cloud)[:, 1].mean() == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------- exactness per logit form - #

def test_euclidean_margin_is_exact(protos_unit, cloud):
    """||x-p0||^2 - ||x-p1||^2 == 2||w|| u, exactly."""
    f = build_frame(protos_unit, cloud, logit_form="euclidean")
    d0 = np.sum((cloud - protos_unit[0]) ** 2, axis=1)
    d1 = np.sum((cloud - protos_unit[1]) ** 2, axis=1)
    assert np.allclose(d0 - d1, f.margin(cloud), atol=1e-9)


def test_euclidean_sign_matches_nearest_prototype(protos_unit, cloud):
    """Sign of the drawn margin must equal the actual nearest-prototype decision."""
    f = build_frame(protos_unit, cloud, logit_form="euclidean")
    d0 = np.linalg.norm(cloud - protos_unit[0], axis=1)
    d1 = np.linalg.norm(cloud - protos_unit[1], axis=1)
    predicted_rotten = d1 < d0
    assert np.array_equal(predicted_rotten, f.margin(cloud) > 0)


def test_cosine_margin_is_exactly_linear(protos_unit, rng):
    """For unit-norm prototypes and normalised queries, margin == scale*||w||*u."""
    scale = 100.0
    q = np.vstack(_unit(rng, 30))
    f = build_frame(protos_unit, q, logit_form="cosine", scale=scale)
    true = scale * (q @ protos_unit[1] - q @ protos_unit[0])
    assert np.allclose(true, f.margin(q), atol=1e-9)


def test_cosine_boundary_sits_at_u_zero(protos_unit, rng):
    q = np.vstack(_unit(rng, 60))
    f = build_frame(protos_unit, q, logit_form="cosine", scale=10.0)
    u = f.project(q)[:, 0]
    predicted_rotten = (q @ protos_unit[1]) > (q @ protos_unit[0])
    assert np.array_equal(predicted_rotten, u > 0)


def test_linear_head_margin_is_exact(rng, cloud):
    """logit_1 - logit_0 == ||w|| u for a linear classifier."""
    W = rng.normal(size=(2, D))
    b = rng.normal(size=2)
    f = build_frame_from_linear_head(W, b, cloud)
    true = (cloud @ W[1] + b[1]) - (cloud @ W[0] + b[0])
    assert np.allclose(true, f.margin(cloud), atol=1e-8)


def test_linear_head_boundary_sits_at_u_zero(rng, cloud):
    W = rng.normal(size=(2, D))
    b = rng.normal(size=2)
    f = build_frame_from_linear_head(W, b, cloud)
    u = f.project(cloud)[:, 0]
    true = (cloud @ W[1] + b[1]) - (cloud @ W[0] + b[0])
    assert np.array_equal(true > 0, u > 0)


def test_linear_head_without_bias(rng, cloud):
    W = rng.normal(size=(2, D))
    f = build_frame_from_linear_head(W, None, cloud)
    true = cloud @ W[1] - cloud @ W[0]
    assert np.allclose(true, f.margin(cloud), atol=1e-8)


# ------------------------------------------------------------ fidelity ----- #

def test_margin_fidelity_is_one_in_its_own_frame(protos_unit, rng):
    """The headline claim: in the current frame for the current method, 1.00."""
    q = np.vstack(_unit(rng, 40))
    scale = 100.0
    f = build_frame(protos_unit, q, logit_form="cosine", scale=scale)
    true = scale * (q @ protos_unit[1] - q @ protos_unit[0])
    proj = project(f, q, protos_unit, true_margins=true)
    assert proj.margin_fidelity == pytest.approx(1.0, abs=1e-9)


def test_margin_fidelity_drops_in_a_borrowed_frame(rng):
    """A frame built for one method must not silently claim fidelity for another."""
    a = np.vstack(_unit(rng, 2))
    b = np.vstack(_unit(rng, 2))
    q = np.vstack(_unit(rng, 50))
    frame_a = build_frame(a, q, logit_form="cosine", scale=1.0)
    margins_b = q @ b[1] - q @ b[0]
    proj = project(frame_a, q, a, true_margins=margins_b)
    assert proj.margin_fidelity < 0.99


def test_explained_variance_is_a_fraction(protos_unit, cloud):
    proj = project(build_frame(protos_unit, cloud), cloud, protos_unit)
    assert 0.0 <= proj.explained_variance <= 1.0


def test_explained_variance_is_one_for_planar_data(protos_unit):
    """Data already lying in the plane must report full fidelity."""
    f = build_frame(protos_unit, None)
    grid = np.array([[u, v] for u in (-2, -1, 0, 1, 2) for v in (-1, 0, 1)], dtype=float)
    pts = f.origin + grid[:, :1] * f.e1 + grid[:, 1:] * f.e2
    proj = project(f, pts, protos_unit)
    assert proj.explained_variance == pytest.approx(1.0, abs=1e-9)


def test_projection_round_trips_planar_points(protos_unit):
    f = build_frame(protos_unit, None)
    target = np.array([[1.5, -0.75], [-2.0, 0.25]])
    pts = f.origin + target[:, :1] * f.e1 + target[:, 1:] * f.e2
    assert np.allclose(f.project(pts), target, atol=1e-10)


# ----------------------------------------------------------- degenerate ---- #

def test_identical_prototypes_raise_with_a_useful_message():
    p = np.vstack(_unit(np.random.default_rng(1), 1))
    with pytest.raises(DegenerateFrameError, match="no decision axis"):
        build_frame(np.vstack([p, p]), None)


def test_no_points_still_builds_a_usable_frame(protos_unit):
    f = build_frame(protos_unit, None)
    assert abs(f.e1 @ f.e2) < 1e-12
    assert any("arbitrary" in w for w in f.warnings)


def test_single_point_still_builds(protos_unit, rng):
    f = build_frame(protos_unit, rng.normal(size=(1, D)))
    assert abs(f.e1 @ f.e2) < 1e-12
    assert f.warnings


def test_points_collinear_with_decision_axis(protos_unit):
    """No orthogonal spread: e2 is arbitrary but must stay orthonormal."""
    f0 = build_frame(protos_unit, None)
    pts = np.array([f0.origin + t * f0.e1 for t in (-2.0, -1.0, 0.0, 1.0, 2.0)])
    f = build_frame(protos_unit, pts)
    assert abs(f.e1 @ f.e2) < 1e-12
    assert any("decision axis" in w for w in f.warnings)


def test_non_unit_prototypes_warn_under_cosine(rng, cloud):
    protos = rng.normal(size=(2, D)) * 3.0
    f = build_frame(protos, cloud, logit_form="cosine")
    assert any("unit-norm" in w for w in f.warnings)


def test_bounds_are_symmetric_about_the_boundary(protos_unit, cloud):
    proj = project(build_frame(protos_unit, cloud), cloud, protos_unit)
    assert proj.bounds["u_min"] == pytest.approx(-proj.bounds["u_max"])
    assert proj.bounds["u_min"] < 0 < proj.bounds["u_max"]


def test_true_margin_length_mismatch_raises(protos_unit, cloud):
    f = build_frame(protos_unit, cloud)
    with pytest.raises(ValueError, match="entries but"):
        project(f, cloud, protos_unit, true_margins=np.zeros(3))


# ------------------------------------------------------------- episode ----- #

def test_project_episode_splits_support_and_query(protos_unit, rng):
    support, query = rng.normal(size=(6, D)), rng.normal(size=(10, D))
    proj, n_support = project_episode(protos_unit, support, query)
    assert n_support == 6
    assert proj.coords.shape == (16, 2)


def test_locked_frame_keeps_coordinates_stable(protos_unit, rng):
    """Reusing a frame lets points animate rather than teleport between k values."""
    support, query = rng.normal(size=(4, D)), rng.normal(size=(8, D))
    proj_a, _ = project_episode(protos_unit, support, query, k=1)
    proj_b, _ = project_episode(
        protos_unit, support, query, frame=proj_a.frame, k=5
    )
    assert np.allclose(proj_a.coords, proj_b.coords, atol=1e-12)


def test_project_episode_handles_empty_query(protos_unit, rng):
    proj, n = project_episode(protos_unit, rng.normal(size=(4, D)), np.zeros((0, D)))
    assert n == 4 and proj.coords.shape == (4, 2)


def test_to_dict_is_json_serialisable(protos_unit, cloud):
    import json

    proj = project(build_frame(protos_unit, cloud), cloud, protos_unit)
    payload = json.dumps(proj.to_dict())
    assert '"boundary_u": 0.0' in payload
    assert '"margin_fidelity"' in payload
