"""
The decision plane: a 2-D projection in which the model's boundary is exact.

Why not t-SNE
-------------
t-SNE has no notion of a decision boundary, its axes carry no units, and its
layout depends on perplexity and a random seed. Showing a classifier's behaviour
on a t-SNE plot means showing a picture that cannot be checked against the
model. (The pre-refactor code went further and computed silhouette and 5-NN
purity *on the 2-D t-SNE output*, with the kNN scored on its own fit data.)

A linear projection, by contrast, preserves every linear decision function
exactly. We project onto the plane spanned by:

    e1 = normalize(p_rotten - p_fresh)      the decision axis
    e2 = leading PC of the residuals orthogonal to e1

With the origin chosen as below, the decision boundary is **exactly** the
vertical line ``u = 0`` for every method in the demo, and the two prototypes sit
symmetrically at ``(-h, 0)`` and ``(+h, 0)`` with ``h = ||w||/2``.

Construction
------------
    w  = p_1 - p_0
    e1 = w / ||w||
    o  = (p_0 + p_1) / 2                        origin = prototype midpoint
    R  = (X - o) - ((X - o) . e1) e1^T          residuals, all orthogonal to e1
    e2 = leading eigenvector of cov(R)
    u  = (x - o) . e1        v = (x - o) . e2

Two facts make this work, and both are asserted in the tests:

1. **e2 is orthogonal to e1 automatically.** Every row of ``R`` lies in the
   orthogonal complement of ``e1``, so the column space of ``R - mean(R)`` does
   too, so every eigenvector of its covariance with non-zero eigenvalue does.
   No Gram-Schmidt is needed.
2. **The origin is the prototype midpoint, not the cloud centroid.** It is
   tempting to shift the origin by the residual centroid ``mean(R)`` so the
   point cloud is vertically centred. That shift leaves ``u`` untouched
   (``mean(R)`` is orthogonal to ``e1``) but it pushes the prototypes *off* the
   ``v = 0`` axis, because ``mean(R)`` is not in general orthogonal to ``e2``.
   Anchoring the prototypes at exactly ``(-h, 0)`` and ``(+h, 0)`` is worth far
   more to the picture than cosmetic centring, so the midpoint wins and the
   viewport handles framing instead.

Exactness per logit form
------------------------
* Euclidean prototypes (ProtoNet)::

      ||x-p_0||^2 - ||x-p_1||^2 = 2 (x-m).w = 2||w|| u

  so the sign of the margin is exactly the sign of u. ProtoNet's logits use -d
  rather than -d^2, which is monotone in u at fixed orthogonal spread and
  identical in sign.

* Scaled cosine with unit-norm prototypes (SAP, ncc_*, clip_text_zeroshot)::

      logit_1 - logit_0 = scale (x_hat . w) = scale ||w|| u

  exactly linear, because ||p_0|| == ||p_1|| forces m.e1 == 0.

* Linear head (zeroshot_supervised), with w = W_1 - W_0 and the origin taken as
  the cloud mean projected onto the decision hyperplane::

      logit_1 - logit_0 = ||w|| u

What is lost, and how it is disclosed
-------------------------------------
Metric fidelity in the directions orthogonal to the plane: two points that
overlap on screen may be far apart in R^d. This is reported numerically rather
than hidden, via ``explained_variance`` and ``margin_fidelity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

# Below this, a quantity is treated as numerically zero.
_EPS = 1e-12

LogitForm = Literal["euclidean", "cosine", "linear"]


class DegenerateFrameError(ValueError):
    """The prototypes coincide, so no decision axis exists."""


@dataclass
class PlaneFrame:
    """An orthonormal 2-D frame with an exact boundary at ``u = 0``."""

    origin: np.ndarray            # [d]
    e1: np.ndarray                # [d] decision axis
    e2: np.ndarray                # [d] leading orthogonal direction
    w_norm: float                 # ||p_1 - p_0||
    logit_form: LogitForm
    scale: float = 1.0
    built_at_k: int | None = None
    warnings: list[str] = field(default_factory=list)

    # -- coordinates ---------------------------------------------------- #
    def project(self, X: np.ndarray) -> np.ndarray:
        """Map points in R^d to (u, v). Accepts [d] or [n, d]."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        centred = X - self.origin
        return np.stack([centred @ self.e1, centred @ self.e2], axis=1)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Signed logit margin (positive = 'rotten') implied by the frame.

        Exact for every supported logit form; see the module docstring.
        """
        u = self.project(X)[:, 0]
        if self.logit_form == "cosine":
            return self.scale * self.w_norm * u
        if self.logit_form == "linear":
            return self.w_norm * u
        return 2.0 * self.w_norm * u          # euclidean, squared-distance margin

    @property
    def prototype_half_gap(self) -> float:
        """Prototypes sit at u = -h and u = +h."""
        return self.w_norm / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_norm": float(self.w_norm),
            "logit_form": self.logit_form,
            "scale": float(self.scale),
            "prototype_half_gap": float(self.prototype_half_gap),
            "built_at_k": self.built_at_k,
            "warnings": list(self.warnings),
        }


@dataclass
class Projection:
    """Everything the frontend needs to draw one plot."""

    coords: np.ndarray                       # [n, 2]
    prototypes_2d: np.ndarray                # [2, 2] rows = (fresh, rotten)
    frame: PlaneFrame
    explained_variance: float
    margin_fidelity: float
    sign_agreement: float = 1.0
    boundary_u: float = 0.0
    bounds: dict[str, float] = field(default_factory=dict)
    degenerate: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords": [[float(a), float(b)] for a, b in self.coords],
            "prototypes": [[float(a), float(b)] for a, b in self.prototypes_2d],
            "boundary_u": float(self.boundary_u),
            "explained_variance": float(self.explained_variance),
            "margin_fidelity": float(self.margin_fidelity),
            "sign_agreement": float(self.sign_agreement),
            "bounds": {k: float(v) for k, v in self.bounds.items()},
            "degenerate": list(self.degenerate),
            "frame": self.frame.to_dict(),
        }


# --------------------------------------------------------------------------- #
#  Frame construction
# --------------------------------------------------------------------------- #

def build_frame(
    prototypes: np.ndarray,
    points: np.ndarray | None = None,
    *,
    logit_form: LogitForm = "cosine",
    scale: float = 1.0,
    built_at_k: int | None = None,
) -> PlaneFrame:
    """Build the decision plane from two prototypes and a point cloud.

    ``prototypes`` is [2, d] with row 0 = fresh (negative) and row 1 = rotten
    (positive). ``points`` is [n, d]; when omitted or degenerate, ``e2`` falls
    back to an arbitrary unit vector orthogonal to ``e1`` so the plot still
    renders.
    """
    protos = np.asarray(prototypes, dtype=np.float64)
    if protos.shape[0] != 2:
        raise ValueError(f"Expected 2 prototypes, got {protos.shape[0]}")

    p0, p1 = protos[0], protos[1]
    w = p1 - p0
    w_norm = float(np.linalg.norm(w))
    warnings: list[str] = []

    if w_norm < _EPS:
        raise DegenerateFrameError(
            "The two prototypes are identical, so there is no decision axis. "
            "This usually means the support set for one class is empty or all "
            "support embeddings collapsed."
        )

    e1 = w / w_norm
    # The prototype midpoint anchors both prototypes at v = 0. See note 2 in the
    # module docstring for why the cloud is not re-centred here.
    origin = (p0 + p1) / 2.0

    e2, _residual_centroid, note = _orthogonal_direction(points, origin, e1)
    if note:
        warnings.append(note)

    if logit_form == "cosine":
        norms = np.linalg.norm(protos, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            warnings.append(
                f"cosine logit form expects unit-norm prototypes, got "
                f"norms {norms.round(4).tolist()}; the boundary is still at u=0 "
                f"but the margin scale is approximate"
            )

    return PlaneFrame(
        origin=origin, e1=e1, e2=e2, w_norm=w_norm,
        logit_form=logit_form, scale=scale, built_at_k=built_at_k,
        warnings=warnings,
    )


def build_frame_from_linear_head(
    weight: np.ndarray,
    bias: np.ndarray | None,
    points: np.ndarray,
    *,
    built_at_k: int | None = None,
) -> PlaneFrame:
    """Frame for a linear classifier, which has no prototypes.

    Uses ``w = W[1] - W[0]`` and places the origin at the cloud mean projected
    onto the decision hyperplane, so the margin is again exactly ``||w|| u``.
    """
    W = np.asarray(weight, dtype=np.float64)
    if W.shape[0] != 2:
        raise ValueError(f"Expected a 2-class head, got {W.shape[0]} rows")

    w = W[1] - W[0]
    w_norm = float(np.linalg.norm(w))
    if w_norm < _EPS:
        raise DegenerateFrameError("The two class weight vectors are identical.")

    delta_b = 0.0
    if bias is not None:
        b = np.asarray(bias, dtype=np.float64)
        delta_b = float(b[1] - b[0])

    e1 = w / w_norm
    X = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x_bar = X.mean(axis=0)
    # Project the cloud mean onto {x : w.x + delta_b = 0}.
    origin = x_bar - ((w @ x_bar + delta_b) / (w_norm ** 2)) * w

    e2, residual_centroid, note = _orthogonal_direction(X, origin, e1)
    warnings = [note] if note else []
    origin = origin + residual_centroid

    return PlaneFrame(
        origin=origin, e1=e1, e2=e2, w_norm=w_norm,
        logit_form="linear", scale=1.0, built_at_k=built_at_k, warnings=warnings,
    )


def _orthogonal_direction(
    points: np.ndarray | None, centre: np.ndarray, e1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Leading PC of the residuals orthogonal to ``e1``.

    Returns (e2, residual_centroid, warning). Falls back to an arbitrary unit
    vector orthogonal to e1 when there is not enough spread to define one.
    """
    d = e1.shape[0]
    zero = np.zeros(d)

    if points is None:
        return _arbitrary_orthogonal(e1), zero, "no points supplied; e2 is arbitrary"

    X = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if X.shape[0] < 2:
        return _arbitrary_orthogonal(e1), zero, "fewer than 2 points; e2 is arbitrary"

    centred = X - centre
    residuals = centred - np.outer(centred @ e1, e1)
    residual_centroid = residuals.mean(axis=0)
    deviations = residuals - residual_centroid

    spread = float(np.linalg.norm(deviations))
    if spread < _EPS:
        return (
            _arbitrary_orthogonal(e1),
            residual_centroid,
            "all points lie on the decision axis; e2 is arbitrary",
        )

    # Economy SVD is more stable than forming the covariance explicitly, and d
    # is only a few hundred here.
    _, _, vt = np.linalg.svd(deviations, full_matrices=False)
    e2 = vt[0]

    # Guard against round-off leaving a component along e1.
    e2 = e2 - (e2 @ e1) * e1
    norm = float(np.linalg.norm(e2))
    if norm < _EPS:  # pragma: no cover - defensive
        return _arbitrary_orthogonal(e1), residual_centroid, "e2 degenerate after re-orthogonalisation"
    return e2 / norm, residual_centroid, None


def _arbitrary_orthogonal(e1: np.ndarray) -> np.ndarray:
    """Any unit vector orthogonal to e1, chosen deterministically."""
    d = e1.shape[0]
    # Start from the standard basis vector least aligned with e1.
    basis = np.zeros(d)
    basis[int(np.argmin(np.abs(e1)))] = 1.0
    v = basis - (basis @ e1) * e1
    norm = float(np.linalg.norm(v))
    if norm < _EPS:  # pragma: no cover - only if d == 1
        v = np.zeros(d)
        v[0] = 1.0
        return v
    return v / norm


# --------------------------------------------------------------------------- #
#  Projection + fidelity
# --------------------------------------------------------------------------- #

def project(
    frame: PlaneFrame,
    points: np.ndarray,
    prototypes: np.ndarray,
    *,
    true_margins: np.ndarray | None = None,
    padding: float = 0.12,
) -> Projection:
    """Project a cloud and its prototypes, with honest fidelity reporting.

    ``true_margins`` are the model's actual logit differences for ``points``. If
    given, ``margin_fidelity`` is their correlation with the in-plane signed
    distance to the drawn boundary — 1.0 when the frame belongs to the method
    being plotted, lower when the frame is locked or borrowed from another arm.
    """
    X = np.atleast_2d(np.asarray(points, dtype=np.float64))
    coords = frame.project(X)
    protos_2d = frame.project(np.asarray(prototypes, dtype=np.float64))

    degenerate = list(frame.warnings)

    # Fraction of the cloud's total spread captured by the plane.
    if X.shape[0] >= 2:
        total = float(np.var(X, axis=0, ddof=1).sum())
        in_plane = float(np.var(coords, axis=0, ddof=1).sum())
        explained = in_plane / total if total > _EPS else 1.0
    else:
        explained = 1.0
        degenerate.append("fewer than 2 points; explained variance undefined")

    # Fidelity of the drawn boundary to the model's real margin.
    implied = frame.margin(X)
    if true_margins is None:
        fidelity = 1.0
    else:
        truth = np.asarray(true_margins, dtype=np.float64).ravel()
        if len(truth) != len(implied):
            raise ValueError(
                f"true_margins has {len(truth)} entries but {len(implied)} points were given"
            )
        fidelity = _safe_corr(truth, implied)

    return Projection(
        coords=coords,
        prototypes_2d=protos_2d,
        frame=frame,
        explained_variance=float(np.clip(explained, 0.0, 1.0)),
        margin_fidelity=float(fidelity),
        bounds=_bounds(np.vstack([coords, protos_2d]), padding),
        degenerate=degenerate,
    )


def sign_agreement(true_margins: np.ndarray, implied_margins: np.ndarray) -> float:
    """Fraction of points on which the drawn boundary agrees with the model.

    This is the claim that is *exact* for distance-based logits. ProtoNet and
    the pixel baseline score with ``-d`` rather than ``-d^2``, so their margin
    is monotone in ``u`` but not linear -- Pearson correlation therefore sits
    below 1.0 even though every prediction falls on the correct side of the
    line. Reporting sign agreement alongside makes that distinction visible
    instead of looking like an error.
    """
    a = np.asarray(true_margins, dtype=np.float64).ravel()
    b = np.asarray(implied_margins, dtype=np.float64).ravel()
    if not len(a):
        return 1.0
    return float((np.sign(a) == np.sign(b)).mean())


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 1.0 for exactly-proportional inputs.

    A constant series has undefined correlation; when both are constant (or one
    is an exact scalar multiple of the other) the frame is still faithful, so we
    report 1.0 rather than NaN.
    """
    if len(a) < 2:
        return 1.0
    sa, sb = float(a.std()), float(b.std())
    if sa < _EPS and sb < _EPS:
        return 1.0
    if sa < _EPS or sb < _EPS:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _bounds(coords: np.ndarray, padding: float) -> dict[str, float]:
    """Axis limits with padding, always containing the boundary at u = 0."""
    if coords.size == 0:  # pragma: no cover - defensive
        return {"u_min": -1.0, "u_max": 1.0, "v_min": -1.0, "v_max": 1.0}

    u_min, v_min = coords.min(axis=0)
    u_max, v_max = coords.max(axis=0)

    # The boundary must always be visible, and the view symmetric about it so
    # the two classes are never given unequal visual weight.
    reach_u = max(abs(float(u_min)), abs(float(u_max)), _EPS)
    span_v = max(float(v_max) - float(v_min), _EPS)
    pad_u = reach_u * padding
    pad_v = span_v * padding

    return {
        "u_min": -(reach_u + pad_u),
        "u_max": reach_u + pad_u,
        "v_min": float(v_min) - pad_v,
        "v_max": float(v_max) + pad_v,
    }


# --------------------------------------------------------------------------- #
#  Convenience
# --------------------------------------------------------------------------- #

def project_episode(
    prototypes: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    *,
    logit_form: LogitForm = "cosine",
    scale: float = 1.0,
    true_margins: np.ndarray | None = None,
    frame: PlaneFrame | None = None,
    k: int | None = None,
) -> tuple[Projection, int]:
    """Project one episode. Returns (projection, n_support).

    Support and query are stacked so both live in one frame; the caller splits
    ``coords`` at the returned index. Pass ``frame`` to reuse a locked frame
    across k values so points animate rather than teleport.
    """
    support = np.atleast_2d(np.asarray(support, dtype=np.float64)) if len(support) else np.zeros((0, prototypes.shape[1]))
    query = np.atleast_2d(np.asarray(query, dtype=np.float64)) if len(query) else np.zeros((0, prototypes.shape[1]))
    cloud = np.vstack([support, query]) if len(support) or len(query) else np.zeros((0, prototypes.shape[1]))

    if frame is None:
        frame = build_frame(
            prototypes, cloud if len(cloud) else None,
            logit_form=logit_form, scale=scale, built_at_k=k,
        )

    # Fidelity is measured on the query points only -- they are what the model
    # actually classified.
    proj = project(frame, cloud, prototypes, true_margins=None)
    if true_margins is not None and len(query):
        q_implied = frame.margin(query)
        proj.margin_fidelity = _safe_corr(
            np.asarray(true_margins, dtype=np.float64).ravel(), q_implied
        )
    return proj, len(support)
