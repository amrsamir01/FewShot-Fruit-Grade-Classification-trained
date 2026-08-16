"""
Classification metrics for episodic evaluation.

Two aggregations exist and they are *not* interchangeable:

``episode_metrics``
    computed within one episode, then averaged over episodes. This is the
    few-shot convention and the right headline for accuracy-family metrics.

``pooled_metrics``
    computed once over every query prediction from every episode. The right
    headline for rank- and calibration-based metrics.

An important algebraic fact, easy to get wrong in both directions: with a
**balanced** query set (the default 15 fresh / 15 rotten), pooled and
episode-mean are *numerically identical* for any metric that is a plain
per-query mean -- accuracy, balanced accuracy, macro recall, sensitivity,
specificity, Brier, NLL. Reporting both as if they were separate findings would
be an error. They diverge only for nonlinear functionals:

    identical : accuracy, balanced_accuracy, recall_*, sensitivity,
                specificity, recall_macro, brier, nll
    differ    : precision_* , f1_*, roc_auc, pr_auc, cohen_kappa, mcc, ece

The ``pooled_equals_episode_mean`` flag in the output records which regime
applies, so a reader of ``metrics.json`` cannot mistake one for the other.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        cohen_kappa_score,
        confusion_matrix,
        matthews_corrcoef,
        roc_auc_score,
    )

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAVE_SKLEARN = False


_EPS = 1e-12


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) for labels {0, 1}, positive class = 1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 15,
) -> tuple[float, float, dict[str, list[float]]]:
    """Equal-width-binned ECE and MCE, plus the reliability curve.

    ``y_score`` is the predicted probability of the *predicted* class
    (confidence), not of the positive class.
    """
    y_true = np.asarray(y_true).astype(int)
    conf = np.asarray(y_score, dtype=float)
    correct = y_true.astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    bin_count: list[float] = []

    ece = 0.0
    mce = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            bin_count.append(0.0)
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        gap = abs(avg_acc - avg_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        bin_count.append(float(count))

    curve = {
        "bin_edges": edges.tolist(),
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
    }
    return float(ece), float(mce), curve


def _core_counts_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    """Metrics derivable purely from the 2x2 confusion matrix."""
    total = tn + fp + fn + tp
    accuracy = _safe_div(tp + tn, total)

    recall_pos = _safe_div(tp, tp + fn)      # sensitivity, recall of 'rotten'
    recall_neg = _safe_div(tn, tn + fp)      # specificity, recall of 'fresh'
    prec_pos = _safe_div(tp, tp + fp)
    prec_neg = _safe_div(tn, tn + fn)

    def _f1(p: float, r: float) -> float:
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    f1_pos = _f1(prec_pos, recall_pos)
    f1_neg = _f1(prec_neg, recall_neg)

    valid_recalls = [r for r in (recall_neg, recall_pos) if not math.isnan(r)]
    balanced_acc = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity": recall_pos,
        "specificity": recall_neg,
        "recall_fresh": recall_neg,
        "recall_rotten": recall_pos,
        "precision_fresh": prec_neg,
        "precision_rotten": prec_pos,
        "f1_fresh": f1_neg,
        "f1_rotten": f1_pos,
        "precision_macro": float(np.nanmean([prec_neg, prec_pos])),
        "recall_macro": float(np.nanmean([recall_neg, recall_pos])),
        "f1_macro": float(np.nanmean([f1_neg, f1_pos])),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def _rank_metrics(y_true: np.ndarray, y_score_pos: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC. NaN when only one class is present (undefined, not zero)."""
    out = {"roc_auc": float("nan"), "pr_auc": float("nan")}
    if not _HAVE_SKLEARN:  # pragma: no cover
        return out
    if len(np.unique(y_true)) < 2:
        return out
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score_pos))
        out["pr_auc"] = float(average_precision_score(y_true, y_score_pos))
    except ValueError:  # pragma: no cover
        pass
    return out


def _agreement_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Cohen's kappa and MCC. Both are NaN when a marginal is degenerate."""
    out = {"cohen_kappa": float("nan"), "mcc": float("nan")}
    if not _HAVE_SKLEARN:  # pragma: no cover
        return out
    # Undefined when predictions (or truths) are constant: the denominator
    # vanishes. sklearn returns 0.0 with a warning, which is misleading -- a
    # degenerate episode has no measurable agreement, it does not have zero
    # agreement. We report NaN and count it.
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return out
    try:
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1]))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:  # pragma: no cover
        pass
    return out


def _proper_scores(y_true: np.ndarray, y_score_pos: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(y_score_pos, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y_true, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"brier": brier, "nll": nll}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def episode_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_pos: np.ndarray | None = None,
    *,
    class_names: Sequence[str] = ("fresh", "rotten"),
) -> dict[str, float]:
    """Metrics for a single episode.

    ``y_score_pos`` is P(class = 1). Rank metrics and proper scores are omitted
    when it is not supplied.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = binary_confusion(y_true, y_pred)
    out = _core_counts_metrics(tn, fp, fn, tp)
    out.update(_agreement_metrics(y_true, y_pred))

    if y_score_pos is not None:
        y_score_pos = np.asarray(y_score_pos, dtype=float)
        out.update(_rank_metrics(y_true, y_score_pos))
        out.update(_proper_scores(y_true, y_score_pos))

    return out


def pooled_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_pos: np.ndarray | None = None,
    *,
    class_names: Sequence[str] = ("fresh", "rotten"),
    ece_bins: int = 15,
    episode_sizes_equal: bool = True,
    query_balanced: bool = True,
) -> dict[str, Any]:
    """Metrics over every query prediction from every episode, pooled.

    ``episode_sizes_equal`` and ``query_balanced`` determine the
    ``pooled_equals_episode_mean`` flag: the identity holds for per-query-mean
    metrics exactly when every episode contributes the same number of queries.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = binary_confusion(y_true, y_pred)
    out: dict[str, Any] = dict(_core_counts_metrics(tn, fp, fn, tp))
    out.update(_agreement_metrics(y_true, y_pred))
    out["n_queries"] = int(len(y_true))
    out["confusion_matrix"] = [[tn, fp], [fn, tp]]
    out["confusion_matrix_labels"] = list(class_names)
    out["confusion_matrix_layout"] = "rows = true (fresh, rotten), cols = predicted"

    if y_score_pos is not None:
        y_score_pos = np.asarray(y_score_pos, dtype=float)
        rank = _rank_metrics(y_true, y_score_pos)
        out["roc_auc_pooled"] = rank["roc_auc"]
        out["pr_auc_pooled"] = rank["pr_auc"]
        out.update(_proper_scores(y_true, y_score_pos))

        # Confidence = probability assigned to the predicted class.
        conf = np.where(y_pred == 1, y_score_pos, 1.0 - y_score_pos)
        correct = (y_pred == y_true).astype(int)
        ece, mce, curve = expected_calibration_error(correct, conf, n_bins=ece_bins)
        out[f"ece_{ece_bins}bin"] = ece
        out[f"mce_{ece_bins}bin"] = mce
        out["reliability"] = curve

    out["pooled_equals_episode_mean"] = bool(episode_sizes_equal)
    out["pooled_identity_note"] = (
        "With equal-size episodes, accuracy / balanced_accuracy / recall_* / "
        "sensitivity / specificity / brier / nll are numerically identical to "
        "their episode-mean counterparts; only their uncertainty differs. "
        "precision_*, f1_*, roc_auc, pr_auc, cohen_kappa, mcc and ece do differ."
        if episode_sizes_equal else
        "Episode sizes differ, so no pooled/episode-mean identity holds."
    )
    if query_balanced:
        out["balanced_note"] = (
            "Query set is class-balanced, so accuracy == balanced_accuracy == "
            "recall_macro identically. Chance level is 1/n_way."
        )
    return out


# Metrics for which pooled == episode-mean under equal-size episodes.
LINEAR_METRICS: frozenset[str] = frozenset({
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "recall_fresh", "recall_rotten", "recall_macro", "brier", "nll",
})

# Metrics whose headline should be the pooled value.
PREFER_POOLED: frozenset[str] = frozenset({
    "roc_auc", "pr_auc", "cohen_kappa", "mcc", "ece", "mce",
})


def temperature_scale_logits(
    support_logits: np.ndarray,
    support_labels: np.ndarray,
    query_logits: np.ndarray,
    *,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Fit a scalar temperature on the support set and apply it to query logits.

    Legal in the few-shot setting: support labels are given by definition. This
    separates "the encoder is miscalibrated on an unseen species" from "the
    learned global temperature transferred badly" -- a distinction that matters
    for a sorting-line deployment, where the operating threshold is chosen from
    predicted confidence.
    """
    from scipy.optimize import minimize_scalar

    support_logits = np.asarray(support_logits, dtype=float)
    support_labels = np.asarray(support_labels).astype(int)

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        z = support_logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(support_labels)), support_labels].mean())

    res = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                          options={"maxiter": max_iter})
    temperature = float(math.exp(res.x))
    return np.asarray(query_logits, dtype=float) / temperature, temperature
