"""
Metric and statistics tests.

Metrics are checked against hand-computed values and sklearn references.
Statistics are checked for the failure the original code shipped: a paired test
applied to unpaired data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, precision_score

from fsgrade.data.episodes import EpisodeAlignmentError
from fsgrade.evaluation.aggregate import EpisodeRecord, MetricAggregator, mean_ci_t
from fsgrade.evaluation.metrics import (
    binary_confusion,
    episode_metrics,
    expected_calibration_error,
    pooled_metrics,
)
from fsgrade.evaluation.stats import (
    EpisodeSeries,
    compare_against_reference,
    holm_bonferroni,
    paired_compare,
)

# TN=8, FP=2, FN=3, TP=7
Y_TRUE = np.array([0] * 10 + [1] * 10)
Y_PRED = np.array([0] * 8 + [1] * 2 + [0] * 3 + [1] * 7)


def test_confusion_matrix_orientation():
    assert binary_confusion(Y_TRUE, Y_PRED) == (8, 2, 3, 7)


def test_core_metrics_match_hand_computation():
    m = episode_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(15 / 20)
    assert m["sensitivity"] == pytest.approx(7 / 10)      # TP/(TP+FN)
    assert m["specificity"] == pytest.approx(8 / 10)      # TN/(TN+FP)
    assert m["balanced_accuracy"] == pytest.approx((0.7 + 0.8) / 2)
    assert m["precision_rotten"] == pytest.approx(7 / 9)
    assert m["recall_rotten"] == pytest.approx(7 / 10)


def test_metrics_match_sklearn():
    m = episode_metrics(Y_TRUE, Y_PRED)
    assert m["mcc"] == pytest.approx(matthews_corrcoef(Y_TRUE, Y_PRED))
    assert m["cohen_kappa"] == pytest.approx(cohen_kappa_score(Y_TRUE, Y_PRED))
    assert m["f1_rotten"] == pytest.approx(f1_score(Y_TRUE, Y_PRED, pos_label=1))
    assert m["precision_rotten"] == pytest.approx(precision_score(Y_TRUE, Y_PRED, pos_label=1))


def test_mcc_matches_closed_form():
    tn, fp, fn, tp = binary_confusion(Y_TRUE, Y_PRED)
    expected = (tp * tn - fp * fn) / np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    assert episode_metrics(Y_TRUE, Y_PRED)["mcc"] == pytest.approx(expected)


def test_degenerate_episode_gives_nan_not_zero():
    """Constant predictions make kappa/MCC undefined. sklearn returns 0.0, which
    reads as 'no agreement' rather than 'not measurable'."""
    m = episode_metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]))
    assert np.isnan(m["cohen_kappa"])
    assert np.isnan(m["mcc"])


def test_single_class_truth_gives_nan_auc():
    m = episode_metrics(np.zeros(4, int), np.zeros(4, int), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(m["roc_auc"])


def test_pooled_equals_episode_mean_for_linear_metrics():
    """The identity holds exactly when episodes are equal-sized."""
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(40):
        yt = np.array([0] * 15 + [1] * 15)
        yp = np.where(rng.random(30) < 0.85, yt, 1 - yt)
        episodes.append((yt, yp))

    ep_mean = np.mean([episode_metrics(a, b)["accuracy"] for a, b in episodes])
    pooled = pooled_metrics(
        np.concatenate([a for a, _ in episodes]),
        np.concatenate([b for _, b in episodes]),
    )
    assert pooled["accuracy"] == pytest.approx(ep_mean, abs=1e-12)
    assert pooled["pooled_equals_episode_mean"] is True


def test_balanced_queries_make_accuracy_equal_balanced_accuracy():
    yt = np.array([0] * 15 + [1] * 15)
    yp = np.array([0] * 12 + [1] * 3 + [0] * 4 + [1] * 11)
    m = episode_metrics(yt, yp)
    assert m["accuracy"] == pytest.approx(m["balanced_accuracy"])
    assert m["accuracy"] == pytest.approx(m["recall_macro"])


def test_ece_is_zero_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = (rng.random(20000) < conf).astype(int)
    ece, mce, curve = expected_calibration_error(correct, conf, n_bins=10)
    assert ece < 0.02
    assert len(curve["bin_accuracy"]) == 10


def test_mean_ci_records_its_unit():
    ci = mean_ci_t([0.8, 0.82, 0.85], unit="fold", ci_method="t_over_folds")
    assert ci.n == 3 and ci.unit == "fold" and ci.ci_method == "t_over_folds"
    assert "fold" in ci.format()


def test_mean_ci_t_matches_scipy():
    vals = [0.81, 0.86, 0.79, 0.9, 0.84]
    ci = mean_ci_t(vals, unit="fold", ci_method="t_over_folds")
    arr = np.asarray(vals)
    crit = sp_stats.t.ppf(0.975, df=len(arr) - 1)
    half = crit * arr.std(ddof=1) / np.sqrt(len(arr))
    assert ci.ci_lo == pytest.approx(arr.mean() - half)
    assert ci.ci_hi == pytest.approx(arr.mean() + half)


def test_nan_values_are_counted_not_silently_dropped():
    ci = mean_ci_t([0.8, np.nan, 0.9], unit="episode", ci_method="t_over_episodes")
    assert ci.n == 2 and ci.n_undefined == 1


# ---------------------------------------------------------------- statistics #

def _series(name, ids, values):
    return EpisodeSeries(name, "accuracy", list(ids), np.asarray(values, dtype=float))


def test_paired_compare_matches_scipy():
    rng = np.random.default_rng(0)
    ids = [f"ep{i}" for i in range(200)]
    a = rng.normal(0.86, 0.05, 200)
    b = rng.normal(0.83, 0.05, 200)
    c = paired_compare(_series("a", ids, a), _series("b", ids, b))
    ref = sp_stats.ttest_rel(a, b)
    assert c.t_stat == pytest.approx(ref.statistic)
    assert c.t_p_raw == pytest.approx(ref.pvalue)
    assert c.n_pairs == 200


def test_cohens_dz_is_the_paired_effect_size():
    ids = [f"ep{i}" for i in range(50)]
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.9, 0.03, 50), rng.normal(0.85, 0.03, 50)
    c = paired_compare(_series("a", ids, a), _series("b", ids, b))
    d = a - b
    assert c.cohens_dz == pytest.approx(d.mean() / d.std(ddof=1))


def test_unpaired_series_raise_rather_than_truncate():
    """The original did a[:min_len] vs b[:min_len] on independently resampled
    episodes, pairing unrelated episodes together."""
    a = _series("a", [f"ep{i}" for i in range(10)], np.linspace(0.8, 0.9, 10))
    b = _series("b", [f"zz{i}" for i in range(10)], np.linspace(0.7, 0.8, 10))
    with pytest.raises(EpisodeAlignmentError, match="No shared episodes"):
        paired_compare(a, b)


def test_partial_overlap_requires_explicit_opt_in():
    ids = [f"ep{i}" for i in range(10)]
    a = _series("a", ids, np.linspace(0.8, 0.9, 10))
    b = _series("b", ids[:5], np.linspace(0.7, 0.8, 5))
    with pytest.raises(EpisodeAlignmentError, match="Partial episode overlap"):
        paired_compare(a, b)
    c = paired_compare(a, b, allow_partial=True)
    assert c.n_pairs == 5 and c.coverage_a == pytest.approx(0.5)


def test_alignment_is_by_id_not_position():
    """Shuffling one series must not change the result."""
    ids = [f"ep{i}" for i in range(30)]
    rng = np.random.default_rng(5)
    va, vb = rng.random(30), rng.random(30)
    straight = paired_compare(_series("a", ids, va), _series("b", ids, vb))

    order = rng.permutation(30)
    shuffled = paired_compare(
        _series("a", ids, va),
        _series("b", [ids[i] for i in order], vb[order]),
    )
    assert shuffled.mean_diff == pytest.approx(straight.mean_diff)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate episode ids"):
        _series("a", ["ep0", "ep0"], [0.5, 0.6])


def test_wilcoxon_uses_pratt_for_ties():
    """Episode accuracy over 30 queries is discrete, so exact ties are common.
    scipy's default zero_method='wilcox' discards them and inflates significance."""
    ids = [f"ep{i}" for i in range(60)]
    v = np.round(np.linspace(0.6, 1.0, 60) * 30) / 30
    w = v.copy()
    w[:30] = np.clip(w[:30] - 1 / 30, 0, 1)
    c = paired_compare(_series("a", ids, v), _series("b", ids, w))
    assert c.wilcoxon_zero_method == "pratt"
    assert c.wilcoxon_zero_pairs == 30


def test_holm_is_monotone_and_conservative():
    ids = [f"ep{i}" for i in range(100)]
    rng = np.random.default_rng(2)
    base = rng.normal(0.8, 0.05, 100)
    series = {"ours": _series("ours", ids, base + 0.05)}
    for i, shift in enumerate([0.0, 0.02, 0.04, 0.20]):
        series[f"m{i}"] = _series(f"m{i}", ids, base - shift)

    cmps = compare_against_reference(series, "ours")
    assert len(cmps) == 4
    for c in cmps:
        assert c.family_size == 4
        assert c.p_holm_t >= c.t_p_raw - 1e-15
    assert len({c.episode_ids_hash for c in cmps}) == 1


def test_holm_ordering_against_reference_values():
    """Holm on p = [0.01, 0.02, 0.03] with m=3 -> [0.03, 0.04, 0.04]."""
    ids = [f"ep{i}" for i in range(20)]
    cmps = []
    for p in (0.01, 0.02, 0.03):
        c = paired_compare(_series("a", ids, np.zeros(20)), _series("b", ids, np.ones(20)))
        c.t_p_raw = p
        cmps.append(c)
    holm_bonferroni(cmps, alpha=0.05)
    got = [round(c.p_holm_t, 10) for c in cmps]
    assert got == [pytest.approx(0.03), pytest.approx(0.04), pytest.approx(0.04)]


# ---------------------------------------------------------------- aggregator #

def _record(eid, method, correct=25, n=30, paths=None):
    yt = np.array([0] * (n // 2) + [1] * (n // 2))
    yp = yt.copy()
    for i in range(n - correct):
        yp[i] = 1 - yp[i]
    return EpisodeRecord(
        episode_id=eid, method=method, fold_id="f1", species="mango", split="test",
        n_shot=5, n_query_per_class=n // 2, seed=1,
        y_true=yt, y_pred=yp, y_score_pos=np.where(yp == 1, 0.8, 0.2),
        query_paths=paths or [f"mango/fresh/img_{eid}_{i}.jpg" for i in range(n)],
    )


def test_aggregator_is_not_replaced_when_empty():
    """MetricAggregator defines __len__, so an empty one is falsy.
    `aggregator or MetricAggregator()` would silently discard the caller's object."""
    agg = MetricAggregator()
    assert len(agg) == 0
    assert not agg              # falsy -- the trap
    assert agg is not None      # the correct check

    agg.add(_record("ep0", "m"))
    assert len(agg) == 1


def test_aggregator_summary_shape():
    agg = MetricAggregator(bootstrap_n=100)
    for i in range(8):
        agg.add(_record(f"ep{i}", "m", correct=24 + (i % 3)))
    summary = agg.summary()
    assert summary["n_episodes"] == 8
    acc = summary["episode_mean"]["accuracy"]
    assert acc["unit"] == "episode" and acc["ci_method"] == "t_over_episodes"
    assert summary["pooled"]["n_queries"] == 8 * 30
    img = summary["image_bootstrap"]["accuracy"]
    assert img["unit"] == "image" and img["ci_method"] == "bootstrap_over_images"


def test_image_bootstrap_clusters_on_distinct_images():
    """n must equal the number of distinct query images, not episodes."""
    agg = MetricAggregator(bootstrap_n=100)
    shared = [f"mango/fresh/img_{i}.jpg" for i in range(30)]
    for i in range(5):
        agg.add(_record(f"ep{i}", "m", correct=25, paths=shared))
    img = agg.image_bootstrap("accuracy")
    assert img["n"] == 30, "should cluster on 30 distinct images, not 5 episodes"
    assert img["unit"] == "image"
