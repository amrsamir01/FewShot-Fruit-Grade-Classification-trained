"""
Statistical comparison of methods on a shared episode bank.

The original ``statistical_significance_tests`` did::

    a = ours_accs[:min_len]
    b = other_accs[:min_len]
    ttest_rel(a, b); wilcoxon(a, b)

but each method's accuracies came from its own independently resampled episodes.
Episode *i* of one run and episode *i* of another were different images of
different species, so the pairing was arbitrary and the reported p-values (down
to 2.68e-125) were not interpretable.

Here, values only ever enter an ``EpisodeSeries`` keyed by ``episode_id``, and
``paired_compare`` aligns on the *intersection* of ids. Misaligned inputs raise
rather than silently truncating. ``episode_ids_hash`` is recorded on every
comparison so a reviewer can verify the pairing was genuine.

Three further corrections over the original:

* **Wilcoxon zero handling.** Episode accuracy over 30 queries takes only 31
  distinct values, so paired differences contain many exact zeros. SciPy's
  default ``zero_method="wilcox"`` discards them, which inflates significance.
  We use ``"pratt"`` and record how many zero pairs occurred.
* **Multiplicity.** Holm-Bonferroni within an explicitly declared family.
* **Effect size and interval.** The primary inference is the paired bootstrap
  CI of the mean difference; the tests corroborate it. A 95% CI on the accuracy
  difference that excludes zero is a better thesis sentence than a p-value, and
  it is robust to the ties problem.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

try:
    from scipy import stats as sp_stats

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

from fsgrade.data.episodes import EpisodeAlignmentError

STATS_SCHEMA_VERSION = "1.0.0"


@dataclass
class EpisodeSeries:
    """One method's per-episode values for one metric, keyed by episode id."""

    method: str
    metric: str
    episode_ids: list[str]
    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if len(self.episode_ids) != len(self.values):
            raise ValueError(
                f"{self.method}/{self.metric}: {len(self.episode_ids)} ids but "
                f"{len(self.values)} values"
            )
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError(f"{self.method}/{self.metric}: duplicate episode ids")

    def as_map(self) -> dict[str, float]:
        return dict(zip(self.episode_ids, self.values.tolist()))

    @classmethod
    def from_aggregator(cls, agg: Any, method: str, metric: str) -> "EpisodeSeries":
        return cls(
            method=method,
            metric=metric,
            episode_ids=agg.episode_ids(),
            values=agg.series(metric),
        )


def _ids_hash(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(i.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


@dataclass
class PairedComparison:
    method_a: str
    method_b: str
    metric: str

    n_pairs: int
    coverage_a: float
    coverage_b: float
    episode_ids_hash: str

    mean_a: float
    mean_b: float
    mean_diff: float

    diff_ci_lo: float
    diff_ci_hi: float
    diff_ci_method: str
    diff_ci_level: float

    t_stat: float
    t_df: int
    t_p_raw: float

    wilcoxon_stat: float
    wilcoxon_p_raw: float
    wilcoxon_zero_pairs: int
    wilcoxon_zero_method: str

    cohens_dz: float
    hedges_g: float

    alternative: str
    family: str = ""
    family_size: int = 0
    alpha: float = 0.05
    p_holm_t: float = float("nan")
    p_holm_wilcoxon: float = float("nan")
    reject_holm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_compare(
    a: EpisodeSeries,
    b: EpisodeSeries,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    bootstrap_n: int = 10_000,
    seed: int = 7,
    level: float = 0.95,
    allow_partial: bool = False,
) -> PairedComparison:
    """Compare two methods on the episodes they genuinely share.

    Raises ``EpisodeAlignmentError`` when the series do not cover the same
    episodes, unless ``allow_partial=True`` (used for methods evaluated on a
    prefix of the bank, where reduced coverage is recorded).
    """
    if a.metric != b.metric:
        raise ValueError(f"Cannot compare different metrics: {a.metric} vs {b.metric}")

    map_a, map_b = a.as_map(), b.as_map()
    shared = [eid for eid in a.episode_ids if eid in map_b]

    if not shared:
        raise EpisodeAlignmentError(
            f"No shared episodes between {a.method!r} and {b.method!r}. "
            "Both methods must be evaluated on the same episode bank for a "
            "paired test to be valid."
        )

    coverage_a = len(shared) / len(a.episode_ids)
    coverage_b = len(shared) / len(b.episode_ids)
    if not allow_partial and (coverage_a < 1.0 or coverage_b < 1.0):
        raise EpisodeAlignmentError(
            f"Partial episode overlap between {a.method!r} ({coverage_a:.1%}) and "
            f"{b.method!r} ({coverage_b:.1%}). Pass allow_partial=True only if "
            "this is intentional (e.g. an expensive method run on a bank prefix)."
        )

    va = np.asarray([map_a[e] for e in shared], dtype=float)
    vb = np.asarray([map_b[e] for e in shared], dtype=float)

    finite = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[finite], vb[finite]
    kept_ids = [e for e, keep in zip(shared, finite) if keep]
    n = len(va)

    diff = va - vb
    mean_diff = float(diff.mean()) if n else float("nan")
    sd_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")

    # --- tests -------------------------------------------------------- #
    t_stat = t_p = float("nan")
    w_stat = w_p = float("nan")
    n_zero = int(np.sum(diff == 0.0))

    if _HAVE_SCIPY and n > 1:
        res = sp_stats.ttest_rel(va, vb, alternative=alternative)
        t_stat, t_p = float(res.statistic), float(res.pvalue)
        if np.any(diff != 0.0):
            try:
                wres = sp_stats.wilcoxon(
                    va, vb, alternative=alternative, zero_method="pratt"
                )
                w_stat, w_p = float(wres.statistic), float(wres.pvalue)
            except ValueError:  # pragma: no cover
                pass

    # --- paired bootstrap CI of the mean difference (primary) ---------- #
    if n > 1:
        rng = np.random.default_rng(seed)
        draws = np.empty(bootstrap_n, dtype=float)
        for i in range(bootstrap_n):
            idx = rng.integers(0, n, size=n)
            draws[i] = diff[idx].mean()
        alpha_q = (1.0 - level) / 2.0
        ci_lo = float(np.quantile(draws, alpha_q))
        ci_hi = float(np.quantile(draws, 1.0 - alpha_q))
    else:  # pragma: no cover
        ci_lo = ci_hi = float("nan")

    # --- effect sizes -------------------------------------------------- #
    # dz is the correct paired effect size: mean difference over the SD of the
    # differences. Reported as primary.
    dz = mean_diff / sd_diff if sd_diff and np.isfinite(sd_diff) and sd_diff > 0 else float("nan")
    correction = 1.0 - 3.0 / (4.0 * n - 5.0) if n > 2 else 1.0
    hedges = dz * correction if np.isfinite(dz) else float("nan")

    return PairedComparison(
        method_a=a.method,
        method_b=b.method,
        metric=a.metric,
        n_pairs=n,
        coverage_a=coverage_a,
        coverage_b=coverage_b,
        episode_ids_hash=_ids_hash(kept_ids),
        mean_a=float(va.mean()) if n else float("nan"),
        mean_b=float(vb.mean()) if n else float("nan"),
        mean_diff=mean_diff,
        diff_ci_lo=ci_lo,
        diff_ci_hi=ci_hi,
        diff_ci_method="paired_bootstrap_over_episodes",
        diff_ci_level=level,
        t_stat=t_stat,
        t_df=max(n - 1, 0),
        t_p_raw=t_p,
        wilcoxon_stat=w_stat,
        wilcoxon_p_raw=w_p,
        wilcoxon_zero_pairs=n_zero,
        wilcoxon_zero_method="pratt",
        cohens_dz=dz,
        hedges_g=hedges,
        alternative=alternative,
    )


def holm_bonferroni(
    comparisons: Sequence[PairedComparison],
    *,
    alpha: float = 0.05,
    p_field: str = "t_p_raw",
    family: str = "main_comparison",
) -> list[PairedComparison]:
    """Holm step-down correction within an explicitly declared family.

    The family is declared, never inferred: reporting which comparisons were
    corrected together is part of the method, and a reviewer will ask.
    """
    items = list(comparisons)
    m = len(items)
    if m == 0:
        return items

    for field_name in ("t_p_raw", "wilcoxon_p_raw"):
        raw = np.asarray([getattr(c, field_name) for c in items], dtype=float)
        order = np.argsort(np.where(np.isnan(raw), np.inf, raw))
        adjusted = np.full(m, float("nan"))
        running = 0.0
        for rank, idx in enumerate(order):
            p = raw[idx]
            if not np.isfinite(p):
                continue
            candidate = (m - rank) * p
            running = max(running, candidate)
            adjusted[idx] = min(1.0, running)
        target = "p_holm_t" if field_name == "t_p_raw" else "p_holm_wilcoxon"
        for i, c in enumerate(items):
            setattr(c, target, float(adjusted[i]))

    decisive = "p_holm_t" if p_field == "t_p_raw" else "p_holm_wilcoxon"
    for c in items:
        c.family = family
        c.family_size = m
        c.alpha = alpha
        p = getattr(c, decisive)
        c.reject_holm = bool(np.isfinite(p) and p < alpha)
    return items


def compare_against_reference(
    series_by_method: dict[str, EpisodeSeries],
    reference: str,
    *,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    family: str = "main_comparison",
    bootstrap_n: int = 10_000,
    seed: int = 7,
    allow_partial: bool = False,
) -> list[PairedComparison]:
    """Compare every method against ``reference``, then apply Holm correction."""
    if reference not in series_by_method:
        raise KeyError(
            f"Reference method {reference!r} not found. "
            f"Available: {sorted(series_by_method)}"
        )
    ref = series_by_method[reference]
    comparisons = [
        paired_compare(
            ref, other,
            alternative=alternative,
            bootstrap_n=bootstrap_n,
            seed=seed,
            allow_partial=allow_partial,
        )
        for name, other in series_by_method.items()
        if name != reference
    ]
    return holm_bonferroni(comparisons, alpha=alpha, family=family)


def comparisons_to_payload(
    comparisons: Sequence[PairedComparison],
    *,
    reference: str,
    metric: str,
    alpha: float = 0.05,
    correction: str = "holm_bonferroni",
    source_runs: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the ``comparisons.json`` payload."""
    hashes = {c.episode_ids_hash for c in comparisons}
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "family": {
            "name": comparisons[0].family if comparisons else "",
            "alpha": alpha,
            "size": len(comparisons),
            "correction": correction,
        },
        "reference_method": reference,
        "metric": metric,
        "pairing_unit": "episode",
        "episode_ids_hash": sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes),
        "all_comparisons_share_episodes": len(hashes) == 1,
        "source_runs": list(source_runs),
        "comparisons": [c.to_dict() for c in comparisons],
    }
