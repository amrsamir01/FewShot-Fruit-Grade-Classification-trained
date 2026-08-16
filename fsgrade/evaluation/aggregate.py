"""
Metric aggregation with self-describing confidence intervals.

The single most dangerous reporting bug in the original code was that the
headline "95% CI" was computed over **5 trial means** while every other table's
"95% CI" was computed over **600 episodes** -- and the two were printed side by
side under the same column heading. ``86.3% +/- 0.4%`` and ``86.0% +/- 0.6%``
were not comparable quantities.

Here it is *impossible* to construct an interval without declaring its method,
its ``n``, and its unit of analysis. The table renderer prints the unit in the
column header, so two differently-scoped intervals can never again share one.

Three intervals with genuinely different meanings are supported:

    t_over_episodes / bootstrap_over_episodes
        episode-sampler noise, conditional on this image pool
    bootstrap_over_images
        would this hold on a different sample of mangoes?
    t_over_folds
        would this hold on a different unseen *species*?

The last is the one that actually supports "generalises to unseen species", and
it is much wider. 600 test episodes drawn from ~700 images overlap heavily, so
episodes are not independent and the episode-level interval is anticonservative.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

try:
    from scipy import stats as sp_stats

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

from fsgrade.evaluation.metrics import (
    LINEAR_METRICS,
    episode_metrics,
    pooled_metrics,
)

CIMethod = Literal[
    "t_over_episodes",
    "bootstrap_over_episodes",
    "bootstrap_over_images",
    "t_over_folds",
    "t_over_seeds",
]

Unit = Literal["episode", "fold", "seed", "image", "query"]


@dataclass
class MeanCI:
    """A mean with an interval that knows what it is an interval over."""

    mean: float
    sd: float
    sem: float
    ci_lo: float
    ci_hi: float
    ci_level: float
    ci_method: CIMethod
    n: int
    unit: Unit
    n_undefined: int = 0

    @property
    def half_width(self) -> float:
        return (self.ci_hi - self.ci_lo) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "sd": self.sd,
            "sem": self.sem,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "ci_level": self.ci_level,
            "ci_method": self.ci_method,
            "n": self.n,
            "unit": self.unit,
            "n_undefined": self.n_undefined,
        }

    def format(self, *, percent: bool = True, digits: int = 1) -> str:
        scale = 100.0 if percent else 1.0
        suffix = "%" if percent else ""
        return (
            f"{self.mean * scale:.{digits}f}{suffix} "
            f"+/- {self.half_width * scale:.{digits}f}{suffix} "
            f"(95% CI, {self.ci_method}, n={self.n} {self.unit}s)"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"MeanCI({self.format()})"


def mean_ci_t(
    values: Sequence[float] | np.ndarray,
    *,
    unit: Unit,
    ci_method: CIMethod,
    level: float = 0.95,
) -> MeanCI:
    """Student-t interval. Correct for small n (folds, seeds) as well as large."""
    arr = np.asarray(list(values), dtype=float)
    n_total = len(arr)
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    n_undefined = n_total - n

    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), float("nan"),
                      float("nan"), level, ci_method, 0, unit, n_undefined)
    if n == 1:
        v = float(finite[0])
        return MeanCI(v, 0.0, 0.0, v, v, level, ci_method, 1, unit, n_undefined)

    mean = float(finite.mean())
    sd = float(finite.std(ddof=1))
    sem = sd / float(np.sqrt(n))
    if _HAVE_SCIPY:
        crit = float(sp_stats.t.ppf(0.5 + level / 2.0, df=n - 1))
    else:  # pragma: no cover
        crit = 1.96
    half = crit * sem
    return MeanCI(mean, sd, sem, mean - half, mean + half, level, ci_method, n, unit, n_undefined)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    unit: Unit,
    ci_method: CIMethod,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    seed: int = 7,
    level: float = 0.95,
    clusters: Sequence[Any] | None = None,
) -> MeanCI:
    """Percentile bootstrap, optionally clustered.

    ``clusters`` enables the *image-level* bootstrap: resample images (with all
    the episodes that used them) rather than episodes. Because 600 episodes are
    drawn from ~700 images, episodes share images heavily and an episode-level
    interval understates the true uncertainty about a different sample of fruit.
    """
    arr = np.asarray(list(values), dtype=float)
    n_total = len(arr)
    mask = np.isfinite(arr)
    finite = arr[mask]
    n = len(finite)
    n_undefined = n_total - n

    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), float("nan"),
                      float("nan"), level, ci_method, 0, unit, n_undefined)

    rng = np.random.default_rng(seed)
    point = float(statistic(finite))
    sd = float(finite.std(ddof=1)) if n > 1 else 0.0
    sem = sd / float(np.sqrt(n)) if n > 1 else 0.0

    if clusters is not None:
        cluster_arr = np.asarray(list(clusters))[mask]
        groups: dict[Any, list[int]] = defaultdict(list)
        for i, c in enumerate(cluster_arr):
            groups[c].append(i)
        keys = list(groups.keys())
        idx_by_key = [np.asarray(groups[k]) for k in keys]
        n_groups = len(keys)
        draws = np.empty(n_resamples, dtype=float)
        for b in range(n_resamples):
            pick = rng.integers(0, n_groups, size=n_groups)
            sel = np.concatenate([idx_by_key[p] for p in pick])
            draws[b] = statistic(finite[sel])
        effective_n = n_groups
    else:
        draws = np.empty(n_resamples, dtype=float)
        for b in range(n_resamples):
            sel = rng.integers(0, n, size=n)
            draws[b] = statistic(finite[sel])
        effective_n = n

    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(draws, alpha))
    hi = float(np.quantile(draws, 1.0 - alpha))
    return MeanCI(point, sd, sem, lo, hi, level, ci_method, effective_n, unit, n_undefined)


# --------------------------------------------------------------------------- #
#  Episode records
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeRecord:
    """Everything one episode produced. The unit of storage and of pairing."""

    episode_id: str
    method: str
    fold_id: str
    species: str
    split: str
    n_shot: int
    n_query_per_class: int
    seed: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score_pos: np.ndarray
    logits: np.ndarray | None = None
    query_paths: list[str] = field(default_factory=list)
    adapt_seconds: float = 0.0

    def metrics(self) -> dict[str, float]:
        return episode_metrics(self.y_true, self.y_pred, self.y_score_pos)


# --------------------------------------------------------------------------- #
#  Aggregator
# --------------------------------------------------------------------------- #

class MetricAggregator:
    """Collects episode records and produces every aggregation the thesis needs."""

    def __init__(
        self,
        *,
        class_names: Sequence[str] = ("fresh", "rotten"),
        ece_bins: int = 15,
        bootstrap_n: int = 10_000,
        bootstrap_seed: int = 7,
        ci_level: float = 0.95,
    ) -> None:
        self.class_names = list(class_names)
        self.ece_bins = ece_bins
        self.bootstrap_n = bootstrap_n
        self.bootstrap_seed = bootstrap_seed
        self.ci_level = ci_level
        self.records: list[EpisodeRecord] = []
        self._episode_metrics: list[dict[str, float]] = []

    # ------------------------------------------------------------------ #
    def add(self, record: EpisodeRecord) -> None:
        self.records.append(record)
        self._episode_metrics.append(record.metrics())

    def extend(self, records: Iterable[EpisodeRecord]) -> None:
        for r in records:
            self.add(r)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def metric_names(self) -> list[str]:
        if not self._episode_metrics:
            return []
        return sorted(self._episode_metrics[0].keys())

    # ------------------------------------------------------------------ #
    def series(self, metric: str) -> np.ndarray:
        return np.asarray([m.get(metric, float("nan")) for m in self._episode_metrics], dtype=float)

    def episode_ids(self) -> list[str]:
        return [r.episode_id for r in self.records]

    def _equal_episode_sizes(self) -> bool:
        sizes = {len(r.y_true) for r in self.records}
        return len(sizes) <= 1

    # ------------------------------------------------------------------ #
    def episode_mean(self, *, metrics: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        """Mean over episodes with a t-interval over episodes."""
        names = list(metrics) if metrics else self.metric_names
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            if name in ("tn", "fp", "fn", "tp"):
                continue
            out[name] = mean_ci_t(
                self.series(name),
                unit="episode",
                ci_method="t_over_episodes",
                level=self.ci_level,
            ).to_dict()
        return out

    def episode_bootstrap(self, metric: str = "accuracy") -> dict[str, Any]:
        return bootstrap_ci(
            self.series(metric),
            unit="episode",
            ci_method="bootstrap_over_episodes",
            n_resamples=self.bootstrap_n,
            seed=self.bootstrap_seed,
            level=self.ci_level,
        ).to_dict()

    def image_bootstrap(self, metric: str = "accuracy") -> dict[str, Any]:
        """Genuine image-level bootstrap over *query images*.

        600 test episodes are drawn from roughly 700 images, so episodes overlap
        heavily and are not independent -- the episode-level t-interval is
        anticonservative. Here we resample distinct query images with
        replacement (carrying every prediction made on each) and recompute the
        metric, which answers "would this hold on a different sample of fruit?"
        rather than "on a different draw of episodes from these same images?".

        Defined only for per-query-mean metrics; returns NaN otherwise.
        """
        if metric not in LINEAR_METRICS or not self.records:
            return MeanCI(
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                self.ci_level, "bootstrap_over_images", 0, "image", 0,
            ).to_dict()

        # Flatten to per-query outcomes keyed by the query image itself.
        paths: list[str] = []
        correct: list[float] = []
        for rec in self.records:
            per_query = (rec.y_pred == rec.y_true).astype(float)
            if len(rec.query_paths) == len(per_query):
                paths.extend(rec.query_paths)
            else:  # pragma: no cover - paths unavailable; fall back to episode id
                paths.extend([f"{rec.episode_id}#{i}" for i in range(len(per_query))])
            correct.extend(per_query.tolist())

        values = np.asarray(correct, dtype=float)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(paths):
            groups[p].append(i)

        keys = list(groups)
        idx_by_key = [np.asarray(groups[k]) for k in keys]
        n_groups = len(keys)

        rng = np.random.default_rng(self.bootstrap_seed)
        n_resamples = min(self.bootstrap_n, 2000)
        draws = np.empty(n_resamples, dtype=float)
        for b in range(n_resamples):
            pick = rng.integers(0, n_groups, size=n_groups)
            sel = np.concatenate([idx_by_key[p] for p in pick])
            draws[b] = values[sel].mean()

        alpha = (1.0 - self.ci_level) / 2.0
        point = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return MeanCI(
            mean=point,
            sd=sd,
            sem=sd / float(np.sqrt(n_groups)) if n_groups else float("nan"),
            ci_lo=float(np.quantile(draws, alpha)),
            ci_hi=float(np.quantile(draws, 1.0 - alpha)),
            ci_level=self.ci_level,
            ci_method="bootstrap_over_images",
            n=n_groups,
            unit="image",
        ).to_dict()

    # ------------------------------------------------------------------ #
    def pooled(self) -> dict[str, Any]:
        """Metrics over every query prediction, pooled across episodes."""
        if not self.records:
            return {}
        y_true = np.concatenate([r.y_true for r in self.records])
        y_pred = np.concatenate([r.y_pred for r in self.records])
        y_score = np.concatenate([r.y_score_pos for r in self.records])

        result = pooled_metrics(
            y_true, y_pred, y_score,
            class_names=self.class_names,
            ece_bins=self.ece_bins,
            episode_sizes_equal=self._equal_episode_sizes(),
        )
        # The rank metrics also get an episode-mean sibling, because pooling
        # ranks across episodes mixes within-episode discrimination with
        # between-episode calibration. Report both; never one alone.
        result["roc_auc_episode_mean"] = float(np.nanmean(self.series("roc_auc")))
        result["pr_auc_episode_mean"] = float(np.nanmean(self.series("pr_auc")))
        result["roc_auc_note"] = (
            "roc_auc_pooled ranks all queries together, so scores from different "
            "support sets are compared directly. roc_auc_episode_mean averages "
            "within-episode AUC. They answer different questions."
        )
        return result

    # ------------------------------------------------------------------ #
    def by(self, key: str) -> dict[str, "MetricAggregator"]:
        """Split into sub-aggregators by 'species', 'fold_id', 'n_shot' or 'method'."""
        groups: dict[str, MetricAggregator] = {}
        for rec in self.records:
            value = str(getattr(rec, key))
            if value not in groups:
                groups[value] = MetricAggregator(
                    class_names=self.class_names,
                    ece_bins=self.ece_bins,
                    bootstrap_n=self.bootstrap_n,
                    bootstrap_seed=self.bootstrap_seed,
                    ci_level=self.ci_level,
                )
            groups[value].add(rec)
        return dict(sorted(groups.items()))

    def fold_mean(self, metric: str = "accuracy") -> dict[str, Any]:
        """Mean of per-fold means, with a t-interval over folds.

        This is the interval that supports the thesis claim. With LOSO on 5
        species, n=5 and the interval is wide -- which is the honest answer.
        """
        per_fold = self.by("fold_id")
        fold_values = [float(np.nanmean(agg.series(metric))) for agg in per_fold.values()]
        return mean_ci_t(
            fold_values, unit="fold", ci_method="t_over_folds", level=self.ci_level
        ).to_dict()

    # ------------------------------------------------------------------ #
    def summary(self, *, headline_metric: str = "accuracy") -> dict[str, Any]:
        """The full block written into ``metrics.json`` for one method."""
        out: dict[str, Any] = {
            "n_episodes": len(self.records),
            "episode_mean": self.episode_mean(),
            "pooled": self.pooled(),
            "episode_bootstrap": {headline_metric: self.episode_bootstrap(headline_metric)},
            "image_bootstrap": {headline_metric: self.image_bootstrap(headline_metric)},
        }
        if len({r.fold_id for r in self.records}) > 1:
            out["fold_mean"] = {headline_metric: self.fold_mean(headline_metric)}

        out["by_species"] = {
            name: {
                "n_episodes": len(agg),
                "episode_mean": agg.episode_mean(metrics=[headline_metric, "balanced_accuracy"]),
            }
            for name, agg in self.by("species").items()
        }
        out["by_fold"] = {
            name: {
                "n_episodes": len(agg),
                "episode_mean": agg.episode_mean(metrics=[headline_metric, "balanced_accuracy"]),
            }
            for name, agg in self.by("fold_id").items()
        }
        out["adapt_seconds"] = {
            "mean": float(np.mean([r.adapt_seconds for r in self.records])) if self.records else 0.0,
            "total": float(np.sum([r.adapt_seconds for r in self.records])) if self.records else 0.0,
        }
        return out
