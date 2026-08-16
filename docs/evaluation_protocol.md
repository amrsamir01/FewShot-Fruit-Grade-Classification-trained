# Evaluation protocol

What is measured, how it is aggregated, and how significance is established.
Implemented in `fsgrade/evaluation/`.

---

## 1. Metrics

Positive class = **rotten** (index 1). Chance = 50 % (2-way).

| Group | Metrics |
|---|---|
| Core | accuracy, balanced accuracy, precision / recall / F1 per class and macro, sensitivity, specificity |
| Threshold-free | ROC-AUC, PR-AUC |
| Agreement | Cohen's κ, Matthews correlation coefficient |
| Confusion | per-species and pooled matrices, raw and normalised |
| Calibration | Brier, NLL, ECE, MCE, reliability curve, support-set temperature scaling |
| Cost | params (total / trainable / frozen), train wall-clock, adaptation seconds, inference latency |

### Undefined ≠ zero

Cohen's κ and MCC have a vanishing denominator when an episode's predictions (or
truths) are constant. `sklearn` returns `0.0` with a warning, which reads as "no
agreement" rather than "not measurable". We return `NaN`, aggregate with
`nanmean`, and report `n_undefined` alongside every mean. The count is itself a
diagnostic: many degenerate episodes means a method is collapsing to one class.

---

## 2. Two aggregations, and when they coincide

**Episode mean** — compute within an episode, then average over episodes.
**Pooled** — compute once over every query prediction from every episode.

With a **balanced, equal-size** query set (the default 15 / 15), the two are
*numerically identical* for any metric that is a plain per-query mean. Reporting
both as separate findings would be an error:

| Metric | Pooled vs episode-mean | Why |
|---|---|---|
| accuracy | **identical** | mean of per-query 0/1, equal episode sizes |
| balanced accuracy | **identical, and equal to accuracy** | (TPR+TNR)/2 = (TP+TN)/30 under 15/15 |
| recall per class, sensitivity, specificity | **identical** | fixed denominator of 15 |
| macro recall | **identical to accuracy** | same algebra |
| Brier, NLL | **identical** | per-query means |
| precision, F1 | **differ** | denominator is the predicted count, which varies |
| ROC-AUC, PR-AUC | **differ** | rank statistic over a different comparison set |
| κ, MCC, ECE | **differ** | nonlinear in the confusion cells / score binning |

`metrics.json` carries a `pooled_equals_episode_mean` flag so no reader can
mistake one for the other.

**Headline choice.** Accuracy-family metrics are reported as the episode mean
(the FSL convention). ROC-AUC, κ, MCC and ECE are reported **pooled**, because
with 30 query points a per-episode AUC takes few distinct values and per-episode
κ/MCC is frequently undefined.

**Pooled ROC-AUC carries a caveat.** Scores from different episodes come from
different support sets and are not on a common scale, so pooling ranks mixes
within-episode discrimination with between-episode calibration. Both
`roc_auc_pooled` and `roc_auc_episode_mean` are emitted; report both, never one
alone.

### A note on the balanced query set

Because queries are balanced, accuracy ≡ balanced accuracy ≡ macro recall
identically, and the richer metrics add little. To make them informative, the
episode builder supports `query_prevalence` (e.g. 27 fresh / 3 rotten). Under
skew, balanced accuracy, MCC, PR-AUC and calibration separate sharply from
accuracy — and the skewed setting is the realistic deployment condition, since
rot is rare on a sorting line.

---

## 3. Uncertainty: three intervals, three questions

Every interval is a `MeanCI` carrying `ci_method`, `n` and `unit`. It is
**impossible to construct one without declaring them**, and the table renderer
prints the unit in the column header — so two differently-scoped intervals can
never appear under the same heading.

| Interval | `ci_method` | Answers |
|---|---|---|
| over episodes | `t_over_episodes` | episode-sampler noise, *conditional on this image pool* |
| over images | `bootstrap_over_images` | would this hold on a different sample of fruit? |
| over folds | `t_over_folds` (n = 5) | would this hold on a different unseen *species*? |

**The episode-level interval is anticonservative.** 600 test episodes are drawn
from roughly 700 images, so episodes share images heavily and are not
independent. The image-level bootstrap resamples *distinct query images*
(carrying every prediction made on each) and recomputes the metric.

**The fold-level interval is the one that supports the thesis claim.** With LOSO
over 5 species, n = 5 and the interval is wide. That width is the honest answer
to "does this generalise to an unseen species?", and reporting it is the
difference between a defensible thesis and a fragile one.

> This is the defect that most needs stating in the thesis: the pre-refactor
> headline `86.3 % ± 0.4 %` was a CI over **5 trial means**, while every other
> table's `± CI` was over **600 episodes**. They were printed side by side as
> the same quantity. They are not comparable.

---

## 4. Significance testing

### Pairing is by construction

All methods are evaluated on one shared episode bank. `paired_compare` aligns
two result series on the **intersection of their episode ids**, and raises
`EpisodeAlignmentError` if they do not overlap, or if coverage is partial
without an explicit opt-in. `episode_ids_hash` is recorded on every comparison
so a reviewer can verify the pairing was genuine.

> The pre-refactor code did `a[:min_len]` vs `b[:min_len]` on two
> *independently resampled* episode lists, pairing episode 17 of one run with an
> unrelated episode 17 of another. Paired tests on unpaired data are invalid;
> the resulting p-values (down to 2.68e-125) cannot be interpreted.

### Tests

- **Primary:** 95 % paired bootstrap CI of the mean difference (10 000
  resamples). A CI on Δaccuracy that excludes 0 is a better thesis sentence than
  a p-value, and it is robust to ties.
- **Corroborating:** paired *t*-test and Wilcoxon signed-rank.
- **Ties.** Episode accuracy over 30 queries takes only 31 distinct values, so
  paired differences contain many exact zeros — in a representative check, **300
  of 600 pairs**. SciPy's default `zero_method="wilcox"` discards zeros and
  inflates significance; we use `"pratt"` and record `wilcoxon_zero_pairs`.
- **Effect size.** Cohen's *d*<sub>z</sub> = mean(diff) / sd(diff) — the correct
  paired form — plus Hedges' *g* small-sample correction.
- **Multiplicity.** Holm–Bonferroni within an **explicitly declared family**
  (`stats.family` in config). The family and its size are serialized on every
  comparison row; they are never inferred.

---

## 5. Artifacts

Every run writes:

| File | Contents |
|---|---|
| `per_query.csv.gz` | **source of truth** — one row per query prediction, with logits, score and image path |
| `per_episode.csv` | per-episode metrics |
| `metrics.json` | aggregated metrics, all intervals, freeze report, parameter counts |
| `comparisons.json` | paired tests, effect sizes, Holm-adjusted p-values |
| `config.yaml`, `config_hash.json` | fully resolved config including the resolved data root |
| `env.json` | versions, GPU, CUDA/cuDNN, git commit + dirty-diff hash, determinism report |
| `status.json` | `running` / `ok` / `failed` with traceback |
| `run.log` | complete log |
| `checkpoints/<fold>/<method>/<spec_hash>.pt` | keyed weights (never a single overwritten file) |

Everything in `metrics.json` is recomputable from `per_query.csv.gz` on a CPU in
seconds. A new metric requested at the viva costs one command, not a retraining
run — and a crashed run still leaves every completed episode on disk, which is
precisely what did not happen when three of the original eight experiments were
lost to a Jupyter output-rate limit.
