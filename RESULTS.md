# Results — Cross-Species Fruit Quality Grading

Source: `src/`, driven by [`scripts/reproduce_thesis.py`](scripts/reproduce_thesis.py).
Open defects: [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md).

> ## ⚠ These numbers are pending regeneration
>
> They were produced before three corrections landed and **will move**:
>
> 1. **Layer freezing was over-broad.** A substring match froze the first
>    convolution and its norm in *every* residual block, not just the stem and
>    `layer1` — 4,648,448 parameters (40.7%) frozen beyond what the method
>    describes. Every number below comes from that over-frozen model.
> 2. **Significance was tested one-sided**, so baselines that beat the proposed
>    method were reported as "not significant" rather than "significantly better".
> 3. **Episode sampling used the global RNG**, so results depended on notebook
>    cell execution order — which is why the same model appears below at 85.0,
>    85.4, 85.6 and 85.7.
>
> The artifacts that produced them — `results/*.json`, `Imgs/*.png`,
> `notebooks/checkpoints/best_model.pth` — have been **deleted** ahead of a clean
> re-run, and the notebook's injected outputs stripped. The numbers below are
> retained only so the re-run can be compared against them. Recover the old
> artifacts if ever needed with:
> `git checkout c1d4301 -- results Imgs notebooks`
>
> Regenerate with `python scripts/reproduce_thesis.py --seed 42` on the GPU
> machine, then replace this file wholesale. Do not cite these figures in the
> thesis until then.

---

## Headline: cross-species generalization (unseen species)

Trained on **apple, banana, grape** → tested on **mango, orange**, no fine-tuning.

| | Accuracy | 95% CI |
|---|---|---|
| **Overall (unseen)** | **85.4%** | ±0.4% ⚠ |
| Mango | 80.8% | ±0.4% ⚠ |
| Orange | 90.1% | ±0.3% ⚠ |

> ⚠ **These intervals are not comparable with any other table on this page.**
> They are t-intervals over **5 trial means**; every other `±` here is over
> **600 episodes**. Averaging 600 episodes before taking the interval removes
> almost all the variance, making these the narrowest numbers in the document by
> construction. `test_on_unseen_fruits` now returns both `ci_95_episode` and
> `ci_95_trial` with explicit `n`; quote the **episode-level** one.

Chance is 50% (balanced binary task). Note that within-species fresh/rotten
classification is a solved problem at 94–98% in the literature — those figures
are **not comparable**, because they train and test on the same species.

## Baselines (unseen species, 600 episodes)

| Method | Accuracy | 95% CI | vs Ours |
|---|---|---|---|
| Nearest Centroid (raw pixels) | 69.6% | ±0.8 | ours better, p ≈ 2e-117 |
| ProtoNet (standard) | 76.1% | ±0.9 | ours better, p ≈ 5e-49 |
| ProtoNet + Temp. Scaling | 83.4% | ±1.1 | ours better, p ≈ 1e-03 |
| Ours (Full Model) | 85.6% | ±0.7 | — |
| **Siamese Network** | **88.0%** | ±0.5 | **ours WORSE**, t = −5.51, p ≈ 5e-08 |
| **Matching Network** | **88.2%** | ±0.8 | **ours WORSE**, t = −4.97, p ≈ 9e-07 |

**The proposed method does not win.** Siamese and Matching Networks beat it, and
the paired t-tests say the gap is significant. An earlier version of this table
reported those two rows as "not significant" because the Wilcoxon test was run
one-sided (`alternative="greater"`), which can only detect *ours better* and
returns p = 1.00 otherwise. That is fixed.

This is consistent with the thesis's stated conclusion: cross-species quality
transfer is a property of episodic metric learning on a pretrained backbone, not
of the specific refinements collected under "ProtoNet++". It also matches the
cross-domain few-shot literature (Chen et al. 2019; Tian et al. 2020), where
elaborate meta-learners routinely fail to beat simple ones under domain shift.

Caveat that cuts the other way: baselines were trained on a **worse budget**
(20 epochs, single LR, no freezing vs 30 epochs, discriminative LRs, frozen
layers — `src/experiments.py:136-139`). They were handicapped and still won.

## Species-split cross-validation — C(5,3) = 10 folds

| # | Train | Test (unseen) | Accuracy |
|---|---|---|---|
| 1 | apple, banana, grape | mango, orange | 85.3% ± 0.6% |
| 2 | apple, banana, mango | grape, orange | 76.3% ± 0.9% |
| 3 | apple, banana, orange | grape, mango | 79.7% ± 0.7% |
| 4 | apple, grape, mango | banana, orange | 87.2% ± 0.6% |
| 5 | apple, grape, orange | banana, mango | 75.7% ± 0.7% |
| 6 | apple, mango, orange | banana, grape | 82.6% ± 0.8% |
| 7 | banana, grape, mango | apple, orange | 77.9% ± 1.2% |
| 8 | banana, grape, orange | apple, mango | 79.5% ± 0.7% |
| 9 | banana, mango, orange | apple, grape | 74.3% ± 0.8% |
| 10 | grape, mango, orange | apple, banana | 80.6% ± 0.8% |
| | | **Mean ± std** | **79.9% ± 4.0%** |

**This is the most defensible result in the project and should lead the results
chapter.** Every fold clears chance by a wide margin, which answers the "you
picked an easy split" objection with evidence.

Note that the headline split (fold 1, 85.3%) is the **2nd best of 10**. Quoting
it as *the* result overstates the method; quoting the CV mean does not.

## Component ablation

| Variant | Accuracy | 95% CI | Δ vs Full |
|---|---|---|---|
| Full Model | 85.7% | ±0.7 | — |
| − Contrastive Loss | 86.5% | ±0.6 | **+0.7** |
| − Temperature Scaling | 80.5% | ±0.6 | −5.3 |
| − Frozen Layers | 84.0% | ±0.9 | −1.7 |
| − Dropout | **89.1%** | ±0.5 | **+3.4** |

Temperature scaling is the only component that clearly earns its place.
**Removing dropout gains 3.4 points and removing the contrastive loss gains 0.7**
— two of the four components of the "full model" hurt, and the best
configuration in this table is not the one being proposed.

The `− Frozen Layers` row must be re-run and reinterpreted: under the old
substring freezing it compared *no freezing* against *40.7% of the backbone
frozen*, not against the stem-and-`layer1` freezing the method describes.

## N-shot ablation (unseen species)

| K (shots) | Accuracy | 95% CI |
|---|---|---|
| 1 | 83.5% | ±0.7 |
| 3 | 85.4% | ±1.4 |
| 5 | 85.6% | ±0.7 |
| 10 | 85.7% | ±0.4 |

**+2.2 points for 10× the support data.** Worth stating directly: the practical
claim ("ten labelled photographs and no retraining") is well supported, because
performance is nearly flat in K — most of the capability comes from the
pretrained backbone and episodic training, not from the support examples.

## Backbone ablation

Present in the notebook, previously omitted from this file.

| Backbone | Accuracy | 95% CI | Params |
|---|---|---|---|
| ResNet-18 (used) | 85.7% | ±0.6 | 6.8M |
| ResNet-50 | 86.8% | ±0.8 | 20.2M |
| **EfficientNet-B0** | **92.3%** | ±0.4 | **4.8M** |

**EfficientNet-B0 is both smaller and 6.6 points better than the backbone the
thesis uses.** This must appear in the thesis. The defensible position is that
the research question concerns whether episodic metric learning transfers
quality concepts across species, not which backbone maximises accuracy — and
that ResNet-18 was fixed in advance for comparability. State it; do not omit it.

## Single-species training baselines

Present in the notebook, previously omitted from this file.

| Training set | Accuracy on unseen |
|---|---|
| Apple only | 87.6% |
| **Banana only** | **90.9%** |
| Grape only | 78.1% |
| Multi-species (apple+banana+grape) | 85.0% |

**Training on banana alone beats multi-species training by 5.9 points.** This
undercuts the assumption that species diversity in training drives transfer, and
it is a finding worth discussing rather than an embarrassment: it suggests what
transfers is a *spoilage appearance prior*, which some species teach better than
others, rather than a diversity-induced invariance.

## Other reported quantities

- Best validation accuracy 0.943 at epoch 15; early stop at epoch 22.
- Learned temperature ≈ 0.507.
- Embedding space: silhouette 0.480, 5-NN purity 0.965.
- Classification report (unseen): Fresh P/R/F1 = 0.90/0.79/0.84;
  Rotten = 0.81/0.91/0.86; accuracy 0.85.
- **Mango fresh recall is 0.645** — the model rejects over a third of acceptable
  mangoes. Raise this before an examiner does; per-species threshold calibration
  is the mitigation.

## Known gaps

- **Single seed (42) throughout.** No `±` here reflects training variance. Run
  `--seed 42 1337 2024` then `--aggregate` for seed-level spread.
- **No fine-tuning control.** Fine-tune on the seen species, then
  nearest-centroid on the support set. The CD-FSL literature suggests it may beat
  everything in this document; it is the most exposed omission.
- **Figure/text divergence.** `Imgs/confusion_matrices.png` totals 86.2% while
  the classification report from the same cell says 85% — different runs. Both
  now come from one call to `plot_confusion_matrices`.

## Files

- [`scripts/reproduce_thesis.py`](scripts/reproduce_thesis.py) — the sole producer of these numbers
- `results/run_<ts>_seed<N>/{metrics.json,env.json,summary.csv,figures/}`
- `notebooks/main_experiment.ipynb` — outputs were injected, not executed
  (duplicate `execution_count` values); superseded by the driver script
- `notebooks/checkpoints/best_model.pth` — trained under `legacy_substring`
  freezing; load with `PrototypicalNetwork(freeze_mode="legacy_substring")`
