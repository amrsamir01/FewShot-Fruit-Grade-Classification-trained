# Results — Cross-Species Fruit Quality Grading

Produced by [`scripts/reproduce_thesis.py`](scripts/reproduce_thesis.py) over **3 seeds** (42, 1337, 2024), regenerated from `results/run_*/metrics.json` by `scripts/make_results_md.py`.

Dataset: **6,978 images**, manifest `sha256:8354959198214354…`. Trainable parameters: **11,414,017** of 11,571,521.

> **Read the interval carefully.** Across seeds the standard deviation of the headline is **0.94 points**, while the episode-level 95% interval within a single run is **±0.21**. Training variance dominates episode sampling by roughly 4×, so quote the cross-seed figure and treat any single-run ranking of methods as unreliable.

---

## Headline: accuracy on unseen species

Trained on apple/banana/grape, evaluated on mango/orange with no retraining.

| | Accuracy | Spread |
|---|---|---|
| **Overall, mean of 3 seeds** | **89.21%** | sd 0.94 (88.18 / 90.01 / 89.45) |
| Mango | 89.44% | sd 2.57 |
| Orange | 89.03% | sd 2.32 |

**Lead with the cross-validated mean below, not this number.** The fixed apple/banana/grape split is the most favourable of the ten possible species splits at every seed.

## Species-split cross-validation — C(5,3) = 10 folds

| Train | Test (unseen) | Accuracy, mean of 3 seeds |
|---|---|---|
| apple, banana, grape | mango, orange | 91.96% |
| apple, grape, mango | banana, orange | 85.59% |
| apple, banana, orange | grape, mango | 83.61% |
| apple, grape, orange | banana, mango | 82.12% |
| apple, mango, orange | banana, grape | 80.12% |
| banana, grape, mango | apple, orange | 78.64% |
| apple, banana, mango | grape, orange | 78.26% |
| banana, grape, orange | apple, mango | 77.79% |
| banana, mango, orange | apple, grape | 72.98% |
| grape, mango, orange | apple, banana | 72.56% |
| | **Mean over 10 folds** | **80.36%** (sd across folds 5.82) |

The apple/banana/grape → mango/orange split ranks **1 of 10** and scores 91.96%, against a 10-fold mean of 80.36%. Reporting only the fixed split overstates cross-species transfer by roughly 12 points.

## Baselines and transfer controls (two-sided paired tests)

| Method | Accuracy (mean of 3) | sd | Verdict vs ours |
|---|---|---|---|
| **Fine-tuned + Nearest Centroid** *(control)* | 94.34% | 0.49 | **ours WORSE** at all 3 seeds |
| **Matching Network** | 93.13% | 0.99 | **ours WORSE** at all 3 seeds |
| ProtoNet + Temp. Scaling | 91.61% | 2.68 | inconsistent (ours WORSE / ours WORSE / ns) |
| Siamese Network | 90.76% | 2.24 | inconsistent (ns / ours WORSE / ours WORSE) |
| Ours (Full Model) | 89.26% | 0.57 | — |
| Supervised transfer (zero-shot) *(control)* | 88.63% | 2.14 | inconsistent (ours better / ns / ns) |
| ProtoNet (standard) | 86.87% | 12.78 | inconsistent (ours WORSE / ours better / ours WORSE) |
| Nearest Centroid (pixels) | 69.23% | 0.48 | **ours better** at all 3 seeds |

**Baselines that beat this method.** *Fine-tuned + Nearest Centroid* — ordinary supervised fine-tuning with a nearest-centroid head and no episodic training — and the *Matching Network* outperform the proposed model at **every seed**, with two-sided Wilcoxon p-values below 1e-28 in every case. This is reported rather than omitted, and it is the expected outcome under the cross-domain few-shot literature (Chen et al. 2019, Baseline++; Wang et al. 2019, SimpleShot; Tian et al. 2020), where a good embedding with a simple classifier rivals meta-learning under domain shift.

*Supervised transfer (zero-shot)* is the control that asks whether few-shot adaptation is needed at all. It is **not reliably beaten**: the method leads at one seed and is statistically indistinguishable at the others.

*ProtoNet (standard)* is unstable across seeds and should not be quoted from a single run in either direction.

## Component ablation — inconclusive

| Variant | Δ vs Full Model, per seed (42, 1337, 2024) | Consistent? |
|---|---|---|
| - Contrastive Loss | -0.17 / +0.57 / -0.29 | **no** |
| - Temperature Scaling | +0.43 / -2.15 / -5.00 | **no** |
| - Frozen Layers | +3.13 / +3.22 / -2.59 | **no** |
| - Dropout | +0.35 / +1.51 / -2.12 | **no** |

**No component shows a consistent effect across seeds**, including frozen layers, which looked like a real gain at two seeds before reversing at the third. Report this ablation as inconclusive at this sample size rather than as evidence for or against any component.

## Backbone ablation

| Backbone | Accuracy (mean of 3) | sd |
|---|---|---|
| resnet50 | 94.79% | 0.92 |
| resnet18 | 89.16% | 0.85 |
| efficientnet_b0 | 86.37% | 2.86 |

## Single-species baselines

| Trained on | Accuracy (mean of 3) | sd |
|---|---|---|
| Banana-trained | 91.96% | 1.56 |
| Apple-trained | 90.17% | 1.20 |
| Multi-species (Ours) | 89.09% | 0.72 |
| Grape-trained | 71.45% | 1.93 |

Both EfficientNet-B0 and the banana-only baseline are reported above rather than omitted. Changing the backbone to ResNet-50 buys more accuracy than the method contributes, and single-species training on banana or apple beats multi-species training.

## N-shot ablation

| K (shots) | Accuracy (mean of 3) | sd |
|---|---|---|
| 1 | 86.80% | 1.65 |
| 3 | 88.87% | 1.15 |
| 5 | 89.12% | 1.05 |
| 10 | 89.13% | 0.94 |

## Variant configurations (one seed each)

| Configuration | Validation set | Best val acc | Unseen-species test |
|---|---|---|---|
| baseline (mean of 3) | held-out images, 3 species | 0.9306 | 89.21% |
| bnfrozen | held-out images, 3 species | 0.9890 | 85.65% |
| loso | grape episodes | 0.7603 | 87.44% |
| loso+bnfrozen | grape episodes | 0.7502 | 92.13% |

**No variant was shown to be better.** Two cautions belong with this table. First, configurations can only be ranked against each other when they share a validation set, so the LOSO rows and the seen-holdout rows are not comparable. Second, among the two LOSO rows the higher unseen-species score belongs to the configuration with the *lower* LOSO validation accuracy — selecting it would mean choosing on the held-out test species, which is exactly what the evaluation protocol exists to prevent. With one seed each and a validation gap far smaller than the seed-to-seed spread, the pre-registered baseline stands.

---

Every number above is a key in a `results/run_*/metrics.json`; `summary.csv` in each run directory is the flat index.
