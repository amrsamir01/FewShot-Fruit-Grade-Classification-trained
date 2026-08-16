# Results — Cross-Species Fruit Quality Grading
Auto-generated summary of the trained model and all experiments. The full notebook with inline outputs is [`notebooks/main_experiment.ipynb`](notebooks/main_experiment.ipynb).
## Headline: cross-species generalization (unseen fruits)
Trained on **apple, banana, grape** → tested on **mango, orange** with no fine-tuning.
| | Accuracy (95% CI) |
|---|---|
| **Overall (unseen)** | **85.4% ± 0.4%** |
| Mango | 80.8% ± 0.4% |
| Orange | 90.1% ± 0.3% |

## Baselines (unseen fruits)
| Method | Accuracy | 95% CI |
|---|---|---|
| Nearest Centroid (pixels) | 69.6% | ±0.8 |
| ProtoNet (standard) | 76.1% | ±0.9 |
| ProtoNet + Temp. Scaling | 83.4% | ±1.1 |
| **Ours (Full Model)** | **85.6%** | ±0.7 |
| Siamese Network | 88.0% | ±0.5 |
| Matching Network | 88.2% | ±0.8 |

## N-shot ablation (unseen fruits)
| K (shots) | Accuracy | 95% CI |
|---|---|---|
| 1 | 83.5% | ±0.7 |
| 3 | 85.4% | ±1.4 |
| 5 | 85.6% | ±0.7 |
| 10 | 85.7% | ±0.4 |

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

## Component ablation (Table 6)
| Variant | Accuracy | 95% CI | Δ vs Full |
|---|---|---|---|
| **Full Model** | 85.7% | ±0.7 | — |
| - Contrastive Loss | 86.5% | ±0.6 | +0.7 |
| - Temperature Scaling | 80.5% | ±0.6 | -5.3 |
| - Frozen Layers | 84.0% | ±0.9 | -1.7 |
| - Dropout | 89.1% | ±0.5 | +3.4 |

**Takeaways:** temperature scaling is the most important component (−5.3 without it); dropout at 0.4 hurts (−dropout is best, +3.4); contrastive loss is within noise (+0.7).

## Files
- `notebooks/main_experiment.ipynb` — full notebook with inline outputs
- `results/cv_results.json`, `results/ablation_results.json`, `results/final_combined_results.json`, `results/training_history.json`
- `notebooks/checkpoints/best_model.pth` — trained weights (val acc 0.943)
