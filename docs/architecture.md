# System architecture

Vector figure for the thesis: `docs/figures/architecture.pdf`
(regenerate with `python scripts/make_architecture_figure.py`).

```mermaid
flowchart TB
    subgraph DATA["1 · Data layer"]
        DS[("Dataset registry<br/>5 species × {fresh, rotten}<br/>6,978 images")]
        SPLIT["LOSO species splitter<br/>5 folds, seeded<br/>leakage assertions"]
        BANK["<b>Deterministic episode bank</b><br/>frozen (support, query) specs<br/>nested shots 1 ⊆ 3 ⊆ 5 ⊆ 10"]
        SHARE["Shared across EVERY method<br/>⇒ paired tests valid"]
        DS --> SPLIT --> BANK --> SHARE
    end

    subgraph ENC["2 · Encoder layer"]
        CNN["ResNet-18/50, EfficientNet<br/>fine-tuned, freeze depth k"]
        DINO["DINOv2 ViT-S/14<br/>frozen + feature cache"]
        CLIPV["CLIP ViT-B/16 image tower<br/>frozen + feature cache"]
        CLIPT["CLIP text tower<br/>'a photo of a {quality} {species}'"]
    end

    subgraph HEAD["3 · Adaptation layer — FewShotMethod"]
        ZS["<b>zero-shot supervised (K=0)</b><br/>THE PIVOTAL CONTROL"]
        TRANSFER["nearest centroid / linear probe<br/>/ support fine-tune"]
        EPISODIC["Siamese / Matching<br/>ProtoNet ± temperature"]
        SAP["<b>SAP (proposed)</b><br/>α·visual + (1−α)·text"]
    end

    subgraph REG["Train-time regularisers"]
        SUPCON["supervised contrastive"]
        GRL["species-adversarial GRL"]
    end

    subgraph EVAL["4 · Evaluation layer"]
        MET["Metric suite<br/>Acc BalAcc P/R/F1 Sens Spec<br/>ROC-AUC PR-AUC κ MCC ECE"]
        CI["Intervals, 3 units<br/>over episodes (t)<br/>over images (cluster bootstrap)<br/>over folds (t, n=5)"]
        STAT["Paired statistics<br/>same episodes only<br/>Wilcoxon (Pratt ties)<br/>Holm–Bonferroni"]
        COST["Cost<br/>params / FLOPs / latency"]
    end

    subgraph OUT["5 · Serialized artifacts"]
        PQ[("<b>per_query.csv.gz</b><br/>SOURCE OF TRUTH")]
        JSON["metrics.json<br/>comparisons.json<br/>per_episode.csv"]
        PROV["config.yaml env.json<br/>status.json run.log<br/>checkpoints/"]
        FIG["figures/ tables/<br/>regenerated ONLY from artifacts"]
    end

    SHARE --> ENC
    CNN --> HEAD
    DINO --> HEAD
    CLIPV --> HEAD
    CLIPT -.-> SAP
    REG -.-> CNN
    HEAD --> EVAL
    MET --> CI --> STAT --> COST
    EVAL --> PQ --> JSON --> PROV --> FIG
    FIG -.->|"any new metric recomputed on CPU, no retraining"| PQ
```

---

## Component reference

### 1 · Data layer

| Module | Responsibility |
|---|---|
| `fsgrade/data/index.py` | Scan the dataset into a hashable manifest keyed by paths **relative to** `data_root`, so banks are portable across machines. Duplicate and corrupt-file detection. Seeded train/val split of seen species with disjointness assertions. |
| `fsgrade/data/episodes.py` | `FoldSpec` (LOSO / C(5,3) / fixed) and `EpisodeBank`. Episodes are frozen artifacts with a content hash. |
| `fsgrade/data/loaders.py` | Turn specs into tensors. Draws no random numbers, so it is safe with `num_workers > 0`. |

The episode bank is the keystone. Because every method replays byte-identical
episodes, paired statistics become valid and the same model can no longer report
three different numbers for the same test set.

### 2 · Encoder layer

| Module | Responsibility |
|---|---|
| `fsgrade/models/backbones.py` | torchvision backbones + projection head. Carries per-backbone normalisation constants (CLIP and DINOv2 do **not** use ImageNet statistics). |
| `fsgrade/models/freezing.py` | Stage-wise freeze by module identity, with the exact parameter ladder asserted in tests. Also forces frozen BatchNorm into eval mode. |
| `fsgrade/models/foundation.py` | Frozen DINOv2 / CLIP with an on-disk feature cache keyed by the manifest hash, plus the text-prototype builder for SAP. |

### 3 · Adaptation layer

| Module | Responsibility |
|---|---|
| `fsgrade/methods/base.py` | The `FewShotMethod` protocol and the **logits contract**: unbounded, shift-invariant scores. |
| `fsgrade/methods/trained.py` | Zero-shot supervised control, supervised centroid / fine-tune, episodic wrappers. The transfer family shares one checkpoint per fold. |
| `fsgrade/methods/frozen.py` | Cached-feature methods: nearest centroid, linear probe, zero-shot text, **SAP**. |
| `fsgrade/methods/registry.py` | The ordered ladder — single source of truth. |

### 4 · Evaluation layer

| Module | Responsibility |
|---|---|
| `fsgrade/evaluation/metrics.py` | Per-episode and pooled metrics; documents where the two coincide. |
| `fsgrade/evaluation/aggregate.py` | `MeanCI` — an interval cannot exist without `ci_method`, `n`, `unit`. Episode / image / fold aggregations. |
| `fsgrade/evaluation/stats.py` | Paired comparison on shared episodes, Holm–Bonferroni, bootstrap CIs, Cohen's *d*<sub>z</sub>. |
| `fsgrade/evaluation/runner.py` | The evaluation loop; streams predictions to disk as they are produced. |

### 5 · Serialization

`fsgrade/io/results.py` — run directories, atomic JSON writes, and a streaming
CSV writer that flushes periodically so a crashed run leaves partial results
rather than none.

---

## Execution flow

```
0  check_data      validate layout, capacity, duplicates      (CPU, seconds; exits non-zero)
1  build_banks     freeze episodes for every fold             (CPU, seconds)
2  cache_features  one GPU pass per foundation encoder        (optional)
3  evaluate        train + evaluate the ladder, write all     (the long step)
4  stats           paired tests on the shared bank            (folded into step 3)
5  figures/tables  regenerated ONLY from serialized artifacts (CPU)
```

`scripts/run_all.sh` runs these in dependency order; `--smoke` exercises every
code path in minutes on CPU.
