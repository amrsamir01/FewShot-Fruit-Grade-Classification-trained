"""
All experiment routines for the thesis.

  test_on_unseen_fruits          – multi-trial evaluation on held-out species
  ablation_n_shot                – vary K and report accuracy ± CI
  run_all_fsl_baselines          – train each baseline FSL method and compare
  backbone_ablation              – compare ResNet18 / ResNet50 / EfficientNet-B0
  single_species_baseline        – train on one species → test on unseen
  run_single_species_baselines   – loop over all training species
  species_split_cross_validation – every 3-train / 2-test permutation
  statistical_significance_tests – Wilcoxon + paired t-test
  component_ablation             – ablate contrastive, temperature, freeze, dropout
  collect_dataset_statistics     – print + LaTeX table of image counts
  generate_thesis_summary        – full text summary of all results
  generate_latex_tables          – publication-ready LaTeX tables
"""

import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from src.dataset import FruitQualityDataset, EpisodicDataLoader
from src.models import (
    PrototypicalNetwork,
    SiameseNetwork,
    MatchingNetwork,
    StandardProtoNet,
    ProtoNetWithTemp,
    _build_backbone,
)
from src.losses import PrototypicalLoss
from src.train import train_epoch, evaluate


# ====================================================================== #
#  CI helper
# ====================================================================== #

def _ci95(accuracies):
    """Return the half-width (margin of error) of the 95% CI using the t-distribution.

    Uses the t-distribution with (n-1) degrees of freedom and sample standard
    deviation (ddof=1), which is correct for any sample size — including small
    numbers of trials (e.g. n=3 or n=5) where the Gaussian z=1.96 approximation
    is inaccurate.
    """
    n = len(accuracies)
    if n <= 1:
        return 0.0
    return sp_stats.t.ppf(0.975, df=n - 1) * np.std(accuracies, ddof=1) / np.sqrt(n)


# ====================================================================== #
#  Core evaluation helpers
# ====================================================================== #

def test_on_unseen_fruits(model, test_dataset, device, config, n_trials=5):
    """
    Test model on unseen fruits, reporting BOTH interval types explicitly.

    Two different quantities were previously both written as "+/-":

      ci_95_trial   – t-interval over `n_trials` trial means (n = 5).
                      Narrow, because averaging 600 episodes first removes
                      almost all of the variance.
      ci_95_episode – t-interval over the pooled per-episode accuracies
                      (n = n_trials * N_EPISODES_TEST). This is the quantity
                      every baseline and ablation table reports, and it is the
                      one the thesis must quote so the numbers are comparable.

    Both are returned with an explicit `n` so a reader can tell them apart.
    """
    print("Testing on unseen fruits...")
    all_results = {"overall": {"accuracies": [], "episode_accs": []}, "per_fruit": {}}
    for fruit in config.TEST_FRUITS:
        all_results["per_fruit"][fruit] = {"accuracies": [], "episode_accs": []}

    for trial in range(n_trials):
        test_loader = EpisodicDataLoader(
            test_dataset, config.N_SHOT, config.N_QUERY,
            config.N_EPISODES_TEST, fruits=config.TEST_FRUITS,
        )
        mean_acc, per_fruit_acc, all_accs, fruit_accs = evaluate(
            model, test_loader, device, f"Trial {trial+1}/{n_trials}", return_all=True,
        )
        all_results["overall"]["accuracies"].append(mean_acc)
        all_results["overall"]["episode_accs"].extend(list(all_accs))
        for fruit in config.TEST_FRUITS:
            if fruit in per_fruit_acc:
                all_results["per_fruit"][fruit]["accuracies"].append(per_fruit_acc[fruit])
            if fruit in fruit_accs:
                all_results["per_fruit"][fruit]["episode_accs"].extend(list(fruit_accs[fruit]))

    def _summarise(block):
        trial_accs = block["accuracies"]
        episode_accs = block["episode_accs"]
        block["mean"] = float(np.mean(trial_accs))
        block["std"] = float(np.std(trial_accs, ddof=1)) if len(trial_accs) > 1 else 0.0
        block["ci_95_trial"] = float(_ci95(trial_accs))
        block["n_trials"] = len(trial_accs)
        block["ci_95_episode"] = float(_ci95(episode_accs)) if episode_accs else 0.0
        block["n_episodes"] = len(episode_accs)
        # Back-compat alias. Points at the EPISODE-level interval so that any
        # caller still reading `ci_95` gets the quantity comparable with the
        # baseline tables, not the artificially narrow trial-level one.
        block["ci_95"] = block["ci_95_episode"]

    _summarise(all_results["overall"])
    for fruit in config.TEST_FRUITS:
        _summarise(all_results["per_fruit"][fruit])

    o = all_results["overall"]
    print(f"\nOverall: {o['mean']*100:.1f}%"
          f"  +/- {o['ci_95_episode']*100:.1f}% (episode-level, n={o['n_episodes']})"
          f"  +/- {o['ci_95_trial']*100:.1f}% (trial-level, n={o['n_trials']})")
    return all_results


def ablation_n_shot(model, test_dataset, device, config, shots=None, n_episodes=None, n_trials=3):
    """Run N-shot ablation study."""
    shots = shots if shots is not None else config.ABLATION_SHOTS
    n_episodes = n_episodes if n_episodes is not None else config.ABLATION_EPISODES

    print("Running N-shot ablation study...")
    results = {}
    for k in shots:
        print(f"\n  Testing {k}-shot...")
        accuracies = []
        for trial in range(n_trials):
            loader = EpisodicDataLoader(test_dataset, k, config.N_QUERY, n_episodes)
            acc, _ = evaluate(model, loader, device, f"{k}-shot trial {trial+1}")
            accuracies.append(acc)
        results[k] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies, ddof=1),
            "ci_95": _ci95(accuracies),
        }
        print(f"    {k}-shot: {results[k]['mean']*100:.1f}% +/- {results[k]['ci_95']*100:.1f}%")
    return results


# ====================================================================== #
#  Baseline training
# ====================================================================== #

def train_baseline_model(model_class, model_name, train_dataset, val_dataset,
                         test_dataset, config, device, epochs=20):
    """Train a baseline FSL model and return episode-level test accuracies."""
    print(f"\n{'='*60}")
    print(f"  Training baseline: {model_name}")
    print(f"{'='*60}")

    baseline_model = model_class(
        backbone="resnet18", embedding_dim=config.EMBEDDING_DIM, pretrained=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        baseline_model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    best_val_acc, patience_counter = 0.0, 0
    best_state = None

    for epoch in range(1, epochs + 1):
        baseline_model.train()
        train_accs = []
        loader = EpisodicDataLoader(
            train_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_TRAIN,
        )
        for support_imgs, support_lbls, query_imgs, query_lbls, fruit in loader:
            support_imgs = support_imgs.to(device)
            support_lbls = support_lbls.to(device)
            query_imgs = query_imgs.to(device)
            query_lbls = query_lbls.to(device)

            logits, _, _, _ = baseline_model(support_imgs, support_lbls, query_imgs)
            loss = ce_loss_fn(logits, query_lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            train_accs.append((logits.argmax(1) == query_lbls).float().mean().item())

        val_loader = EpisodicDataLoader(
            val_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_VAL,
        )
        val_acc, _ = evaluate(baseline_model, val_loader, device, desc=f"{model_name} Val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in baseline_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train {np.mean(train_accs):.3f} | "
                  f"Val {val_acc:.3f} | Best {best_val_acc:.3f}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    baseline_model.load_state_dict(best_state)
    test_loader = EpisodicDataLoader(
        test_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_TEST,
    )
    mean_acc, per_fruit, all_accs, fruit_accs = evaluate(
        baseline_model, test_loader, device, f"{model_name} Test", return_all=True,
    )

    print(f"\n  {model_name} Test Accuracy: {mean_acc*100:.1f}%")
    for fruit, acc in per_fruit.items():
        print(f"    {fruit}: {acc*100:.1f}%")

    return {
        "mean": mean_acc,
        "std": np.std(all_accs, ddof=1),
        "ci_95": _ci95(all_accs),
        "episode_accs": all_accs,
        "per_fruit": per_fruit,
    }


def run_all_fsl_baselines(our_model, train_dataset, val_dataset, test_dataset,
                          config, device):
    """Train and evaluate all baseline FSL methods. Returns per-episode accuracies."""
    baseline_specs = [
        (SiameseNetwork, "Siamese Network"),
        (MatchingNetwork, "Matching Network"),
        (StandardProtoNet, "ProtoNet (standard)"),
        (ProtoNetWithTemp, "ProtoNet + Temp. Scaling"),
    ]
    results = {}

    # Nearest Centroid baseline (no training needed)
    print("Computing Nearest Centroid (pixel) baseline...")
    nc_accs = []
    for _ in tqdm(range(config.N_EPISODES_TEST), desc="NC Baseline", leave=False, mininterval=30, miniters=50):
        s_imgs, s_lbls, q_imgs, q_lbls, _ = test_dataset.get_episode(config.N_SHOT, config.N_QUERY)
        s_flat = s_imgs.view(s_imgs.size(0), -1)
        q_flat = q_imgs.view(q_imgs.size(0), -1)
        centroids = []
        for c in range(2):
            centroids.append(s_flat[s_lbls == c].mean(dim=0))
        centroids = torch.stack(centroids)
        dists = torch.cdist(q_flat, centroids)
        preds = dists.argmin(dim=1)
        nc_accs.append((preds == q_lbls).float().mean().item())

    results["Nearest Centroid (pixels)"] = {
        "mean": np.mean(nc_accs),
        "std": np.std(nc_accs, ddof=1),
        "ci_95": _ci95(nc_accs),
        "episode_accs": nc_accs,
    }
    print(f"  Nearest Centroid: {np.mean(nc_accs)*100:.1f}%")

    for model_class, model_name in baseline_specs:
        results[model_name] = train_baseline_model(
            model_class, model_name, train_dataset, val_dataset,
            test_dataset, config, device, epochs=20,
        )

    # Our method (already trained)
    print("\nEvaluating our full model...")
    test_loader = EpisodicDataLoader(
        test_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_TEST,
    )
    our_mean, our_per_fruit, our_all_accs, _ = evaluate(
        our_model, test_loader, device, "Ours Test", return_all=True,
    )
    results["Ours (Full Model)"] = {
        "mean": our_mean,
        "std": np.std(our_all_accs, ddof=1),
        "ci_95": _ci95(our_all_accs),
        "episode_accs": our_all_accs,
        "per_fruit": our_per_fruit,
    }

    print(f"\n{'='*65}")
    print(f"  {'Method':<30} {'Accuracy':>10} {'95% CI':>10}")
    print(f"  {'-'*55}")
    for name, data in results.items():
        marker = " ***" if "Ours" in name else ""
        print(f"  {name:<30} {data['mean']*100:>9.1f}% {data['ci_95']*100:>9.1f}%{marker}")
    print(f"{'='*65}")
    return results


# ====================================================================== #
#  Transfer controls: does episodic training buy anything?
# ====================================================================== #

class _FlatImageDataset(TorchDataset):
    """Flat (image, quality_label) view over a FruitQualityDataset's file lists."""

    def __init__(self, fruit_dataset, transform):
        self.transform = transform
        self.samples = []
        for fruit in fruit_dataset.fruit_types:
            for class_idx, quality in enumerate(fruit_dataset.classes):
                for path in fruit_dataset.data[fruit][quality]:
                    self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class _SupervisedBackbone(nn.Module):
    """ResNet-18 + linear fresh/rotten head. Trained non-episodically."""

    def __init__(self, backbone="resnet18", pretrained=True, n_classes=2):
        super().__init__()
        self.encoder, in_features = _build_backbone(backbone, pretrained)
        self.classifier = nn.Linear(in_features, n_classes)

    def features(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def transfer_controls(train_dataset, test_dataset, transforms_dict, config, device,
                      epochs=15, batch_size=32, lr=1e-4):
    """
    The two controls the thesis most needs, and currently lacks.

    Both start from ONE conventionally trained model: ResNet-18 with a linear
    fresh/rotten head, trained on the seen species with ordinary mini-batches
    and no episodes at all.

      "Fine-tuned + Nearest Centroid" — discard the head, embed the support set,
          classify queries by nearest class centroid. This is the Baseline++ /
          SimpleShot control (Chen et al. 2019; Tian et al. 2020; Wang et al.
          2019). The cross-domain few-shot literature repeatedly finds it matches
          or beats meta-learning under domain shift. If it wins here, that is a
          real finding and should be reported as one.

      "Supervised transfer (zero-shot)" — apply the trained binary classifier
          directly to unseen species, using no support set whatsoever. This is
          the control that decides whether the few-shot framing is *necessary*.
          If a plain classifier transfers across the species boundary on its own,
          then episodes buy nothing and the thesis must say so.

    Both are evaluated on the same episode stream as every other baseline, so
    their per-episode accuracies are paired and can go into the significance
    tests unchanged.
    """
    print(f"\n{'='*60}")
    print("  Transfer controls (non-episodic supervised training)")
    print(f"{'='*60}")

    flat_train = _FlatImageDataset(train_dataset, transforms_dict["train"])
    loader = DataLoader(flat_train, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    print(f"  {len(flat_train)} training images from {train_dataset.fruit_types}")

    model = _SupervisedBackbone(config.BACKBONE, config.PRETRAINED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        running = 0.0
        for imgs, labels in tqdm(loader, desc=f"Supervised epoch {epoch}/{epochs}",
                                 leave=False, mininterval=30):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss {running/total:.4f} | "
                  f"train acc {correct/total:.3f}")

    # ---- evaluate both controls on a shared episode stream ---------------- #
    model.eval()
    ncc_accs, zero_accs = [], []
    ncc_fruit, zero_fruit = defaultdict(list), defaultdict(list)

    for _ in tqdm(range(config.N_EPISODES_TEST), desc="Transfer controls",
                  leave=False, mininterval=30, miniters=50):
        s_imgs, s_lbls, q_imgs, q_lbls, fruit = test_dataset.get_episode(
            config.N_SHOT, config.N_QUERY)
        s_imgs, q_imgs = s_imgs.to(device), q_imgs.to(device)
        s_lbls_d, q_lbls_d = s_lbls.to(device), q_lbls.to(device)

        with torch.no_grad():
            s_feat = model.features(s_imgs)
            q_feat = model.features(q_imgs)
            centroids = torch.stack([
                s_feat[s_lbls_d == c].mean(0) for c in range(config.N_CLASSES)
            ])
            ncc_pred = torch.cdist(q_feat, centroids).argmin(1)
            zero_pred = model(q_imgs).argmax(1)

        ncc_acc = (ncc_pred == q_lbls_d).float().mean().item()
        zero_acc = (zero_pred == q_lbls_d).float().mean().item()
        ncc_accs.append(ncc_acc)
        zero_accs.append(zero_acc)
        ncc_fruit[fruit].append(ncc_acc)
        zero_fruit[fruit].append(zero_acc)

    def _pack(accs, per_fruit):
        return {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)),
            "ci_95": float(_ci95(accs)),
            "episode_accs": accs,
            "per_fruit": {f: float(np.mean(a)) for f, a in per_fruit.items()},
        }

    results = {
        "Fine-tuned + Nearest Centroid": _pack(ncc_accs, ncc_fruit),
        "Supervised transfer (zero-shot)": _pack(zero_accs, zero_fruit),
    }

    print(f"\n  {'Control':<34} {'Accuracy':>10} {'95% CI':>10}")
    print(f"  {'-'*56}")
    for name, data in results.items():
        print(f"  {name:<34} {data['mean']*100:>9.1f}% {data['ci_95']*100:>9.1f}%")
    print("\n  Read these against the proposed method. If either is competitive,")
    print("  the honest conclusion is that episodic training is not what carries")
    print("  cross-species transfer -- report that rather than omitting them.")
    return results


# ====================================================================== #
#  Backbone ablation
# ====================================================================== #

def backbone_ablation(train_dataset, val_dataset, test_dataset, config, device):
    """Train the full pipeline with different backbones and compare."""
    backbones = ["resnet18", "resnet50", "efficientnet_b0"]
    results = {}

    for backbone_name in backbones:
        print(f"\n{'='*60}")
        print(f"  Backbone Ablation: {backbone_name}")
        print(f"{'='*60}")

        abl_model = PrototypicalNetwork(
            backbone=backbone_name, embedding_dim=config.EMBEDDING_DIM,
            pretrained=True, dropout_rate=config.DROPOUT_RATE,
            temperature=config.TEMPERATURE,
        ).to(device)

        abl_criterion = PrototypicalLoss(
            label_smoothing=config.LABEL_SMOOTHING,
            contrastive_weight=config.CONTRASTIVE_WEIGHT,
            temperature=config.TEMPERATURE,
        )

        optimizer = torch.optim.AdamW([
            {"params": abl_model.encoder.encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": abl_model.encoder.projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": [abl_model.temperature], "lr": config.LEARNING_RATE * 0.5},
        ], weight_decay=config.WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        best_val_acc, patience_counter = 0.0, 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                abl_model,
                EpisodicDataLoader(train_dataset, config.N_SHOT, config.N_QUERY,
                                   config.N_EPISODES_TRAIN),
                abl_criterion, optimizer, device, config.GRADIENT_CLIP,
            )
            val_loader = EpisodicDataLoader(
                val_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_VAL,
            )
            val_acc, _ = evaluate(abl_model, val_loader, device, f"{backbone_name} Val")
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in abl_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                print(f"  Epoch {epoch} | Train {train_acc:.3f} | Val {val_acc:.3f}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        abl_model.load_state_dict(best_state)
        test_loader = EpisodicDataLoader(
            test_dataset, config.N_SHOT, config.N_QUERY, config.N_EPISODES_TEST,
        )
        mean_acc, per_fruit, all_accs, _ = evaluate(
            abl_model, test_loader, device, f"{backbone_name} Test", return_all=True,
        )

        trainable = sum(p.numel() for p in abl_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in abl_model.parameters())

        results[backbone_name] = {
            "mean": mean_acc, "std": np.std(all_accs, ddof=1),
            "ci_95": _ci95(all_accs),
            "per_fruit": per_fruit,
            "trainable_params": trainable, "total_params": total,
            "episode_accs": all_accs,
        }
        print(f"  {backbone_name}: {mean_acc*100:.1f}% +/- {results[backbone_name]['ci_95']*100:.1f}%")
        for fruit, acc in per_fruit.items():
            print(f"    {fruit}: {acc*100:.1f}%")

        del abl_model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  BACKBONE ABLATION SUMMARY")
    print(f"  {'Backbone':<20} {'Accuracy':>10} {'95% CI':>10} {'Params':>15}")
    print(f"  {'-'*60}")
    for bb, data in results.items():
        print(f"  {bb:<20} {data['mean']*100:>9.1f}% "
              f"{data['ci_95']*100:>9.1f}% {data['trainable_params']:>12,}")
    print(f"{'='*70}")
    return results


# ====================================================================== #
#  Single-species baselines
# ====================================================================== #

def single_species_baseline(single_fruit, test_dataset, transforms_dict,
                            config, device, epochs=20):
    """Train our full model on a single fruit species and test on unseen ones."""
    print(f"\n  Training on [{single_fruit}] only ...")

    tr_data = FruitQualityDataset(
        config.DATA_ROOT, [single_fruit],
        support_transform=transforms_dict["support"],
        query_transform=transforms_dict["train"],
        split="train", val_ratio=config.VAL_SPLIT_RATIO, seed=42,
    )
    vl_data = FruitQualityDataset(
        config.DATA_ROOT, [single_fruit],
        support_transform=transforms_dict["eval"],
        query_transform=transforms_dict["eval"],
        split="val", val_ratio=config.VAL_SPLIT_RATIO, seed=42,
    )

    ss_model = PrototypicalNetwork(
        backbone="resnet18", embedding_dim=config.EMBEDDING_DIM,
        pretrained=True, dropout_rate=config.DROPOUT_RATE,
        temperature=config.TEMPERATURE,
    ).to(device)

    ss_criterion = PrototypicalLoss(
        label_smoothing=config.LABEL_SMOOTHING,
        contrastive_weight=config.CONTRASTIVE_WEIGHT,
        temperature=config.TEMPERATURE,
    )
    optimizer = torch.optim.AdamW([
        {"params": ss_model.encoder.encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
        {"params": ss_model.encoder.projection.parameters(), "lr": config.LEARNING_RATE},
        {"params": [ss_model.temperature], "lr": config.LEARNING_RATE * 0.5},
    ], weight_decay=config.WEIGHT_DECAY)

    best_val, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        ss_model.train()
        loader = EpisodicDataLoader(tr_data, config.N_SHOT, config.N_QUERY,
                                    config.N_EPISODES_TRAIN)
        for s_imgs, s_lbls, q_imgs, q_lbls, _ in loader:
            s_imgs, s_lbls = s_imgs.to(device), s_lbls.to(device)
            q_imgs, q_lbls = q_imgs.to(device), q_lbls.to(device)
            logits, q_emb, s_emb, _ = ss_model(s_imgs, s_lbls, q_imgs)
            all_emb = torch.cat([s_emb, q_emb], dim=0)
            all_lbl = torch.cat([s_lbls, q_lbls], dim=0)
            loss, _, _ = ss_criterion(logits, q_lbls, all_emb, all_lbl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ss_model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

        val_loader = EpisodicDataLoader(vl_data, config.N_SHOT, config.N_QUERY,
                                        config.N_EPISODES_VAL)
        va, _ = evaluate(ss_model, val_loader, device, f"{single_fruit} Val")
        if va > best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in ss_model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= config.EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    ss_model.load_state_dict(best_state)
    te_loader = EpisodicDataLoader(test_dataset, config.N_SHOT, config.N_QUERY,
                                   config.N_EPISODES_TEST)
    m_acc, pf, all_accs, _ = evaluate(ss_model, te_loader, device,
                                       f"{single_fruit}->unseen Test", return_all=True)

    print(f"    {single_fruit}-trained -> unseen: {m_acc*100:.1f}% +/- "
          f"{_ci95(all_accs)*100:.1f}%")

    del ss_model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean": m_acc, "std": np.std(all_accs, ddof=1),
        "ci_95": _ci95(all_accs),
        "per_fruit": pf,
    }


def run_single_species_baselines(our_model, test_dataset, transforms_dict,
                                 config, device):
    """Run single-species baselines for all training species."""
    print(f"\n{'='*70}")
    print(f"  SINGLE-SPECIES BASELINES (Table 4)")
    print(f"{'='*70}")

    results = {}
    for fruit in config.TRAIN_FRUITS:
        results[f"{fruit.capitalize()}-trained"] = single_species_baseline(
            fruit, test_dataset, transforms_dict, config, device,
        )

    te_loader = EpisodicDataLoader(test_dataset, config.N_SHOT, config.N_QUERY,
                                   config.N_EPISODES_TEST)
    m_acc, pf, all_accs, _ = evaluate(our_model, te_loader, device,
                                       "Multi-species", return_all=True)
    results["Multi-species (Ours)"] = {
        "mean": m_acc, "std": np.std(all_accs, ddof=1),
        "ci_95": _ci95(all_accs),
        "per_fruit": pf,
    }

    print(f"\n{'='*70}")
    print(f"  {'Training Species':<25} {'Accuracy':>10} {'95% CI':>10}")
    print(f"  {'-'*50}")
    for name, data in results.items():
        marker = " ***" if "Ours" in name else ""
        print(f"  {name:<25} {data['mean']*100:>9.1f}% {data['ci_95']*100:>9.1f}%{marker}")
    print(f"{'='*70}")
    return results


# ====================================================================== #
#  Species-split cross-validation
# ====================================================================== #

def species_split_cross_validation(transforms_dict, config, device, epochs=20):
    """Train and evaluate on every possible 3-train / 2-test split."""
    all_fruits = sorted(set(config.TRAIN_FRUITS + config.TEST_FRUITS))
    n_train = len(config.TRAIN_FRUITS)
    splits = list(combinations(all_fruits, n_train))
    split_results = []

    for idx, train_fruits in enumerate(splits):
        test_fruits = [f for f in all_fruits if f not in train_fruits]
        print(f"\n{'='*60}")
        print(f"  Split {idx+1}/{len(splits)}: Train={list(train_fruits)}, Test={test_fruits}")
        print(f"{'='*60}")

        tr_data = FruitQualityDataset(
            config.DATA_ROOT, list(train_fruits),
            support_transform=transforms_dict["support"],
            query_transform=transforms_dict["train"],
            split="train", val_ratio=config.VAL_SPLIT_RATIO, seed=42,
        )
        vl_data = FruitQualityDataset(
            config.DATA_ROOT, list(train_fruits),
            support_transform=transforms_dict["eval"],
            query_transform=transforms_dict["eval"],
            split="val", val_ratio=config.VAL_SPLIT_RATIO, seed=42,
        )
        te_data = FruitQualityDataset(
            config.DATA_ROOT, test_fruits,
            support_transform=transforms_dict["eval"],
            query_transform=transforms_dict["eval"],
            split="all",
        )

        split_model = PrototypicalNetwork(
            backbone="resnet18", embedding_dim=config.EMBEDDING_DIM,
            pretrained=True, dropout_rate=config.DROPOUT_RATE,
            temperature=config.TEMPERATURE,
        ).to(device)

        split_criterion = PrototypicalLoss(
            label_smoothing=config.LABEL_SMOOTHING,
            contrastive_weight=config.CONTRASTIVE_WEIGHT,
            temperature=config.TEMPERATURE,
        )
        optimizer = torch.optim.AdamW([
            {"params": split_model.encoder.encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": split_model.encoder.projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": [split_model.temperature], "lr": config.LEARNING_RATE * 0.5},
        ], weight_decay=config.WEIGHT_DECAY)

        best_val, patience_ctr = 0.0, 0
        best_state = None
        for epoch in range(1, epochs + 1):
            tl, ta = train_epoch(
                split_model,
                EpisodicDataLoader(tr_data, config.N_SHOT, config.N_QUERY,
                                   config.N_EPISODES_TRAIN),
                split_criterion, optimizer, device, config.GRADIENT_CLIP,
            )
            va, _ = evaluate(
                split_model,
                EpisodicDataLoader(vl_data, config.N_SHOT, config.N_QUERY,
                                   config.N_EPISODES_VAL),
                device, "Val",
            )
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in split_model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        split_model.load_state_dict(best_state)
        te_loader = EpisodicDataLoader(te_data, config.N_SHOT, config.N_QUERY,
                                       config.N_EPISODES_TEST)
        m_acc, pf_acc, all_accs, _ = evaluate(
            split_model, te_loader, device, "Test", return_all=True,
        )

        split_results.append({
            "train_fruits": list(train_fruits),
            "test_fruits": test_fruits,
            "mean": m_acc,
            "std": np.std(all_accs, ddof=1),
            "ci_95": _ci95(all_accs),
            "per_fruit": pf_acc,
        })
        print(f"  Result: {m_acc*100:.1f}% +/- {split_results[-1]['ci_95']*100:.1f}%")

        del split_model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*75}")
    print(f"  SPECIES SPLIT CROSS-VALIDATION SUMMARY")
    print(f"  {'Train Species':<35} {'Test Species':<20} {'Acc':>8} {'CI':>8}")
    print(f"  {'-'*70}")
    for r in split_results:
        tr_str = ", ".join(f.capitalize() for f in r["train_fruits"])
        te_str = ", ".join(f.capitalize() for f in r["test_fruits"])
        print(f"  {tr_str:<35} {te_str:<20} {r['mean']*100:>7.1f}% {r['ci_95']*100:>7.1f}%")

    all_means = [r["mean"] for r in split_results]
    print(f"\n  Mean across splits: {np.mean(all_means)*100:.1f}% +/- {np.std(all_means)*100:.1f}%")
    print(f"  Min: {np.min(all_means)*100:.1f}%  Max: {np.max(all_means)*100:.1f}%")
    print(f"{'='*75}")
    return split_results


# ====================================================================== #
#  Statistical significance
# ====================================================================== #

def statistical_significance_tests(baseline_results):
    """Paired Wilcoxon + t-tests between our method and each baseline."""
    if "Ours (Full Model)" not in baseline_results:
        print("Run baseline evaluation first.")
        return None

    ours_accs = np.array(baseline_results["Ours (Full Model)"]["episode_accs"])
    n = len(ours_accs)

    print(f"{'='*75}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS  (n = {n} episodes)")
    print(f"{'='*75}")
    print(f"  {'Baseline':<30} {'delta':>8} {'t-stat':>8} {'p (t)':>10} "
          f"{'W-stat':>8} {'p (W)':>10} {'Outcome':>12}")
    print(f"  {'-'*90}")

    sig_table = []
    min_len = n

    for name, data in baseline_results.items():
        if name == "Ours (Full Model)":
            continue
        other_accs = np.array(data.get("episode_accs", []))
        if len(other_accs) == 0:
            continue

        min_len = min(n, len(other_accs))
        a = ours_accs[:min_len]
        b = other_accs[:min_len]

        t_stat, t_p = sp_stats.ttest_rel(a, b)

        # Two-sided. The previous alternative="greater" could only ever answer
        # "is ours better?", so every baseline that BEAT ours returned p = 1.00
        # and printed "No" — which reads as "no significant difference" when the
        # truth is "significantly worse". Direction is now reported explicitly
        # via `outcome` rather than inferred from an untestable one-sided p.
        try:
            w_stat, w_p = sp_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        delta = (np.mean(a) - np.mean(b)) * 100
        significant = t_p < 0.05 and w_p < 0.05
        if not significant:
            outcome = "ns"
        elif delta > 0:
            outcome = "ours better"
        else:
            outcome = "ours WORSE"

        print(f"  {name:<30} {delta:>+8.2f} {t_stat:>8.2f} {t_p:>10.2e} "
              f"{w_stat:>8.0f} {w_p:>10.2e} {outcome:>12}")

        sig_table.append({
            "baseline": name, "delta_acc": delta,
            "t_stat": t_stat, "t_p": t_p,
            "w_stat": w_stat, "w_p": w_p,
            "significant": bool(significant),
            "outcome": outcome,
        })

    print(f"{'='*75}")
    return sig_table


# ====================================================================== #
#  Component ablation
# ====================================================================== #

def train_ablation_variant(variant_name, train_dataset, val_dataset, test_dataset,
                           config, device, epochs=20,
                           use_contrastive=True, use_temperature=True,
                           freeze_early=True, dropout_rate=0.4):
    """Train a single ablated variant from scratch and evaluate on test set."""
    print(f"\n  -- {variant_name} --")

    abl_model = PrototypicalNetwork(
        backbone="resnet18", embedding_dim=config.EMBEDDING_DIM,
        pretrained=True, dropout_rate=dropout_rate,
        temperature=config.TEMPERATURE,
        freeze_early=freeze_early, use_temperature=use_temperature,
    ).to(device)

    contrastive_w = config.CONTRASTIVE_WEIGHT if use_contrastive else 0.0
    abl_criterion = PrototypicalLoss(
        label_smoothing=config.LABEL_SMOOTHING,
        contrastive_weight=contrastive_w,
        temperature=config.TEMPERATURE,
    )

    param_groups = [
        {"params": abl_model.encoder.encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
        {"params": abl_model.encoder.projection.parameters(), "lr": config.LEARNING_RATE},
    ]
    if use_temperature:
        param_groups.append({"params": [abl_model.temperature], "lr": config.LEARNING_RATE * 0.5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)

    best_val_acc, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        abl_model.train()
        loader = EpisodicDataLoader(train_dataset, config.N_SHOT, config.N_QUERY,
                                    config.N_EPISODES_TRAIN)
        for s_imgs, s_lbls, q_imgs, q_lbls, fruit in loader:
            s_imgs, s_lbls = s_imgs.to(device), s_lbls.to(device)
            q_imgs, q_lbls = q_imgs.to(device), q_lbls.to(device)

            logits, q_emb, s_emb, _ = abl_model(s_imgs, s_lbls, q_imgs)
            all_emb = torch.cat([s_emb, q_emb], dim=0)
            all_lbl = torch.cat([s_lbls, q_lbls], dim=0)
            loss, _, _ = abl_criterion(logits, q_lbls, all_emb, all_lbl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(abl_model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()

        val_loader = EpisodicDataLoader(val_dataset, config.N_SHOT, config.N_QUERY,
                                        config.N_EPISODES_VAL)
        val_acc, _ = evaluate(abl_model, val_loader, device, f"{variant_name} Val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in abl_model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= config.EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    abl_model.load_state_dict(best_state)
    test_loader = EpisodicDataLoader(test_dataset, config.N_SHOT, config.N_QUERY,
                                     config.N_EPISODES_TEST)
    m_acc, pf, all_accs, _ = evaluate(abl_model, test_loader, device,
                                       f"{variant_name} Test", return_all=True)

    print(f"    {variant_name}: {m_acc*100:.1f}% +/- "
          f"{_ci95(all_accs)*100:.1f}%")

    del abl_model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean": m_acc, "std": np.std(all_accs, ddof=1),
        "ci_95": _ci95(all_accs),
        "per_fruit": pf,
    }


def component_ablation(our_model, train_dataset, val_dataset, test_dataset,
                       config, device):
    """Run the full component ablation study (Table 6)."""
    print(f"\n{'='*70}")
    print(f"  COMPONENT ABLATION STUDY (Table 6)")
    print(f"{'='*70}")

    # Full model reference
    print("\n  Evaluating full model (reference) ...")
    test_loader = EpisodicDataLoader(test_dataset, config.N_SHOT, config.N_QUERY,
                                     config.N_EPISODES_TEST)
    full_acc, full_pf, full_all, _ = evaluate(our_model, test_loader, device,
                                               "Full model", return_all=True)
    results = {
        "Full Model": {
            "mean": full_acc, "std": np.std(full_all, ddof=1),
            "ci_95": _ci95(full_all),
            "per_fruit": full_pf,
        }
    }

    ablation_variants = [
        ("- Contrastive Loss",     dict(use_contrastive=False, use_temperature=True,
                                        freeze_early=True, dropout_rate=config.DROPOUT_RATE)),
        ("- Temperature Scaling",  dict(use_contrastive=True, use_temperature=False,
                                        freeze_early=True, dropout_rate=config.DROPOUT_RATE)),
        ("- Frozen Layers",        dict(use_contrastive=True, use_temperature=True,
                                        freeze_early=False, dropout_rate=config.DROPOUT_RATE)),
        ("- Dropout",              dict(use_contrastive=True, use_temperature=True,
                                        freeze_early=True, dropout_rate=0.0)),
    ]
    for name, kwargs in ablation_variants:
        results[name] = train_ablation_variant(
            name, train_dataset, val_dataset, test_dataset,
            config, device, epochs=20, **kwargs,
        )

    full_mean = results["Full Model"]["mean"]
    print(f"\n{'='*70}")
    print(f"  COMPONENT ABLATION SUMMARY (Table 6)")
    print(f"  {'Variant':<30} {'Accuracy':>10} {'95% CI':>10} {'Delta':>8}")
    print(f"  {'-'*60}")
    for name, data in results.items():
        delta = (data["mean"] - full_mean) * 100
        delta_str = f"{delta:+.1f}" if name != "Full Model" else "---"
        print(f"  {name:<30} {data['mean']*100:>9.1f}% {data['ci_95']*100:>9.1f}% {delta_str:>8}")
    print(f"{'='*70}")
    return results


# ====================================================================== #
#  Dataset statistics
# ====================================================================== #

def collect_dataset_statistics(data_root, train_fruits, test_fruits,
                               classes=("fresh", "rotten")):
    """Collect comprehensive dataset statistics."""
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    stats = {"train": {}, "test": {}, "total": {"images": 0, "per_class": defaultdict(int)}}

    print("\nTRAINING SET (Seen Fruits):")
    print("-" * 50)
    total_train = 0
    train_lines = []
    for fruit in train_fruits:
        stats["train"][fruit] = {}
        fruit_counts = []
        for quality in classes:
            folder = os.path.join(data_root, fruit, quality)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder)
                             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            else:
                count = 0
            stats["train"][fruit][quality] = count
            stats["total"]["per_class"][quality] += count
            total_train += count
            fruit_counts.append(f"{quality}: {count}" if os.path.exists(folder) else f"{quality}: 0 (missing)")
        train_lines.append(f"  {fruit.upper()}: {' | '.join(fruit_counts)}")
    print("\n".join(train_lines))

    stats["total"]["train"] = total_train
    print(f"\n  TOTAL TRAINING: {total_train} images")

    print("\nTEST SET (Unseen Fruits):")
    print("-" * 50)
    total_test = 0
    test_lines = []
    for fruit in test_fruits:
        stats["test"][fruit] = {}
        fruit_counts = []
        for quality in classes:
            folder = os.path.join(data_root, fruit, quality)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder)
                             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            else:
                count = 0
            stats["test"][fruit][quality] = count
            stats["total"]["per_class"][quality] += count
            total_test += count
            fruit_counts.append(f"{quality}: {count}" if os.path.exists(folder) else f"{quality}: 0 (missing)")
        test_lines.append(f"  {fruit.upper()}: {' | '.join(fruit_counts)}")
    print("\n".join(test_lines))

    stats["total"]["test"] = total_test
    stats["total"]["images"] = total_train + total_test
    print(f"\n  TOTAL TEST: {total_test} images")

    print(f"\nSUMMARY: {stats['total']['images']} total images "
          f"(train: {total_train}, test: {total_test}) | "
          f"Fresh: {stats['total']['per_class']['fresh']}, "
          f"Rotten: {stats['total']['per_class']['rotten']}")
    return stats


# ====================================================================== #
#  Thesis summary / LaTeX
# ====================================================================== #

def generate_thesis_summary(all_results, dataset_stats, config):
    """Generate complete thesis results summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPLETE THESIS RESULTS SUMMARY")
    lines.append("=" * 80)

    lines.append("\n" + "-" * 80)
    lines.append("1. RESEARCH FOCUS")
    lines.append("-" * 80)
    lines.append(f"  Task: Cross-Species Fruit Quality Grading")
    lines.append(f"  Problem: Binary classification "
                 f"({' vs '.join(c.capitalize() for c in config.CLASSES)}) "
                 f"that generalizes to unseen fruits")
    lines.append(f"  Method: Few-Shot Learning with Prototypical Networks")
    lines.append(f"  Key Claim: Model trained on "
                 f"{{{', '.join(f.capitalize() for f in config.TRAIN_FRUITS)}}} "
                 f"generalizes to "
                 f"{{{', '.join(f.capitalize() for f in config.TEST_FRUITS)}}}")

    lines.append("\n" + "-" * 80)
    lines.append("2. DATASET STATISTICS")
    lines.append("-" * 80)
    if dataset_stats:
        lines.append(f"  Training Fruits (Seen): {config.TRAIN_FRUITS}")
        lines.append(f"  Test Fruits (Unseen):   {config.TEST_FRUITS}")
        lines.append(f"  Classes: {' vs '.join(c.capitalize() for c in config.CLASSES)}")
        lines.append(f"  Total Training Images:  {dataset_stats['total'].get('train', 'N/A')}")
        lines.append(f"  Total Test Images:      {dataset_stats['total'].get('test', 'N/A')}")

    lines.append("\n" + "-" * 80)
    lines.append("3. MAIN RESULTS: Cross-Species Generalization")
    lines.append("-" * 80)
    if "main_experiment" in all_results and all_results["main_experiment"]:
        main = all_results["main_experiment"]
        lines.append(f"\n  Overall Accuracy on UNSEEN Fruits:")
        lines.append(f"    {main['overall']['mean']*100:.1f}% +/- "
                     f"{main['overall']['ci_95']*100:.1f}% (95% CI)")
        lines.append(f"\n  Per-Fruit Results:")
        for fruit, data in main.get("per_fruit", {}).items():
            lines.append(f"    {fruit.capitalize():10}: {data['mean']*100:.1f}% +/- "
                         f"{data['ci_95']*100:.1f}%")

    lines.append("\n" + "-" * 80)
    lines.append("4. N-SHOT ABLATION STUDY")
    lines.append("-" * 80)
    if "ablation_nshot" in all_results and all_results["ablation_nshot"]:
        lines.append(f"\n  {'K-Shot':<10} {'Accuracy':<15} {'95% CI':<15}")
        lines.append("  " + "-" * 40)
        for k, data in sorted(all_results["ablation_nshot"].items()):
            lines.append(f"  {k:<10} {data['mean']*100:>10.1f}%    +/-{data['ci_95']*100:>10.1f}%")

    lines.append("\n" + "-" * 80)
    lines.append("5. BASELINE COMPARISONS")
    lines.append("-" * 80)
    if "baselines" in all_results and all_results["baselines"]:
        lines.append(f"\n  {'Method':<30} {'Accuracy':<12} {'95% CI':<12}")
        lines.append("  " + "-" * 55)
        for method, data in all_results["baselines"].items():
            marker = " ***" if "Ours" in method else ""
            lines.append(f"  {method:<30} {data['mean']*100:>10.1f}%  "
                         f"+/-{data['ci_95']*100:>10.1f}%{marker}")

    lines.append("\n" + "-" * 80)
    lines.append("6. FIGURES GENERATED")
    lines.append("-" * 80)
    figures = [
        ("training_curves.png", "Training and validation accuracy/loss curves"),
        ("ablation_nshot.png", "N-shot ablation study results"),
        # baseline_comparison.png was listed here but no function in the project
        # ever wrote it, so this check reported [Missing] on every run by design.
        # The baseline comparison is a table (generate_latex_tables), not a figure.
        ("embedding_tsne.png", "t-SNE visualization of learned embeddings"),
        ("confusion_matrices.png", "Confusion matrices for unseen fruits"),
    ]
    for fname, desc in figures:
        path = os.path.join(config.RESULTS_DIR, fname)
        status = "OK" if os.path.exists(path) else "Missing"
        lines.append(f"  [{status}] {fname}: {desc}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF RESULTS SUMMARY")
    lines.append("=" * 80)

    print("\n".join(lines))


def generate_latex_tables(all_results, config):
    """Generate all LaTeX tables needed for thesis."""
    lines = []
    lines.append("=" * 70)
    lines.append("LATEX TABLES FOR THESIS")
    lines.append("=" * 70)

    # Table 1: Main Results
    lines.append("\n% Table: Main Results on Unseen Fruits")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-species generalization results on unseen fruit types. "
                 f"The model was trained exclusively on "
                 f"{', '.join(f.capitalize() for f in config.TRAIN_FRUITS)} "
                 f"images and tested on "
                 f"{', '.join(f.capitalize() for f in config.TEST_FRUITS)} "
                 f"without any fine-tuning.}}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Fruit Species} & \\textbf{Accuracy (\\%)} & \\textbf{95\\% CI} \\\\")
    lines.append("\\midrule")

    if "main_experiment" in all_results:
        main = all_results["main_experiment"]
        for fruit in config.TEST_FRUITS:
            if fruit in main.get("per_fruit", {}):
                d = main["per_fruit"][fruit]
                lines.append(f"{fruit.capitalize()} (unseen) & {d['mean']*100:.1f} "
                             f"& $\\pm${d['ci_95']*100:.1f} \\\\")
        lines.append("\\midrule")
        lines.append(f"\\textbf{{Overall}} & "
                     f"\\textbf{{{main['overall']['mean']*100:.1f}}} "
                     f"& $\\pm${main['overall']['ci_95']*100:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Table 2: N-Shot Ablation
    lines.append("\n\n% Table: N-Shot Ablation Study")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Effect of the number of support examples (shots) on "
                 "classification accuracy for unseen fruit species.}")
    lines.append("\\label{tab:nshot_ablation}")
    lines.append("\\begin{tabular}{ccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{K (shots)} & \\textbf{Accuracy (\\%)} & \\textbf{95\\% CI} \\\\")
    lines.append("\\midrule")

    if "ablation_nshot" in all_results:
        for k in sorted(all_results["ablation_nshot"].keys()):
            d = all_results["ablation_nshot"][k]
            lines.append(f"{k} & {d['mean']*100:.1f} & $\\pm${d['ci_95']*100:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Table 3: Baseline Comparison
    lines.append("\n\n% Table: Baseline Comparison")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of our few-shot prototypical network approach "
                 "against baseline methods on unseen fruit species "
                 f"({', '.join(f.capitalize() for f in config.TEST_FRUITS)}).}}")
    lines.append("\\label{tab:baseline_comparison}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Accuracy (\\%)} & \\textbf{95\\% CI} \\\\")
    lines.append("\\midrule")

    if "baselines" in all_results:
        for method, d in all_results["baselines"].items():
            method_clean = method.replace("_", "\\_")
            bold = "\\textbf{" if "Ours" in method else ""
            bold_end = "}" if "Ours" in method else ""
            lines.append(f"{bold}{method_clean}{bold_end} & "
                         f"{bold}{d['mean']*100:.1f}{bold_end} "
                         f"& $\\pm${d['ci_95']*100:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    lines.append("\n" + "=" * 70)
    lines.append("Copy the above LaTeX code into your thesis document")
    lines.append("=" * 70)

    print("\n".join(lines))
