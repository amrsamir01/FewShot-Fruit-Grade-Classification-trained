"""
Training and evaluation utilities.

Includes:
  EarlyStopping       – patience-based stopper
  compute_accuracy    – episode accuracy from logits
  train_epoch         – one epoch of episodic training
  evaluate            – evaluation loop (returns per-fruit stats)
  train_protonet      – full training loop with warmup, cosine LR, early stopping
  save_checkpoint / load_checkpoint
"""

import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


# ====================================================================== #
#  Helpers
# ====================================================================== #

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            (score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def compute_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]


# ====================================================================== #
#  Training / evaluation loops
# ====================================================================== #

def train_epoch(model, dataloader, criterion, optimizer, device, gradient_clip=1.0):
    model.train()
    total_loss, total_acc, n_episodes = 0.0, 0.0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False, mininterval=30, miniters=50)
    for support_imgs, support_lbls, query_imgs, query_lbls, fruit in pbar:
        support_imgs = support_imgs.to(device)
        support_lbls = support_lbls.to(device)
        query_imgs = query_imgs.to(device)
        query_lbls = query_lbls.to(device)

        logits, query_emb, support_emb, _ = model(support_imgs, support_lbls, query_imgs)

        all_embeddings = torch.cat([support_emb, query_emb], dim=0)
        all_labels = torch.cat([support_lbls, query_lbls], dim=0)
        loss, _, _ = criterion(logits, query_lbls, all_embeddings, all_labels)

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        acc = compute_accuracy(logits, query_lbls)
        total_loss += loss.item()
        total_acc += acc
        n_episodes += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

    return total_loss / n_episodes, total_acc / n_episodes


@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating", return_all=False):
    model.eval()
    all_accuracies = []
    fruit_accuracies = defaultdict(list)

    pbar = tqdm(dataloader, desc=desc, leave=False, mininterval=30, miniters=50)
    for support_imgs, support_lbls, query_imgs, query_lbls, fruit in pbar:
        support_imgs = support_imgs.to(device)
        support_lbls = support_lbls.to(device)
        query_imgs = query_imgs.to(device)
        query_lbls = query_lbls.to(device)

        logits, _, _, _ = model(support_imgs, support_lbls, query_imgs)
        acc = compute_accuracy(logits, query_lbls)
        all_accuracies.append(acc)
        fruit_accuracies[fruit].append(acc)
        pbar.set_postfix({"acc": f"{acc:.3f}"})

    mean_acc = np.mean(all_accuracies)
    per_fruit_acc = {f: np.mean(a) for f, a in fruit_accuracies.items()}

    if return_all:
        return mean_acc, per_fruit_acc, all_accuracies, dict(fruit_accuracies)
    return mean_acc, per_fruit_acc


# ====================================================================== #
#  Full training loop
# ====================================================================== #

def train_protonet(model, train_loader, val_loader, criterion, config, device):
    """Training loop with warmup, cosine LR, and early stopping."""

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.encoder.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": model.encoder.projection.parameters(), "lr": config.LEARNING_RATE},
            {"params": [model.temperature], "lr": config.LEARNING_RATE * 0.5},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / config.WARMUP_EPOCHS
        progress = (epoch - config.WARMUP_EPOCHS) / (config.EPOCHS - config.WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, min_delta=0.003)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": [], "temperature": [], "gap": []}
    best_val_acc = 0.0
    best_state = None

    print("=" * 70)
    print("TRAINING WITH WARMUP AND EARLY STOPPING")
    print("=" * 70)

    for epoch in range(1, config.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config.GRADIENT_CLIP
        )
        val_acc, _ = evaluate(model, val_loader, device, "Validating")
        scheduler.step()

        gap = train_acc - val_acc
        temp = model.temperature.item()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["temperature"].append(temp)
        history["gap"].append(gap)

        if epoch <= config.WARMUP_EPOCHS:
            status = "WARMUP"
        elif gap > 0.15:
            status = "OVERFITTING"
        elif gap > 0.08:
            status = "SLIGHT OVERFIT"
        elif val_acc < 0.55:
            status = "UNDERFITTING"
        else:
            status = "GOOD"

        if epoch == 1 or epoch % 5 == 0 or epoch == config.EPOCHS:
            print(
                f"\nEpoch {epoch}/{config.EPOCHS} [{status}]"
                f"\n  Train: Loss={train_loss:.4f}, Acc={train_acc:.3f}"
                f"\n  Val:   Acc={val_acc:.3f} | Gap={gap:+.3f} | Temp={temp:.3f}"
                f"\n  LR:    {current_lr:.2e}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": val_acc, "train_acc": train_acc, "gap": gap},
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
            )
            print("  -> Best model saved")

        if epoch > config.WARMUP_EPOCHS and early_stopping(val_acc, epoch):
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"  Best val acc: {early_stopping.best_score:.3f} @ epoch {early_stopping.best_epoch}")
            break

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final train/val gap:      {history['gap'][-1]:+.3f}")

    # Restore the best model weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        print("  -> Best model weights restored.")

    return history
