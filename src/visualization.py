"""
Plotting and visualisation helpers for thesis figures.

  plot_training_history     – loss, accuracy, gap, LR/temperature curves
  visualize_embeddings      – t-SNE of test-set embeddings
  plot_confusion_matrices   – per-fruit and overall confusion matrices
  plot_ablation_nshot       – N-shot ablation bar chart
"""

import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


# ====================================================================== #
#  Training curves
# ====================================================================== #

def plot_training_history(history, config, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_acc"], "b-", lw=2, label="Train")
    axes[0, 1].plot(epochs, history["val_acc"], color="orange", lw=2, label="Validation")
    axes[0, 1].fill_between(epochs, history["val_acc"], history["train_acc"],
                             alpha=0.3, color="red", label="Overfit Gap")
    axes[0, 1].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random")
    axes[0, 1].set(xlabel="Epoch", ylabel="Accuracy", title="Train vs Validation Accuracy")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_ylim(0.4, 1.0)

    axes[1, 0].plot(epochs, history["gap"], "r-", lw=2)
    axes[1, 0].axhline(0.1, color="orange", ls="--", alpha=0.7, label="Warning")
    axes[1, 0].axhline(0.15, color="red", ls="--", alpha=0.7, label="Overfitting")
    axes[1, 0].axhline(0, color="green", ls="--", alpha=0.7, label="Ideal")
    axes[1, 0].fill_between(epochs, 0, history["gap"],
                             where=[g > 0.1 for g in history["gap"]], alpha=0.3, color="red")
    axes[1, 0].set(xlabel="Epoch", ylabel="Train - Val Accuracy", title="Overfitting Gap")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(epochs, history["lr"], "g-", lw=2, label="Learning Rate")
    ax4.set(xlabel="Epoch", ylabel="Learning Rate", title="LR & Temperature Schedule")
    ax4.tick_params(axis="y", labelcolor="green"); ax4.grid(True, alpha=0.3)
    if "temperature" in history:
        ax_t = ax4.twinx()
        ax_t.plot(epochs, history["temperature"], "m--", lw=2, label="Temperature")
        ax_t.set_ylabel("Temperature", color="magenta")
        ax_t.tick_params(axis="y", labelcolor="magenta")
        lines1, l1 = ax4.get_legend_handles_labels()
        lines2, l2 = ax_t.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, l1 + l2, loc="upper right")
    else:
        ax4.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    thesis = os.path.join(config.THESIS_FIGS_DIR, "training_curves.png")
    plt.savefig(thesis, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved to {thesis}")


# ====================================================================== #
#  Embedding visualisation (t-SNE)
# ====================================================================== #

def visualize_embeddings(model, test_dataset, eval_transform, device, config,
                         n_samples_per_class=100):
    model.eval()
    embeddings_list, labels_list, fruits_list = [], [], []

    with torch.no_grad():
        for fruit in config.TEST_FRUITS:
            for class_idx, quality in enumerate(config.CLASSES):
                images = test_dataset.data[fruit][quality][:n_samples_per_class]
                for img_path in tqdm(images, desc=f"{fruit}/{quality}", leave=False, mininterval=30, miniters=20):
                    img = Image.open(img_path).convert("RGB")
                    t = eval_transform(img).unsqueeze(0).to(device)
                    emb = model.encoder(t)
                    embeddings_list.append(emb.cpu().numpy().flatten())
                    labels_list.append(class_idx)
                    fruits_list.append(fruit)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    fruits = np.array(fruits_list)

    # scikit-learn renamed TSNE's iteration cap from `n_iter` to `max_iter` in
    # 1.5. Passing the wrong one is a hard TypeError, so the same code crashes on
    # one machine and works on another -- exactly the kind of environment drift
    # that makes a result unreproducible. Pick whichever the installed version
    # accepts rather than pinning a version here.
    import inspect
    tsne_params = inspect.signature(TSNE.__init__).parameters
    iter_kwarg = "max_iter" if "max_iter" in tsne_params else "n_iter"

    # perplexity must be < n_samples; sklearn raises otherwise. This matters for
    # small or partial datasets, where the default 30 is not always satisfiable.
    perplexity = min(30, max(5, len(embeddings) - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                **{iter_kwarg: 1000})
    emb2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(config.CLASSES)))
    for lbl, name in enumerate(config.CLASSES):
        m = labels == lbl
        axes[0].scatter(emb2d[m, 0], emb2d[m, 1], c=[colors[lbl]], label=name.capitalize(), alpha=0.6, s=50)
    axes[0].set_title("By Quality Class", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=12); axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    fc = plt.cm.tab10(np.linspace(0, 1, len(config.TEST_FRUITS)))
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
    for i, fruit in enumerate(config.TEST_FRUITS):
        m = fruits == fruit
        axes[1].scatter(emb2d[m, 0], emb2d[m, 1], c=[fc[i]],
                        marker=markers[i % len(markers)], label=fruit.capitalize(), alpha=0.6, s=50)
    axes[1].set_title("By Fruit Type (Unseen)", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=12); axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "embedding_tsne.png"), dpi=300, bbox_inches="tight")
    thesis = os.path.join(config.THESIS_FIGS_DIR, "embedding_tsne.png")
    plt.savefig(thesis, dpi=300, bbox_inches="tight"); plt.show()

    sil = silhouette_score(emb2d, labels)
    knn = KNeighborsClassifier(n_neighbors=5); knn.fit(emb2d, labels)
    print(f"  Silhouette: {sil:.3f}  |  5-NN purity: {knn.score(emb2d, labels):.3f}")
    return emb2d, labels, fruits


# ====================================================================== #
#  Confusion matrices
# ====================================================================== #

def plot_confusion_matrices(model, test_dataset, device, config):
    model.eval()
    class_names = [c.capitalize() for c in config.CLASSES]
    fig, axes = plt.subplots(1, len(config.TEST_FRUITS) + 1,
                              figsize=(6 * (len(config.TEST_FRUITS) + 1), 5))
    all_preds, all_labels, all_fruits = [], [], []

    for idx, fruit in enumerate(config.TEST_FRUITS):
        preds, labels = [], []
        for _ in tqdm(range(100), desc=f"Testing {fruit}", leave=False, mininterval=30, miniters=20):
            si, sl, qi, ql, _ = test_dataset.get_episode(config.N_SHOT, config.N_QUERY, fruit=fruit)
            si, sl, qi = si.to(device), sl.to(device), qi.to(device)
            with torch.no_grad():
                logits, _, _, _ = model(si, sl, qi)
            batch_preds = logits.argmax(1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(ql.numpy())
            all_preds.extend(batch_preds)
            all_labels.extend(ql.numpy())
            all_fruits.extend([fruit] * len(batch_preds))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f"{fruit.capitalize()}\n(Unseen)", fontsize=12, fontweight="bold")
        axes[idx].set(xlabel="Predicted", ylabel="True")

    cm_all = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_all, annot=True, fmt="d", cmap="Greens", ax=axes[-1],
                xticklabels=class_names, yticklabels=class_names)
    axes[-1].set_title("Overall\n(All Unseen)", fontsize=12, fontweight="bold")
    axes[-1].set(xlabel="Predicted", ylabel="True")

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    thesis = os.path.join(config.THESIS_FIGS_DIR, "confusion_matrices.png")
    plt.savefig(thesis, dpi=300, bbox_inches="tight"); plt.show()

    print("\nClassification Report (All Unseen Fruits):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Return the matrix AND the report derived from the SAME prediction arrays.
    # The committed Imgs/confusion_matrices.png (86.2% overall) disagreed with
    # the classification report printed beside it (85%), because the figure and
    # the text came from different runs. Handing both back from one call makes
    # that divergence impossible to reintroduce.
    return {
        "confusion_matrix_overall": cm_all.tolist(),
        "per_fruit_confusion": {
            fruit: confusion_matrix(
                [l for l, f in zip(all_labels, all_fruits) if f == fruit],
                [p for p, f in zip(all_preds, all_fruits) if f == fruit],
            ).tolist()
            for fruit in config.TEST_FRUITS
        },
        "classification_report": classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        ),
        "n_query_predictions": len(all_labels),
    }


# ====================================================================== #
#  N-shot ablation plot
# ====================================================================== #

def plot_ablation_nshot(results, config):
    """Plot N-shot ablation from a {k: {mean, ci_95, ...}} dict."""
    k_values = sorted(results.keys())
    means = [results[k]["mean"] * 100 for k in k_values]
    cis = [results[k]["ci_95"] * 100 for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, means, yerr=cis, marker="o", capsize=5, lw=2, ms=8)
    plt.xlabel("Number of Shots (K)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("N-Shot Ablation on Unseen Fruits", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3); plt.xticks(k_values); plt.ylim(40, 100)
    plt.savefig(os.path.join(config.RESULTS_DIR, "ablation_nshot.png"), dpi=300, bbox_inches="tight")
    thesis = os.path.join(config.THESIS_FIGS_DIR, "ablation_nshot.png")
    plt.savefig(thesis, dpi=300, bbox_inches="tight"); plt.show()
