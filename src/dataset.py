"""
Episodic dataset and data-loader for few-shot learning.

FruitQualityDataset  – loads image paths organised by fruit/quality with train/val split.
EpisodicDataLoader   – yields (support, query) episodes.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image


class FruitQualityDataset:
    """
    Loads fruit images organised as  <data_root>/<fruit>/<quality>/*.jpg
    Supports a reproducible train / val split to prevent data leakage.
    """

    def __init__(
        self,
        data_root,
        fruit_types,
        classes=("fresh", "rotten"),
        transform=None,
        support_transform=None,
        query_transform=None,
        split="all",
        val_ratio=0.0,
        seed=42,
    ):
        self.data_root = data_root
        self.fruit_types = fruit_types
        self.classes = classes
        self.transform = transform
        self.support_transform = support_transform or transform
        self.query_transform = query_transform or transform
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed

        self.data = defaultdict(lambda: defaultdict(list))
        self._load_data()

    # ------------------------------------------------------------------ #
    def _load_data(self):
        rng = np.random.RandomState(self.seed)

        for fruit in self.fruit_types:
            for quality in self.classes:
                folder_path = os.path.join(self.data_root, fruit, quality)
                if os.path.exists(folder_path):
                    all_images = sorted(
                        [
                            os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        ]
                    )

                    if self.val_ratio > 0 and self.split in ("train", "val"):
                        n_val = max(1, int(len(all_images) * self.val_ratio))
                        indices = rng.permutation(len(all_images))
                        selected = indices[:n_val] if self.split == "val" else indices[n_val:]
                        images = [all_images[i] for i in selected]
                    else:
                        images = all_images

                    self.data[fruit][quality] = images
                    print(f"  [{self.split:5}] Loaded {len(images):4d} images: {fruit}/{quality}")
                else:
                    print(f"  WARNING  Missing folder: {folder_path}")

    # ------------------------------------------------------------------ #
    def get_episode(self, n_shot, n_query, fruit=None):
        """Sample a single episode with separate transforms for support / query."""
        if fruit is None:
            fruit = random.choice(self.fruit_types)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for class_idx, quality in enumerate(self.classes):
            all_images = self.data[fruit][quality]
            required = n_shot + n_query

            if len(all_images) < required:
                if len(all_images) < n_shot:
                    raise ValueError(
                        f"Not enough images for support: {fruit}/{quality}. "
                        f"Need {n_shot}, have {len(all_images)}"
                    )
                support_paths = random.sample(all_images, n_shot)
                remaining = [p for p in all_images if p not in support_paths]
                if len(remaining) < n_query:
                    query_paths = remaining + random.choices(
                        all_images, k=n_query - len(remaining)
                    )
                else:
                    query_paths = random.sample(remaining, n_query)
            else:
                sampled = random.sample(all_images, required)
                support_paths = sampled[:n_shot]
                query_paths = sampled[n_shot:]

            for path in support_paths:
                img = Image.open(path).convert("RGB")
                if self.support_transform:
                    img = self.support_transform(img)
                support_images.append(img)
                support_labels.append(class_idx)

            for path in query_paths:
                img = Image.open(path).convert("RGB")
                if self.query_transform:
                    img = self.query_transform(img)
                query_images.append(img)
                query_labels.append(class_idx)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels),
            fruit,
        )


class EpisodicDataLoader:
    """DataLoader that yields episodes instead of mini-batches."""

    def __init__(self, dataset, n_shot, n_query, n_episodes, fruits=None):
        self.dataset = dataset
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.fruits = fruits if fruits else dataset.fruit_types

    def __iter__(self):
        for _ in range(self.n_episodes):
            fruit = random.choice(self.fruits)
            yield self.dataset.get_episode(self.n_shot, self.n_query, fruit=fruit)

    def __len__(self):
        return self.n_episodes
