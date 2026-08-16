"""
Datasets that turn episode specs and manifests into tensors.

Loading is decoupled from sampling. ``EpisodeDataset`` never draws random
numbers -- the episode was fully determined when the bank was built -- so it is
safe with ``num_workers > 0``. The original ``get_episode`` called the global
``random`` module inside the loader, which made results depend on worker count
and iteration order.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from fsgrade.data.episodes import EpisodeBank, EpisodeSpec
from fsgrade.data.index import ImageIndex


@dataclass
class EpisodeBatch:
    """One episode as tensors, plus the spec it came from."""

    spec: EpisodeSpec
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor

    @property
    def n_way(self) -> int:
        return self.spec.n_way

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(
            spec=self.spec,
            support_x=self.support_x.to(device, non_blocking=True),
            support_y=self.support_y.to(device, non_blocking=True),
            query_x=self.query_x.to(device, non_blocking=True),
            query_y=self.query_y.to(device, non_blocking=True),
        )


def _load_image(path: pathlib.Path, transform: Callable | None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return transform(img) if transform is not None else torch.as_tensor(img)


class EpisodeDataset(Dataset):
    """Yields an ``EpisodeBatch`` for each spec in a bank."""

    def __init__(
        self,
        bank: EpisodeBank,
        data_root: str | pathlib.Path,
        *,
        support_transform: Callable | None = None,
        query_transform: Callable | None = None,
        k_shot: int | None = None,
    ) -> None:
        self.bank = bank.with_shots(k_shot) if k_shot is not None else bank
        self.data_root = pathlib.Path(data_root)
        self.support_transform = support_transform
        self.query_transform = query_transform

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, i: int) -> EpisodeBatch:
        spec = self.bank[i]
        sx, sy = [], []
        for label, rel in spec.support:
            sx.append(_load_image(self.data_root / rel, self.support_transform))
            sy.append(label)
        qx, qy = [], []
        for label, rel in spec.query:
            qx.append(_load_image(self.data_root / rel, self.query_transform))
            qy.append(label)
        return EpisodeBatch(
            spec=spec,
            support_x=torch.stack(sx),
            support_y=torch.tensor(sy, dtype=torch.long),
            query_x=torch.stack(qx),
            query_y=torch.tensor(qy, dtype=torch.long),
        )


def episode_collate(batch: Sequence[EpisodeBatch]) -> EpisodeBatch:
    """Collate that passes a single episode through unchanged.

    Episodes are the unit of computation, so ``batch_size=1`` is the norm.
    """
    if len(batch) != 1:
        raise ValueError(
            f"EpisodeDataset expects batch_size=1 (one episode per step), got {len(batch)}"
        )
    return batch[0]


class FlatImageDataset(Dataset):
    """Flat (image, quality_label, species_label) dataset for supervised training.

    Used by the zero-shot supervised baseline and the species-adversarial arm --
    neither of which is episodic.
    """

    def __init__(
        self,
        index: ImageIndex,
        pools: dict[str, dict[str, list[str]]],
        *,
        species: Sequence[str],
        transform: Callable | None = None,
    ) -> None:
        self.index = index
        self.transform = transform
        self.species_names = list(species)
        self.species_to_idx = {s: i for i, s in enumerate(self.species_names)}

        self.samples: list[tuple[str, int, int]] = []
        for sp in self.species_names:
            for label, cls in enumerate(index.classes):
                for rel in pools.get(sp, {}).get(cls, []):
                    self.samples.append((rel, label, self.species_to_idx[sp]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        rel, quality, species = self.samples[i]
        return _load_image(self.index.abs_path(rel), self.transform), quality, species

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, quality, _ in self.samples:
            counts[quality] = counts.get(quality, 0) + 1
        return counts


class PathListDataset(Dataset):
    """Minimal dataset over relative paths -- used for feature caching."""

    def __init__(
        self,
        relpaths: Sequence[str],
        data_root: str | pathlib.Path,
        transform: Callable | None = None,
    ) -> None:
        self.relpaths = list(relpaths)
        self.data_root = pathlib.Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.relpaths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return _load_image(self.data_root / self.relpaths[i], self.transform), i


def build_transforms(cfg: dict[str, Any], normalization: tuple | None = None) -> dict[str, Any]:
    """Transform pipelines for support / query-train / eval.

    ``normalization`` defaults to ImageNet statistics but is overridden for CLIP
    and DINOv2, which expect their own constants. The original code hardcoded
    ImageNet values, which would silently degrade a frozen foundation model.
    """
    from torchvision import transforms

    from fsgrade.config import get_in
    from fsgrade.models.backbones import IMAGENET_MEAN, IMAGENET_STD

    size = int(get_in(cfg, "data.image_size", 224))
    resize = int(get_in(cfg, "data.resize_size", 256))
    mean, std = normalization if normalization else (IMAGENET_MEAN, IMAGENET_STD)
    norm = transforms.Normalize(mean=list(mean), std=list(std))
    aug = get_in(cfg, "data.augmentation", {}) or {}

    train = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=float(aug.get("hflip", 0.5))),
        transforms.RandomVerticalFlip(p=float(aug.get("vflip", 0.3))),
        transforms.RandomRotation(degrees=float(aug.get("rotation", 20))),
        transforms.ColorJitter(
            brightness=float(aug.get("brightness", 0.2)),
            contrast=float(aug.get("contrast", 0.2)),
            saturation=float(aug.get("saturation", 0.2)),
            # Hue jitter on a colour-driven fresh/rotten task is aggressive:
            # browning IS the signal. Default lowered from the original 0.05.
            hue=float(aug.get("hue", 0.02)),
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=float(aug.get("erasing", 0.1)), scale=(0.02, 0.08)),
    ])

    support = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        norm,
    ])

    evaluation = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        norm,
    ])

    return {"train": train, "support": support, "eval": evaluation}
