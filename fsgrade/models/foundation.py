"""
Frozen foundation-model encoders (DINOv2, CLIP) with an on-disk feature cache.

Motivation. The literature is consistent that a strong frozen embedding plus a
simple classifier is very hard to beat, especially under domain shift -- Tian et
al. (2020) "Rethinking Few-Shot Image Classification", Hu et al. (2022) "Pushing
the Limits of Simple Pipelines" (PMF), Huang et al. (2024) LP++. A thesis in
2026 that compares only ResNet-18 variants invites the obvious question.

The cache is what makes the whole baseline ladder affordable: one GPU pass over
~7k images, after which nearest-centroid, linear probe and SAP are all seconds
of CPU work. Cache keys include the manifest hash, so changing the dataset
invalidates it automatically.

CLIP additionally provides a *text* encoder, which is the basis of SAP: text
embeddings of "a photo of a fresh mango" give a species-conditioned prior for an
unseen species without a single labelled image of it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from fsgrade.models.backbones import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


class FoundationUnavailable(RuntimeError):
    """Optional dependency (timm / open_clip) is not installed."""


@dataclass
class FoundationEncoder:
    """A frozen image encoder, plus an optional text encoder."""

    name: str
    module: Any
    out_dim: int
    normalization: tuple[Sequence[float], Sequence[float]]
    input_size: int
    kind: str                        # "dinov2" | "clip"
    tokenizer: Any = None
    _text_module: Any = None

    @torch.no_grad()
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "clip":
            feats = self.module.encode_image(x)
        else:
            feats = self.module(x)
        return feats.float()

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Embed text prompts. CLIP only."""
        if self.kind != "clip" or self.tokenizer is None:
            raise FoundationUnavailable(
                f"Encoder {self.name!r} has no text tower; SAP requires a CLIP-style model."
            )
        tokens = self.tokenizer(list(prompts)).to(device)
        return self.module.encode_text(tokens).float()


def build_foundation_encoder(
    name: str, *, device: torch.device | str = "cpu"
) -> FoundationEncoder:
    """Build a frozen DINOv2 or CLIP encoder.

    Names: ``dinov2_vits14``, ``dinov2_vitb14`` (timm),
           ``clip_vitb16``, ``clip_vitb32`` (open_clip, LAION-2B weights).
    """
    device = torch.device(device)

    if name.startswith("dinov2"):
        try:
            import timm
        except ImportError as exc:
            raise FoundationUnavailable(
                "DINOv2 requires timm. Install with:  pip install timm"
            ) from exc

        timm_name = {
            "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
        }.get(name, name)
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        cfg = model.default_cfg
        return FoundationEncoder(
            name=name,
            module=model,
            out_dim=model.num_features,
            normalization=(cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD)),
            input_size=cfg.get("input_size", (3, 518, 518))[-1],
            kind="dinov2",
        )

    if name.startswith("clip"):
        try:
            import open_clip
        except ImportError as exc:
            raise FoundationUnavailable(
                "CLIP requires open_clip_torch. Install with:  pip install open_clip_torch"
            ) from exc

        arch, pretrained = {
            "clip_vitb16": ("ViT-B-16", "laion2b_s34b_b88k"),
            "clip_vitb32": ("ViT-B-32", "laion2b_s34b_b79k"),
            "clip_vitl14": ("ViT-L-14", "laion2b_s32b_b82k"),
        }.get(name, ("ViT-B-16", "laion2b_s34b_b88k"))

        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        tokenizer = open_clip.get_tokenizer(arch)
        return FoundationEncoder(
            name=name,
            module=model,
            out_dim=model.visual.output_dim,
            normalization=(CLIP_MEAN, CLIP_STD),
            input_size=model.visual.image_size[0]
            if isinstance(model.visual.image_size, (tuple, list))
            else model.visual.image_size,
            kind="clip",
            tokenizer=tokenizer,
        )

    raise ValueError(
        f"Unknown foundation encoder {name!r}. "
        "Known: dinov2_vits14, dinov2_vitb14, clip_vitb16, clip_vitb32, clip_vitl14"
    )


# --------------------------------------------------------------------------- #
#  Feature cache
# --------------------------------------------------------------------------- #

class FeatureCache:
    """Memory-mapped cache of frozen features, keyed by (encoder, manifest hash)."""

    def __init__(self, cache_root: str | pathlib.Path, encoder_name: str, manifest_hash: str):
        digest = hashlib.sha1(manifest_hash.encode("utf-8")).hexdigest()[:16]
        self.directory = pathlib.Path(cache_root) / "features" / encoder_name / digest
        self.features_path = self.directory / "features.npy"
        self.paths_path = self.directory / "paths.json"
        self.encoder_name = encoder_name
        self.manifest_hash = manifest_hash
        self._features: np.ndarray | None = None
        self._index: dict[str, int] | None = None

    def exists(self) -> bool:
        return self.features_path.exists() and self.paths_path.exists()

    def save(self, relpaths: Sequence[str], features: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.features_path, features.astype(np.float32))
        with open(self.paths_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "encoder": self.encoder_name,
                    "manifest_hash": self.manifest_hash,
                    "dim": int(features.shape[1]),
                    "n": int(features.shape[0]),
                    "relpaths": list(relpaths),
                },
                fh,
            )

    def load(self) -> tuple[list[str], np.ndarray]:
        if self._features is None:
            with open(self.paths_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self._features = np.load(self.features_path, mmap_mode="r")
            self._relpaths = meta["relpaths"]
            self._index = {rel: i for i, rel in enumerate(self._relpaths)}
        return self._relpaths, self._features

    def lookup(self, relpaths: Sequence[str]) -> np.ndarray:
        """Fetch features for specific images, in the order given."""
        self.load()
        assert self._index is not None and self._features is not None
        missing = [r for r in relpaths if r not in self._index]
        if missing:
            raise KeyError(
                f"{len(missing)} paths absent from the {self.encoder_name} cache "
                f"(e.g. {missing[:3]}). Rebuild with: python -m fsgrade.cli.cache_features"
            )
        idx = np.asarray([self._index[r] for r in relpaths], dtype=np.int64)
        return np.asarray(self._features[idx], dtype=np.float32)


@torch.no_grad()
def extract_features(
    encoder: FoundationEncoder,
    relpaths: Sequence[str],
    data_root: str | pathlib.Path,
    *,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int | None = None,
    logger: Any = None,
) -> np.ndarray:
    """Run every image through a frozen encoder once."""
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from fsgrade.data.loaders import PathListDataset

    size = image_size or encoder.input_size
    mean, std = encoder.normalization
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])

    dataset = PathListDataset(relpaths, data_root, tf)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    chunks: list[np.ndarray] = []
    for step, (images, _) in enumerate(loader):
        feats = encoder.encode_images(images.to(device))
        chunks.append(feats.cpu().numpy().astype(np.float32))
        if logger is not None and step % 20 == 0:
            logger.info(
                "  %s: %d/%d images", encoder.name,
                min((step + 1) * batch_size, len(dataset)), len(dataset),
            )
    return np.concatenate(chunks, axis=0)


def get_or_build_cache(
    encoder_name: str,
    index: Any,
    *,
    cache_root: str | pathlib.Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    logger: Any = None,
) -> FeatureCache:
    """Return a populated cache, extracting features only if needed."""
    cache = FeatureCache(cache_root, encoder_name, index.manifest_hash)
    if cache.exists():
        if logger:
            logger.info("Feature cache hit: %s", cache.directory)
        return cache

    if logger:
        logger.info("Building feature cache for %s (%d images)", encoder_name, len(index))
    encoder = build_foundation_encoder(encoder_name, device=device)
    relpaths = [r.relpath for r in index.records]
    features = extract_features(
        encoder, relpaths, index.data_root,
        device=device, batch_size=batch_size, num_workers=num_workers, logger=logger,
    )
    cache.save(relpaths, features)
    if logger:
        logger.info("Cached %s features -> %s", features.shape, cache.directory)
    return cache


# --------------------------------------------------------------------------- #
#  Prompts for the text branch (SAP)
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATES: dict[str, list[str]] = {
    "simple": ["a photo of a {quality} {species}"],
    "descriptive": [
        "a photo of a {quality} {species}",
        "a close-up photo of a {quality} {species}",
        "a {quality} {species} fruit",
    ],
    "ensemble": [
        "a photo of a {quality} {species}",
        "a close-up photo of a {quality} {species}",
        "a {quality} {species} fruit",
        "an image of a {quality} {species} on a plain background",
        "a photograph of {quality} {species} fruit for quality inspection",
    ],
}

# Synonyms matter: CLIP's training distribution contains "rotten"/"spoiled"
# far more often than a dataset folder name like "rotten". Prompt-template
# sensitivity is a standard reviewer question and is swept as experiment E7.
QUALITY_SYNONYMS: dict[str, list[str]] = {
    "fresh": ["fresh", "ripe and fresh", "unspoiled", "good quality"],
    "rotten": ["rotten", "spoiled", "moldy and rotten", "decayed"],
}


def build_prompts(
    species: str,
    classes: Sequence[str],
    *,
    template: str = "descriptive",
    use_synonyms: bool = False,
) -> dict[str, list[str]]:
    """Prompts per quality class for one species."""
    templates = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["simple"])
    out: dict[str, list[str]] = {}
    for cls in classes:
        words = QUALITY_SYNONYMS.get(cls, [cls]) if use_synonyms else [cls]
        out[cls] = [t.format(quality=w, species=species) for t in templates for w in words]
    return out


@torch.no_grad()
def text_prototypes(
    encoder: FoundationEncoder,
    species: str,
    classes: Sequence[str],
    device: torch.device,
    *,
    template: str = "descriptive",
    use_synonyms: bool = False,
) -> torch.Tensor:
    """One L2-normalised text prototype per class, averaged over the prompt ensemble.

    This is the species-conditioned prior that requires **zero** labelled images
    of the unseen species -- the core of SAP.
    """
    import torch.nn.functional as F

    prompts = build_prompts(species, classes, template=template, use_synonyms=use_synonyms)
    protos = []
    for cls in classes:
        emb = encoder.encode_text(prompts[cls], device)
        emb = F.normalize(emb, dim=-1).mean(dim=0)
        protos.append(F.normalize(emb, dim=-1))
    return torch.stack(protos)
