"""
Turn uploaded bytes into model-ready tensors, safely.

Two things must be right here or the demo lies:

1. **Per-encoder normalisation.** ResNet backbones expect ImageNet statistics;
   CLIP and DINOv2 expect their own. ``build_transforms`` in the training code
   defaults to ImageNet, and ``extract_features`` (which built the feature
   cache) instead derives its transform from ``encoder.normalization`` and
   ``encoder.input_size``. Using the wrong constants silently degrades a frozen
   foundation model, so the transform used here must match the one that built
   the cache exactly -- there is a parity test for this.

2. **Untrusted input.** Uploads come from a file picker, so this module assumes
   nothing: decompression-bomb guards, EXIF orientation, mode conversion, and a
   per-file rejection reason rather than a 500.
"""

from __future__ import annotations

import base64
import io
import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from fsgrade.models.backbones import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD

EncoderFamily = Literal["imagenet", "clip", "dinov2"]

# Pillow refuses images above this pixel count -- a decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = 64_000_000

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
THUMBNAIL_PX = 96

ACCEPTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "GIF", "TIFF"}


class ImageRejected(ValueError):
    """One uploaded file could not be used. Carries a human-readable reason."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


@dataclass(frozen=True)
class TransformSpec:
    """Everything needed to reproduce a preprocessing pipeline."""

    size: int
    resize: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    family: EncoderFamily

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size, "resize": self.resize,
            "mean": list(self.mean), "std": list(self.std), "family": self.family,
        }


def transform_spec(
    family: EncoderFamily,
    *,
    image_size: int = 224,
    resize_size: int = 256,
    encoder: Any = None,
) -> TransformSpec:
    """Resolve the preprocessing spec for an encoder family.

    When a live ``FoundationEncoder`` is supplied its own ``normalization`` and
    ``input_size`` win, which is what keeps uploads byte-identical to the cached
    feature path.
    """
    if encoder is not None:
        mean, std = encoder.normalization
        size = int(encoder.input_size)
        # extract_features() resizes to (size, size) then centre-crops to size,
        # so resize == size on that path. Mirror it exactly.
        return TransformSpec(size=size, resize=size, mean=tuple(mean), std=tuple(std),
                             family=family)

    if family == "clip":
        return TransformSpec(224, 224, CLIP_MEAN, CLIP_STD, "clip")
    if family == "dinov2":
        return TransformSpec(518, 518, IMAGENET_MEAN, IMAGENET_STD, "dinov2")
    return TransformSpec(image_size, resize_size, IMAGENET_MEAN, IMAGENET_STD, "imagenet")


def build_transform(spec: TransformSpec):
    """torchvision pipeline for a spec. Deterministic -- no augmentation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((spec.resize, spec.resize)),
        transforms.CenterCrop(spec.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(spec.mean), std=list(spec.std)),
    ])


# --------------------------------------------------------------------------- #
#  Decoding
# --------------------------------------------------------------------------- #

def decode_image(data: bytes, filename: str = "upload") -> Image.Image:
    """Decode untrusted bytes into an RGB image, or raise ``ImageRejected``."""
    if not data:
        raise ImageRejected(filename, "file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected(
            filename, f"file is {len(data) / 1e6:.1f} MB, limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB"
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()          # cheap structural check; consumes the file object
    except UnidentifiedImageError:
        raise ImageRejected(filename, "not a recognised image format") from None
    except Image.DecompressionBombError:
        raise ImageRejected(filename, "image dimensions are implausibly large") from None
    except Exception as exc:  # noqa: BLE001
        raise ImageRejected(filename, f"corrupt image ({type(exc).__name__})") from exc

    if fmt and fmt.upper() not in ACCEPTED_FORMATS:
        raise ImageRejected(filename, f"unsupported format {fmt}")

    # verify() invalidates the handle, so reopen to actually read pixels.
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)      # honour camera rotation
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageRejected(filename, f"could not decode ({type(exc).__name__})") from exc


def load_path(path: str | pathlib.Path) -> Image.Image:
    """Load a dataset image. Trusted input, so no size guard."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def to_tensor(images: Sequence[Image.Image], spec: TransformSpec) -> torch.Tensor:
    """Stack images into a [n, 3, H, W] batch."""
    if not images:
        return torch.zeros(0, 3, spec.size, spec.size)
    tf = build_transform(spec)
    return torch.stack([tf(img) for img in images])


def thumbnail_data_uri(img: Image.Image, px: int = THUMBNAIL_PX, quality: int = 78) -> str:
    """A small inline JPEG for the UI.

    Returned as a data: URI so the frontend never issues a second request for
    an image the server is already holding in memory.
    """
    thumb = img.copy()
    thumb.thumbnail((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ingest(
    data: bytes,
    filename: str,
    specs: dict[str, TransformSpec],
) -> tuple[dict[str, torch.Tensor], str, tuple[int, int]]:
    """Decode once, then produce a tensor per encoder family.

    Decoding is the expensive part, so it happens once even when several arms
    with different normalisations are active.
    """
    img = decode_image(data, filename)
    tensors = {family: to_tensor([img], spec)[0] for family, spec in specs.items()}
    return tensors, thumbnail_data_uri(img), img.size
