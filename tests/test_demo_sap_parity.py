"""
SAP parity and preprocessing tests.

The demo re-implements the SAP blend so it can run on uploaded tensors rather
than cached dataset paths. Two implementations of the same equation will drift
unless something pins them together, so these tests compare the demo's
``blend_prototypes`` against the real
``SpeciesAnchoredPrototypes._blend`` on identical synthetic features.

None of this needs CLIP installed -- the blend is pure numpy, and the method's
``_blend`` only touches the text-prototype cache, which we prefill.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsgrade.demo.preprocess import (
    ImageRejected,
    TransformSpec,
    decode_image,
    ingest,
    thumbnail_data_uri,
    to_tensor,
    transform_spec,
)
from fsgrade.demo.sap import (
    DEFAULT_KAPPA,
    blend_prototypes,
    effective_alpha,
    kappa_grid,
    l2_normalize,
)

DIM = 64
N_WAY = 2


def _real_method(text_protos, kappa, mode="shot_adaptive", alpha=0.5):
    """A SpeciesAnchoredPrototypes with its text cache prefilled (no CLIP needed)."""
    from fsgrade.methods.frozen import SpeciesAnchoredPrototypes

    m = SpeciesAnchoredPrototypes(kappa=kappa, alpha_mode=mode, alpha=alpha)
    m._classes = ["fresh", "rotten"]
    m._text_cache = {"mango": np.asarray(text_protos, dtype=np.float32)}
    return m


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def text_protos(rng):
    return l2_normalize(rng.normal(size=(N_WAY, DIM)).astype(np.float32))


# ------------------------------------------------------------- parity ------ #

@pytest.mark.parametrize("k", [0, 1, 3, 5, 10])
@pytest.mark.parametrize("kappa", [1.0, 5.0, 8.0, 20.0])
def test_blend_matches_the_real_method(rng, text_protos, k, kappa):
    """The demo's blend must equal SpeciesAnchoredPrototypes._blend exactly."""
    sf = l2_normalize(rng.normal(size=(k * N_WAY, DIM)).astype(np.float32)) if k else np.zeros((0, DIM), np.float32)
    sy = np.repeat(np.arange(N_WAY), k) if k else np.zeros(0, dtype=int)

    mine, alpha = blend_prototypes(text_protos, sf, sy, N_WAY, kappa=kappa)
    theirs = _real_method(text_protos, kappa)._blend("mango", sf, sy, N_WAY)

    assert np.allclose(mine, theirs, atol=1e-6), f"drift at k={k}, kappa={kappa}"
    assert alpha == pytest.approx(k / (k + kappa))


def test_blend_matches_in_fixed_alpha_mode(rng, text_protos):
    sf = l2_normalize(rng.normal(size=(6, DIM)).astype(np.float32))
    sy = np.repeat(np.arange(N_WAY), 3)
    mine, alpha = blend_prototypes(
        text_protos, sf, sy, N_WAY, mode="fixed", fixed_alpha=0.3
    )
    theirs = _real_method(text_protos, 5.0, mode="fixed", alpha=0.3)._blend("mango", sf, sy, N_WAY)
    assert np.allclose(mine, theirs, atol=1e-6)
    assert alpha == pytest.approx(0.3)


# ------------------------------------------------------------- alpha ------- #

def test_zero_shot_is_pure_text(text_protos):
    """At K=0 the method must collapse exactly onto the text prototypes."""
    protos, alpha = blend_prototypes(
        text_protos, np.zeros((0, DIM), np.float32), np.zeros(0, int), N_WAY
    )
    assert alpha == 0.0
    assert np.allclose(protos, l2_normalize(text_protos), atol=1e-6)


def test_alpha_increases_monotonically_with_k():
    """More shots means more trust in the visual prototype."""
    alphas = [effective_alpha(k, DEFAULT_KAPPA) for k in range(0, 21)]
    assert alphas[0] == 0.0
    assert all(b > a for a, b in zip(alphas, alphas[1:]))
    assert alphas[-1] < 1.0, "alpha approaches but never reaches 1"


def test_alpha_equals_half_at_k_equals_kappa():
    assert effective_alpha(8, 8.0) == pytest.approx(0.5)


def test_larger_kappa_means_slower_hand_over():
    """This is what the UI dial demonstrates."""
    assert effective_alpha(5, 20.0) < effective_alpha(5, 5.0) < effective_alpha(5, 1.0)


def test_prototypes_stay_unit_norm(rng, text_protos):
    sf = l2_normalize(rng.normal(size=(10, DIM)).astype(np.float32))
    sy = np.repeat(np.arange(N_WAY), 5)
    protos, _ = blend_prototypes(text_protos, sf, sy, N_WAY)
    assert np.allclose(np.linalg.norm(protos, axis=1), 1.0, atol=1e-5)


def test_missing_class_support_falls_back_to_text(rng, text_protos):
    """One-sided support must not produce a NaN prototype."""
    sf = l2_normalize(rng.normal(size=(3, DIM)).astype(np.float32))
    sy = np.zeros(3, dtype=int)          # only class 0 present
    protos, _ = blend_prototypes(text_protos, sf, sy, N_WAY)
    assert np.isfinite(protos).all()
    assert np.allclose(np.linalg.norm(protos, axis=1), 1.0, atol=1e-5)


def test_kappa_grid_matches_the_calibration_grid():
    """The UI dial ticks must be the values _calibrate actually searches."""
    assert kappa_grid() == [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


def test_l2_normalize_handles_zero_vectors():
    out = l2_normalize(np.zeros((2, DIM), dtype=np.float32))
    assert np.isfinite(out).all()


# --------------------------------------------------------- preprocessing --- #

def test_clip_spec_does_not_use_imagenet_constants():
    """Using ImageNet statistics on CLIP silently degrades it."""
    clip = transform_spec("clip")
    imagenet = transform_spec("imagenet")
    assert clip.mean != imagenet.mean
    assert clip.mean[0] == pytest.approx(0.48145466)


def test_encoder_normalisation_wins_over_defaults():
    """Must mirror extract_features(), which builds its transform from the encoder."""
    class FakeEncoder:
        normalization = ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        input_size = 336

    spec = transform_spec("clip", encoder=FakeEncoder())
    assert spec.size == 336 and spec.resize == 336
    assert spec.mean == (0.1, 0.2, 0.3)


def _png_bytes(size=(40, 30), colour=(120, 180, 90)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_accepts_a_normal_image():
    img = decode_image(_png_bytes(), "a.png")
    assert img.mode == "RGB" and img.size == (40, 30)


def test_decode_rejects_non_images_with_a_reason():
    with pytest.raises(ImageRejected) as exc:
        decode_image(b"this is not an image", "notes.txt")
    assert exc.value.filename == "notes.txt"
    assert "recognised" in exc.value.reason


def test_decode_rejects_empty_and_oversized():
    with pytest.raises(ImageRejected, match="empty"):
        decode_image(b"", "empty.png")
    with pytest.raises(ImageRejected, match="limit"):
        decode_image(b"x" * (13 * 1024 * 1024), "huge.png")


def test_grayscale_is_converted_to_rgb():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (20, 20), 128).save(buf, format="PNG")
    assert decode_image(buf.getvalue(), "g.png").mode == "RGB"


def test_to_tensor_shape_and_normalisation():
    spec = TransformSpec(32, 36, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "imagenet")
    batch = to_tensor([decode_image(_png_bytes(), "a.png")] * 3, spec)
    assert batch.shape == (3, 3, 32, 32)
    assert batch.dtype.is_floating_point


def test_to_tensor_handles_empty_list():
    spec = transform_spec("imagenet")
    assert to_tensor([], spec).shape == (0, 3, 224, 224)


def test_thumbnail_is_an_inline_data_uri():
    uri = thumbnail_data_uri(decode_image(_png_bytes((500, 400)), "a.png"))
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) < 40_000, "thumbnails must stay small enough to inline"


def test_ingest_produces_one_tensor_per_family():
    specs = {"imagenet": transform_spec("imagenet"), "clip": transform_spec("clip")}
    tensors, thumb, size = ingest(_png_bytes((64, 64)), "a.png", specs)
    assert set(tensors) == {"imagenet", "clip"}
    assert tensors["imagenet"].shape == (3, 224, 224)
    assert thumb.startswith("data:image/jpeg;base64,")
    assert size == (64, 64)


def test_ingest_families_differ_because_normalisation_differs():
    specs = {"imagenet": transform_spec("imagenet"), "clip": transform_spec("clip")}
    tensors, _, _ = ingest(_png_bytes((64, 64)), "a.png", specs)
    import torch

    assert not torch.allclose(tensors["imagenet"], tensors["clip"])
