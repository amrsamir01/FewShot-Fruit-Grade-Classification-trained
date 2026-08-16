"""
The method registry -- single source of truth for the baseline ladder.

Adding a method here makes it available to every experiment script. The ladder
is deliberately ordered from weakest to strongest so results tables read as a
progression, and so the *interesting* comparisons are adjacent:

    chance                -> what 50% looks like
    nc_pixel              -> training-free floor
    zeroshot_supervised   -> NO target labels           <-- the pivotal control
    ncc_supervised        -> same encoder + K-shot centroid
    probe_supervised      -> same encoder + K-shot linear probe
    finetune_supervised   -> same encoder + K-shot fine-tune
    siamese / matching    -> episodic metric baselines (corrected)
    protonet/_temp/ours   -> prototypical family
    clip_text_zeroshot    -> language only, NO images of the unseen species
    dinov2_* / clip_*     -> frozen foundation features
    sap                   -> text-anchored prototypes (applied, not novel)

CAVEAT: ``ours`` is presently an alias of ``protonet_temp`` -- identical class,
identical kwargs (see ``EpisodicMethod.MODULES``). Any table placing them on
adjacent rows is reporting seed noise as a method delta. Resolve before writing
the results chapter.
"""

from __future__ import annotations

from typing import Any, Callable

from fsgrade.methods.base import BaseMethod

_REGISTRY: dict[str, Callable[..., BaseMethod]] = {}


def register(name: str, builder: Callable[..., BaseMethod]) -> None:
    _REGISTRY[name] = builder


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_method(name: str, **kwargs: Any) -> BaseMethod:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method {name!r}. Available: {available()}")
    method = _REGISTRY[name](**kwargs)
    method.name = name
    return method


def _register_all() -> None:
    from fsgrade.methods.frozen import (
        LinearProbe,
        NearestCentroid,
        PixelNearestCentroid,
        SpeciesAnchoredPrototypes,
        ZeroShotText,
    )
    from fsgrade.methods.trained import (
        ChanceMethod,
        EpisodicMethod,
        SupervisedFineTune,
        SupervisedNearestCentroid,
        ZeroShotSupervised,
    )

    register("chance", ChanceMethod)
    register("nc_pixel", PixelNearestCentroid)

    # Supervised transfer ladder -- all four share one checkpoint per fold.
    register("zeroshot_supervised", ZeroShotSupervised)
    register("ncc_supervised", SupervisedNearestCentroid)
    register("finetune_supervised", SupervisedFineTune)

    # Episodic baselines (Siamese/Matching corrected to emit real logits).
    for variant in ("protonet", "protonet_temp", "siamese", "matching", "ours"):
        register(variant, lambda v=variant, **kw: EpisodicMethod(variant=v, **kw))

    # Frozen foundation features.
    register("dinov2_ncc", lambda **kw: NearestCentroid(encoder_name="dinov2_vits14", **kw))
    register("dinov2_probe", lambda **kw: LinearProbe(encoder_name="dinov2_vits14", **kw))
    register("clip_ncc", lambda **kw: NearestCentroid(encoder_name="clip_vitb16", **kw))
    register("clip_probe", lambda **kw: LinearProbe(encoder_name="clip_vitb16", **kw))

    # Vision-language: zero-shot text, and the proposed method.
    register("clip_text_zeroshot", ZeroShotText)
    register("sap", SpeciesAnchoredPrototypes)


_register_all()


# Ordered ladder used by default when a config says ``methods: all``.
LADDER: list[str] = [
    "chance",
    "nc_pixel",
    "zeroshot_supervised",
    "ncc_supervised",
    "siamese",
    "matching",
    "protonet",
    "protonet_temp",
    "ours",
    "clip_text_zeroshot",
    "dinov2_ncc",
    "dinov2_probe",
    "clip_ncc",
    "clip_probe",
    "sap",
]

# Methods safe to run without optional dependencies (no timm / open_clip).
CORE_LADDER: list[str] = [
    "chance",
    "nc_pixel",
    "zeroshot_supervised",
    "ncc_supervised",
    "siamese",
    "matching",
    "protonet",
    "protonet_temp",
    "ours",
]


def resolve_method_list(spec: Any) -> list[str]:
    """Turn a config value into a concrete method list."""
    if spec in (None, "all"):
        return list(LADDER)
    if spec == "core":
        return list(CORE_LADDER)
    if isinstance(spec, str):
        return [spec]
    return list(spec)
