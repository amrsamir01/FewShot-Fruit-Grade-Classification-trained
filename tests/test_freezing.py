"""
Freeze-depth tests.

The literal parameter-count ladder is asserted so the original substring bug --
which froze 4,805,952 parameters instead of the intended 157,504 -- cannot
silently return.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from torchvision import models

from fsgrade.models.freezing import (
    FreezeSpec,
    apply_freeze,
    set_frozen_bn_eval,
    stage_module_names,
)

# Cumulative frozen parameters when freezing up to each stage (resnet18, fc removed).
RESNET18_LADDER = {
    -1: 0,
    0: 9_536,           # conv1 + bn1
    1: 157_504,         # + layer1
    2: 683_072,         # + layer2
    3: 2_782_784,       # + layer3
    4: 11_176_512,      # + layer4 == the entire backbone
}

# What the original substring matcher actually froze.
ORIGINAL_BUG_FROZEN = 4_805_952


def _resnet18():
    enc = models.resnet18(weights=None)
    enc.fc = nn.Identity()
    return enc


@pytest.mark.parametrize("stage,expected", sorted(RESNET18_LADDER.items()))
def test_freeze_ladder_exact_counts(stage, expected):
    enc = _resnet18()
    spec = FreezeSpec(strategy="up_to_stage" if stage >= 0 else "none", stage=stage)
    report = apply_freeze(enc, "resnet18", spec)
    assert report.frozen_params == expected, (
        f"stage {stage}: expected {expected:,} frozen, got {report.frozen_params:,}"
    )
    assert report.frozen_params + report.trainable_params == report.total_params


def test_full_backbone_freeze_equals_total():
    enc = _resnet18()
    total = sum(p.numel() for p in enc.parameters())
    report = apply_freeze(enc, "resnet18", FreezeSpec(strategy="up_to_stage", stage=4))
    assert report.frozen_params == total == 11_176_512


def test_original_substring_bug_is_not_reproduced():
    """The buggy value must not be reachable from any coherent stage setting."""
    enc = _resnet18()
    buggy = sum(
        p.numel() for name, p in enc.named_parameters()
        if any(s in name for s in ("layer1", "conv1", "bn1"))
    )
    assert buggy == ORIGINAL_BUG_FROZEN, "bug reproduction changed; update the reference"
    assert buggy not in RESNET18_LADDER.values(), (
        "the buggy freeze size corresponds to no coherent architectural choice"
    )
    # It sits between stage 3 and stage 4, confirming it is not a valid depth.
    assert RESNET18_LADDER[3] < buggy < RESNET18_LADDER[4]


def test_stage_names_are_cumulative():
    assert stage_module_names("resnet18", 0) == ["conv1", "bn1"]
    assert stage_module_names("resnet18", 1) == ["conv1", "bn1", "layer1"]
    assert stage_module_names("resnet18", -1) == []
    for stage in range(4):
        assert set(stage_module_names("resnet18", stage)).issubset(
            set(stage_module_names("resnet18", stage + 1))
        )


def test_frozen_batchnorm_stats_are_tagged_and_eval():
    """requires_grad=False does not stop running-stat updates; the tag does."""
    enc = _resnet18()
    report = apply_freeze(
        enc, "resnet18", FreezeSpec(stage=1, freeze_bn_stats=True)
    )
    assert report.frozen_bn_modules > 0

    enc.train()
    n = set_frozen_bn_eval(enc)
    assert n == report.frozen_bn_modules
    assert not enc.bn1.training, "frozen BatchNorm must be in eval mode during training"
    assert enc.layer2[0].bn1.training, "unfrozen BatchNorm must remain in train mode"


def test_freeze_bn_stats_disabled_leaves_bn_training():
    enc = _resnet18()
    apply_freeze(enc, "resnet18", FreezeSpec(stage=1, freeze_bn_stats=False))
    enc.train()
    set_frozen_bn_eval(enc)
    assert enc.bn1.training


def test_unknown_backbone_raises():
    with pytest.raises(KeyError):
        stage_module_names("not_a_backbone", 1)
