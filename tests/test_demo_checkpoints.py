"""
Discovery and checkpoint-loading tests.

These cover the three traps that make loading a trained model non-trivial:
Windows-flavoured paths recorded at training time, the supervised checkpoint
shared under a different method name, and the fact that checkpoints carry no
architecture metadata at all.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from fsgrade.demo.checkpoints import (
    CheckpointError,
    build_module,
    infer_backbone,
    inspect_state_dict,
    load_checkpoint_file,
    load_for_inference,
)
from fsgrade.demo.discovery import (
    NO_CHECKPOINT,
    best_method_pair,
    discover_runs,
    find_checkpoint,
    load_run,
    normalise_checkpoint_path,
)

CONFIG = {
    "model": {
        "backbone": "resnet18", "embedding_dim": 32, "pretrained": True,
        "dropout": 0.0, "use_batchnorm": False, "transductive_forward": False,
        "freeze": {"strategy": "up_to_stage", "stage": 3, "freeze_bn_stats": True},
        "temperature": {"init": 0.5},
    },
    "data": {"classes": ["fresh", "rotten"]},
    "protocol": {"episodes": {"n_way": 2}},
}


# ------------------------------------------------------------ path traps --- #

def test_windows_path_is_reanchored_to_the_run_dir(tmp_path):
    """Recorded paths use backslashes and the training-time CWD."""
    run = tmp_path / "run1"
    ckpt = run / "checkpoints" / "loso-mango" / "ours" / "abc123.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    recorded = r"results\smoke\some_other_run\checkpoints\loso-mango\ours\abc123.pt"
    assert normalise_checkpoint_path(recorded, run) == ckpt


def test_normalise_handles_none_and_missing():
    assert normalise_checkpoint_path(None, pathlib.Path(".")) is None
    out = normalise_checkpoint_path(r"nowhere\checkpoints\f\m\h.pt", pathlib.Path("/tmp"))
    assert out is not None and not out.exists()


def test_supervised_checkpoint_is_shared_across_three_methods(tmp_path):
    """zeroshot/ncc/finetune all read checkpoints/<fold>/supervised/<hash>.pt."""
    run = tmp_path / "run"
    shared = run / "checkpoints" / "loso-mango" / "supervised" / "deadbeef.pt"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"x")

    for method in ("zeroshot_supervised", "ncc_supervised", "finetune_supervised"):
        assert find_checkpoint(run, "loso-mango", method) == shared

    assert find_checkpoint(run, "loso-mango", "ours") is None


def test_find_checkpoint_prefers_the_matching_spec_hash(tmp_path):
    run = tmp_path / "run"
    d = run / "checkpoints" / "f1" / "ours"
    d.mkdir(parents=True)
    (d / "aaa.pt").write_bytes(b"x")
    (d / "bbb.pt").write_bytes(b"x")
    assert find_checkpoint(run, "f1", "ours", "bbb").name == "bbb.pt"
    assert find_checkpoint(run, "f1", "ours", "zzz").name == "aaa.pt"   # falls back


# -------------------------------------------------------------- discovery -- #

def _write_run(root: pathlib.Path, experiment: str, run_id: str, methods: dict) -> pathlib.Path:
    run = root / experiment / run_id
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "experiment": experiment, "chance_level": 0.5,
        "protocol": {"name": "fixed", "folds": ["f1"]},
        "methods": methods,
    }), encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"status": "ok", "started_utc": "2026-08-03T00:00:00Z"}), encoding="utf-8"
    )
    return run


def test_methods_without_weights_are_always_available(tmp_path):
    run = _write_run(tmp_path, "exp", "r1", {
        "chance": {"n_episodes": 10, "state": {}},
        "sap": {"n_episodes": 10, "state": {"kappa": 8.0}},
        "nc_pixel": {"n_episodes": 10, "state": {}},
    })
    ref = load_run(run)
    for name in ("chance", "sap", "nc_pixel"):
        assert ref.methods[name].available
        assert not ref.methods[name].needs_checkpoint
        assert name in NO_CHECKPOINT


def test_method_with_missing_checkpoint_is_unavailable_with_a_reason(tmp_path):
    run = _write_run(tmp_path, "exp", "r1", {
        "ours": {"n_episodes": 10, "state": {"checkpoint": r"gone\checkpoints\f1\ours\h.pt"}},
    })
    ref = load_run(run)
    assert not ref.methods["ours"].available
    assert "not found" in ref.methods["ours"].reason


def test_failed_runs_are_hidden_by_default(tmp_path):
    run = _write_run(tmp_path, "exp", "r1", {"chance": {"state": {}}})
    (run / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert discover_runs(tmp_path) == []
    assert len(discover_runs(tmp_path, include_failed=True)) == 1


def test_underscore_directories_are_skipped(tmp_path):
    (tmp_path / "_episodes").mkdir()
    (tmp_path / "_episodes" / "metrics.json").write_text("{}", encoding="utf-8")
    _write_run(tmp_path, "exp", "r1", {"chance": {"state": {}}})
    assert [r.experiment for r in discover_runs(tmp_path)] == ["exp"]


def test_missing_results_root_returns_empty(tmp_path):
    assert discover_runs(tmp_path / "nope") == []


def test_kappa_is_read_never_fitted(tmp_path):
    """Calibrating kappa on the demo species would leak target information."""
    run = _write_run(tmp_path, "exp", "r1", {
        "sap": {"state": {"kappa": 8.0, "calibration": {
            "parameter": "kappa", "value": 8.0,
            "n_val_episodes": 200, "val_accuracy": 0.871,
        }}},
    })
    ref = load_run(run).methods["sap"]
    assert ref.kappa == 8.0
    assert "calibrated on seen species" in ref.kappa_source
    assert "200 val episodes" in ref.kappa_source


def test_kappa_absent_reports_no_source(tmp_path):
    run = _write_run(tmp_path, "exp", "r1", {"sap": {"state": {}}})
    ref = load_run(run).methods["sap"]
    assert ref.kappa is None and ref.kappa_source == ""


def test_default_pair_prefers_the_thesis_comparison(tmp_path):
    run = _write_run(tmp_path, "exp", "r1", {
        "ours": {"state": {}}, "zeroshot_supervised": {"state": {}},
    })
    ref = load_run(run)
    for m in ref.methods.values():          # pretend their checkpoints resolved
        m.available = True
    assert best_method_pair([ref]) == ("ours", "zeroshot_supervised")


def test_default_pair_falls_back_to_the_training_free_pair(tmp_path):
    """With no runs at all, SAP vs CLIP-text still works -- neither needs weights."""
    assert best_method_pair([]) == ("sap", "clip_text_zeroshot")


# --------------------------------------------------- architecture recovery - #

def test_inspect_recovers_shape_facts_from_an_episodic_model():
    from fsgrade.models.episodic import PrototypicalNetwork

    m = PrototypicalNetwork(backbone="resnet18", embedding_dim=64, pretrained=False)
    facts = inspect_state_dict(m.state_dict())
    assert facts["embedding_dim"] == 64
    assert facts["backbone_features"] == 512
    assert facts["has_temperature"] is True
    assert facts["has_classifier"] is False
    assert facts["uses_batchnorm_head"] is False


def test_inspect_detects_batchnorm_head():
    from fsgrade.models.episodic import PrototypicalNetwork

    m = PrototypicalNetwork(
        backbone="resnet18", embedding_dim=32, pretrained=False, use_batchnorm=True
    )
    assert inspect_state_dict(m.state_dict())["uses_batchnorm_head"] is True


def test_inspect_recovers_supervised_heads():
    from fsgrade.models.episodic import SupervisedClassifier

    m = SupervisedClassifier(
        backbone="resnet18", embedding_dim=32, pretrained=False, n_classes=2, n_species=3
    )
    facts = inspect_state_dict(m.state_dict())
    assert facts["n_classes"] == 2 and facts["n_species"] == 3
    assert facts["has_classifier"] and facts["has_species_head"]


def test_backbone_inferred_from_feature_width():
    assert infer_backbone({"backbone_features": 512}, "resnet18")[0] == "resnet18"
    assert infer_backbone({"backbone_features": 2048}, "resnet50")[0] == "resnet50"
    assert infer_backbone({"backbone_features": 1280}, "efficientnet_b0")[0] == "efficientnet_b0"


def test_checkpoint_wins_when_config_disagrees():
    """A config edited after training must not break the load."""
    backbone, conflicts = infer_backbone({"backbone_features": 2048}, "resnet18")
    assert backbone == "resnet50"
    assert conflicts and "resnet50" in conflicts[0]


def test_build_module_never_requests_pretrained_weights(monkeypatch):
    """Downloading ImageNet weights would break the offline guarantee."""
    import fsgrade.models.backbones as backbones

    seen = {}
    original = backbones.build_backbone

    def spy(name, *, pretrained=True):
        seen["pretrained"] = pretrained
        return original(name, pretrained=False)

    monkeypatch.setattr(backbones, "build_backbone", spy)
    build_module("ours", CONFIG, {"embedding_dim": 32, "backbone_features": 512})
    assert seen["pretrained"] is False


# ------------------------------------------------------------ round trip --- #

@pytest.mark.parametrize("method,cls", [
    ("ours", "PrototypicalNetwork"),
    ("protonet", "PrototypicalNetwork"),
    ("siamese", "SiameseNetwork"),
    ("matching", "MatchingNetwork"),
    ("zeroshot_supervised", "SupervisedClassifier"),
])
def test_round_trip_every_method(tmp_path, method, cls):
    """Save a model, reload it through the loader, and get identical outputs."""
    module, kind, _ = build_module(method, CONFIG, {})
    path = tmp_path / f"{method}.pt"
    torch.save({
        "epoch": 7, "model_state_dict": module.state_dict(),
        "spec_hash": "abc123", "metrics": {"val_balanced_accuracy": 0.83},
    }, path)

    loaded = load_for_inference(method, path, CONFIG, device="cpu")
    assert loaded.card.module_class == cls
    assert loaded.card.epoch == 7
    assert loaded.card.best_val == pytest.approx(0.83)
    assert not loaded.module.training, "model must come back in eval mode"
    assert all(not p.requires_grad for p in loaded.module.parameters())

    torch.manual_seed(0)
    sx, sy = torch.randn(4, 3, 32, 32), torch.tensor([0, 0, 1, 1])
    qx = torch.randn(6, 3, 32, 32)
    module.eval()
    with torch.no_grad():
        if kind == "supervised":
            a, b = module(qx), loaded.module(qx)
        else:
            a, b = module(sx, sy, qx, 2).logits, loaded.module(sx, sy, qx, 2).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_missing_file_raises_with_remediation(tmp_path):
    with pytest.raises(CheckpointError) as exc:
        load_checkpoint_file(tmp_path / "nope.pt")
    assert exc.value.remediation


def test_bare_state_dict_is_accepted(tmp_path):
    from fsgrade.models.episodic import PrototypicalNetwork

    m = PrototypicalNetwork(backbone="resnet18", embedding_dim=32, pretrained=False)
    path = tmp_path / "bare.pt"
    torch.save(m.state_dict(), path)
    assert "model_state_dict" in load_checkpoint_file(path)


def test_non_checkpoint_payload_is_rejected(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"hello": "world"}, path)
    with pytest.raises(CheckpointError, match="does not look like"):
        load_checkpoint_file(path)


def test_architecture_mismatch_explains_itself(tmp_path):
    """A wrong config must produce an actionable message, not a torch wall of text."""
    module, _, _ = build_module("ours", CONFIG, {})
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": module.state_dict()}, path)

    wrong = {**CONFIG, "model": {**CONFIG["model"], "embedding_dim": 999}}
    # embedding_dim is recovered from the state dict, so force a real conflict
    # by asking for a structurally different method instead.
    with pytest.raises(CheckpointError) as exc:
        load_for_inference("siamese", path, wrong, device="cpu")
    text = str(exc.value)
    assert "does not match" in text
    assert "missing" in text or "unexpected" in text
    assert exc.value.remediation


def test_legacy_checkpoint_is_diagnosed(tmp_path):
    """Legacy src/ checkpoints use encoder.encoder.* and must say so."""
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {
        "encoder.encoder.conv1.weight": torch.zeros(64, 3, 7, 7),
        "encoder.projection.0.weight": torch.zeros(512, 512),
    }}, path)
    with pytest.raises(CheckpointError, match="LEGACY"):
        load_for_inference("ours", path, CONFIG, device="cpu")
