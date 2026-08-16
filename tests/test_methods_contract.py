"""
The logits contract, and the transductive-leakage guarantee.

Siamese and Matching originally emitted values in [0, 1] (sigmoid output and a
softmax-weighted one-hot respectively) and fed them to CrossEntropyLoss, which
applies log_softmax on top. A softmax over a range of width <= 1 is nearly
uniform, so both baselines trained on tiny, badly-scaled gradients -- in the very
comparison they existed to provide.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from fsgrade.models.episodic import (
    MatchingNetwork,
    PrototypicalNetwork,
    SiameseNetwork,
    SupervisedClassifier,
    class_prototypes,
)

EPISODIC = ["protonet", "siamese", "matching"]


def _build(kind: str):
    common = dict(backbone="resnet18", embedding_dim=32, pretrained=False, dropout=0.0)
    if kind == "protonet":
        return PrototypicalNetwork(**common)
    if kind == "siamese":
        return SiameseNetwork(**common)
    return MatchingNetwork(**common)


def _episode(n_shot=3, n_query=4, size=32):
    torch.manual_seed(0)
    sx = torch.randn(2 * n_shot, 3, size, size)
    sy = torch.tensor([0] * n_shot + [1] * n_shot)
    qx = torch.randn(2 * n_query, 3, size, size)
    qy = torch.tensor([0] * n_query + [1] * n_query)
    return sx, sy, qx, qy


@pytest.mark.parametrize("kind", EPISODIC)
def test_output_shape_and_dtype(kind):
    model = _build(kind).eval()
    sx, sy, qx, qy = _episode()
    out = model(sx, sy, qx, 2)
    assert out.logits.shape == (qx.size(0), 2)
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("kind", EPISODIC)
def test_logits_are_shift_invariant_in_argmax(kind):
    """A defining property of logits: adding a constant cannot change the decision."""
    model = _build(kind).eval()
    sx, sy, qx, _ = _episode()
    with torch.no_grad():
        logits = model(sx, sy, qx, 2).logits
    assert torch.equal(logits.argmax(1), (logits + 7.5).argmax(1))


@pytest.mark.parametrize("kind", EPISODIC)
def test_logits_are_not_probabilities(kind):
    """Rows must not sum to 1 -- that would mean a probability vector was
    returned, which is exactly the original Matching Network bug."""
    model = _build(kind).eval()
    sx, sy, qx, _ = _episode()
    with torch.no_grad():
        logits = model(sx, sy, qx, 2).logits
    row_sums = logits.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


@pytest.mark.parametrize("kind", EPISODIC)
def test_cross_entropy_produces_encoder_gradients(kind):
    """The broken baselines produced vanishing gradients through the encoder."""
    model = _build(kind).train()
    sx, sy, qx, qy = _episode()
    loss = nn.CrossEntropyLoss()(model(sx, sy, qx, 2).logits, qy)
    model.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters()
             if p.requires_grad and p.grad is not None]
    assert grads, "no gradient reached the encoder"
    assert sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize("kind", EPISODIC)
def test_loss_decreases_on_a_separable_episode(kind):
    """Sanity: each method can fit a trivially separable task."""
    torch.manual_seed(0)
    model = _build(kind).train()
    size = 32
    sx = torch.cat([torch.zeros(3, 3, size, size), torch.ones(3, 3, size, size)])
    sy = torch.tensor([0] * 3 + [1] * 3)
    qx = torch.cat([torch.zeros(4, 3, size, size), torch.ones(4, 3, size, size)])
    qy = torch.tensor([0] * 4 + [1] * 4)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    first = last = None
    for step in range(25):
        opt.zero_grad()
        loss = ce(model(sx, sy, qx, 2).logits, qy)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, f"{kind}: loss did not decrease ({first:.4f} -> {last:.4f})"


def test_separate_passes_prevent_query_to_support_leakage():
    """With BatchNorm in the head, a concatenated forward pass lets query batch
    statistics change the support embeddings. Dropout is disabled so BatchNorm
    is the only possible coupling."""
    torch.manual_seed(0)
    sx, sy, _, _ = _episode()
    q_a = torch.randn(8, 3, 32, 32)
    q_b = torch.randn(8, 3, 32, 32) * 5 + 3

    for transductive, expect_leak in ((False, False), (True, True)):
        model = PrototypicalNetwork(
            backbone="resnet18", embedding_dim=32, pretrained=False, dropout=0.0,
            use_batchnorm=True, transductive=transductive,
        )
        model.train()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            emb_a = model(sx, sy, q_a, 2).support_embeddings.clone()
        model.load_state_dict(state)
        with torch.no_grad():
            emb_b = model(sx, sy, q_b, 2).support_embeddings.clone()

        delta = float((emb_a - emb_b).abs().max())
        if expect_leak:
            assert delta > 1e-6, "transductive path should couple support to query"
        else:
            assert delta == pytest.approx(0.0, abs=1e-6), (
                f"support embeddings changed by {delta} when only the query set changed"
            )


def test_class_prototypes_match_manual_means():
    emb = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])
    protos = class_prototypes(emb, labels, 2)
    assert torch.allclose(protos[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(protos[1], torch.tensor([0.0, 3.0]))


def test_supervised_classifier_needs_no_support_set():
    """The pivotal control: predicts from the query image alone."""
    model = SupervisedClassifier(
        backbone="resnet18", pretrained=False, embedding_dim=32, dropout=0.0
    ).eval()
    with torch.no_grad():
        logits = model(torch.randn(6, 3, 32, 32))
    assert logits.shape == (6, 2)


def test_species_adversary_reverses_gradient():
    from fsgrade.training.grl import dann_lambda, grad_reverse

    x = torch.randn(4, 8, requires_grad=True)
    grad_reverse(x, 1.0).sum().backward()
    reversed_grad = x.grad.clone()

    x2 = torch.randn(4, 8, requires_grad=True)
    x2.data = x.data.clone()
    x2.sum().backward()
    assert torch.allclose(reversed_grad, -x2.grad)

    assert dann_lambda(0.0) == pytest.approx(0.0)
    assert 0.0 < dann_lambda(0.5) < 1.0
    assert dann_lambda(1.0) > 0.99
