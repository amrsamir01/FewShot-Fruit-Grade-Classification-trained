"""
Methods that train an encoder on the seen species.

Contains the single most important baseline in the thesis:
``ZeroShotSupervised``. The label set (fresh/rotten) is *fixed* across species,
so nothing about the task requires a support set. A plain binary classifier
trained on the seen species and applied directly to an unseen one -- with no
target labels at all -- is the control that determines whether few-shot
adaptation is buying anything.

The original repository never ran it. Its own N-shot ablation hints at the
answer: accuracy moved only 84.3% -> 86.6% going from 1-shot to 10-shot, a
+2.3-point return on ten times the support data. If the support set barely
matters, the prototypical machinery has to justify itself.

``ZeroShotSupervised`` and ``SupervisedNearestCentroid`` deliberately **share one
checkpoint per fold**, which makes the comparison exact: same weights, same
features, the only difference being whether the 5-shot target support is used.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from fsgrade.data.loaders import EpisodeBatch
from fsgrade.methods.base import BaseMethod, EpisodeOutput, FitContext


class EpisodicMethod(BaseMethod):
    """Wraps an episodically-trained module (ProtoNet / Siamese / Matching)."""

    family = "episodic"
    trains_episodically = True
    adapts_per_episode = False

    # WARNING -- "ours" and "protonet_temp" are currently IDENTICAL: same class,
    # same kwargs, and no config in configs/ or scripts/run_all.sh distinguishes
    # them. They differ only by the seed derived from the method name.
    #
    # Consequence: a reported `ours` vs `protonet_temp` delta measures
    # training-run noise, not a method difference, and must not be presented as
    # an ablation. Either give "ours" a real differentiating factor here, or drop
    # it from the ladder and use "protonet_temp" as the reference method.
    # (`stats.reference_method: ours` in configs/base.yaml, and every E4/E6 arm
    # in scripts/run_all.sh, currently point at it.)
    MODULES = {
        "protonet": ("PrototypicalNetwork", {"use_temperature": False}),
        "protonet_temp": ("PrototypicalNetwork", {"use_temperature": True}),
        "ours": ("PrototypicalNetwork", {"use_temperature": True}),  # == protonet_temp
        "siamese": ("SiameseNetwork", {}),
        "matching": ("MatchingNetwork", {}),
    }

    def __init__(self, variant: str = "ours", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.variant = variant
        self.name = kwargs.get("name", variant)
        self.train_summary: dict[str, Any] = {}

    def _build(self, cfg: dict[str, Any]) -> torch.nn.Module:
        from fsgrade.config import get_in
        from fsgrade.models import episodic as ep
        from fsgrade.models.freezing import FreezeSpec

        cls_name, defaults = self.MODULES[self.variant]
        cls = getattr(ep, cls_name)

        kwargs: dict[str, Any] = {
            "backbone": get_in(cfg, "model.backbone", "resnet18"),
            "embedding_dim": int(get_in(cfg, "model.embedding_dim", 256)),
            "pretrained": bool(get_in(cfg, "model.pretrained", True)),
            "dropout": float(get_in(cfg, "model.dropout", 0.4)),
            "freeze": FreezeSpec.from_dict(get_in(cfg, "model.freeze", {})),
            "transductive": bool(get_in(cfg, "model.transductive_forward", False)),
        }
        kwargs.update(defaults)
        if cls_name == "PrototypicalNetwork":
            kwargs["temperature"] = float(get_in(cfg, "model.temperature.init", 0.5))
            kwargs["use_batchnorm"] = bool(get_in(cfg, "model.use_batchnorm", False))
        kwargs.update({k: v for k, v in self.config.items() if k in kwargs})
        return cls(**kwargs)

    # ------------------------------------------------------------------ #
    def prepare(self, ctx: FitContext) -> None:
        from torch.utils.data import DataLoader

        from fsgrade.config import get_in
        from fsgrade.data.loaders import EpisodeDataset, build_transforms, episode_collate
        from fsgrade.training.harness import Trainer, TrainSpec
        from fsgrade.training.losses import PrototypicalLoss

        self.device = ctx.device
        self.model = self._build(ctx.cfg).to(ctx.device)

        spec = TrainSpec.from_config(
            ctx.cfg,
            mode="episodic",
            contrastive_weight=float(self.config.get(
                "contrastive_weight", get_in(ctx.cfg, "train.contrastive_weight", 0.1)
            )),
        )
        loss_fn = PrototypicalLoss(
            label_smoothing=spec.label_smoothing,
            contrastive_weight=spec.contrastive_weight,
            temperature=spec.supcon_temperature,
        )

        tfs = build_transforms(ctx.cfg)
        n_way = int(get_in(ctx.cfg, "protocol.episodes.n_way", 2))
        k = int(get_in(ctx.cfg, "protocol.episodes.n_shot", 5))
        workers = int(get_in(ctx.cfg, "num_workers", 0))

        def make(bank, support_tf, query_tf):
            ds = EpisodeDataset(
                bank, ctx.data_root,
                support_transform=support_tf, query_transform=query_tf, k_shot=k,
            )
            return DataLoader(
                ds, batch_size=1, shuffle=False, num_workers=workers, collate_fn=episode_collate
            )

        ckpt = None
        if ctx.run is not None:
            ckpt = ctx.run.checkpoint_path(ctx.fold.fold_id, self.name, spec.spec_hash())

        trainer = Trainer(
            self.model, loss_fn, spec,
            device=ctx.device, logger=ctx.logger, checkpoint_path=ckpt, n_way=n_way,
        )
        self.train_summary = trainer.fit(
            train_loader_fn=lambda epoch: make(ctx.train_bank, tfs["support"], tfs["train"]),
            val_loader_fn=lambda: make(ctx.val_bank, tfs["support"], tfs["eval"]),
        )
        self.model.eval()
        self._summary.update({
            "variant": self.variant,
            "train_spec_hash": self.train_summary["spec_hash"],
            "best_val": self.train_summary["best_score"],
            "checkpoint": str(ckpt) if ckpt else None,
            "freeze_report": self.train_summary.get("freeze_report"),
        })
        self.n_way = n_way

    @torch.no_grad()
    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        t0 = time.perf_counter()
        self.model.eval()
        b = batch.to(self.device)
        out = self.model(b.support_x, b.support_y, b.query_x, batch.n_way)
        return EpisodeOutput(
            logits=self._to_numpy(out.logits),
            query_embeddings=self._to_numpy(out.query_embeddings),
            support_embeddings=self._to_numpy(out.support_embeddings),
            prototypes=self._to_numpy(out.prototypes) if out.prototypes is not None else None,
            adapt_seconds=time.perf_counter() - t0,
        )


class SupervisedBackboneMixin:
    """Trains (or reuses) a plain binary classifier on the seen species."""

    _SHARED: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}

    def _train_supervised(self, ctx: FitContext) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Train once per (fold, train-spec) and share across dependent methods.

        Sharing is what makes 'zero-shot vs centroid vs probe vs fine-tune' an
        exact comparison rather than three separately-trained encoders.
        """
        from torch.utils.data import DataLoader

        from fsgrade.config import get_in
        from fsgrade.data.loaders import FlatImageDataset, build_transforms
        from fsgrade.models.episodic import SupervisedClassifier
        from fsgrade.models.freezing import FreezeSpec
        from fsgrade.training.harness import Trainer, TrainSpec
        from fsgrade.training.losses import SupervisedLoss

        adv = float(self.config.get(
            "adversarial_weight", get_in(ctx.cfg, "train.adversarial_weight", 0.0)
        ))
        spec = TrainSpec.from_config(
            ctx.cfg,
            mode="supervised",
            epochs=int(get_in(ctx.cfg, "train.supervised_epochs",
                              get_in(ctx.cfg, "train.epochs", 30))),
            adversarial_weight=adv,
        )
        key = f"{ctx.fold.fold_id}|{spec.spec_hash()}|adv{adv}"
        if key in self._SHARED:
            if ctx.logger:
                ctx.logger.info("reusing shared supervised encoder for %s", key)
            return self._SHARED[key]

        species = list(ctx.fold.train_species)
        model = SupervisedClassifier(
            backbone=get_in(ctx.cfg, "model.backbone", "resnet18"),
            n_classes=len(ctx.index.classes),
            pretrained=bool(get_in(ctx.cfg, "model.pretrained", True)),
            dropout=float(get_in(ctx.cfg, "model.dropout", 0.4)),
            freeze=FreezeSpec.from_dict(get_in(ctx.cfg, "model.freeze", {})),
            embedding_dim=int(get_in(ctx.cfg, "model.embedding_dim", 256)),
            n_species=len(species) if adv > 0 else 0,
        ).to(ctx.device)

        tfs = build_transforms(ctx.cfg)
        workers = int(get_in(ctx.cfg, "num_workers", 0))
        train_ds = FlatImageDataset(ctx.index, ctx.train_pools, species=species,
                                    transform=tfs["train"])
        val_ds = FlatImageDataset(ctx.index, ctx.val_pools, species=species,
                                  transform=tfs["eval"])

        def train_loader(_epoch: int):
            return DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True,
                              num_workers=workers, drop_last=True)

        def val_loader():
            return DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False,
                              num_workers=workers)

        ckpt = None
        if ctx.run is not None:
            ckpt = ctx.run.checkpoint_path(ctx.fold.fold_id, "supervised", spec.spec_hash())

        trainer = Trainer(model, SupervisedLoss(spec.label_smoothing, adv), spec,
                          device=ctx.device, logger=ctx.logger, checkpoint_path=ckpt)
        summary = trainer.fit(train_loader, val_loader)
        summary["checkpoint"] = str(ckpt) if ckpt else None
        summary["n_train_images"] = len(train_ds)
        summary["adversarial_weight"] = adv
        model.eval()

        self._SHARED[key] = (model, summary)
        return model, summary

    @torch.no_grad()
    def _embed(self, model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
        return model.embed(images.to(self.device)).detach().cpu().float().numpy()


class ZeroShotSupervised(BaseMethod, SupervisedBackboneMixin):
    """Binary classifier trained on seen species, applied with NO support set.

    The control that determines whether the thesis has a subject.
    """

    name = "zeroshot_supervised"
    family = "zero_shot"
    requires_support = False
    trains_episodically = False
    adapts_per_episode = False

    def prepare(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self.model, summary = self._train_supervised(ctx)
        self._summary.update({
            "train_spec_hash": summary["spec_hash"],
            "best_val": summary["best_score"],
            "checkpoint": summary.get("checkpoint"),
            "shared_encoder": True,
            "n_support_used": 0,
            "freeze_report": summary.get("freeze_report"),
        })

    @torch.no_grad()
    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        t0 = time.perf_counter()
        self.model.eval()
        logits = self.model(batch.query_x.to(self.device))
        return EpisodeOutput(
            logits=self._to_numpy(logits),
            adapt_seconds=time.perf_counter() - t0,
            extras={"n_support_used": 0},
        )


class SupervisedNearestCentroid(BaseMethod, SupervisedBackboneMixin):
    """Same encoder as ZeroShotSupervised, but a centroid over the target support.

    Isolates exactly what the K labelled target images contribute.
    """

    name = "ncc_supervised"
    family = "transfer"
    requires_support = True
    adapts_per_episode = True

    def __init__(self, scale: float = 10.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def prepare(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self.model, summary = self._train_supervised(ctx)
        self._summary.update({
            "train_spec_hash": summary["spec_hash"],
            "shared_encoder_with": "zeroshot_supervised",
            "freeze_report": summary.get("freeze_report"),
        })

    @torch.no_grad()
    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        import torch.nn.functional as F

        t0 = time.perf_counter()
        self.model.eval()
        b = batch.to(self.device)
        se = F.normalize(self.model.embed(b.support_x), dim=1)
        qe = F.normalize(self.model.embed(b.query_x), dim=1)
        protos = torch.stack([
            F.normalize(se[b.support_y == c].mean(0), dim=0) for c in range(batch.n_way)
        ])
        return EpisodeOutput(
            logits=self._to_numpy((qe @ protos.t()) * self.scale),
            query_embeddings=self._to_numpy(qe),
            support_embeddings=self._to_numpy(se),
            prototypes=self._to_numpy(protos),
            adapt_seconds=time.perf_counter() - t0,
        )


class SupervisedFineTune(BaseMethod, SupervisedBackboneMixin):
    """Per-episode fine-tuning on the K target support images.

    The transfer baseline of Chen et al. (2019) and Guo et al. (2020), which
    showed fine-tuning beats meta-learning under large domain shift. Expensive:
    a fresh optimisation per episode, so it is typically run on a bank prefix
    with coverage recorded so paired tests stay valid.
    """

    name = "finetune_supervised"
    family = "transfer"
    requires_support = True
    adapts_per_episode = True

    def __init__(self, steps: int = 50, lr: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.steps = steps
        self.lr = lr

    def prepare(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self.model, summary = self._train_supervised(ctx)
        self._base_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._summary.update({
            "train_spec_hash": summary["spec_hash"],
            "adapt_steps": self.steps,
            "adapt_lr": self.lr,
            "shared_encoder_with": "zeroshot_supervised",
        })

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        import copy

        import torch.nn as nn

        t0 = time.perf_counter()
        # Fresh copy per episode: adaptation must never leak across episodes.
        model = copy.deepcopy(self.model).to(self.device)
        model.load_state_dict(self._base_state)
        model.train()

        b = batch.to(self.device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr, weight_decay=1e-4
        )
        ce = nn.CrossEntropyLoss()
        torch.manual_seed(int(batch.spec.seed) % 2**31)

        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = ce(model(b.support_x), b.support_y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(b.query_x)
        return EpisodeOutput(
            logits=self._to_numpy(logits),
            adapt_seconds=time.perf_counter() - t0,
            extras={"adapt_steps": self.steps},
        )


class ChanceMethod(BaseMethod):
    """Uniform logits -- prints the chance row so 86% is read against 50%, not 0%."""

    name = "chance"
    family = "reference"
    requires_support = False
    adapts_per_episode = False

    def prepare(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self._summary["params"] = {"total": 0, "trainable": 0, "frozen": 0}

    def predict_episode(self, batch: EpisodeBatch) -> EpisodeOutput:
        n_q = batch.query_y.numel()
        return EpisodeOutput(logits=np.zeros((n_q, batch.n_way), dtype=np.float32))
