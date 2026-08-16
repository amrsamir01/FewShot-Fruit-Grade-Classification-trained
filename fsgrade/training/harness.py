"""
The single training loop.

The original repository had five near-duplicate loops (``train_protonet`` plus
four copies inside ``experiments.py``). They disagreed on learning rate
(backbone 5e-6 vs 5e-5), epoch budget (30 vs 20), and which regularisers were
active. Any "our method beats baseline X" conclusion was therefore confounded on
at least five axes at once.

There is now exactly one loop. Methods differ only in the ``TrainSpec`` handed
to it, and ``assert_single_factor`` mechanically refuses to run an ablation arm
that changes more than the factor under study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


class UncontrolledComparisonError(RuntimeError):
    """An ablation arm differs from its base config in more than the studied factor."""


@dataclass(frozen=True)
class EarlyStoppingSpec:
    monitor: str = "val_balanced_accuracy"
    patience: int = 7
    min_delta: float = 0.003
    mode: str = "max"


@dataclass(frozen=True)
class TrainSpec:
    """Everything that determines a training run. Hashable, serialized, compared."""

    mode: str = "episodic"                 # "episodic" | "supervised"
    epochs: int = 30
    episodes_per_epoch: int = 500
    batch_size: int = 32
    optimizer: str = "adamw"
    lr_backbone: float = 5e-6
    lr_head: float = 5e-5
    lr_temperature: float = 2.5e-5
    weight_decay: float = 5e-4
    no_decay_on_norm_and_bias: bool = True
    warmup_epochs: int = 3
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    contrastive_weight: float = 0.1
    supcon_temperature: float = 0.5
    adversarial_weight: float = 0.0
    adversarial_gamma: float = 10.0
    amp: bool = False
    seed: int = 42
    early_stopping: EarlyStoppingSpec = field(default_factory=EarlyStoppingSpec)

    def spec_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "TrainSpec":
        from fsgrade.config import get_in

        train = get_in(cfg, "train", {}) or {}
        es = train.get("early_stopping", {}) or {}
        base = {
            k: train[k] for k in asdict(cls()).keys()
            if k in train and k != "early_stopping"
        }
        base["early_stopping"] = EarlyStoppingSpec(
            monitor=es.get("monitor", "val_balanced_accuracy"),
            patience=int(es.get("patience", 7)),
            min_delta=float(es.get("min_delta", 0.003)),
            mode=es.get("mode", "max"),
        )
        base.update(overrides)
        return cls(**base)


class EarlyStopping:
    """Patience-based stopper. Carried over from ``src/train.py`` with a reason field."""

    def __init__(self, spec: EarlyStoppingSpec) -> None:
        self.spec = spec
        self.best_score: float | None = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.stop_reason = ""

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score, self.best_epoch = score, epoch
            return False
        improved = (
            score > self.best_score + self.spec.min_delta
            if self.spec.mode == "max"
            else score < self.best_score - self.spec.min_delta
        )
        if improved:
            self.best_score, self.best_epoch, self.counter = score, epoch, 0
        else:
            self.counter += 1
            if self.counter >= self.spec.patience:
                self.early_stop = True
                self.stop_reason = (
                    f"no improvement in {self.spec.monitor} for {self.spec.patience} epochs "
                    f"(best {self.best_score:.4f} @ epoch {self.best_epoch})"
                )
        return self.early_stop


def build_optimizer(
    param_groups: dict[str, list[nn.Parameter]], spec: TrainSpec, model: nn.Module | None = None
) -> torch.optim.Optimizer:
    """One optimizer construction for every method.

    Weight decay is not applied to norm/bias parameters -- standard practice the
    original omitted, which decayed BatchNorm scales toward zero.
    """
    lr_by_group = {
        "backbone": spec.lr_backbone,
        "head": spec.lr_head,
        "temperature": spec.lr_temperature,
    }

    no_decay_ids: set[int] = set()
    if spec.no_decay_on_norm_and_bias and model is not None:
        for module in model.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)):
                no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                no_decay_ids.add(id(p))

    groups: list[dict[str, Any]] = []
    for name, params in param_groups.items():
        params = [p for p in params if p.requires_grad]
        if not params:
            continue
        lr = lr_by_group.get(name, spec.lr_head)
        decay = [p for p in params if id(p) not in no_decay_ids]
        plain = [p for p in params if id(p) in no_decay_ids]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": spec.weight_decay,
                           "name": name})
        if plain:
            groups.append({"params": plain, "lr": lr, "weight_decay": 0.0,
                           "name": f"{name}_no_decay"})

    if not groups:  # pragma: no cover
        raise ValueError("No trainable parameters. Check the freeze specification.")

    if spec.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, spec: TrainSpec):
    """Linear warmup then cosine decay, matching the original schedule shape."""
    if spec.scheduler == "none":
        return None

    def lr_lambda(epoch: int) -> float:
        if epoch < spec.warmup_epochs:
            return (epoch + 1) / max(spec.warmup_epochs, 1)
        denom = max(spec.epochs - spec.warmup_epochs, 1)
        progress = (epoch - spec.warmup_epochs) / denom
        return float(0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def assert_single_factor(
    base: dict[str, Any],
    arm: dict[str, Any],
    allowed: Sequence[str],
    arm_name: str,
) -> dict[str, Any]:
    """Refuse to run an ablation arm that varies more than the studied factor.

    Returns the diff, which is serialized into ``metrics.json`` so a reader can
    see exactly what changed in each row of an ablation table.
    """
    from fsgrade.config import get_in

    def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, f"{key}."))
            else:
                out[key] = v
        return out

    flat_base, flat_arm = flatten(base), flatten(arm)
    diff = {
        k: {"base": flat_base.get(k), "arm": v}
        for k, v in flat_arm.items()
        if flat_base.get(k) != v
    }
    for k in flat_base:
        if k not in flat_arm:
            diff[k] = {"base": flat_base[k], "arm": None}

    allowed_set = set(allowed)
    violations = [k for k in diff if k not in allowed_set]
    if violations:
        raise UncontrolledComparisonError(
            f"Ablation arm {arm_name!r} changes {sorted(violations)} but is only "
            f"permitted to change {sorted(allowed_set)}. "
            "An ablation table row that varies two things at once is uninterpretable."
        )
    return diff


class Trainer:
    """The one training loop, in episodic or supervised mode."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        spec: TrainSpec,
        *,
        device: torch.device,
        logger: Any = None,
        checkpoint_path: Any = None,
        n_way: int = 2,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.spec = spec
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.n_way = n_way

        groups = model.param_groups() if hasattr(model, "param_groups") else {
            "head": list(model.parameters())
        }
        self.optimizer = build_optimizer(groups, spec, model)
        self.scheduler = build_scheduler(self.optimizer, spec)
        self.early_stopping = EarlyStopping(spec.early_stopping)
        self.history: dict[str, list[Any]] = {}
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_score = -float("inf")

    # ------------------------------------------------------------------ #
    def _set_train_mode(self) -> None:
        """train() then re-freeze tagged BatchNorm modules.

        Without this, ``requires_grad=False`` still lets running statistics drift
        on fruit data, so a "frozen" stage is not actually frozen.
        """
        from fsgrade.models.freezing import set_frozen_bn_eval

        self.model.train()
        set_frozen_bn_eval(self.model)

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    # ------------------------------------------------------------------ #
    def train_epoch_episodic(self, loader: Iterable, epoch: int) -> dict[str, float]:
        self._set_train_mode()
        losses, accs, components = [], [], {}

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.support_x, batch.support_y, batch.query_x, self.n_way)

            embeddings = torch.cat([out.support_embeddings, out.query_embeddings], dim=0)
            labels = torch.cat([batch.support_y, batch.query_y], dim=0)
            parts = self.loss_fn(out.logits, batch.query_y, embeddings, labels)
            loss = parts["total"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.spec.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.spec.grad_clip)
            self.optimizer.step()

            losses.append(float(loss.item()))
            accs.append(float((out.logits.argmax(1) == batch.query_y).float().mean()))
            for k, v in parts.items():
                components.setdefault(k, []).append(float(v.item()))

        stats = {"train_loss": float(np.mean(losses)), "train_acc": float(np.mean(accs))}
        stats.update({f"train_{k}": float(np.mean(v)) for k, v in components.items()})
        return stats

    def train_epoch_supervised(self, loader: Iterable, epoch: int) -> dict[str, float]:
        from fsgrade.training.grl import dann_lambda

        self._set_train_mode()
        losses, correct, total, components = [], 0, 0, {}
        progress = epoch / max(self.spec.epochs, 1)
        grl_lambda = (
            dann_lambda(progress, gamma=self.spec.adversarial_gamma)
            if self.spec.adversarial_weight > 0 else 0.0
        )

        for images, quality, species in loader:
            images = images.to(self.device)
            quality = quality.to(self.device)
            species = species.to(self.device)

            if self.spec.adversarial_weight > 0 and hasattr(self.model, "forward_with_species"):
                q_logits, s_logits, _ = self.model.forward_with_species(images, grl_lambda)
                parts = self.loss_fn(q_logits, quality, s_logits, species)
            else:
                q_logits = self.model(images)
                parts = self.loss_fn(q_logits, quality)

            loss = parts["total"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.spec.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.spec.grad_clip)
            self.optimizer.step()

            losses.append(float(loss.item()))
            correct += int((q_logits.argmax(1) == quality).sum())
            total += int(quality.numel())
            for k, v in parts.items():
                components.setdefault(k, []).append(float(v.item()))

        stats = {
            "train_loss": float(np.mean(losses)),
            "train_acc": correct / total if total else float("nan"),
            "grl_lambda": grl_lambda,
        }
        stats.update({f"train_{k}": float(np.mean(v)) for k, v in components.items()})
        return stats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate_episodic(self, loader: Iterable) -> dict[str, float]:
        from fsgrade.evaluation.metrics import episode_metrics

        self.model.eval()
        per_episode = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.support_x, batch.support_y, batch.query_x, self.n_way)
            y_pred = out.logits.argmax(1).cpu().numpy()
            y_true = batch.query_y.cpu().numpy()
            per_episode.append(episode_metrics(y_true, y_pred))
        if not per_episode:
            return {"val_accuracy": float("nan"), "val_balanced_accuracy": float("nan")}
        return {
            "val_accuracy": float(np.nanmean([m["accuracy"] for m in per_episode])),
            "val_balanced_accuracy": float(np.nanmean([m["balanced_accuracy"] for m in per_episode])),
        }

    @torch.no_grad()
    def validate_supervised(self, loader: Iterable) -> dict[str, float]:
        from fsgrade.evaluation.metrics import episode_metrics

        self.model.eval()
        preds, trues = [], []
        for images, quality, _ in loader:
            logits = self.model(images.to(self.device))
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(quality.numpy())
        if not preds:
            return {"val_accuracy": float("nan"), "val_balanced_accuracy": float("nan")}
        y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
        m = episode_metrics(y_true, y_pred)
        return {
            "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"],
        }

    # ------------------------------------------------------------------ #
    def fit(self, train_loader_fn: Callable[[int], Iterable], val_loader_fn: Callable[[], Iterable]) -> dict[str, Any]:
        """Run training. ``train_loader_fn(epoch)`` returns a fresh iterable per epoch."""
        monitor = self.spec.early_stopping.monitor
        started = time.time()

        for epoch in range(1, self.spec.epochs + 1):
            lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(self.optimizer.param_groups)}
            t0 = time.time()

            if self.spec.mode == "episodic":
                stats = self.train_epoch_episodic(train_loader_fn(epoch), epoch)
                stats.update(self.validate_episodic(val_loader_fn()))
            else:
                stats = self.train_epoch_supervised(train_loader_fn(epoch), epoch)
                stats.update(self.validate_supervised(val_loader_fn()))

            if self.scheduler is not None:
                self.scheduler.step()

            score = stats.get(monitor, stats.get("val_accuracy", float("nan")))
            stats.update({
                "epoch": epoch,
                "seconds": time.time() - t0,
                "lr": lrs,
                "gap": stats.get("train_acc", float("nan")) - stats.get("val_accuracy", float("nan")),
            })
            if hasattr(self.model, "temperature"):
                stats["temperature"] = float(self.model.temperature.item())

            for k, v in stats.items():
                self.history.setdefault(k, []).append(v)

            self._log(
                "epoch %3d/%d | loss %.4f | train %.3f | val %.3f | %s %.3f | gap %+0.3f | %.0fs",
                epoch, self.spec.epochs, stats.get("train_loss", float("nan")),
                stats.get("train_acc", float("nan")), stats.get("val_accuracy", float("nan")),
                monitor, score, stats.get("gap", float("nan")), stats["seconds"],
            )

            if np.isfinite(score) and score > self.best_score:
                self.best_score = score
                self.best_state = copy.deepcopy(
                    {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                )
                if self.checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "metrics": {k: v for k, v in stats.items() if not isinstance(v, dict)},
                            "train_spec": self.spec.to_dict(),
                            "spec_hash": self.spec.spec_hash(),
                        },
                        self.checkpoint_path,
                    )

            if epoch > self.spec.warmup_epochs and self.early_stopping(score, epoch):
                self._log("early stopping: %s", self.early_stopping.stop_reason)
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self._log("restored best weights (%s = %.4f)", monitor, self.best_score)

        from fsgrade.models.freezing import freeze_report

        summary: dict[str, Any] = {
            "history": self.history,
            "best_score": self.best_score,
            "best_epoch": self.early_stopping.best_epoch,
            "monitor": monitor,
            "stopped_early": self.early_stopping.early_stop,
            "stop_reason": self.early_stopping.stop_reason,
            "train_spec": self.spec.to_dict(),
            "spec_hash": self.spec.spec_hash(),
            "total_seconds": time.time() - started,
        }
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None and hasattr(encoder, "freeze_report"):
            summary["freeze_report"] = encoder.freeze_report.to_dict()
        return summary
