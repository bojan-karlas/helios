"""Training traces for A-MIL: per-step and per-epoch probes of a training run.

The tracer reads values the training step already computes (forward-hook outputs,
gradients after ``backward``, weights around ``optimizer.step``). It runs no extra
forward passes and draws no random numbers, so traced and untraced runs train
identical weights. All reductions stay on the device, with one host transfer per step.

When given a directory, the tables below are rewritten atomically after every epoch,
so an interrupted run keeps complete tables up to its last finished epoch:

* ``steps.parquet``: one row per training bag (identity, loss terms, output,
  gradient and update sizes per block, attention, quality gate, activations, spikes).
* ``epochs.parquet``: one row per epoch (training and validation summaries, weight
  sizes per block, timing, event counts).
* ``val_predictions.parquet``: every validation bag's probability, every epoch.
* ``test_predictions.parquet``: every traced held-out slide's probability, every
  epoch (only when trace slides are given; analysis only, never used for selection).
  Its ``weights`` column is ``current`` for the epoch's weights and ``average`` for the
  running weight average, when weight averaging is on.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

BLOCKS = ("feature_enhancer", "quality_estimator", "branches", "branch_fusion", "classifier")
#: Rolling window (steps) and factors for spike events.
SPIKE_WINDOW = 200
SPIKE_MIN_HISTORY = 20
LOSS_SPIKE_FACTOR = 5.0
GRAD_SPIKE_FACTOR = 10.0
#: Epoch events: mean training loss this far above its best so far; validation prediction spread below this.
COLLAPSE_LOSS_RISE = 0.05
COLLAPSE_PRED_STD = 0.01
#: Probability clamp used by the training objective; beyond it the classification gradient is zero.
PROB_CLAMP = 1e-5


@dataclass(frozen=True)
class BagIdentity:
    """Which slide a bag comes from and which of its versions it is."""

    image_id: str
    bag: str  # "original", "augmented", or "mixed" (tiles from several versions)
    augmentation: int  # 0 for the original, else 1-based augmentation index; -1 for mixed bags


class MilTrainingTracer:
    """Collects probes during :meth:`MilAttentionModel.fit` and writes them as Parquet tables."""

    def __init__(self, net: torch.nn.Module, *, out_dir: Path | None, n_branches: int) -> None:
        self.net = net
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.n_branches = n_branches
        self.blocks = {name: [p for p in getattr(net, name).parameters() if p.requires_grad] for name in BLOCKS}
        self._params = [p for name in BLOCKS for p in self.blocks[name]]
        if len(self._params) != sum(1 for p in net.parameters() if p.requires_grad):
            raise ValueError(f"Traced blocks {BLOCKS} must cover every trainable parameter.")
        self._sizes = [len(self.blocks[name]) for name in BLOCKS]
        self._before: list[torch.Tensor] | None = None
        self._active = False
        self._probes: dict[str, torch.Tensor] = {}
        self._branch_att: list[torch.Tensor | None] = [None] * n_branches
        self._handles: list[Any] = []
        self._register_hooks()

        self.steps: list[dict[str, Any]] = []
        self.epochs: list[dict[str, Any]] = []
        self.val_rows: list[dict[str, Any]] = []
        self.test_rows: list[dict[str, Any]] = []
        self._loss_window: deque[float] = deque(maxlen=SPIKE_WINDOW)
        self._grad_window: deque[float] = deque(maxlen=SPIKE_WINDOW)
        self._epoch_steps: list[dict[str, Any]] = []
        self._best_epoch_loss = math.inf

    # -- hooks --------------------------------------------------------------
    def _register_hooks(self) -> None:
        net = self.net

        def hook(fn):
            def wrapped(module, inputs, output):
                if self._active:
                    with torch.no_grad():
                        fn(inputs, output)
            return wrapped

        def enhancer(inputs, output):
            x = inputs[0]
            self._probes["enhancer_residual_ratio"] = output.norm() / x.norm().clamp_min(1e-12)

        def gate(inputs, output):
            g = output.reshape(-1)
            self._probes["gate_mean"] = g.mean()
            self._probes["gate_min"] = g.min()
            self._probes["gate_frac_below_0.05"] = (g < 0.05).float().mean()
            self._probes["gate_frac_above_0.95"] = (g > 0.95).float().mean()

        def attention(k):
            def fn(inputs, output):
                scores = output.reshape(-1)
                att = F.softmax(scores, dim=0)
                n = att.numel()
                entropy = -(att * att.clamp_min(1e-30).log()).sum()
                self._probes[f"att{k}_entropy"] = entropy / math.log(n) if n > 1 else torch.ones((), device=att.device)
                self._probes[f"att{k}_max_x_n"] = att.max() * n
                self._probes[f"att{k}_frac_saturated"] = (scores.abs() > 0.99).float().mean()
                self._branch_att[k] = att
            return fn

        def dead(name):
            def fn(inputs, output):
                self._probes[name] = (output == 0).float().mean()
            return fn

        def logit(inputs, output):
            self._probes["logit"] = output.reshape(())

        targets = [
            (net.feature_enhancer, enhancer),
            (net.quality_estimator, gate),
            (net.branch_fusion[2], dead("dead_frac_branch_fusion")),
            (net.classifier[2], dead("dead_frac_classifier_512")),
            (net.classifier[6], dead("dead_frac_classifier_256")),
            (net.classifier[9], dead("dead_frac_classifier_128")),
            (net.classifier[11], logit),
        ]
        targets += [(branch["attention_fusion"], attention(k)) for k, branch in enumerate(net.branches)]
        self._handles = [module.register_forward_hook(hook(fn)) for module, fn in targets]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._before = None

    # -- per step -----------------------------------------------------------
    def begin_step(self) -> None:
        self._probes = {}
        self._branch_att = [None] * self.n_branches
        self._active = True

    def end_forward(self) -> None:
        self._active = False

    def before_optimizer_step(self) -> None:
        """Record gradient sizes and snapshot weights; call after ``backward``."""
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self._params]
        norms = torch.stack(torch._foreach_norm(grads))
        for name, chunk in zip(BLOCKS, norms.split(self._sizes), strict=True):
            self._probes[f"grad_{name}"] = chunk.norm()
        self._probes["grad_total"] = norms.norm()
        with torch.no_grad():
            if self._before is None:
                self._before = [torch.empty_like(p) for p in self._params]
            torch._foreach_copy_(self._before, [p.detach() for p in self._params])

    def end_step(
        self, *, epoch: int, step: int, identity: BagIdentity, label: float, n_tiles: int,
        total: torch.Tensor, classification: torch.Tensor, diversity: torch.Tensor, probability: torch.Tensor,
        class_weight: float,
    ) -> None:
        """Record update sizes and everything else for one step; call after ``optimizer.step``."""
        with torch.no_grad():
            assert self._before is not None
            torch._foreach_sub_(self._before, [p.detach() for p in self._params])
            deltas = torch.stack(torch._foreach_norm(self._before))
            weights = torch.stack(torch._foreach_norm([p.detach() for p in self._params]))
            for name, d, w in zip(BLOCKS, deltas.split(self._sizes), weights.split(self._sizes), strict=True):
                self._probes[f"update_ratio_{name}"] = d.norm() / w.norm().clamp_min(1e-12)
            atts = [a for a in self._branch_att if a is not None]
            if len(atts) > 1:
                stacked = F.normalize(torch.stack(atts), p=2, dim=1)
                sim = stacked @ stacked.t()
                k = len(atts)
                self._probes["branch_similarity"] = (sim.sum() - sim.diagonal().sum()) / (k * (k - 1))
            self._probes["loss"] = total.detach().reshape(())
            self._probes["loss_classification"] = classification.detach().reshape(())
            self._probes["loss_diversity"] = diversity.detach().reshape(())
            self._probes["probability"] = probability.detach().reshape(())
            names = list(self._probes)
            values = torch.stack([self._probes[n].float() for n in names]).tolist()
        row: dict[str, Any] = {
            "epoch": epoch, "step": step, "image_id": identity.image_id, "bag": identity.bag,
            "augmentation": identity.augmentation, "label": label, "n_tiles": n_tiles, "class_weight": class_weight,
            **dict(zip(names, values, strict=True)),
        }
        p = row["probability"]
        row["clamped"] = bool(p < PROB_CLAMP or p > 1.0 - PROB_CLAMP)
        row["nonfinite"] = not all(math.isfinite(v) for v in values)
        row["loss_spike"] = _spike(row["loss"], self._loss_window, LOSS_SPIKE_FACTOR)
        row["grad_spike"] = _spike(row["grad_total"], self._grad_window, GRAD_SPIKE_FACTOR)
        self._loss_window.append(row["loss"])
        self._grad_window.append(row["grad_total"])
        self.steps.append(row)
        self._epoch_steps.append(row)

    # -- per epoch ----------------------------------------------------------
    def end_epoch(
        self, *, epoch: int, val: Sequence[tuple[BagIdentity, float, float | None]],
        test: Sequence[tuple[BagIdentity, float, float | None]], w_pos: float, w_neg: float, lr: float,
        eligible: bool, improved: bool, skipped: int, seconds: dict[str, float], averaged: bool = False,
        test_average: Sequence[tuple[BagIdentity, float, float | None]] = (),
    ) -> dict[str, Any]:
        """Summarize the epoch, store prediction rows, and rewrite the tables; returns the epoch row.

        ``averaged`` marks an epoch folded into the running weight average; ``test_average`` holds
        the traced slides' probabilities under the averaged weights so far.
        """
        steps = self._epoch_steps
        losses = [s["loss"] for s in steps]
        mean_loss = float(np.mean(losses)) if losses else math.nan
        weights = {}
        with torch.no_grad():
            for name in BLOCKS:
                params = self.blocks[name]
                weights[f"weight_{name}"] = float(torch.stack(torch._foreach_norm(
                    [p.detach() for p in params])).norm().item()) if params else 0.0
        row: dict[str, Any] = {
            "epoch": epoch, "n_steps": len(steps), "skipped": skipped, "train_loss": mean_loss,
            "loss_spikes": sum(s["loss_spike"] for s in steps), "grad_spikes": sum(s["grad_spike"] for s in steps),
            "max_grad_total": max((s["grad_total"] for s in steps), default=math.nan),
            "clamped_steps": sum(s["clamped"] for s in steps), "nonfinite_steps": sum(s["nonfinite"] for s in steps),
            "training_collapse": bool(
                math.isfinite(mean_loss) and mean_loss > self._best_epoch_loss + COLLAPSE_LOSS_RISE
            ),
            "w_pos": w_pos, "w_neg": w_neg, "lr": lr, "eligible": eligible, "improved": improved,
            "averaged": averaged, **weights,
            **{f"seconds_{k}": v for k, v in seconds.items()},
        }
        if math.isfinite(mean_loss):
            self._best_epoch_loss = min(self._best_epoch_loss, mean_loss)
        row |= _prediction_summary("val", val, w_pos, w_neg)
        originals = _prediction_summary("val_original", [v for v in val if v[0].bag == "original"], w_pos, w_neg)
        row |= {k: originals[k] for k in ("val_original_auprc", "val_original_auroc")}
        row["prediction_collapse"] = bool(row["val_prob_std"] < COLLAPSE_PRED_STD)
        if test:
            row |= _prediction_summary("test", test, w_pos, w_neg)
        if test_average:
            row |= _prediction_summary("test_average", test_average, w_pos, w_neg)
        self.epochs.append(row)
        self.val_rows += _prediction_rows(epoch, val)
        self.test_rows += [r | {"weights": "current"} for r in _prediction_rows(epoch, test)]
        self.test_rows += [r | {"weights": "average"} for r in _prediction_rows(epoch, test_average)]
        self._epoch_steps = []
        self.flush()
        return row

    def summary(self) -> str:
        """One log line of event counts and worst values over the whole run."""
        steps = self.steps
        return (
            f"loss_spikes={sum(s['loss_spike'] for s in steps)} grad_spikes={sum(s['grad_spike'] for s in steps)} "
            f"max_grad_total={max((s['grad_total'] for s in steps), default=math.nan):.4g} "
            f"clamped_steps={sum(s['clamped'] for s in steps)} nonfinite_steps={sum(s['nonfinite'] for s in steps)} "
            f"collapse_epochs={[e['epoch'] for e in self.epochs if e['training_collapse']]} "
            f"prediction_collapse_epochs={[e['epoch'] for e in self.epochs if e['prediction_collapse']]}"
            + (f" trace_dir={self.out_dir}" if self.out_dir is not None else "")
        )

    def flush(self) -> None:
        if self.out_dir is None:
            return
        from helios.data.io import write

        tables = {"steps": self.steps, "epochs": self.epochs, "val_predictions": self.val_rows}
        if self.test_rows:
            tables["test_predictions"] = self.test_rows
        for name, rows in tables.items():
            write(pd.DataFrame(rows), self.out_dir / f"{name}.parquet", "parquet")


def _spike(value: float, window: deque[float], factor: float) -> bool:
    return len(window) >= SPIKE_MIN_HISTORY and value > factor * median(window)


def _prediction_rows(epoch: int, scored: Sequence[tuple[BagIdentity, float, float | None]]) -> list[dict[str, Any]]:
    return [
        {"epoch": epoch, "image_id": ident.image_id, "bag": ident.bag, "augmentation": ident.augmentation,
         "label": label, "probability": prob}
        for ident, label, prob in scored
    ]


def _prediction_summary(
    prefix: str, scored: Sequence[tuple[BagIdentity, float, float | None]], w_pos: float, w_neg: float,
) -> dict[str, float]:
    pairs = [(label, prob) for _, label, prob in scored if prob is not None]
    keys = ("auprc", "auroc", "loss", "prob_std", "mean_prob_pos", "mean_prob_neg")
    if not pairs:
        return {f"{prefix}_{k}": math.nan for k in keys}
    y = np.array([p[0] for p in pairs], dtype=float)
    p = np.array([p[1] for p in pairs], dtype=float)
    clipped = np.clip(p, PROB_CLAMP, 1.0 - PROB_CLAMP)
    loss = np.where(y == 1, w_pos, w_neg) * -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    both = len(set(y.tolist())) == 2
    return {
        f"{prefix}_auprc": float(average_precision_score(y, p)) if both else math.nan,
        f"{prefix}_auroc": float(roc_auc_score(y, p)) if both else math.nan,
        f"{prefix}_loss": float(loss.mean()),
        f"{prefix}_prob_std": float(p.std()),
        f"{prefix}_mean_prob_pos": float(p[y == 1].mean()) if (y == 1).any() else math.nan,
        f"{prefix}_mean_prob_neg": float(p[y == 0].mean()) if (y == 0).any() else math.nan,
    }
