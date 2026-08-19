"""Components: whole-slide A-MIL (alignment + attention + pooling + classifier).

* :func:`train_mil` — fit one A-MIL model on a set of training slides.
* :func:`infer_mil` — score ONE slide with a fitted model.
* :func:`reduce_whole_image_risk` — collapse the K per-fold whole-image scores to
  one shipped value per image (out-of-fold on a CV cohort / ensemble on deploy).

Tile features are large (tens of thousands of tiles per slide), so training and
inference take *resolved paths* and open them lazily (per the streaming seam,
:meth:`helios.data.io.ArtifactStore.path`) rather than receiving fully-loaded
arrays. The owning stages live in :mod:`helios.stages.mil`.

The multi-branch gated-attention architecture and training loop are ported from
the project's ``train_visiomel_visiomel_augmented_04202026.py`` script; one
difference from that script is required by the HELIOS stage contract:
``train_mil`` only ever sees one fold's *train* slides (the CV test fold is
scored later by :func:`infer_mil`), so early stopping is done against an
internal train/val split carved out of the training slides rather than against
the held-out test fold.

:func:`infer_mil`'s gradient-based attribution outputs (``patch_embeddings``,
the gradient-weighted per-tile ``attention`` + ``tile_gradient`` saliency, and
the ``slide_embedding`` + its gradient) are ported from four further scripts
that ran a *second*, separate extraction pass over an already-trained model:
``extract_tile_features_training.py`` / ``extract_tile_features.py`` (tile-level
saliency: ``forward_tile_level``) and ``generate_penultimate.py`` /
``generate_tcav_files.py`` (slide-level TCAV bottleneck: ``forward_with_bottleneck``).
Here they are folded into one method (:meth:`_MultiBranchGatedAMIL.forward_with_attribution`)
called from :func:`infer_mil` instead of being a separate stage, because they
share the exact same GRAIN (one slide, one fold's already-fitted model) and the
same forward pass :func:`infer_mil` already needs for the risk score — a
separate stage would reload the same model bundle and re-stream the same
tile-feature HDF5s just to redo work already in hand.

Two representation choices, ambiguous or inconsistent across those four source
scripts, were resolved as follows:

* **Slide embedding** = the 128-dim penultimate bottleneck activation (after
  ``classifier[:10]``, i.e. post ``Linear(256, 128)`` + ``ReLU``), matching
  ``generate_penultimate.py``. ``generate_tcav_files.py``'s ``forward_with_bottleneck``
  captures a *different*, 256-dim intermediate under the same "penultimate"
  name and additionally treats the pre-classifier ``M_fused`` (``L*K``-dim) as
  a separate "embedding" — ``generate_penultimate.py`` reads as the deliberate
  fix, and a tighter bottleneck closer to the decision is the more standard
  choice for concept probes.
* **Attention** = gradient-weighted branch attention (branch contribution
  weights from ``branch_reps`` gradient norms w.r.t. the logit, as in
  ``extract_tile_features_training.py``'s ``weighted_attention``), not a naive
  mean across branches.
* Both gradient computations backward through the TRUE pre-sigmoid logit
  (``classifier[:12]``, i.e. everything except the final ``Sigmoid``).
  ``extract_tile_features_training.py``/``extract_tile_features.py`` instead
  called ``.backward()`` on ``self.classifier(M_fused)``, whose Sequential
  already ends in ``Sigmoid`` — so their "logit" was already a probability,
  sigmoided a second time for the reported ``prob``. That double-sigmoid is a
  bug in the source scripts and is not reproduced here;
  ``generate_penultimate.py``/``generate_tcav_files.py`` already avoid it by
  calling ``classifier[11]`` (the ``Linear(128, 1)`` layer only) directly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from helios.models.base import Model, register_model


@dataclass(frozen=True)
class SlideFeatureRef:
    """A training slide's resolved feature paths + label (opened lazily)."""

    image_id: str
    features_path: Path
    background_path: Path | None = None
    aug_features_path: Path | None = None
    label: object | None = None


@dataclass(frozen=True)
class MilSlideOutputs:
    """Per-slide A-MIL inference products.

    ``attention`` carries one row per tile with ``tile_id``, gradient-weighted
    ``attention``, and per-tile saliency ``tile_gradient`` columns.
    ``slide_embedding_grad`` is the gradient of the pre-sigmoid logit w.r.t.
    ``slide_embedding`` (a TCAV-style concept-sensitivity signal).
    """

    patch_embeddings: NDArray[np.float32]
    attention: pd.DataFrame
    slide_embedding: NDArray[np.float32]
    slide_embedding_grad: NDArray[np.float32]
    risk_score: float


# =================================================================
# Architecture (ported from train_visiomel_visiomel_augmented_04202026.py)
# =================================================================
class _MultiBranchGatedAMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        n_branches: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.L = input_dim
        self.D, self.K = 128, n_branches
        self.num_heads = num_heads
        self.head_dim = self.D // num_heads

        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.LayerNorm(self.L),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.L, self.L),
        )
        self.quality_estimator = nn.Sequential(
            nn.Linear(self.L, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.branches = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention_V_heads": nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(self.L, self.head_dim),
                                    nn.LayerNorm(self.head_dim),
                                    nn.Tanh(),
                                    nn.Dropout(dropout_rate),
                                )
                                for _ in range(num_heads)
                            ]
                        ),
                        "attention_U_heads": nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(self.L, self.head_dim),
                                    nn.LayerNorm(self.head_dim),
                                    nn.Sigmoid(),
                                    nn.Dropout(dropout_rate),
                                )
                                for _ in range(num_heads)
                            ]
                        ),
                        "attention_w_heads": nn.ModuleList(
                            [nn.Linear(self.head_dim, 1) for _ in range(num_heads)]
                        ),
                        "attention_fusion": nn.Sequential(nn.Linear(num_heads, 1), nn.Tanh()),
                        "feature_processor": nn.Sequential(
                            nn.Linear(self.L, self.L), nn.ReLU(), nn.Dropout(dropout_rate)
                        ),
                    }
                )
                for _ in range(self.K)
            ]
        )

        self.branch_fusion = nn.Sequential(
            nn.Linear(self.L * self.K, self.L * self.K),
            nn.LayerNorm(self.L * self.K),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one slide's tile bag ``x`` (shape ``(N, L)``) through the network.

        Returns ``(risk_prob, branch_attention[K,N], branch_reps[K,L],
        patch_embeddings[N,L])``.
        """
        h_enhanced = x + self.feature_enhancer(x)
        h_weighted = h_enhanced * self.quality_estimator(h_enhanced)

        branch_reps, branch_atts = [], []
        for branch in self.branches:
            head_scores = [
                branch["attention_w_heads"][h](
                    branch["attention_V_heads"][h](h_weighted) * branch["attention_U_heads"][h](h_weighted)
                )
                for h in range(self.num_heads)
            ]
            att = F.softmax(branch["attention_fusion"](torch.cat(head_scores, dim=1)), dim=0)
            branch_reps.append(torch.sum(att * branch["feature_processor"](h_weighted), dim=0))
            branch_atts.append(att.squeeze(1))

        reps_stacked = torch.stack(branch_reps, dim=0)  # [K, L]
        attention = torch.stack(branch_atts, dim=0)  # [K, N]

        m_fused = self.branch_fusion(reps_stacked.view(-1))
        y_prob = self.classifier(m_fused).squeeze(-1)

        return y_prob, attention, reps_stacked, h_weighted

    def forward_with_attribution(
        self, x: torch.Tensor
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score one slide AND compute gradient-based tile/slide attribution.

        Inference-only (requires grad tracking + one ``backward()``, unlike the
        plain :meth:`forward` used during training). Returns ``(risk_prob,
        patch_embeddings [N,L], weighted_attention [N], tile_gradients [N],
        slide_embedding [128], slide_embedding_grad [128])``.
        """
        h_enhanced = x + self.feature_enhancer(x)
        h_weighted = h_enhanced * self.quality_estimator(h_enhanced)

        # A per-tile scalar multiplier (init 1, so it doesn't change the forward
        # value) purely to get a differentiable handle for per-tile saliency.
        tile_scores = torch.ones(h_weighted.shape[0], device=h_weighted.device, requires_grad=True)
        h_scaled = h_weighted * tile_scores.unsqueeze(1)

        branch_reps, branch_atts = [], []
        for branch in self.branches:
            head_scores = [
                branch["attention_w_heads"][h](
                    branch["attention_V_heads"][h](h_scaled) * branch["attention_U_heads"][h](h_scaled)
                )
                for h in range(self.num_heads)
            ]
            att = F.softmax(branch["attention_fusion"](torch.cat(head_scores, dim=1)), dim=0)
            branch_atts.append(att.squeeze(1))
            branch_reps.append(torch.sum(att * branch["feature_processor"](h_scaled), dim=0))

        reps_stacked = torch.stack(branch_reps, dim=0)  # [K, L]
        reps_stacked.retain_grad()
        att_matrix = torch.stack(branch_atts, dim=0)  # [K, N]

        m_fused = self.branch_fusion(reps_stacked.view(-1))

        penultimate = m_fused
        for layer in self.classifier[:10]:  # through the 128-dim ReLU (post Linear(256, 128))
            penultimate = layer(penultimate)
        penultimate.retain_grad()

        logit = penultimate
        for layer in self.classifier[10:12]:  # Dropout(0.1) (no-op in eval), Linear(128, 1) -- pre-Sigmoid
            logit = layer(logit)
        logit = logit.squeeze(-1)
        risk_prob = torch.sigmoid(logit)

        logit.backward()

        branch_grad_norms = reps_stacked.grad.norm(dim=1)  # [K]
        branch_weights = branch_grad_norms / (branch_grad_norms.sum() + 1e-8)
        weighted_attention = (branch_weights.unsqueeze(1) * att_matrix).sum(dim=0)  # [N]

        return (
            float(risk_prob.item()),
            h_weighted.detach(),
            weighted_attention.detach(),
            tile_scores.grad.detach().clone(),
            penultimate.detach(),
            penultimate.grad.detach().clone(),
        )

    def objective(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w_pos: float,
        w_neg: float,
        *,
        lambda_div: float = 0.01,
    ) -> torch.Tensor:
        y_prob, attention, _, _ = self.forward(x)
        y_prob = torch.clamp(y_prob, 1e-5, 1.0 - 1e-5)
        weight = w_pos if float(y.item()) == 1 else w_neg
        nll = weight * (-(y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob)))

        att_norm = F.normalize(attention, p=2, dim=1)
        sim = torch.mm(att_norm, att_norm.t())
        eye = torch.eye(attention.size(0), device=attention.device)
        div_loss = lambda_div * torch.sum(sim * (1 - eye))
        return nll + div_loss


@register_model("mil_amil")
class MilAttentionModel(Model):
    """Fitted A-MIL bundle: architecture hyperparameters + trained weights.

    Construct via :func:`train_mil`; :meth:`predict` scores ONE slide's tile
    bag and returns ``(risk_prob, patch_embeddings[N,L], weighted_attention[N],
    tile_gradients[N], slide_embedding[128], slide_embedding_grad[128])``.
    """

    def __init__(
        self,
        *,
        n_branches: int = 4,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        lambda_div: float = 0.01,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        patience: int = 5,
        val_frac: float = 0.15,
        seed: int = 42,
        aug_swap_prob: float = 0.5,
        aug_mode: Literal["replace", "augment"] = "replace",
        device: str = "cuda",
    ) -> None:
        self.n_branches = n_branches
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.lambda_div = lambda_div
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.aug_swap_prob = aug_swap_prob
        self.aug_mode = aug_mode
        self.device = device
        self.input_dim: int | None = None
        self._state_dict: dict[str, torch.Tensor] | None = None

    def _build_net(self) -> _MultiBranchGatedAMIL:
        if self.input_dim is None:
            raise RuntimeError("MilAttentionModel is not fitted yet.")
        net = _MultiBranchGatedAMIL(
            self.input_dim,
            n_branches=self.n_branches,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        if self._state_dict is not None:
            net.load_state_dict(self._state_dict)
        return net

    def fit(self, inputs: list[SlideFeatureRef], **params: Any) -> MilAttentionModel:
        slides = inputs
        if not slides:
            raise ValueError("train_mil requires at least one training slide.")

        _seed_everything(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(_resolve_device(self.device))

        train_refs, val_refs = _split_train_val(slides, val_frac=self.val_frac, seed=self.seed)
        self.input_dim = _peek_feature_dim(train_refs[0].features_path)

        net = self._build_net().to(dev)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = np.array([float(s.label) for s in train_refs])
        n_pos, n_neg = float((labels == 1).sum()), float((labels == 0).sum())
        w_pos = len(labels) / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = len(labels) / (2.0 * n_neg) if n_neg > 0 else 1.0

        best_state: dict[str, torch.Tensor] | None = None
        best_auprc = -1.0
        epochs_no_improve = 0

        for _epoch in range(self.max_epochs):
            net.train()
            for idx in rng.permutation(len(train_refs)):
                ref = train_refs[idx]
                bag = _load_training_bag(
                    ref, aug_swap_prob=self.aug_swap_prob, aug_mode=self.aug_mode, rng=rng
                )
                if bag.shape[0] == 0:
                    continue
                x = torch.from_numpy(bag).to(dev)
                y = torch.tensor(float(ref.label), device=dev)
                optimizer.zero_grad()
                loss = net.objective(x, y, w_pos, w_neg, lambda_div=self.lambda_div)
                loss.backward()
                optimizer.step()

            if val_refs:
                auprc = _validation_auprc(net, val_refs, device=dev)
                if auprc > best_auprc:
                    best_auprc = auprc
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                # Fold too small for an internal val split: keep the final epoch's weights.
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        self._state_dict = best_state
        return self

    def predict(
        self, inputs: NDArray[np.float32], *, device: str | None = None
    ) -> tuple[
        float, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
    ]:
        """Score one slide's tile bag with gradient-based tile/slide attribution.

        Returns ``(risk_prob, patch_embeddings[N,L], weighted_attention[N],
        tile_gradients[N], slide_embedding[128], slide_embedding_grad[128])``.
        """
        dev = torch.device(_resolve_device(device or self.device))
        net = self._build_net().to(dev)
        net.eval()
        net.zero_grad()
        x = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(dev)
        (
            risk_prob,
            patch_embeddings,
            weighted_attention,
            tile_gradients,
            slide_embedding,
            slide_embedding_grad,
        ) = net.forward_with_attribution(x)
        return (
            risk_prob,
            patch_embeddings.cpu().numpy(),
            weighted_attention.cpu().numpy(),
            tile_gradients.cpu().numpy(),
            slide_embedding.cpu().numpy(),
            slide_embedding_grad.cpu().numpy(),
        )


def train_mil(
    slides: list[SlideFeatureRef],
    *,
    aug_swap_prob: float = 0.5,
    aug_mode: Literal["replace", "augment"] = "replace",
    n_branches: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    lambda_div: float = 0.01,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 20,
    patience: int = 5,
    val_frac: float = 0.15,
    seed: int = 42,
    device: str = "cuda",
) -> Model:
    """Fit one A-MIL model on a fold's training slides.

    GRAIN: one fold (all of its train slides) per call. Tiles are streamed from
    each slide's ``features_path`` (background tiles excluded via
    ``background_path``). If a tile has a stain-augmented counterpart
    (``aug_features_path``), each tile independently swaps to it with
    probability ``aug_swap_prob``: ``aug_mode="replace"`` (default) substitutes
    the tile's embedding in place; ``aug_mode="augment"`` keeps the original
    tile and appends the augmented one as an extra bag entry instead.

    Early stopping monitors AUPRC on an internal ``val_frac`` split carved out
    of ``slides`` (this fold's CV test slides are never visible here — they are
    scored later by :func:`infer_mil`).

    Returns the fitted :class:`~helios.models.base.Model` (persisted as a bundle).
    """
    model = MilAttentionModel(
        n_branches=n_branches,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        lambda_div=lambda_div,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        val_frac=val_frac,
        seed=seed,
        aug_swap_prob=aug_swap_prob,
        aug_mode=aug_mode,
        device=device,
    )
    return model.fit(slides)


def infer_mil(
    model: Model,
    *,
    features_path: Path,
    background_path: Path | None = None,
    device: str = "cuda",
) -> MilSlideOutputs:
    """Score ONE slide with a fitted A-MIL model.

    GRAIN: one slide per call; tiles are streamed from ``features_path``
    (background excluded via ``background_path``). One gradient backward pass
    (through the pre-sigmoid logit) additionally produces gradient-weighted
    per-tile attention/saliency and the 128-dim penultimate slide embedding +
    its gradient.
    """
    feats, tile_ids = _load_inference_bag(features_path, background_path)
    if feats.shape[0] == 0:
        raise ValueError(f"No foreground tiles found for {features_path}.")

    (
        risk_score,
        patch_embeddings,
        weighted_attention,
        tile_gradients,
        slide_embedding,
        slide_embedding_grad,
    ) = model.predict(feats, device=device)  # type: ignore[call-arg]

    attention_df = pd.DataFrame(
        {"tile_id": tile_ids, "attention": weighted_attention, "tile_gradient": tile_gradients}
    )

    return MilSlideOutputs(
        patch_embeddings=patch_embeddings,
        attention=attention_df,
        slide_embedding=slide_embedding.astype(np.float32),
        slide_embedding_grad=slide_embedding_grad.astype(np.float32),
        risk_score=risk_score,
    )


def reduce_whole_image_risk(
    per_fold_scores: pd.DataFrame,
    cv_splits: pd.DataFrame | None,
) -> pd.DataFrame:
    """Collapse K per-fold whole-image scores to one row per image.

    Out-of-fold (each image scored by its test-fold model) on a CV cohort;
    ensemble-average when ``cv_splits`` is ``None`` (deploy). Emits a ``fold``
    provenance column (test fold; null on deploy).
    """
    if per_fold_scores.empty:
        return pd.DataFrame(columns=["image_id", "risk_score", "fold"])

    if cv_splits is None:
        reduced = per_fold_scores.groupby("image_id", as_index=False)["risk_score"].mean()
        reduced["fold"] = None
        return reduced

    test_rows = cv_splits.loc[cv_splits["split"] == "test", ["image_id", "fold"]]
    reduced = test_rows.merge(per_fold_scores, on=["image_id", "fold"], how="left")
    return reduced[["image_id", "risk_score", "fold"]]


# =================================================================
# Internal helpers
# =================================================================
def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peek_feature_dim(path: Path) -> int:
    with h5py.File(path, "r") as h:
        return int(h["features"].shape[-1])


def _read_hdf5_features(path: Path) -> NDArray[np.float32]:
    with h5py.File(path, "r") as h:
        return np.asarray(h["features"][()], dtype=np.float32)


def _read_tile_background(path: Path | None, n_tiles: int) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    if path is None or not Path(path).exists():
        return np.zeros(n_tiles, dtype=bool), np.arange(n_tiles, dtype=np.int64)
    table = pd.read_parquet(path).sort_values("tile_id")
    flag_column = "tile_background" if "tile_background" in table.columns else "is_background"
    return table[flag_column].to_numpy(dtype=bool), table["tile_id"].to_numpy(dtype=np.int64)


def _load_training_bag(
    ref: SlideFeatureRef,
    *,
    aug_swap_prob: float,
    aug_mode: Literal["replace", "augment"],
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    feats = _read_hdf5_features(ref.features_path)
    is_background, _ = _read_tile_background(ref.background_path, len(feats))
    foreground = ~is_background
    feats = feats[foreground]

    if ref.aug_features_path is not None and aug_swap_prob > 0 and feats.shape[0] > 0:
        aug_feats = _read_hdf5_features(ref.aug_features_path)[foreground]
        swap = rng.random(feats.shape[0]) < aug_swap_prob
        if aug_mode == "replace":
            feats = feats.copy()
            feats[swap] = aug_feats[swap]
        elif swap.any():
            feats = np.concatenate([feats, aug_feats[swap]], axis=0)

    return feats


def _load_inference_bag(
    features_path: Path, background_path: Path | None
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    feats = _read_hdf5_features(features_path)
    is_background, tile_ids = _read_tile_background(background_path, len(feats))
    foreground = ~is_background
    return feats[foreground], tile_ids[foreground]


def _split_train_val(
    slides: list[SlideFeatureRef], *, val_frac: float, seed: int
) -> tuple[list[SlideFeatureRef], list[SlideFeatureRef]]:
    if val_frac <= 0 or len(slides) < 4:
        return list(slides), []
    labels = [s.label for s in slides]
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) > 1 and counts.min() >= 2 else None
    train, val = train_test_split(slides, test_size=val_frac, random_state=seed, stratify=stratify)
    return list(train), list(val)


def _validation_auprc(
    net: _MultiBranchGatedAMIL, val_refs: list[SlideFeatureRef], *, device: torch.device
) -> float:
    net.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for ref in val_refs:
            bag = _read_hdf5_features(ref.features_path)
            is_background, _ = _read_tile_background(ref.background_path, len(bag))
            bag = bag[~is_background]
            if bag.shape[0] == 0:
                continue
            x = torch.from_numpy(bag).to(device)
            prob, _, _, _ = net(x)
            probs.append(float(prob.item()))
            labels.append(int(ref.label))  # type: ignore[arg-type]
    if len(set(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probs))
