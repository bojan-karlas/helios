"""Components: patch morphology analysis.

* :func:`train_morphology` — attention-cutoff filter → PCA → FAISS k-means
  over-clustering → Leiden clustering → cluster exclusion (significance test)
  → cluster-presence risk model.
* :func:`infer_morphology` — assign tiles to clusters, score cluster presence,
  and produce the patch-morphology risk score.

Patch embeddings are per-tile (large), so they are passed as resolved paths and
streamed lazily; attention tables are small and passed loaded. The owning stages
live in :mod:`helios.stages.morphology`.

Ported from ``utility.py`` (the shared module behind
``tile_analysis_v3_lean.ipynb`` / ``tile_analysis_external.ipynb`` /
``tile_analysis_presence_table.ipynb``), with the following choices made to fit
the HELIOS per-fold contract:

* **Attention filter**: the *cumulative attention mass* cutoff
  (``cumulative_attention_mask`` — keep the smallest set of tiles whose
  attention mass reaches ``attention_cutoff``), not a percentile-by-count
  filter.
* **Resolution**: the source notebooks sweep six Leiden resolutions and
  hand-pick clusters per resolution; here ``resolution`` is a single
  configurable value (one clustering per fold, like every other HELIOS
  hyperparameter).
* **No multi-model concatenation**: the notebooks concatenate embeddings across
  5 *independently trained* models (12800-dim) because that was an
  ensemble-of-separate-runs setup. HELIOS's fold structure is different — each
  fold already has its own single embedding space — so PCA runs directly on
  that fold's ``aligned_patch_embeddings`` (no concatenation).
* **PCA backend**: the notebooks used GPU cuML PCA for the (much larger,
  concatenated) embeddings; this uses ``sklearn.decomposition.PCA`` (CPU),
  matching ``utility.preprocess``'s portable fallback.
* **Cluster-presence feature**: only the plain cosine-weighted, per-slide
  L1-normalized ``presence`` (not the ``_abovemed``/``_abovep25`` variants) —
  this is the one ``tile_analysis_external.ipynb``'s own final model comparison
  actually used.
* **Risk estimator**: configurable (``estimator``), defaulting to
  ``random_forest``, matching the notebook's model-zoo comparison; also
  supports ``logreg_l2``, ``logreg_l1``, ``elasticnet``, ``gradient_boosting``.
* **Target column**: like ``helios.stages.mil``/``helios.stages.risk``, the
  outcome column is picked by the stage via a ``target`` parameter (the
  notebooks hardcoded ``disease_pfs_recurred``; the stub's ``labels`` argument
  is the raw per-slide metadata table with no target column named, so this adds
  ``target`` rather than guessing a hardcoded column inside the component).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import faiss
import h5py
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from helios.models.base import Model, register_model

#: Outcome column used for cluster-significance testing + the risk model,
#: matching helios.stages.mil's DEFAULT_TARGET.
DEFAULT_TARGET: str = "disease_pfs_recurrence_5yfu"

#: Risk estimators available for the cluster-presence model.
MorphologyEstimator = Literal["random_forest", "logreg_l2", "logreg_l1", "elasticnet", "gradient_boosting"]
MORPHOLOGY_ESTIMATORS: tuple[str, ...] = (
    "random_forest",
    "logreg_l2",
    "logreg_l1",
    "elasticnet",
    "gradient_boosting",
)


def train_morphology(
    patch_embedding_paths: dict[str, Path],
    attention: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    attention_cutoff: float = 0.8,
    n_pca: int = 50,
    k_micro: int = 3000,
    k_neighbors: int = 20,
    resolution: float = 1.0,
    significance_alpha: float = 0.05,
    estimator: MorphologyEstimator = "random_forest",
    seed: int = 42,
) -> Model:
    """Fit the morphology clustering + cluster-presence risk model for one fold.

    GRAIN: one fold (its per-fold patch embeddings/attention) per call.

    Parameters
    ----------
    patch_embedding_paths:
        ``image_id -> aligned_patch_embeddings path`` (streamed lazily).
    attention:
        Per-tile attention scores, one row per (image_id, tile_id) — used for
        the cumulative-attention-mass filter before clustering.
    labels:
        Per-slide metadata (one row per image_id); ``target`` selects the
        outcome column used for cluster significance + the risk model.
    attention_cutoff:
        Keep the smallest set of each slide's tiles whose attention mass
        reaches this fraction of the slide's total attention.
    n_pca:
        PCA component count prior to clustering.
    k_micro:
        FAISS spherical k-means over-cluster count (micro-clusters).
    k_neighbors:
        k-NN graph neighbors per micro-cluster centroid, for Leiden.
    resolution:
        Leiden resolution parameter (one clustering, not a sweep).
    significance_alpha:
        BH-adjusted p-value threshold for a cluster's presence to be
        significantly associated with ``target`` (Mann-Whitney U); clusters
        failing this are excluded as risk-model features (but still reported
        in ``patch_morphology_presence``).
    estimator:
        Which classifier fits the cluster-presence risk model (see
        :data:`MORPHOLOGY_ESTIMATORS`).
    seed:
        Random seed (PCA, k-means, Leiden, estimator).

    Returns the fitted bundle (PCA + Leiden + cluster-exclusion + risk model).
    """
    if estimator not in MORPHOLOGY_ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}. Available: {list(MORPHOLOGY_ESTIMATORS)}.")

    tile_reps, tile_meta = _load_and_filter_tiles(patch_embedding_paths, attention, attention_cutoff=attention_cutoff)
    if tile_reps.shape[0] == 0:
        raise ValueError("No tiles survived the attention-cutoff filter across all slides.")

    n_pca_eff = min(n_pca, tile_reps.shape[0], tile_reps.shape[1])
    reps_pca, pca_mean, pca_components = _fit_pca(tile_reps, n_pca=n_pca_eff, seed=seed)

    k_micro_eff = max(1, min(k_micro, reps_pca.shape[0]))
    micro_labels, micro_centroids = _faiss_kmeans_overcluster(reps_pca, k=k_micro_eff, seed=seed)
    micro_to_community = _leiden_communities(micro_centroids, k_neighbors=k_neighbors, resolution=resolution, seed=seed)

    n_clusters = int(micro_to_community.max()) + 1 if len(micro_to_community) else 0
    tile_meta = tile_meta.copy()
    tile_meta["cluster"] = micro_to_community[micro_labels]

    cluster_centroids, valid = _compute_cluster_centroids(reps_pca, tile_meta["cluster"].to_numpy(), n_clusters)
    tile_meta["cos_weight"] = _cosine_weight(reps_pca, tile_meta["cluster"].to_numpy(), cluster_centroids, valid)

    presence = _build_presence_table(tile_meta, n_clusters, valid)
    y_by_image = labels.set_index("image_id")[target]
    presence[target] = presence["image_id"].map(y_by_image)
    presence = presence.dropna(subset=[target])
    presence[target] = presence[target].astype(int)

    significant = _significant_clusters(presence, valid, target=target, alpha=significance_alpha)
    feature_cluster_ids = sorted(significant)
    feat_cols = [f"cluster_{c}_presence" for c in feature_cluster_ids]

    est, needs_scaling = _make_estimator(estimator, seed=seed)
    scaler: StandardScaler | None = None
    x = presence[feat_cols].to_numpy(dtype=np.float64)
    y = presence[target].to_numpy(dtype=int)
    if needs_scaling:
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
    est.fit(x, y)

    return MorphologyModel(
        pca_mean=pca_mean,
        pca_components=pca_components,
        micro_centroids=micro_centroids,
        micro_to_community=micro_to_community,
        cluster_centroids=cluster_centroids,
        valid_clusters=valid,
        feature_cluster_ids=feature_cluster_ids,
        estimator_name=estimator,
        estimator=est,
        scaler=scaler,
        attention_cutoff=attention_cutoff,
    )


def infer_morphology(
    models: dict[int | None, Model],
    patch_embedding_paths: dict[int | None, dict[str, Path]],
    attention: dict[int | None, pd.DataFrame],
    cv_splits: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign clusters, score cluster presence, and produce morphology risk.

    GRAIN: the scored cohort per call. Inputs are keyed by fold (each fold's
    model + that fold's per-image patch-embedding paths + attention).

    Reduction differs by output, because cluster IDs live in a *per-fold*
    embedding space (fold A's cluster 5 is not fold B's cluster 5) while a risk
    score is a comparable scalar across folds:

    * ``tile_clusters`` / ``patch_morphology_presence`` — OOF-select the
      image's test-fold assignment on a CV cohort; on deploy (``cv_splits`` is
      ``None``, no test-fold concept) fall back to the first fold's model as
      the single canonical clustering.
    * ``patch_morphology_risk_score`` — OOF-select the test-fold score on a CV
      cohort / ensemble-average across folds on deploy, carrying a ``fold``
      provenance column (matching every other HELIOS risk reduction).

    Returns
    -------
    tuple
        ``(tile_clusters, patch_morphology_presence, patch_morphology_risk_score)``;
        clusters/presence are fold-free, the risk score carries a ``fold``
        provenance column.
    """
    folds = sorted(models.keys(), key=lambda f: (f is None, f))
    if not folds:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=["image_id", "risk_score", "fold"])

    image_ids = sorted(next(iter(patch_embedding_paths.values())).keys())

    clusters_per_fold: dict[int | None, pd.DataFrame] = {}
    presence_per_fold: dict[int | None, pd.DataFrame] = {}
    risk_rows: list[dict[str, object]] = []

    for fold in folds:
        model = models[fold]
        assert isinstance(model, MorphologyModel)
        fold_attention = attention[fold]
        fold_clusters, fold_presence, fold_risk = [], [], []
        for image_id in image_ids:
            path = patch_embedding_paths[fold].get(image_id)
            if path is None:
                continue
            att = fold_attention[fold_attention["image_id"] == image_id]
            clusters_df, presence_row, risk_score = model.predict((path, att))
            clusters_df = clusters_df.assign(image_id=image_id)
            fold_clusters.append(clusters_df)
            fold_presence.append({"image_id": image_id, **presence_row})
            fold_risk.append({"image_id": image_id, "risk_score": risk_score, "fold": fold})
        clusters_per_fold[fold] = (
            pd.concat(fold_clusters, ignore_index=True) if fold_clusters else pd.DataFrame()
        )
        presence_per_fold[fold] = pd.DataFrame(fold_presence)
        risk_rows.extend(fold_risk)

    risk_df = pd.DataFrame(risk_rows)
    risk_reduced = _reduce_risk(risk_df, cv_splits)

    canonical_fold = _oof_fold_map(image_ids, cv_splits) if cv_splits is not None else None
    tile_clusters = _select_canonical(clusters_per_fold, image_ids, canonical_fold, id_col="image_id")
    presence = _select_canonical(presence_per_fold, image_ids, canonical_fold, id_col="image_id")

    return tile_clusters, presence, risk_reduced


# =================================================================
# Model bundle
# =================================================================
@register_model("morphology_cluster_presence")
class MorphologyModel(Model):
    """Fitted morphology bundle: PCA + FAISS/Leiden clustering + risk estimator."""

    def __init__(
        self,
        *,
        pca_mean: NDArray[np.float32],
        pca_components: NDArray[np.float32],
        micro_centroids: NDArray[np.float32],
        micro_to_community: NDArray[np.int32],
        cluster_centroids: NDArray[np.float32],
        valid_clusters: NDArray[np.bool_],
        feature_cluster_ids: list[int],
        estimator_name: str,
        estimator: Any,
        scaler: StandardScaler | None,
        attention_cutoff: float,
    ) -> None:
        self.pca_mean = pca_mean
        self.pca_components = pca_components
        self.micro_centroids = micro_centroids
        self.micro_to_community = micro_to_community
        self.cluster_centroids = cluster_centroids
        self.valid_clusters = valid_clusters
        self.feature_cluster_ids = feature_cluster_ids
        self.estimator_name = estimator_name
        self.estimator = estimator
        self.scaler = scaler
        self.attention_cutoff = attention_cutoff

    def fit(self, inputs: Any, **params: Any) -> MorphologyModel:
        raise NotImplementedError("Use train_morphology to construct a fitted MorphologyModel.")

    def predict(self, inputs: tuple[Path, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float], float]:
        """Score ONE image: full-grid tile clusters, cluster presence, risk score.

        ``inputs`` is ``(patch_embeddings_path, attention_df_for_this_image)``.
        Cluster assignment covers ALL foreground tiles (for reporting); the
        presence table and risk score use only the attention-cutoff-filtered
        subset, matching how the model was trained.
        """
        path, image_attention = inputs
        reps = _read_patch_embeddings(path)
        image_attention = image_attention.sort_values("tile_id")
        tile_ids = image_attention["tile_id"].to_numpy()
        attn_values = image_attention["attention"].to_numpy()

        micro, _sim = _project_and_assign(reps, self.pca_mean, self.pca_components, self.micro_centroids)
        community = self.micro_to_community[micro]
        tile_clusters = pd.DataFrame({"tile_id": tile_ids, "cluster": community})

        keep = _cumulative_attention_mask(attn_values, cutoff=self.attention_cutoff)
        reps_pca_filtered = _project_pca(reps[keep], self.pca_mean, self.pca_components)
        community_filtered = community[keep]
        cos_weight = _cosine_weight(
            reps_pca_filtered, community_filtered, self.cluster_centroids, self.valid_clusters
        )
        tile_meta = pd.DataFrame(
            {"image_id": "_single", "cluster": community_filtered, "cos_weight": cos_weight}
        )
        n_clusters = len(self.valid_clusters)
        presence_df = _build_presence_table(tile_meta, n_clusters, self.valid_clusters)
        presence_row = {
            c: (
                float(presence_df.loc[0, f"cluster_{c}_presence"])
                if f"cluster_{c}_presence" in presence_df.columns and len(presence_df)
                else 0.0
            )
            for c in np.where(self.valid_clusters)[0]
        }

        feat_cols = [f"cluster_{c}_presence" for c in self.feature_cluster_ids]
        x = np.array([[presence_row.get(c, 0.0) for c in self.feature_cluster_ids]], dtype=np.float64)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        risk_score = float(self.estimator.predict_proba(x)[0, 1]) if feat_cols else 0.5

        presence_named = {f"cluster_{c}_presence": v for c, v in presence_row.items()}
        return tile_clusters, presence_named, risk_score


# =================================================================
# Internal helpers
# =================================================================
def _read_patch_embeddings(path: Path) -> NDArray[np.float32]:
    with h5py.File(path, "r") as h:
        return np.asarray(h["embeddings"][()], dtype=np.float32)


def _cumulative_attention_mask(attention: NDArray[np.float64], cutoff: float) -> NDArray[np.bool_]:
    """Keep the smallest set of tiles whose attention mass reaches ``cutoff``."""
    total = attention.sum()
    if total <= 0:
        return np.ones(len(attention), dtype=bool)
    order = np.argsort(attention)[::-1]
    cumulative = np.cumsum(attention[order]) / total
    n_keep = min(int(np.searchsorted(cumulative, cutoff)) + 1, len(attention))
    mask = np.zeros(len(attention), dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def _l2_normalize(x: NDArray[np.float32], eps: float = 1e-12) -> NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _load_and_filter_tiles(
    patch_embedding_paths: dict[str, Path], attention: pd.DataFrame, *, attention_cutoff: float
) -> tuple[NDArray[np.float32], pd.DataFrame]:
    reps_list: list[NDArray[np.float32]] = []
    meta_rows: list[pd.DataFrame] = []
    for image_id, path in patch_embedding_paths.items():
        image_attention = attention[attention["image_id"] == image_id].sort_values("tile_id")
        if image_attention.empty:
            continue
        reps = _read_patch_embeddings(path)
        keep = _cumulative_attention_mask(image_attention["attention"].to_numpy(), cutoff=attention_cutoff)
        tile_ids = image_attention["tile_id"].to_numpy()[keep]
        reps_list.append(reps[keep])
        meta_rows.append(pd.DataFrame({"image_id": image_id, "tile_id": tile_ids}))
    if not reps_list:
        return np.empty((0, 0), dtype=np.float32), pd.DataFrame(columns=["image_id", "tile_id"])
    return np.concatenate(reps_list, axis=0), pd.concat(meta_rows, ignore_index=True)


def _fit_pca(
    reps: NDArray[np.float32], *, n_pca: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    reps_n = _l2_normalize(reps.astype(np.float32))
    pca = PCA(n_components=n_pca, random_state=seed)
    reps_pca = pca.fit_transform(reps_n).astype(np.float32)
    reps_pca = _l2_normalize(reps_pca)
    return reps_pca, pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def _project_pca(
    reps: NDArray[np.float32], pca_mean: NDArray[np.float32], pca_components: NDArray[np.float32]
) -> NDArray[np.float32]:
    x = _l2_normalize(reps.astype(np.float32))
    z = (x - pca_mean) @ pca_components.T
    return _l2_normalize(z.astype(np.float32))


def _faiss_kmeans_overcluster(
    reps_pca: NDArray[np.float32], *, k: int, seed: int, niter: int = 25
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    n, d = reps_pca.shape
    kmeans = faiss.Kmeans(d=d, k=k, niter=niter, seed=seed, gpu=False, spherical=True, verbose=False)
    kmeans.train(np.ascontiguousarray(reps_pca))
    centroids = _l2_normalize(kmeans.centroids.astype(np.float32))
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(centroids))
    _, assignment = index.search(np.ascontiguousarray(reps_pca), 1)
    return assignment[:, 0].astype(np.int32), centroids


def _project_and_assign(
    reps: NDArray[np.float32],
    pca_mean: NDArray[np.float32],
    pca_components: NDArray[np.float32],
    centroids: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    z = _project_pca(reps, pca_mean, pca_components)
    index = faiss.IndexFlatIP(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids.astype(np.float32)))
    sims, idx = index.search(np.ascontiguousarray(z), 1)
    return idx[:, 0].astype(np.int32), sims[:, 0].astype(np.float32)


def _build_knn_graph(centroids: NDArray[np.float32], *, k_neighbors: int) -> ig.Graph:
    n, d = centroids.shape
    k = max(1, min(k_neighbors, n - 1))
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(centroids))
    sims, nbrs = index.search(np.ascontiguousarray(centroids), k + 1)
    sims, nbrs = sims[:, 1:], nbrs[:, 1:]  # drop self-loop

    edges, weights, seen = [], [], set()
    for i in range(n):
        for j, s in zip(nbrs[i], sims[i], strict=True):
            if j < 0 or j == i:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b))
            weights.append(float(max(s, 1e-6)))

    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = weights
    return graph


def _leiden_communities(
    centroids: NDArray[np.float32], *, k_neighbors: int, resolution: float, seed: int
) -> NDArray[np.int32]:
    n = centroids.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32)
    graph = _build_knn_graph(centroids, k_neighbors=k_neighbors)
    partition = la.find_partition(
        graph, la.RBConfigurationVertexPartition, weights=graph.es["weight"], resolution_parameter=resolution, seed=seed
    )
    return np.array(partition.membership, dtype=np.int32)


def _compute_cluster_centroids(
    reps_pca: NDArray[np.float32], cluster_labels: NDArray[np.int32], n_clusters: int
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    d = reps_pca.shape[1] if reps_pca.ndim == 2 else 0
    centroids = np.zeros((n_clusters, d), dtype=np.float32)
    valid = np.zeros(n_clusters, dtype=bool)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if not mask.any():
            continue
        mean_vec = reps_pca[mask].mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-12:
            centroids[c] = mean_vec / norm
            valid[c] = True
    return centroids, valid


def _cosine_weight(
    reps_pca: NDArray[np.float32],
    cluster_labels: NDArray[np.int32],
    cluster_centroids: NDArray[np.float32],
    valid: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Per-tile cosine similarity to its OWN cluster's centroid, rescaled to [0, 1]."""
    n = len(cluster_labels)
    weight = np.zeros(n, dtype=np.float32)
    valid_tile = (cluster_labels >= 0) & valid[np.clip(cluster_labels, 0, None)]
    idx = np.where(valid_tile)[0]
    if len(idx) == 0:
        return weight
    cos = np.einsum(
        "ij,ij->i", reps_pca[idx].astype(np.float64), cluster_centroids[cluster_labels[idx]].astype(np.float64)
    )
    weight[idx] = ((cos + 1.0) / 2.0).astype(np.float32)
    return weight


def _build_presence_table(tile_meta: pd.DataFrame, n_clusters: int, valid: NDArray[np.bool_]) -> pd.DataFrame:
    """Per-slide, cosine-weighted, L1-normalized cluster presence (one row per image_id)."""
    slide_ids = tile_meta["image_id"].unique()
    s_row = {s: i for i, s in enumerate(slide_ids)}
    row_per_tile = tile_meta["image_id"].map(s_row).to_numpy()
    n_slides = len(slide_ids)

    clusters = tile_meta["cluster"].to_numpy()
    weights = tile_meta["cos_weight"].to_numpy()
    w = np.zeros((n_slides, max(n_clusters, 1)), dtype=np.float64)
    m = clusters >= 0
    if m.any():
        np.add.at(w, (row_per_tile[m], clusters[m]), weights[m])

    row_denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    w_norm = w / row_denom

    out = pd.DataFrame({"image_id": slide_ids})
    for c in range(n_clusters):
        if valid[c]:
            out[f"cluster_{c}_presence"] = w_norm[:, c]
    return out


def _significant_clusters(presence: pd.DataFrame, valid: NDArray[np.bool_], *, target: str, alpha: float) -> set[int]:
    """Mann-Whitney U per cluster (target=1 vs target=0), BH-corrected."""
    neg = presence[presence[target] == 0]
    pos = presence[presence[target] == 1]
    if len(neg) == 0 or len(pos) == 0:
        return set(np.where(valid)[0].tolist())

    cluster_ids, p_values = [], []
    for c in np.where(valid)[0]:
        col = f"cluster_{c}_presence"
        if col not in presence.columns:
            continue
        x0, x1 = neg[col].to_numpy(), pos[col].to_numpy()
        if x0.std() == 0 and x1.std() == 0:
            continue
        _, p = mannwhitneyu(x1, x0, alternative="two-sided")
        cluster_ids.append(int(c))
        p_values.append(float(p))

    if not p_values:
        return set(np.where(valid)[0].tolist())

    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    significant = {c for c, p in zip(cluster_ids, p_adj, strict=True) if p < alpha}
    return significant if significant else set(np.where(valid)[0].tolist())


def _make_estimator(name: str, *, seed: int) -> tuple[Any, bool]:
    """Return ``(estimator, needs_scaling)``."""
    if name == "logreg_l2":
        return (
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=seed),
            True,
        )
    if name == "logreg_l1":
        return (
            LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=5000, class_weight="balanced", random_state=seed
            ),
            True,
        )
    if name == "elasticnet":
        return (
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                solver="saga",
                max_iter=10000,
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        )
    if name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        )
    if name == "gradient_boosting":
        return (
            GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed
            ),
            False,
        )
    raise ValueError(f"Unknown estimator {name!r}. Available: {list(MORPHOLOGY_ESTIMATORS)}.")


def _reduce_risk(risk_df: pd.DataFrame, cv_splits: pd.DataFrame | None) -> pd.DataFrame:
    if risk_df.empty:
        return pd.DataFrame(columns=["image_id", "risk_score", "fold"])
    if cv_splits is None:
        reduced = risk_df.groupby("image_id", as_index=False)["risk_score"].mean()
        reduced["fold"] = None
        return reduced
    test_rows = cv_splits.loc[cv_splits["split"] == "test", ["image_id", "fold"]]
    reduced = test_rows.merge(risk_df, on=["image_id", "fold"], how="left")
    return reduced[["image_id", "risk_score", "fold"]]


def _oof_fold_map(image_ids: list[str], cv_splits: pd.DataFrame) -> dict[str, int]:
    test_rows = cv_splits.loc[cv_splits["split"] == "test", ["image_id", "fold"]]
    return dict(zip(test_rows["image_id"], test_rows["fold"], strict=False))


def _select_canonical(
    per_fold: dict[int | None, pd.DataFrame],
    image_ids: list[str],
    canonical_fold: dict[str, int] | None,
    *,
    id_col: str,
) -> pd.DataFrame:
    """Pick one fold's rows per image: OOF test-fold (CV) / first fold (deploy)."""
    if not per_fold:
        return pd.DataFrame()
    first_fold = next(iter(per_fold))
    if canonical_fold is None:
        return per_fold[first_fold]

    parts = []
    for image_id in image_ids:
        fold = canonical_fold.get(image_id, first_fold)
        frame = per_fold.get(fold, per_fold[first_fold])
        parts.append(frame[frame[id_col] == image_id])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
