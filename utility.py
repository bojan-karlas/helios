"""
Utility functions for tile-level attention & gradient analysis.

Organized by section:
  1. Tile I/O & per-slide loading
  2. Clustering pipeline (PCA + FAISS k-means + Leiden)
  3. Representative-tile visualization
  4. Marginal contribution analysis (re-run ensemble without each cluster)
  5. Cluster presence per slide + significance testing
  6. Logistic regression on cluster presence (with external validation)
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tile I/O
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tile_images(slide_id: str, tile_dir: str) -> np.ndarray:
    """Load the raw 224x224 tile array for a slide.
    Handles both nested ({tile_dir}/{slide_id}/{slide_id}.hdf5) and
    flat ({tile_dir}/{slide_id}.hdf5) layouts."""
    nested = os.path.join(tile_dir, slide_id, f"{slide_id}.hdf5")
    flat   = os.path.join(tile_dir, f"{slide_id}.hdf5")
    path = nested if os.path.exists(nested) else flat
    with h5py.File(path, "r") as f:
        return f["tile_224"][:]


def load_foreground_indices(slide_id: str, mask_dir: str) -> np.ndarray:
    """
    Load background mask and return the raw indices of foreground tiles (bg == 0).
    These raw indices index into the full tile array from load_tile_images.
    """
    df = pd.read_parquet(os.path.join(mask_dir, f"{slide_id}.parquet"))
    bg = df["tile_background"].astype(np.int64).values
    return np.where(bg == 0)[0]


def load_slide_model_data(
    slide_id: str, tile_features_dir: str, model_num: int
) -> Optional[dict]:
    """
    Load attention, representations, and per-tile gradient magnitude for one
    (slide, model). Returns a dict (or None if the HDF5 doesn't exist) with:
        attention, representations, grad_magnitude, hybrid_score,
        prob, y_true
    The hybrid score is `attention * grad_magnitude` (per-tile, signed).
    """
    path = os.path.join(tile_features_dir, f"{slide_id}_model{model_num}.hdf5")
    if not os.path.exists(path):
        print(f"Tile features not found for {slide_id}")
        return None

    with h5py.File(path, "r") as f:
        att = f["weighted_attention"][:]
        reps = f["tile_representations"][:]
        grad_magnitude = f["tile_gradients"][:]
        hybrid_score = att * grad_magnitude
        return {
            "attention":       att,
            "representations": reps,
            "grad_magnitude":  grad_magnitude,
            "hybrid_score":    hybrid_score,
            "prob":            float(f.attrs["prob"]),
            "y_true":          int(f.attrs["y_true"]),
        }


def cumulative_attention_mask(attention: np.ndarray, cutoff: float = 0.9) -> np.ndarray:
    """
    Keep the smallest set of tiles whose attention mass reaches `cutoff` of the
    slide total. Returns a boolean mask aligned with `attention`.
    """
    total = attention.sum()
    if total == 0:
        return np.ones(len(attention), dtype=bool)

    sorted_idx = np.argsort(attention)[::-1]
    cumulative = np.cumsum(attention[sorted_idx]) / total
    n_keep = min(int(np.searchsorted(cumulative, cutoff)) + 1, len(attention))

    mask = np.zeros(len(attention), dtype=bool)
    mask[sorted_idx[:n_keep]] = True
    return mask


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize a tile image to [0, 1] (HWC) for display."""
    img = np.array(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clustering pipeline
# ─────────────────────────────────────────────────────────────────────────────

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows. After this, Euclidean distance ↔ cosine distance."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def preprocess(reps: np.ndarray, n_pca: int = 256, random_state: int = 42):
    """L2-normalize → PCA → L2-normalize again (so Euclidean ≈ cosine)."""
    from sklearn.decomposition import PCA

    print(f"[preprocess] L2-normalizing {reps.shape[0]:,} × {reps.shape[1]} embeddings...")
    reps_n = l2_normalize(reps.astype(np.float32))

    print(f"[preprocess] PCA → {n_pca} components...")
    pca = PCA(n_components=n_pca, random_state=random_state)
    reps_pca = pca.fit_transform(reps_n).astype(np.float32)
    cum_var = pca.explained_variance_ratio_.cumsum()
    print(f"  Cumulative explained variance: {cum_var[-1]:.3%}")
    print(f"  Variance at 50/100/200 PCs: "
          f"{cum_var[min(49, n_pca-1)]:.2%} / "
          f"{cum_var[min(99, n_pca-1)]:.2%} / "
          f"{cum_var[min(199, n_pca-1)]:.2%}")
    return l2_normalize(reps_pca), pca


def plot_scree(pca, save_path: str = "scree_plot.png") -> None:
    """Cumulative explained variance — use this to pick n_pca."""
    cum = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, linewidth=2)
    for thresh in (0.90, 0.95, 0.99):
        ax.axhline(thresh, color="grey", linestyle="--", alpha=0.5)
        ax.text(len(cum) * 0.98, thresh + 0.005, f"{thresh:.0%}",
                ha="right", fontsize=9, color="grey")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA scree (post L2-normalization)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def faiss_kmeans_overcluster(
    reps_pca: np.ndarray,
    k: int = 3000,
    niter: int = 25,
    nredo: int = 1,
    seed: int = 42,
    use_gpu: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Over-clustering with spherical FAISS k-means. A large k (2000–5000) keeps
    each micro-cluster small enough that k-means' spherical assumption is harmless.
    Returns (micro_labels [N], centroids [k, D] L2-normalized).
    """
    import faiss

    N, D = reps_pca.shape
    print(f"[faiss-kmeans] k={k}, niter={niter}, nredo={nredo} on {N:,}×{D}...")

    kmeans = faiss.Kmeans(
        d=D, k=k, niter=niter, nredo=nredo,
        seed=seed, gpu=use_gpu, verbose=verbose,
        spherical=True,
    )
    kmeans.train(np.ascontiguousarray(reps_pca))

    centroids = l2_normalize(kmeans.centroids.astype(np.float32))
    index = faiss.IndexFlatIP(D)
    index.add(np.ascontiguousarray(centroids))
    _, I = index.search(np.ascontiguousarray(reps_pca), 1)
    micro_labels = I[:, 0].astype(np.int32)

    sizes = np.bincount(micro_labels, minlength=k)
    print(f"  Micro-cluster sizes — min: {sizes.min()}, median: {int(np.median(sizes))}, "
          f"max: {sizes.max()}, empty: {(sizes == 0).sum()}")
    return micro_labels, centroids


def build_knn_graph(centroids: np.ndarray, k_neighbors: int = 20):
    """
    Symmetric k-NN graph on centroids using cosine similarity. Returns an
    igraph.Graph with edge weights = similarities (clipped to >0).
    """
    import faiss
    import igraph as ig

    n, d = centroids.shape
    print(f"[knn-graph] {n} centroids, k_neighbors={k_neighbors}...")
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(centroids))
    sims, nbrs = index.search(np.ascontiguousarray(centroids), k_neighbors + 1)
    sims, nbrs = sims[:, 1:], nbrs[:, 1:]   # drop self-loop

    edges, weights, seen = [], [], set()
    for i in range(n):
        for j, s in zip(nbrs[i], sims[i]):
            if j < 0 or j == i:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b))
            weights.append(float(max(s, 1e-6)))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges, "
          f"avg degree {2 * g.ecount() / g.vcount():.1f}")
    return g


def leiden_multi_resolution(
    g, resolutions: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0), seed: int = 42,
) -> Dict[float, np.ndarray]:
    """Run Leiden at each resolution. Returns {resolution: centroid_community_array}."""
    import leidenalg as la

    results = {}
    for res in resolutions:
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights=g.es["weight"], resolution_parameter=res, seed=seed,
        )
        comm = np.array(part.membership)
        results[res] = comm
        print(f"[leiden] resolution={res}: {comm.max()+1} communities, "
              f"modularity={part.modularity:.3f}")
    return results


def propagate_to_tiles(micro_labels: np.ndarray, centroid_communities: np.ndarray) -> np.ndarray:
    """Map each tile's micro-cluster to its centroid's community label."""
    return centroid_communities[micro_labels]


def stability_check(
    reps_pca: np.ndarray,
    k_micro: int = 3000,
    k_neighbors: int = 20,
    resolution: float = 1.0,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    use_gpu: bool = True,
):
    """
    Run the full pipeline once per seed; report pairwise ARI/AMI on a subsample.
    Returns (runs, ari_matrix, ami_matrix).
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    runs = []
    for seed in seeds:
        print(f"\n=== Stability run, seed={seed} ===")
        micro, cents = faiss_kmeans_overcluster(
            reps_pca, k=k_micro, seed=seed, use_gpu=use_gpu, verbose=False,
        )
        g = build_knn_graph(cents, k_neighbors=k_neighbors)
        comm = leiden_multi_resolution(g, resolutions=(resolution,), seed=seed)[resolution]
        runs.append(propagate_to_tiles(micro, comm))

    n_runs = len(runs)
    ari = np.zeros((n_runs, n_runs))
    ami = np.zeros((n_runs, n_runs))
    sub_idx = np.random.default_rng(0).choice(
        len(runs[0]), size=min(100_000, len(runs[0])), replace=False
    )
    for i in range(n_runs):
        for j in range(i, n_runs):
            ari[i, j] = ari[j, i] = adjusted_rand_score(runs[i][sub_idx], runs[j][sub_idx])
            ami[i, j] = ami[j, i] = adjusted_mutual_info_score(runs[i][sub_idx], runs[j][sub_idx])

    print("\nPairwise ARI (upper triangle is what matters):")
    print(np.round(ari, 3))
    print("\nPairwise AMI:")
    print(np.round(ami, 3))
    print(f"\nMean off-diagonal ARI: {ari[np.triu_indices(n_runs, k=1)].mean():.3f}")
    print(f"Mean off-diagonal AMI: {ami[np.triu_indices(n_runs, k=1)].mean():.3f}")
    print("Rule of thumb: >0.7 is solid, 0.5–0.7 is OK, <0.5 means clusters are unstable.")
    return runs, ari, ami


def run_full_pipeline(
    reps: np.ndarray,
    meta_df: pd.DataFrame,
    n_pca: int = 256,
    k_micro: int = 3000,
    k_neighbors: int = 20,
    resolutions: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    seed: int = 42,
    use_gpu: bool = True,
):
    """
    Run preprocess → over-cluster → kNN graph → multi-resolution Leiden.
    Adds 'micro_cluster' and one 'cluster_res{r}' column per resolution to
    a copy of meta_df, and returns (meta_df, reps_pca, centroids, leiden_results).
    """
    assert len(reps) == len(meta_df), "reps and meta_df must align row-wise"

    reps_pca, pca = preprocess(reps, n_pca=n_pca, random_state=seed)
    plot_scree(pca)

    meta_df = meta_df.copy()
    print(f"\n[main] Clustering {len(meta_df):,} tiles")

    micro_labels, centroids = faiss_kmeans_overcluster(
        reps_pca, k=k_micro, seed=seed, use_gpu=use_gpu,
    )
    g = build_knn_graph(centroids, k_neighbors=k_neighbors)
    leiden_results = leiden_multi_resolution(g, resolutions=resolutions, seed=seed)

    meta_df["micro_cluster"] = micro_labels.astype(np.int32)
    for res, comm in leiden_results.items():
        col = f"cluster_res{res}"
        meta_df[col] = comm[micro_labels].astype(np.int32)
        sizes = meta_df[col].value_counts().sort_index()
        print(f"\n  {col}: {len(sizes)} clusters")
        print(sizes.head(15).to_string())

    return meta_df, reps_pca, centroids, leiden_results

def _load_tile_coords(
    slide_id: str,
    tile_metadata_dir: str,
    row_col_cols: Tuple[str, str] = ("tile_row", "tile_col"),
):
    """
    Return a DataFrame of per-tile coordinates for a slide, indexed by raw
    tile position (tile_idx). Assumes one parquet per slide at
    {tile_metadata_dir}/{slide_id}.parquet, ordered by tile position.
    Returns None if the file doesn't exist.
    """
    if tile_metadata_dir is None:
        return None
    path = os.path.join(tile_metadata_dir, f"{slide_id}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  [!] Could not read tile_metadata for {slide_id}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3. Representative-tile visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_representative_tiles(
    meta_df: pd.DataFrame,
    reps: np.ndarray,
    cluster_col: str = "cluster_res1.0",
    top_k_clusters: int = 30,
    n_tiles: int = 20,
    ncols: int = 10,
    save_dir: str = ".",
    show: bool = True,
    tile_metadata_dirs: Optional[Dict[str, str]] = None,
    row_col_cols: Tuple[str, str] = ("tile_row", "tile_col"),
    csv_path: Optional[str] = "displayed_tiles.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each of the top-K largest clusters, show:
      - Row 1 ("Representative tiles"): the N tiles closest to the cluster
        centroid in cosine distance.
      - Row 2 ("Random tiles"): N tiles sampled at random from the cluster.
    Each tile is numbered 1..N in its top-left corner and titled with its
    full slide name. `reps` must be row-aligned with `meta_df`.

    A CSV is written recording, for every displayed tile:
      cluster_col, cluster_id, panel ('representative'|'random'),
      position (1..N), slide_id, tile_idx, tile_row, tile_col, cohort.

    Parameters
    ----------
    tile_metadata_dirs : {cohort: tile_metadata_dir}. Used to resolve
        tile_row / tile_col from tile_idx. If None or a cohort is missing,
        row/col are recorded as NaN.
    row_col_cols : the (row_column_name, col_column_name) in the tile_metadata
        parquet. Adjust to match your schema.
    """
    assert len(meta_df) == len(reps), "meta_df and reps must align row-wise"
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    row_name, col_name = row_col_cols

    nrows_per_panel = (n_tiles + ncols - 1) // ncols
    plot_df = meta_df.reset_index(drop=True).copy()

    sizes = plot_df[cluster_col].value_counts().drop(index=-1, errors="ignore")
    top_clusters = sizes.head(top_k_clusters).index.tolist()
    print(f"Visualizing top {len(top_clusters)} clusters by size "
          f"(from column '{cluster_col}')")

    # Cache of slide_id -> coord DataFrame so we don't re-read per tile
    coord_cache: Dict[str, Optional[pd.DataFrame]] = {}

    def get_coords(sid, cohort, tile_idx):
        """Return (tile_row, tile_col) for a tile, or (nan, nan)."""
        if sid not in coord_cache:
            tm_dir = None
            if tile_metadata_dirs is not None:
                tm_dir = tile_metadata_dirs.get(cohort)
            coord_cache[sid] = _load_tile_coords(sid, tm_dir, row_col_cols)
        cdf = coord_cache[sid]
        if cdf is None or tile_idx >= len(cdf):
            return (np.nan, np.nan)
        try:
            r = cdf.iloc[tile_idx]
            return (r[row_name], r[col_name])
        except Exception:
            return (np.nan, np.nan)

    csv_records = []

    for rank, cluster_id in enumerate(top_clusters, start=1):
        fname = os.path.join(save_dir, f"representative_tiles_{cluster_col}_c{cluster_id}.png")
        # if there is this file, do not continue
        if Path(fname).exists():
            continue
        idx = np.where(plot_df[cluster_col].values == cluster_id)[0]
        if len(idx) == 0:
            continue

        # --- Representative tiles: closest to cluster centroid (cosine) ---
        cluster_reps = l2_normalize(np.asarray(reps[idx], dtype=np.float32))
        centroid = _l2_normalize_vec(cluster_reps.mean(axis=0))
        cos_dist = 1.0 - cluster_reps @ centroid
        rep_order = np.argsort(cos_dist)[:n_tiles]
        rep_global_idx = idx[rep_order]

        # --- Random tiles: sample from the cluster ---
        n_rand = min(n_tiles, len(idx))
        rand_local = rng.choice(len(idx), size=n_rand, replace=False)
        rand_global_idx = idx[rand_local]

        print(f"\n{'='*70}")
        print(f"[{rank}/{len(top_clusters)}] Cluster {cluster_id} — {len(idx):,} tiles")

        # Figure: 2 panels (representative on top, random below)
        total_rows = nrows_per_panel * 2
        fig, axes = plt.subplots(
            total_rows, ncols,
            figsize=(ncols * 2.2, total_rows * 2.4),
        )
        axes = np.atleast_1d(axes).reshape(total_rows, ncols)

        panels = [
            ("Representative tiles", rep_global_idx, "representative"),
            ("Random tiles",        rand_global_idx, "random"),
        ]

        for panel_i, (panel_label, global_indices, panel_key) in enumerate(panels):
            row_offset    = panel_i * nrows_per_panel
            number_offset = panel_i * n_tiles      # 0 for representative, n_tiles for random

            # Row label on the left of the panel's first row
            axes[row_offset, 0].annotate(
                panel_label,
                xy=(0, 0.5), xytext=(-0.35, 0.5),
                textcoords="axes fraction", xycoords="axes fraction",
                rotation=90, ha="center", va="center",
                fontsize=12, fontweight="bold",
            )

            for display_pos, global_i in enumerate(global_indices):
                grid_r = row_offset + display_pos // ncols
                grid_c = display_pos % ncols
                ax = axes[grid_r, grid_c]

                number = number_offset + display_pos + 1   # continuous: 1..N, then N+1..2N

                row = plot_df.iloc[global_i]
                sid      = row["slide_id"]
                tile_idx = int(row["tile_idx"])
                tile_dir = row["tile_dir"]
                cohort   = row.get("cohort", None)

                tr, tc = get_coords(sid, cohort, tile_idx)

                csv_records.append({
                    "cluster_col": cluster_col,
                    "cluster_id":  int(cluster_id),
                    "panel":       panel_key,
                    "position":    number,            # continuous numbering
                    "slide_id":    sid,
                    "tile_idx":    tile_idx,
                    "tile_row":    tr,
                    "tile_col":    tc,
                    "cohort":      cohort,
                })

                tiles = load_tile_images(sid, tile_dir)
                if tiles is None or tile_idx >= len(tiles):
                    ax.set_visible(False)
                    continue

                ax.imshow(normalize_image(tiles[tile_idx]))
                ax.set_title(str(sid), fontsize=7)
                ax.text(
                    0.04, 0.96, str(number),          # continuous number in overlay
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=11, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6),
                )
                ax.set_xticks([]); ax.set_yticks([])

            # Hide unused cells in this panel
            for display_pos in range(len(global_indices), nrows_per_panel * ncols):
                grid_r = row_offset + display_pos // ncols
                grid_c = display_pos % ncols
                axes[grid_r, grid_c].set_visible(False)

        plt.suptitle(
            f"Cluster {cluster_id} (rank {rank}, n={len(idx):,})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0.03, 0, 1, 0.98])
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show() if show else plt.close(fig)
        print(f"  Saved: {fname}")

    displayed_df = pd.DataFrame(csv_records)
    if csv_path:
        displayed_df.to_csv(csv_path, index=False)
        print(f"\nSaved tile index CSV: {csv_path}  ({len(displayed_df):,} rows)")
    return displayed_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Marginal contribution analysis
# ─────────────────────────────────────────────────────────────────────────────

def _load_fg_features(slide_id: str, feature_dir: str, mask_dir: str) -> Optional[np.ndarray]:
    f_path = os.path.join(feature_dir, slide_id, f"{slide_id}.hdf5")
    bg_path = os.path.join(mask_dir, f"{slide_id}.parquet")
    if not (os.path.exists(f_path) and os.path.exists(bg_path)):
        return None
    try:
        with h5py.File(f_path, "r") as f:
            feats_all = f["tile_224"][:].astype(np.float32)
        bg = pd.read_parquet(bg_path)["tile_background"].astype(np.int64).values
        return feats_all[bg == 0]
    except Exception as e:
        print(f"  [!] Failed to load {slide_id}: {e}")
        return None


def _run_ensemble(models, feats_fg: np.ndarray, device) -> float:
    import torch
    with torch.no_grad():
        x = torch.from_numpy(feats_fg).unsqueeze(0).to(device)
        probs = [float(m(x)[0].item()) for m in models]
    return float(np.mean(probs))


def compute_marginal_contributions(
    meta_df: pd.DataFrame,
    slide_filtered_data: dict,
    cluster_col: str,
    ensemble_models,
    raw_feature_dirs: Dict[str, str],
    mask_dirs: Dict[str, str],
    device,
    cohort_col: str = "cohort",
    slide_col: str = "slide_id",
    y_true_col: str = "y_true",
    save_csv: Optional[str] = "marginal_contributions.csv",
    min_cluster_frac=0.01,
) -> pd.DataFrame:
    """
    For each (slide, cluster) remove the cluster's filtered tiles from the
    slide's foreground feature matrix and re-run the ensemble.
    delta_p = P_full − P_without_cluster   (positive → cluster was pro-recurrence)

    Notes
    -----
    `meta_df` contains the post-cumulative-attention-filter tiles. So
    "remove cluster c" removes only the high-attention members of c from
    that slide, not all of c's tiles. fg_position is recovered from
    slide_filtered_data[sid]["fg_positions"], aligned row-wise with meta_df
    rows for that slide.
    """
    cluster_ids = sorted(int(c) for c in meta_df[cluster_col].unique() if c != -1)
    print(f"Computing marginal contributions for {len(cluster_ids)} clusters: {cluster_ids}")

    available_cohorts = set(raw_feature_dirs.keys())
    df = meta_df[meta_df[cohort_col].isin(available_cohorts)].copy().reset_index(drop=True)
    n_dropped_cohort = len(meta_df) - len(df)
    if n_dropped_cohort:
        print(f"  Dropped {n_dropped_cohort:,} tiles from cohorts not in raw_feature_dirs")
    print(f"  Slides in scope: {df[slide_col].nunique():,}  ({len(df):,} filtered tiles)")

    # Attach fg_position by joining each slide's rows to its fg_positions array.
    fg_pos = np.full(len(df), -1, dtype=np.int64)
    for sid, grp in df.groupby(slide_col, sort=False):
        if sid not in slide_filtered_data:
            print(f"  [!] {sid} not in slide_filtered_data, will skip")
            continue
        positions = slide_filtered_data[sid]["fg_positions"]
        if len(positions) != len(grp):
            print(f"  [!] {sid}: {len(positions)} fg_positions vs {len(grp)} meta_df rows — skipping")
            continue
        fg_pos[grp.index.values] = positions
    df["_fg_position"] = fg_pos
    df = df[df["_fg_position"] >= 0]

    has_label = y_true_col in df.columns
    records = []
    skipped_missing_features = 0
    skipped_below_threshold = 0

    for sid, grp in tqdm(df.groupby(slide_col, sort=False), desc="Marginal inference"):
        cohort = grp[cohort_col].iloc[0]
        y_true = int(grp[y_true_col].iloc[0]) if has_label else np.nan

        feats_fg = _load_fg_features(sid, raw_feature_dirs[cohort], mask_dirs[cohort])
        if feats_fg is None or len(feats_fg) == 0:
            skipped_missing_features += 1
            continue
        n_fg = len(feats_fg)

        p_full = _run_ensemble(ensemble_models, feats_fg, device)

        for c, sub in grp.groupby(cluster_col, sort=False):
            remove_pos = sub["_fg_position"].values.astype(np.int64)
            if remove_pos.size == 0:
                continue

            cluster_frac = remove_pos.size / n_fg
            if cluster_frac < min_cluster_frac:
                skipped_below_threshold += 1
                continue

            if remove_pos.max() >= n_fg or remove_pos.min() < 0:
                print(f"  [!] {sid} cluster {c}: fg_position out of range "
                      f"(max {remove_pos.max()}, n_fg={n_fg}). Skipping.")
                continue

            keep_mask = np.ones(n_fg, dtype=bool)
            keep_mask[remove_pos] = False
            if keep_mask.sum() == 0:
                continue

            p_without = _run_ensemble(ensemble_models, feats_fg[keep_mask], device)
            records.append({
                "slide_id":          sid,
                "cohort":            cohort,
                "y_true":            y_true,
                "cluster":           int(c),
                "p_full":            p_full,
                "p_without_cluster": p_without,
                "delta_p":           p_full - p_without,
                "n_tiles_removed":   int((~keep_mask).sum()),
                "n_tiles_fg_total":  n_fg,
                "pct_fg_removed":    float((~keep_mask).sum()) / n_fg,
            })

    if skipped_missing_features:
        print(f"  [!] Skipped {skipped_missing_features} slides (raw features or mask not found)")

    marginal_df = pd.DataFrame(records)
    if save_csv:
        marginal_df.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}  ({len(marginal_df):,} rows)")
    return marginal_df

def compute_marginal_contributions_cv(
    meta_df,
    slide_filtered_data,
    cluster_col,
    fold_models,                      # dict {fold_idx: model} OR list ordered by fold
    fold_df,                          # DataFrame with columns: image_id, fold
    raw_feature_dirs,
    mask_dirs,
    device,
    cohort_col="cohort",
    slide_col="slide_id",
    y_true_col="y_true",
    save_csv="marginal_contributions_cv.csv",
    min_cluster_frac=0.01,
):
    """
    CV-aware marginal contribution: for each slide, use ONLY the fold model that
    did NOT train on it (the held-out model). Mirrors `compute_marginal_contributions`
    but scores p_full / p_without_cluster with a single model per slide rather
    than an ensemble average.

    fold_models can be a dict {fold_idx: model} or a list indexed by fold.
    fold_df maps slide_id -> fold (0..N-1). Slides not in fold_df are skipped.
    """
    import numpy as np, pandas as pd, torch
    from tqdm import tqdm

    cluster_ids = sorted(int(c) for c in meta_df[cluster_col].unique() if c != -1)
    print(f"Computing CV marginal contributions for {len(cluster_ids)} clusters: {cluster_ids}")
    print(f"  Removal threshold: cluster must occupy >= {min_cluster_frac:.1%} of foreground")

    # Build slide -> fold lookup
    fold_df = fold_df.copy()
    fold_df["image_id"] = fold_df["image_id"].astype(str)
    slide_to_fold = dict(zip(fold_df["image_id"], fold_df["fold"].astype(int)))
    print(f"  Fold assignment covers {len(slide_to_fold)} slides")

    # Normalize fold_models to a dict
    if isinstance(fold_models, list):
        fold_models = {i: m for i, m in enumerate(fold_models)}

    available_cohorts = set(raw_feature_dirs.keys())
    df = meta_df[meta_df[cohort_col].isin(available_cohorts)].copy().reset_index(drop=True)

    # Drop slides that aren't in the fold assignment
    df = df[df[slide_col].isin(slide_to_fold)]
    print(f"  Slides in scope (with fold): {df[slide_col].nunique():,}  ({len(df):,} tiles)")

    # Attach fg_position per tile
    fg_pos = np.full(len(df), -1, dtype=np.int64)
    for sid, grp in df.groupby(slide_col, sort=False):
        if sid not in slide_filtered_data:
            print(f"  [!] {sid} not in slide_filtered_data, will skip")
            continue
        positions = slide_filtered_data[sid]["fg_positions"]
        if len(positions) != len(grp):
            print(f"  [!] {sid}: {len(positions)} fg_positions vs {len(grp)} meta_df rows — skipping")
            continue
        fg_pos[grp.index.values] = positions
    df["_fg_position"] = fg_pos
    df = df[df["_fg_position"] >= 0]

    has_label = y_true_col in df.columns
    records = []
    skipped_missing_features = 0
    skipped_below_threshold = 0
    skipped_no_model = 0

    def run_single(model, feats_fg):
        with torch.no_grad():
            x = torch.from_numpy(feats_fg).unsqueeze(0).to(device)
            out = model(x)
            # model returns (Y_prob, ...) — first element is the probability
            return float(out[0].item())

    for sid, grp in tqdm(df.groupby(slide_col, sort=False), desc="Marginal inference (CV)"):
        cohort = grp[cohort_col].iloc[0]
        y_true = int(grp[y_true_col].iloc[0]) if has_label else np.nan
        fold = slide_to_fold[sid]

        if fold not in fold_models:
            skipped_no_model += 1
            continue
        model = fold_models[fold]

        feats_fg = _load_fg_features(sid, raw_feature_dirs[cohort], mask_dirs[cohort])
        if feats_fg is None or len(feats_fg) == 0:
            skipped_missing_features += 1
            continue
        n_fg = len(feats_fg)

        p_full = run_single(model, feats_fg)

        for c, sub in grp.groupby(cluster_col, sort=False):
            remove_pos = sub["_fg_position"].values.astype(np.int64)
            if remove_pos.size == 0:
                continue

            cluster_frac = remove_pos.size / n_fg
            if cluster_frac < min_cluster_frac:
                skipped_below_threshold += 1
                continue

            if remove_pos.max() >= n_fg or remove_pos.min() < 0:
                print(f"  [!] {sid} cluster {c}: fg_position out of range "
                      f"(max {remove_pos.max()}, n_fg={n_fg}). Skipping.")
                continue

            keep_mask = np.ones(n_fg, dtype=bool)
            keep_mask[remove_pos] = False
            if keep_mask.sum() == 0:
                continue

            p_without = run_single(model, feats_fg[keep_mask])
            records.append({
                "slide_id":          sid,
                "cohort":            cohort,
                "fold":              fold,
                "y_true":            y_true,
                "cluster":           int(c),
                "p_full":            p_full,
                "p_without_cluster": p_without,
                "delta_p":           p_full - p_without,
                "n_tiles_removed":   int((~keep_mask).sum()),
                "n_tiles_fg_total":  n_fg,
                "pct_fg_removed":    float((~keep_mask).sum()) / n_fg,
            })

    if skipped_missing_features:
        print(f"  [!] Skipped {skipped_missing_features} slides (raw features or mask not found)")
    if skipped_no_model:
        print(f"  [!] Skipped {skipped_no_model} slides (no fold model for their fold)")
    if skipped_below_threshold:
        print(f"  [i] Skipped {skipped_below_threshold} (slide, cluster) pairs below "
              f"{min_cluster_frac:.1%} foreground threshold")

    marginal_df = pd.DataFrame(records)
    if save_csv:
        marginal_df.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}  ({len(marginal_df):,} rows)")
    return marginal_df

def plot_marginal_contribution_histograms(
    marginal_df: pd.DataFrame,
    cluster_ids: Optional[Sequence[int]] = None,
    bins: int = 40,
    save_path: str = "marginal_histograms.png",
) -> None:
    if cluster_ids is None:
        cluster_ids = sorted(marginal_df["cluster"].unique().tolist())

    n = len(cluster_ids)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    gmin, gmax = marginal_df["delta_p"].min(), marginal_df["delta_p"].max()

    for i, c in enumerate(cluster_ids):
        ax = axes[i]
        delta = marginal_df.loc[marginal_df["cluster"] == c, "delta_p"].values
        if len(delta) == 0:
            ax.set_visible(False)
            continue
        median_val = np.median(delta)
        color = "firebrick" if median_val > 0 else "steelblue"
        ax.hist(delta, bins=bins, range=(gmin, gmax),
                color=color, alpha=0.75, edgecolor="none")
        ax.axvline(median_val, color="black", linewidth=2, linestyle="--",
                   label=f"Median ΔP = {median_val:+.4f}")
        ax.axvline(0, color="gray", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.set_title(f"Cluster {c}  (n_slides={len(delta):,})", fontweight="bold")
        ax.set_xlabel("ΔP = P_full − P_without_cluster")
        ax.set_ylabel("Slide count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(
        "Marginal Contribution of Tile Clusters\n"
        "positive → pro-recurrence | negative → anti-recurrence",
        fontsize=13, fontweight="bold", y=0.98
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cluster presence per slide & significance testing
# ─────────────────────────────────────────────────────────────────────────────

def compute_cluster_centroids(
    reps_pca: np.ndarray, cluster_labels: np.ndarray, n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalized centroid per cluster, skipping -1 (artifact)."""
    D = reps_pca.shape[1]
    centroids = np.zeros((n_clusters, D), dtype=np.float32)
    valid = np.zeros(n_clusters, dtype=bool)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        mean_vec = reps_pca[mask].mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-12:
            centroids[c] = mean_vec / norm
            valid[c] = True
    print(f"  Computed {valid.sum()}/{n_clusters} valid centroids")
    return centroids, valid


def compute_cosine_weights(
    reps_pca: np.ndarray,
    centroids: np.ndarray,
    cluster_labels: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-tile cosine similarity to its OWN cluster's centroid; 0 for invalid."""
    N = reps_pca.shape[0]
    weights = np.zeros(N, dtype=np.float32)
    valid_tile_mask = (cluster_labels >= 0) & valid[np.clip(cluster_labels, 0, None)]
    valid_tile_idx = np.where(valid_tile_mask)[0]

    print(f"  Computing cosine weights for {len(valid_tile_idx):,} valid tiles...")
    assigned_centroids = centroids[cluster_labels[valid_tile_idx]]
    weights[valid_tile_idx] = np.einsum(
        "ij,ij->i", reps_pca[valid_tile_idx], assigned_centroids,
    )

    valid_w = weights[valid_tile_idx]
    print(f"  Cosine weights — min: {valid_w.min():.3f}, "
          f"median: {np.median(valid_w):.3f}, "
          f"mean: {valid_w.mean():.3f}, max: {valid_w.max():.3f}")
    n_negative = int((valid_w < 0).sum())
    if n_negative > 0:
        print(f"  ⚠ {n_negative:,} tiles have negative cosine sim to their own centroid "
              f"({n_negative / len(valid_w):.2%}) — clamping to 0.")
        weights = np.clip(weights, 0.0, None)
    return weights


def build_weighted_presence_table(
    meta_df: pd.DataFrame,
    weights: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    valid: np.ndarray,
    cluster_col: str,
) -> pd.DataFrame:
    """
    Per (slide, cluster):
      - weighted_count    : sum of cosine weights of slide's tiles in that cluster
      - weighted_presence : weighted_count / n_tiles in slide
      - hard_count        : count of tiles hard-assigned to that cluster
      - hard_frac         : hard_count / n_tiles in slide
    """
    slide_info = meta_df.groupby("slide_id").agg(
        y_true        = ("y_true",        "first"),
        ensemble_prob = ("ensemble_prob", "first"),
        cohort        = ("cohort",        "first"),
        n_tiles       = (cluster_col,     "count"),
    ).reset_index()

    slide_to_row = {sid: i for i, sid in enumerate(slide_info["slide_id"])}
    slide_row_per_tile = meta_df["slide_id"].map(slide_to_row).values
    n_slides = len(slide_info)

    print(f"  Aggregating across {n_slides} slides...")
    slide_weighted = np.zeros((n_slides, n_clusters), dtype=np.float64)
    slide_hard = np.zeros((n_slides, n_clusters), dtype=np.int32)

    valid_mask = cluster_labels >= 0
    s_rows = slide_row_per_tile[valid_mask]
    c_labs = cluster_labels[valid_mask]
    w_vals = weights[valid_mask]
    np.add.at(slide_weighted, (s_rows, c_labs), w_vals)
    np.add.at(slide_hard, (s_rows, c_labs), 1)

    n_tiles_arr = slide_info["n_tiles"].values.astype(np.float64)
    for c in range(n_clusters):
        if not valid[c]:
            continue
        slide_info[f"cluster_{c}_weighted_count"]    = slide_weighted[:, c]
        slide_info[f"cluster_{c}_weighted_presence"] = slide_weighted[:, c] / n_tiles_arr
        slide_info[f"cluster_{c}_hard_count"]        = slide_hard[:, c]
        slide_info[f"cluster_{c}_hard_frac"]         = slide_hard[:, c] / n_tiles_arr
    return slide_info

def build_normalized_presence_table(
    meta_df,
    reps_pca,
    cl_centroids,          # [n_clusters, D] L2-normalized, from compute_cluster_centroids
    valid,                 # bool [n_clusters]
    cluster_col,
    n_total_map,   # for n_total (foreground tile count per slide)
    eps=1e-12,
):
    """
    Per-slide cluster presence. One row per slide, columns per cluster.

    Weight per tile = (cos_sim_to_own_centroid + 1) / 2   -> [0, 1]

    Denominators use the slide's TOTAL FOREGROUND tile count (n_total from
    slide_filtered_data), not the filtered count.

    Per cluster c, four columns:
      cluster_{c}_presence       weighted, row across clusters sums to 1 per slide
      cluster_{c}_presence_abs   weighted count / n_foreground (does NOT sum to 1)
      cluster_{c}_hard_count     # tiles hard-assigned to c in the slide
      cluster_{c}_hard_frac      hard_count / n_foreground (does NOT sum to 1)
    """
    import numpy as np, pandas as pd

    labels = meta_df[cluster_col].values
    n_clusters = cl_centroids.shape[0]

    # --- per-tile rescaled cosine weight to assigned centroid ---
    weights = np.zeros(len(meta_df), dtype=np.float64)
    valid_tile = (labels >= 0) & valid[np.clip(labels, 0, None)]
    vi = np.where(valid_tile)[0]
    cos = np.einsum("ij,ij->i", reps_pca[vi].astype(np.float64),
                    cl_centroids[labels[vi]].astype(np.float64))
    weights[vi] = (cos + 1.0) / 2.0     # [-1,1] -> [0,1]

    # --- slide bookkeeping ---
    slide_info = meta_df.groupby("slide_id").agg(
        y_true       = ("y_true", "first"),
        cohort       = ("cohort", "first"),
        n_filtered   = (cluster_col, "count"),
    ).reset_index()

    # foreground (total) tile count per slide, from the filtering step
    slide_info["n_foreground"] = slide_info["slide_id"].map(n_total_map)

    missing = slide_info["n_foreground"].isna().sum()
    if missing:
        print(f"  [!] {missing} slides have no n_total in slide_filtered_data "
              f"(will fall back to filtered count)")
        slide_info["n_foreground"] = slide_info["n_foreground"].fillna(
            slide_info["n_filtered"]
        )
    slide_info["n_foreground"] = slide_info["n_foreground"].astype(np.float64)

    s_row = {s: i for i, s in enumerate(slide_info["slide_id"])}
    row_per_tile = meta_df["slide_id"].map(s_row).values
    n_slides = len(slide_info)

    # --- accumulate weighted counts AND hard counts [n_slides, n_clusters] ---
    W    = np.zeros((n_slides, n_clusters), dtype=np.float64)
    Hard = np.zeros((n_slides, n_clusters), dtype=np.int64)
    m = labels >= 0
    np.add.at(W,    (row_per_tile[m], labels[m]), weights[m])
    np.add.at(Hard, (row_per_tile[m], labels[m]), 1)

    n_fg = slide_info["n_foreground"].values[:, None]     # foreground denominator

    # weighted: absolute (/foreground) and normalized (row sums to 1)
    W_abs  = W / np.maximum(n_fg, eps)
    W_norm = W / np.maximum(W.sum(axis=1, keepdims=True), eps)

    # hard fraction over foreground
    Hard_frac = Hard / np.maximum(n_fg, eps)

    for c in range(n_clusters):
        if not valid[c]:
            continue
        slide_info[f"cluster_{c}_presence"]     = W_norm[:, c] # cosine weighted presence such that the sum of all cluster presence = 1
        slide_info[f"cluster_{c}_presence_abs"] = W_abs[:, c] # cosine weighted presence divided by the total number of foreground tiles
        slide_info[f"cluster_{c}_hard_count"]   = Hard[:, c] # count presence
        slide_info[f"cluster_{c}_hard_frac"]    = Hard_frac[:, c] # count presence divided by the total number of foreground tiles

    return slide_info

def test_cluster_significance(
    presence_df: pd.DataFrame, valid: np.ndarray, presence_kind: str = "weighted_presence",
) -> pd.DataFrame:
    """Mann-Whitney U per cluster (recur+ vs recur−), BH-corrected."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    neg = presence_df[presence_df["y_true"] == 0]
    pos = presence_df[presence_df["y_true"] == 1]
    n0, n1 = len(neg), len(pos)
    print(f"  Test groups: neg={n0}, pos={n1}")

    rows = []
    for c in np.where(valid)[0]:
        col = f"cluster_{c}_{presence_kind}"
        if col not in presence_df.columns:
            continue
        x0, x1 = neg[col].values, pos[col].values
        if x0.std() == 0 and x1.std() == 0:
            continue
        u_stat, p_val = mannwhitneyu(x1, x0, alternative="two-sided")
        rows.append({
            "cluster":    int(c),
            "median_neg": float(np.median(x0)),
            "median_pos": float(np.median(x1)),
            "mean_neg":   float(np.mean(x0)),
            "mean_pos":   float(np.mean(x1)),
            "U_stat":     float(u_stat),
            "p_value":    float(p_val),
            "effect_r":   float((2 * u_stat) / (n0 * n1) - 1),
        })

    sig_df = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(sig_df["p_value"].values, method="fdr_bh")
    sig_df["p_adj_bh"] = p_adj
    sig_df["significant"] = sig_df["p_adj_bh"] < 0.05
    return sig_df.sort_values("p_adj_bh").reset_index(drop=True)


def plot_significant_clusters(
    presence_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    resolution: float,
    presence_kind: str = "weighted_presence",
    p_threshold: float = 0.05,
    max_clusters: int = 20,
    bins: int = 30,
    log_x: bool = False,
    save_path: str = "significant_clusters_recur.png",
) -> None:
    """Per-cluster overlapping histograms of presence for recur+ vs recur−."""
    sig_clusters = sig_df[sig_df["p_adj_bh"] < p_threshold].head(max_clusters)
    n_show = len(sig_clusters)
    if n_show == 0:
        print(f"No clusters significant at p_adj < {p_threshold}. "
              f"Try a relaxed threshold or plot top-N by p-value instead.")
        return

    print(f"Plotting {n_show} significant clusters (p_adj < {p_threshold})...")
    ncols = min(6, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten() if n_show > 1 else np.array([axes])

    neg = presence_df[presence_df["y_true"] == 0]
    pos = presence_df[presence_df["y_true"] == 1]

    for i, (_, row) in enumerate(sig_clusters.iterrows()):
        c = int(row["cluster"])
        ax = axes[i]
        col = f"cluster_{c}_{presence_kind}"
        x0, x1 = neg[col].values, pos[col].values

        x_all = np.concatenate([x0, x1])
        x_max = np.percentile(x_all, 99)
        if x_max <= 0:
            x_max = max(x_all.max(), 1e-6)
        bin_edges = np.linspace(0, x_max, bins + 1)

        ax.hist(x0, bins=bin_edges, alpha=0.55, color="steelblue",
                label=f"Recur− (n={len(x0)})", density=True)
        ax.hist(x1, bins=bin_edges, alpha=0.55, color="firebrick",
                label=f"Recur+ (n={len(x1)})", density=True)
        ax.axvline(np.median(x0), color="steelblue", linewidth=2, linestyle="--", alpha=0.9)
        ax.axvline(np.median(x1), color="firebrick", linewidth=2, linestyle="--", alpha=0.9)

        p_adj = row["p_adj_bh"]
        r = row["effect_r"]
        star = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
                else "*" if p_adj < 0.05 else "ns")
        direction = "↑recur+" if r > 0 else "↑recur−"
        ax.set_title(
            f"Cluster {c}  [{star}]  {direction}\n"
            f"r={r:+.3f}  p_adj={p_adj:.2e}",
            fontweight="bold", fontsize=10,
        )
        ax.set_xlabel(presence_kind.replace("_", " "))
        ax.set_ylabel("Density")
        ax.set_xlim(0, x_max)
        if log_x:
            ax.set_xscale("symlog", linthresh=x_max / 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Significant Clusters (BH-corrected p < {p_threshold}) — "
        f"Recurrence+ vs Recurrence−\n",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_volcano(
    sig_df: pd.DataFrame,
    resolution: float,
    p_threshold: float = 0.05,
    save_path: str = "volcano_clusters.png",
) -> None:
    """Effect size (rank-biserial r) vs −log10(p_adj)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    r = sig_df["effect_r"].values
    p = sig_df["p_adj_bh"].values
    neg_log_p = -np.log10(np.clip(p, 1e-300, None))

    sig_mask = p < p_threshold
    pos_dir = r > 0
    ax.scatter(r[~sig_mask], neg_log_p[~sig_mask],
               s=30, color="lightgrey", alpha=0.6, label="ns")
    ax.scatter(r[sig_mask & pos_dir], neg_log_p[sig_mask & pos_dir],
               s=50, color="firebrick", alpha=0.8,
               label=f"Sig, ↑recur+ (n={(sig_mask & pos_dir).sum()})")
    ax.scatter(r[sig_mask & ~pos_dir], neg_log_p[sig_mask & ~pos_dir],
               s=50, color="steelblue", alpha=0.8,
               label=f"Sig, ↑recur− (n={(sig_mask & ~pos_dir).sum()})")
    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", alpha=0.5,
               label=f"p_adj = {p_threshold}")
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)

    top_n = min(15, int(sig_mask.sum()))
    for idx in np.argsort(p)[:top_n]:
        ax.annotate(f"{int(sig_df['cluster'].iloc[idx])}",
                    (r[idx], neg_log_p[idx]), fontsize=8, alpha=0.8,
                    xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel("Rank-biserial r  (+ = more in recur+, − = more in recur−)")
    ax.set_ylabel("−log10(BH-adjusted p)")
    ax.set_title(f"Volcano: cluster presence association with recurrence\n"
                 f"Resolution {resolution}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Logistic regression on cluster presence
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    presence_df: pd.DataFrame, valid: np.ndarray, presence_kind: str = "weighted_presence",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    feat_cols, cluster_ids = [], []
    for c in np.where(valid)[0]:
        col = f"cluster_{c}_{presence_kind}"
        if col in presence_df.columns:
            feat_cols.append(col)
            cluster_ids.append(int(c))
    X = presence_df[feat_cols].values.astype(np.float64)
    y = presence_df["y_true"].values.astype(int)
    return X, y, feat_cols, cluster_ids


def split_by_cohort(
    presence_df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
    train_cohort: str, test_cohorts: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cohort = presence_df["cohort"].values
    train_mask = cohort == train_cohort
    test_masks = {c: (cohort == c) for c in test_cohorts}

    print(f"Train cohort ({train_cohort}): {train_mask.sum()} slides, "
          f"recur+ rate = {y[train_mask].mean():.1%}")
    for c, m in test_masks.items():
        if m.sum() == 0:
            print(f"  ⚠ Test cohort '{c}' has 0 slides — check cohort labels")
        else:
            print(f"Test cohort ({c}): {m.sum()} slides, "
                  f"recur+ rate = {y[m].mean():.1%}")
    return train_mask, test_masks


def bootstrap_test_auc(
    y: np.ndarray, probs: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], probs[idx]))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def fit_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_tests: Dict[str, np.ndarray], y_tests: Dict[str, np.ndarray],
    cluster_ids: Sequence[int], feat_cols: Sequence[str],
    train_cohort: str = "MGB",
    penalty: str = "l2", use_cv_for_C: bool = True, seed: int = 42,
):
    """
    Fit logistic regression on training cohort, evaluate on each test cohort.
    Standardization is fit on training only. Returns (clf, scaler, results, coef_df).
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_tests_s = {c: scaler.transform(X) for c, X in X_tests.items()}

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    if use_cv_for_C:
        clf = LogisticRegressionCV(
            Cs=20, penalty=penalty, solver=solver,
            cv=5, scoring="roc_auc", max_iter=5000,
            random_state=seed, n_jobs=-1, class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        print(f"  Inner CV chose C = {clf.C_[0]:.4f}")
    else:
        clf = LogisticRegression(
            C=1.0, penalty=penalty, solver=solver,
            max_iter=5000, random_state=seed, class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)

    train_probs = clf.predict_proba(X_train_s)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"  Train AUC ({train_cohort}): {train_auc:.4f}  ")

    results = {"train": {"y": y_train, "probs": train_probs, "auc": train_auc}}
    for cohort, X_test_s in X_tests_s.items():
        if len(X_test_s) == 0:
            continue
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_tests[cohort], probs)
        ap = average_precision_score(y_tests[cohort], probs)
        print(f"  Test AUC ({cohort}):  {auc:.4f}    AP: {ap:.4f}")
        results[cohort] = {"y": y_tests[cohort], "probs": probs, "auc": auc, "ap": ap}

    coefs = clf.coef_.ravel()
    coef_df = pd.DataFrame({
        "cluster":   cluster_ids,
        "feat_col":  feat_cols,
        "coef":      coefs,
        "abs_coef":  np.abs(coefs),
        "direction": np.where(coefs > 0, "↑recur+",
                              np.where(coefs < 0, "↑recur−", "—")),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return clf, scaler, results, coef_df


def plot_external_validation(
    results: dict,
    presence_df: pd.DataFrame,
    test_masks: Dict[str, np.ndarray],
    test_cohorts: Sequence[str],
    train_cohort: str,
    save_path: str = "logreg_external_roc.png",
) -> None:
    """ROC curves: each test cohort, with MIL baseline overlaid if available."""
    from sklearn.metrics import roc_curve, roc_auc_score

    has_mil = "ensemble_prob" in presence_df.columns
    ensemble_probs = presence_df["ensemble_prob"].values if has_mil else None
    y_all = presence_df["y_true"].values.astype(int)

    fig, axes = plt.subplots(1, len(test_cohorts), figsize=(7 * len(test_cohorts), 6))
    if len(test_cohorts) == 1:
        axes = [axes]

    for ax, cohort in zip(axes, test_cohorts):
        if cohort not in results:
            ax.set_visible(False)
            continue
        r = results[cohort]
        fpr_lr, tpr_lr, _ = roc_curve(r["y"], r["probs"])
        _, lo, hi = bootstrap_test_auc(r["y"], r["probs"])
        ax.plot(fpr_lr, tpr_lr, linewidth=2,
                label=f"Cluster logistic (AUC={r['auc']:.3f}, "
                      f"95%CI [{lo:.3f}, {hi:.3f}])")
        if has_mil:
            m = test_masks[cohort]
            fpr_e, tpr_e, _ = roc_curve(y_all[m], ensemble_probs[m])
            auc_e = roc_auc_score(y_all[m], ensemble_probs[m])
            ax.plot(fpr_e, tpr_e, linewidth=2, linestyle="--",
                    label=f"MIL ensemble (AUC={auc_e:.3f})")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"External validation: {cohort}\n(trained on {train_cohort})")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_coefficients(
    coef_df: pd.DataFrame, train_cohort: str, save_path: str = "logreg_coefs.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(5, len(coef_df) * 0.28)))
    colors = ["firebrick" if c > 0 else "steelblue" for c in coef_df["coef"]]
    ax.barh(range(len(coef_df)), coef_df["coef"], color=colors, alpha=0.75)
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels([f"Cluster {int(c)}" for c in coef_df["cluster"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient")
    ax.set_title(f"Cluster coefficients (trained on {train_cohort})\n"
                 f"(red = ↑recur+, blue = ↑recur−)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def generate_slide_cluster_file(
    slide_id,
    feature_dir,          # {feature_dir}/{slide_id}/{slide_id}.hdf5  (5 model files: _model{m})
    tile_metadata_dir,    # {tile_metadata_dir}/{slide_id}.parquet
    mask_dir,             # background parquet: {mask_dir}/{slide_id}.parquet (col tile_background)
    pca_projector_path="pca_projector.npz",
    centroids_path="centroids.npy",
    cluster_lookup_path="cluster_lookup.parquet",
    out_path=None,
    n_models=5,
    d_model=2560,
    load_slide_model_data=None,   # pass U.load_slide_model_data
):
    import faiss
    if out_path is None:
        out_path = f"slide_cluster_assignment_{slide_id}.parquet"
 
    # ── Load cluster artifacts ──
    proj = np.load(pca_projector_path)
    pca_components = proj["components"]          # [n_pca, n_models*d_model]
    pca_mean       = proj["mean"]
    centroids      = np.load(centroids_path)     # [k, n_pca], L2-normalized
    cluster_lookup = pd.read_parquet(cluster_lookup_path)
    res_cols = [c for c in cluster_lookup.columns if c.startswith("cluster_res")]
 
    # ── Load this slide's 5-model features, concatenate (foreground tiles) ──
    model_data = [load_slide_model_data(slide_id, feature_dir, m)
                  for m in range(1, n_models + 1)]
    if any(d is None for d in model_data):
        raise FileNotFoundError(f"{slide_id}: missing one or more fold feature files")
    concat = np.concatenate([d["representations"] for d in model_data], axis=1).astype(np.float32)
    n_fg_feats = concat.shape[0]
 
    # ── Per-block L2 -> PCA project -> final L2 (same transform as training) ──
    def l2_per_block(x):
        b = x.shape[0]
        x = x.reshape(b, n_models, d_model)
        x = x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-12)
        return x.reshape(b, n_models * d_model)
 
    xb = l2_per_block(concat)
    reps_pca = (xb - pca_mean) @ pca_components.T
    reps_pca = (reps_pca / np.maximum(np.linalg.norm(reps_pca, axis=1, keepdims=True), 1e-12)
                ).astype(np.float32)
 
    # ── Assign to nearest micro-cluster (inner product = cosine on unit vecs) ──
    index = faiss.IndexFlatIP(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids.astype("float32")))
    sims, I = index.search(np.ascontiguousarray(reps_pca), 1)
    micro = I.flatten().astype(np.int32)
    assign_sim = sims.flatten().astype(np.float32)
 
    # ── Tile metadata (full grid) + background, joined like the reference ──
    tm = pd.read_parquet(os.path.join(tile_metadata_dir, f"{slide_id}.parquet"))
    bg = pd.read_parquet(os.path.join(mask_dir, f"{slide_id}.parquet"))
    tm = tm.join(bg["tile_background"])
 
    fg_positions = np.where(tm["tile_background"].to_numpy() == 0)[0]
    if len(fg_positions) != n_fg_feats:
        raise ValueError(
            f"{slide_id}: foreground tiles from mask ({len(fg_positions)}) != "
            f"feature rows ({n_fg_feats}). Mask/feature mismatch.")
 
    # ── Scatter per-tile cluster info into the FULL-GRID table (bg tiles = -1) ──
    n_all = len(tm)
    tm["micro_cluster"] = -1
    tm["assign_sim"]    = np.nan
    tm.iloc[fg_positions, tm.columns.get_loc("micro_cluster")] = micro
    tm.iloc[fg_positions, tm.columns.get_loc("assign_sim")]    = assign_sim
 
    # community label per resolution
    for rc in res_cols:
        full = np.full(n_all, -1, dtype=np.int32)
        full[fg_positions] = cluster_lookup[rc].reindex(micro).to_numpy()
        tm[rc] = full
 
    tm.to_parquet(out_path)
    print(f"Saved {out_path}: {n_all} tiles ({len(fg_positions)} foreground), "
          f"resolutions: {res_cols}")
    return tm
 