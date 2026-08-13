"""Components: cell detection & classification pipeline.

Pure, inference-only steps of the cell pipeline, run against pretrained
external models:

* :func:`segment_cells` — CellViT++ instance segmentation + typing.
* :func:`cluster_based_filtering` — DBSCAN debris removal over cell centroids.
* :func:`extract_tumor_patches` — cut patches centred on tumor cells.
* :func:`classify_mitoses` — OMG-Net mitotic-figure classification.

The owning stage (:mod:`helios.stages.cells`) chains these per slide and writes
the ``cells``, ``tumor_cell_patches`` and ``mitotic_cells`` artifacts.

References
----------
OMG-Net mitotic-figure classifier (the ``Classifier`` below and the checkpoint
loaded in ``classify_mitoses``; an adapted ResNet18 trained on a pan-cancer
mitotic-figure dataset incl. MIDOG++):

    Shen, Z., Simard, M., Brand, D., Andrei, V., Al-Khader, A., Oumlil, F.,
    ... Collins-Fekete, C.-A. (2024). A deep learning framework deploying
    Segment Anything to detect pan-cancer mitotic figures from haematoxylin
    and eosin-stained slides. Communications Biology, 7(1), 1674.
    https://doi.org/10.1038/s42003-024-07398-6
    Code: https://github.com/SZY1234567/OMG-Net (preprint arXiv:2407.12773)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import lightning as L
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.cluster import DBSCAN
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# -- configurable defaults -----------------------------------------------------
# Cluster-based filtering DBSCAN (runs on all cell centroids, before patch
# extraction).
DEFAULT_DBSCAN_EPS: float = 500.0
DEFAULT_DBSCAN_MIN_SAMPLES: int = 10
# Discard any cluster holding fewer cells than this fraction of the largest
# cluster. 0.1 == 10%.
DEFAULT_MIN_CLUSTER_FRAC: float = 0.1

# Raw PanNuke class name (as emitted by CellViT++) treated as tumor.
TUMOR_CLASS = "Neoplastic"


def segment_cells(
    wsi_path: str,
    *,
    device: str = "cuda",
    outdir: str | None = None,
    image_mpp: float | None = None,
    image_magnification: int | None = None,
) -> pd.DataFrame:
    """Instance-segment and type every cell in ONE slide (CellViT++).

    GRAIN: one whole-slide image per call. CellViT applies its own internal
    background detection, so no tile/background input is needed.

    Parameters
    ----------
    wsi_path
        Path to the whole-slide image file.
    device
        Torch device string (e.g., 'cuda' or 'cpu').
    outdir
        Output directory for CellViT results (GeoJSON + metadata). If None,
        a directory named after the slide is created under the system temp
        dir (e.g. ``/tmp/cellvit_<slide_id>``), so the output is findable
        per slide rather than under a random suffix.
    image_mpp, image_magnification
        Fallback MPP / magnification (typically the ``image_mpp`` /
        ``image_magnification`` columns of ``image_metadata.csv``) used only
        if CellViT's own auto-detection fails on the first attempt. CellViT
        is retried once with these passed as explicit ``--wsi_mpp`` /
        ``--wsi_magnification`` overrides; if neither is available, no retry
        is attempted.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``cell_id, x, y, cell_type``. ``cell_type`` is the
        raw PanNuke class name from CellViT (e.g., Neoplastic, Inflammatory,
        Connective, Epithelial, Dead).

    Notes
    -----
    CellViT writes one GeoJSON *feature per cell type*, each a single
    MultiPoint whose points are the individual cell centroids. This function
    explodes those into one row per cell.
    """
    # Directory name is derived from the slide filename so the output is
    # findable per slide (e.g. ".../cellvit_BWH-SKCM-P00001-S01-B1-I1") instead
    # of a random temp suffix. Pass an explicit `outdir` to override.
    #
    # Because the name is now fixed rather than unique-per-call, a re-run of the
    # same slide would otherwise read alongside the previous run's files; when
    # we own the directory (outdir is None) we clear it first so rglob only ever
    # sees the current run's output.
    slide_id = Path(wsi_path).stem
    if outdir is None:
        outdir_path = Path(tempfile.gettempdir()) / f"cellvit_{slide_id}"
        if outdir_path.exists():
            shutil.rmtree(outdir_path)
    else:
        outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cellvit-inference",
        "--model", "SAM",
        "--outdir", str(outdir_path),
        "--nuclei_taxonomy", "pannuke",
        "--geojson",
        "process_wsi",
        "--wsi_path", wsi_path,
    ]

    # stdout/stderr are inherited (not captured) so CellViT's own tqdm progress
    # bar renders live in the terminal, instead of being buffered and dumped
    # only after the process exits.
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and (image_mpp is not None or image_magnification is not None):
        # CellViT occasionally can't auto-detect MPP/magnification from the
        # WSI's own metadata; retry once, forcing the values from image_metadata.csv.
        fallback_cmd = list(cmd)
        if image_mpp is not None:
            fallback_cmd += ["--wsi_mpp", str(image_mpp)]
        if image_magnification is not None:
            fallback_cmd += ["--wsi_magnification", str(image_magnification)]

        shutil.rmtree(outdir_path, ignore_errors=True)
        outdir_path.mkdir(parents=True, exist_ok=True)
        fallback_proc = subprocess.run(fallback_cmd)
        if fallback_proc.returncode == 0:
            proc = fallback_proc
        else:
            raise RuntimeError(
                f"cellvit-inference failed (exit {proc.returncode}) for {wsi_path}, "
                f"and the retry with image_metadata.csv's mpp/magnification fallback "
                f"also failed (exit {fallback_proc.returncode}). See CellViT's own "
                f"output above for the underlying error."
            )
    elif proc.returncode != 0:
        raise RuntimeError(
            f"cellvit-inference failed (exit {proc.returncode}) for {wsi_path}. "
            f"See CellViT's own output above for the underlying error."
        )

    # CellViT emits several *.geojson files per slide (e.g. tissue vs. detection
    # layers). We want only the cell-detection output. This pattern matches both
    # ".../<slide_id>/cell_detection.geojson" and
    # ".../<slide_id>_cell_detection.geojson".
    geojson_files = sorted(outdir_path.rglob("*cell_detection.geojson"))
    if not geojson_files:
        raise FileNotFoundError(
            f"No *cell_detection.geojson output found in {outdir_path}"
        )

    frames = []
    for geojson_file in geojson_files:
        with open(geojson_file) as f:
            data = json.load(f)

        # Accept a bare list of Features or a FeatureCollection.
        if isinstance(data, dict) and "features" in data:
            features = data["features"]
        elif isinstance(data, list):
            features = data
        else:
            continue

        for feature in features:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                continue

            props = feature.get("properties") or {}
            classification = props.get("classification") or {}
            cell_type = classification.get("name", "unknown")

            geom = feature.get("geometry") or {}
            if geom.get("type") != "MultiPoint":
                # This pipeline only emits MultiPoint; skip anything else.
                continue

            # Vectorized: clean the whole feature's coordinates at once.
            pts = np.asarray(geom.get("coordinates", []), dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            pts = pts[:, :2]                              # drop any z
            pts = pts[np.isfinite(pts).all(axis=1)]       # drop non-finite rows
            if pts.shape[0] == 0:
                continue

            frames.append(
                pd.DataFrame(
                    {"x": pts[:, 0], "y": pts[:, 1], "cell_type": cell_type}
                )
            )

    if not frames:
        return pd.DataFrame(columns=["cell_id", "x", "y", "cell_type"])

    cells = pd.concat(frames, ignore_index=True)
    cells.insert(0, "cell_id", cells.index.astype(str))
    return cells[["cell_id", "x", "y", "cell_type"]]


def cluster_based_filtering(
    cells: pd.DataFrame,
    *,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
    min_cluster_frac: float = DEFAULT_MIN_CLUSTER_FRAC,
) -> pd.DataFrame:
    """Spatially cluster detected cells, then drop stray cells and small clusters.

    A single whole-slide image frequently contains several physically separate
    cross-sections of the *same* tumor. Running DBSCAN over cell centroids
    groups each contiguous section into one cluster. Cells DBSCAN labels as
    noise (``-1``) are discarded, and any cluster holding fewer cells than
    ``min_cluster_frac`` of the largest cluster is dropped — the assumption
    being that genuine sections are cross-sections of one tumor and therefore
    of comparable size, so much-smaller clusters are debris / detached
    fragments rather than real tissue.

    Both cluster assignment and size filtering happen here. Cluster labels are
    internal and never surfaced, so the returned frame always matches the input
    schema.

    GRAIN: one slide's worth of cells per call. Clustering is purely spatial
    and cell-type agnostic (all types cluster together). Intended to run
    immediately after :func:`segment_cells` and before patch extraction /
    mitosis scoring, so every downstream step sees only retained cells.

    Parameters
    ----------
    cells
        DataFrame from :func:`segment_cells` with at least ``x, y`` columns.
    eps
        DBSCAN neighborhood radius, in level-0 pixels.
    min_samples
        DBSCAN core-point threshold (min neighbors to form a dense region).
    min_cluster_frac
        Minimum retained-cluster size as a fraction of the largest cluster's
        cell count, in ``[0, 1]``. ``0.1`` == 10%: keeps clusters holding at
        least a tenth as many cells as the biggest one. ``0`` keeps every
        non-noise cluster.

    Returns
    -------
    pandas.DataFrame
        ``cells`` restricted to the retained clusters, index reset and
        original column order preserved.

    Notes
    -----
    If DBSCAN finds no dense region at all (every point is noise), the result
    is empty — there is no reliable tissue to keep.
    """
    # Guard against a percentage (e.g. 10) being passed where a fraction is
    # expected, which would silently discard every cluster but the largest.
    if not 0.0 <= float(min_cluster_frac) <= 1.0:
        raise ValueError(
            "min_cluster_frac is a fraction in [0, 1] (0.1 == 10%), got "
            f"{min_cluster_frac!r}"
        )

    # Empty-input guard: nothing to cluster.
    if cells.shape[0] == 0:
        return cells.copy()

    xy = cells[["x", "y"]].to_numpy(dtype=np.float64)
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(xy)

    # Sizes are computed over non-noise labels only; noise (-1) never counts
    # toward "largest cluster" and is never retained.
    non_noise = labels != -1
    if not non_noise.any():
        return cells.iloc[:0].copy()

    uniq, counts = np.unique(labels[non_noise], return_counts=True)
    size_threshold = counts.max() * float(min_cluster_frac)
    retained = uniq[counts >= size_threshold]           # largest always kept

    return cells.loc[np.isin(labels, retained)].reset_index(drop=True)


def extract_tumor_patches(
    wsi_path: str,
    cells: pd.DataFrame,
    *,
    patch_px: int = 64,
    tumor_class: str = TUMOR_CLASS,
    level: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cut a fixed-size RGB patch around every tumor cell in ONE slide.

    GRAIN: one whole-slide image per call. Only cells whose ``cell_type``
    equals ``tumor_class`` get a patch, since mitoses are scored in tumor
    nuclei.

    Parameters
    ----------
    wsi_path
        Path to the whole-slide image file (OpenSlide-readable).
    cells
        DataFrame from ``segment_cells`` with columns
        ``cell_id, x, y, cell_type``. Coordinates are level-0 pixels.
    patch_px
        Side length of each square patch, in level-0 pixels.
    tumor_class
        The ``cell_type`` value treated as tumor (PanNuke: "Neoplastic").
    level
        Pyramid level to read from. Coordinates are always interpreted in
        the level-0 frame (OpenSlide's convention); 0 = full resolution.

    Returns
    -------
    patch_index : pandas.DataFrame
        One row per patch, row-aligned with ``patches`` (row i describes
        ``patches[i]``). Columns: ``cell_id, x, y, cell_type``. Used by
        ``classify_mitoses`` to map predictions back to cells.
    patches : numpy.ndarray
        ``(N, patch_px, patch_px, 3)`` array, dtype ``uint8``. Kept as
        uint8 to stay memory-light (~12 KiB/patch); convert to float32 and
        normalize per batch inside ``classify_mitoses``, not here.

    Notes
    -----
    Patches are centered on each cell centroid. Windows that run off the
    slide edge are zero-padded (black) so every patch is exactly
    ``patch_px`` square, keeping the array shape uniform.
    """
    cols = ["cell_id", "x", "y", "cell_type"]

    # Keep only tumor cells, preserving order so patch_index aligns with patches.
    tumor = cells[cells["cell_type"] == tumor_class].reset_index(drop=True)
    n = len(tumor)

    # Empty-slide guard: return correctly-shaped empties so callers don't break.
    if n == 0:
        empty_patches = np.zeros((0, patch_px, patch_px, 3), dtype=np.uint8)
        return pd.DataFrame(columns=cols), empty_patches

    # Preallocate once (uint8): filling in place avoids a transient 2x from
    # list-append + stack, which matters at ~70k+ patches per slide.
    patches = np.zeros((n, patch_px, patch_px, 3), dtype=np.uint8)

    xs = np.rint(tumor["x"].to_numpy()).astype(np.int64)
    ys = np.rint(tumor["y"].to_numpy()).astype(np.int64)

    with openslide.OpenSlide(wsi_path) as slide:
        W, H = slide.level_dimensions[0]  # level-0 width, height

        for i in range(n):
            cx, cy = int(xs[i]), int(ys[i])

            # Desired window (top-left inclusive, bottom-right exclusive),
            # centered on the centroid, in level-0 coordinates.
            x0 = cx - patch_px // 2
            y0 = cy - patch_px // 2
            x1 = x0 + patch_px
            y1 = y0 + patch_px

            # Clamp the *readable* sub-window to the slide, then paste it into
            # the zero patch at the matching offset -> uniform size + padding.
            rx0, ry0 = max(x0, 0), max(y0, 0)
            rx1, ry1 = min(x1, W), min(y1, H)
            rw, rh = rx1 - rx0, ry1 - ry0
            if rw <= 0 or rh <= 0:
                continue  # cell fully outside the slide (essentially never)

            region = slide.read_region((rx0, ry0), level, (rw, rh)).convert("RGB")
            region = np.asarray(region, dtype=np.uint8)  # (rh, rw, 3)

            ox, oy = rx0 - x0, ry0 - y0  # offset of the valid region in the patch
            patches[i, oy:oy + rh, ox:ox + rw, :] = region

    patch_index = tumor[cols].copy()
    return patch_index, patches


# ``Classifier`` is adapted from OMG-Net's ``Models/SAM_Classifier.py``
# (https://github.com/SZY1234567/OMG-Net) — an adapted ResNet18 that classifies
# mitotic figures. See the module-level References for the full citation:
#   Shen et al. (2024), Communications Biology 7:1674,
#   doi:10.1038/s42003-024-07398-6 (preprint arXiv:2407.12773).
class Classifier(L.LightningModule):
    def __init__(self, config, label_encoder=None):
        super().__init__()
        self.config = config
        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()
        if self.config['BASEMODEL']['Loss_Function'] == 'CrossEntropyLoss':
            if "weights" in self.config['DATA']:
                w = torch.tensor(self.config['DATA']['weights'], dtype=torch.float32)
            else:
                w = torch.ones(self.config['DATA']['Num_of_Classes'], dtype=torch.float32)
            self.loss_fcn = torch.nn.CrossEntropyLoss(weight=w,
                                                      label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])

        self.LabelEncoder = label_encoder
        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()
        backbone = getattr(torchvision.models, self.config['BASEMODEL']['Backbone'])
        self.backbone = backbone(weights='DEFAULT')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.config['DATA']['Num_of_Classes'])
        self.mask_encoder = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        self.encoder_4d = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.save_hyperparameters()

    def forward(self, data):
        x = self.backbone.conv1(data['img'])
        if self.config['BASEMODEL']['Mask_Input']:
            if self.config['BASEMODEL']['Input_Type'] == "3_Channel":
                x = x + self.mask_encoder(data['msk'])
            elif self.config['BASEMODEL']['Input_Type'] == "4_Channel":
                x = self.encoder_4d(torch.cat([data['img'], data['msk']], dim=1))

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.backbone.fc(x)
        x = self.activation(x)

        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = softmax(self(batch), dim=1)
        return output, batch['coords'], batch['id']


# --- Checkpoint + config (same as the standalone script) ---
_CKPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints"
MITOSIS_CKPT = str(_CKPT_DIR / "mitosis.ckpt")

# Index of the "mitotic" class in the model's 2-class output. Flip to 0 if your
# training used the opposite convention.
MITOSIS_POSITIVE_CLASS = 1

MITOSIS_CONFIG = {
    "BASEMODEL": {
        "Training_Stratgy": "AllCells",
        "Input_Type": "3_Channel",
        "Mask_Input": False,
        "Activation": "Identity",
        "Backbone": "resnet18",
        "Model": "convnet",
        "Loss_Function": "CrossEntropyLoss",
        "Batch_Size": 600,
        "Precision": "16-mixed",
        "Vis": [0],
        "Num_of_Worker": 20,
        "GPU_ID": [0],
        "Max_Epochs": 30,
        "Random_Seed": 666,
    },
    "DATA": {
        "Input_Size": [64, 64],
        "Num_of_Classes": 2,
    },
    "REGULARIZATION": {
        "Label_Smoothing": 0.03,
    },
}

# Same preprocessing as the reference script: HWC uint8 -> CHW float in [0,1],
# then ImageNet normalization. ToTensor already scales to [0,1] and returns
# float32; the ToDtype/Normalize steps mirror the original exactly.
_VAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _PatchDataset(Dataset):
    """Wraps an (N, H, W, 3) uint8 array; yields normalized CHW float tensors."""

    def __init__(self, patches: np.ndarray, transform):
        self.data = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        # self.data[idx] is (H, W, 3) uint8; transform -> (3, H, W) float32.
        return self.transform(self.data[idx])


@cache
def _load_mitosis_model(ckpt_path: str, device: str):
    """Load + cache the classifier so it isn't re-read from disk per slide.

    prep_cells calls classify_mitoses once per slide in a loop; caching on
    (ckpt_path, device) means the checkpoint is deserialized only once.
    """
    model = Classifier.load_from_checkpoint(ckpt_path, config=MITOSIS_CONFIG)
    model.eval()
    model.to(device)
    return model


def classify_mitoses(
    patches: np.ndarray,
    patch_index: pd.DataFrame,
    *,
    device: str = "cuda",
    ckpt_path: str = MITOSIS_CKPT,
    batch_size: int = 256,
    num_workers: int = 0,
    positive_class: int = MITOSIS_POSITIVE_CLASS,
) -> pd.DataFrame:
    """Classify each tumor-cell patch as mitotic / non-mitotic for ONE slide.

    GRAIN: one whole-slide image per call. ``patches`` and ``patch_index`` are
    the row-aligned outputs of ``extract_tumor_patches`` (already tumor-only),
    so no cell-type filtering happens here.

    Parameters
    ----------
    patches
        ``(N, 64, 64, 3)`` uint8 array of tumor-cell patches.
    patch_index
        DataFrame with one row per patch (``cell_id, x, y, cell_type``),
        aligned so row i describes ``patches[i]``.
    device
        Torch device string ('cuda' or 'cpu').
    ckpt_path
        Path to the trained classifier checkpoint.
    batch_size
        Patches per forward pass. Only one batch is float32-resident at a
        time, so peak GPU/CPU memory is independent of N.
    num_workers
        DataLoader workers. 0 avoids forking a large in-memory array.
    positive_class
        Output-column index treated as "mitotic".

    Returns
    -------
    pandas.DataFrame
        ``patch_index`` plus per-class probabilities (``prob_0`` ... ),
        ``mitosis_score`` (probability of the positive class),
        ``predicted_label`` (argmax), and ``is_mitotic`` (bool).
    """
    n_classes = MITOSIS_CONFIG["DATA"]["Num_of_Classes"]
    prob_cols = [f"prob_{c}" for c in range(n_classes)]
    out_cols = list(patch_index.columns) + prob_cols + [
        "mitosis_score", "predicted_label", "is_mitotic"
    ]

    # Empty-slide guard: no tumor patches -> correctly-typed empty frame.
    if patches.shape[0] == 0:
        return pd.DataFrame(columns=out_cols)

    torch_device = torch.device(device)
    model = _load_mitosis_model(ckpt_path, str(torch_device))

    loader = DataLoader(
        _PatchDataset(patches, _VAL_TRANSFORM),
        batch_size=batch_size,
        shuffle=False,        # keep alignment with patch_index
        num_workers=num_workers,
    )

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch_device)
            logits = model({"img": batch})
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    predicted_probs = np.concatenate(all_probs, axis=0)       # (N, n_classes)
    predicted_labels = predicted_probs.argmax(axis=1)         # (N,)

    out = patch_index.copy().reset_index(drop=True)
    for c, col in enumerate(prob_cols):
        out[col] = predicted_probs[:, c]
    out["mitosis_score"] = predicted_probs[:, positive_class]
    out["predicted_label"] = predicted_labels.astype(np.int64)
    out["is_mitotic"] = predicted_labels == positive_class

    return out[out_cols]