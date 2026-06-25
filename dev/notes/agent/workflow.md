# HELIOS Workflow — Detailed Description

HELIOS is an end-to-end computational pathology system that takes an H&E-stained Whole Slide Image (WSI) as input and produces an individualized recurrence risk score for melanoma patients, along with an interpretable tumor synoptic report.

---

## Top-Level Overview

**Input:** H&E-stained Whole Slide Image (WSI)

**Two upstream pipelines** feed into four downstream analysis modules:

1. **Cell Detection & Classification Pipeline** → Tumor, Immune, and Mitotic Tumor Cell Coordinates
2. **Whole-Slide Multiple-Instance Learning (MIL) Pipeline** → Recurrence-Aligned Slide Embedding, Patch Attention Scores, Recurrence-Aligned Patch Embeddings

**Four analysis modules** each produce a risk score:

| Module | Output |
|---|---|
| Cellular Composition Analysis | Cellular Risk Score |
| Pathological Concept Analysis | Pathological Concept Risk Score |
| Patch Morphology Analysis | Patch Morphology Risk Score |
| (External) Clinical Risk Score | from Patient EHR Data |

**Optional input:** Pathologist-reported Stage → Staging Risk Score

**Final fusion:** A **Weighted Risk Score Fusion Model** combines all risk scores into:
- **HELIOS Risk Score** (image-based only)
- **HELIOS-S Risk Score** (image + staging)

**Final outputs:**
- HELIOS Risk Groups: Low (≈46.7%), Intermediate (≈36.5%), High (≈16.8%)
- Individualized Risk Profile: Recurrence Probability over Time (survival curve)
- HELIOS Tumor Synoptic Report

---

## Pipeline 1: Cell Detection & Classification Pipeline

**Input:** H&E-stained Whole Slide Image

### Steps

1. **CellViT++ Cell Segmentation**
   - Performs instance segmentation of all cells in the WSI
   - Output: Cell Types & Coordinates (tumor cells, immune/lymphocyte cells, other)

2. **Tumor Cell Patch Extraction**
   - Extracts small image patches centered on detected tumor cells
   - Output: Tumor Cell Patches (used as input to mitotic figure classifier)

3. **OMG-Net Mitotic Figure Classifier**
   - A binary classifier trained on the **MIDOG++** dataset
   - Classifies each tumor cell patch as mitotic figure or not
   - Output: Predicted Mitotic Figures (per-cell binary labels)
   - Validated performance:
     - MIDOG++ test set: Accuracy = 0.991, F1 = 0.721
     - MGB-MF (external validation): Accuracy = 0.996, F1 = 0.563

**Output:** Tumor, Immune, and Mitotic Tumor Cell Coordinates (x/y coordinates with cell type labels across the WSI)

---

## Pipeline 2: Whole-Slide Multiple-Instance Learning (MIL) Pipeline

**Input:** H&E-stained Whole Slide Image

### Steps

1. **Background & Artifact Removal**
   - Masks out glass background, pen marks, and non-tissue artifacts
   - Output: Tissue-only masked WSI

2. **Patch Extraction**
   - Divides the tissue region into fixed-size patches
   - Output: Tissue Patches

3. **CycleGAN Stain Augmentation** *(applied during MIL training only)*
   - Trained on the **HELIOS-2K** training dataset (in the paper; **in this repo the generator is pretrained
     externally and loaded for inference only**)
   - Applies stain augmentation to patches to reduce staining variability
   - Output: Augmented Tissue Patches (stored as augmented HELIOS-2K)

4. **Pathology Foundation Model**
   - Encodes each patch into a high-dimensional embedding vector
   - Best-performing model: **Virchow 2** (AUROC 0.828, AUPRC 0.526)
   - Other evaluated models (in descending AUROC order):
     - UNI (0.816), CONCH 1.5 (0.816), UNI 2 (0.808), Phikon 2 (0.813), Prov GigaPath (0.814), CONCH (0.811), Lunit DINO (0.812), CHIEF (0.805), CTransPath (0.801), MUSK (0.799), ResNet 50 (0.792)
   - Evaluated via HELIOS-2K cross-validation
   - Output: Patch Embeddings (one vector per patch)

5. **A-MIL (Attention-based Multiple Instance Learning) Model** *(trainable deep learning, trained on HELIOS-2K)*
   - A sequence of trainable modules:
     - **Alignment** (patch level): Projects patch embeddings into a recurrence-aligned space → **Aligned Patch Embeddings** *(output)*
     - **Attention**: Assigns an attention score to each patch → **Patch Attention Scores** *(output)*
     - **Weighted Pooling**: Aggregates aligned patch embeddings weighted by attention → Slide Embedding
     - **Alignment** (slide level): Projects slide embedding into recurrence-aligned space → **Aligned Slide Embedding** *(output)*
     - **Classifier**: Maps aligned slide embedding to a scalar risk → **Whole Image Risk Score** *(output)*

**Key outputs used downstream:**
- Aligned Patch Embeddings → Patch Morphology Analysis
- Patch Attention Scores → Patch Morphology Analysis
- Aligned Slide Embedding → Pathological Concept Analysis
- Whole Image Risk Score → passed to Pathological Concept Analysis as residual reference

---

## Analysis Module 1: Cellular Composition Analysis

**Input:** Tumor, Immune, and Mitotic Tumor Cell Coordinates

### Steps

1. **Threshold-Based Patch Classification**
   - Divides the WSI into a grid of patches
   - Classifies each patch based on cell density thresholds:
     - TIL-dominant patches (high lymphocyte count)
     - Mitosis-dominant patches (high mitotic figure count)
   - Produces spatially-resolved patch-level maps

2. **Patch Counting** (three parallel streams)

   | Stream | Scope | Counts |
   |---|---|---|
   | Whole-slide | All tissue patches | N_tumor = 36,995 · N_lymph = 31,151 · N_mitosis = 56 |
   | TIL coverage | TIL-classified patches only | N_tumor = 532 · N_lymph = 352 |
   | Mitotic coverage | Mitosis-classified patches only | N_tumor = 532 · N_mitosis = 97 |

3. **Feature Computation**

   - **eTIL** (TIL to Tumor+TIL Ratio): `N_lymph / (N_lymph + N_tumor)` → e.g., 45.7%
   - **Mitosis to Tumor Ratio**: `N_mitosis / N_tumor` → e.g., 0.15%
   - **TIL Coverage**: `N_lymph / N_tumor` in TIL-dominant patches → e.g., 66.2%
   - **Mitotic Coverage**: `N_mitosis / N_tumor` in mitosis-dominant patches → e.g., 18%

**Output (Cell Ratio and Coverage Results):** 4 scalar features — eTIL, Mitosis-to-Tumor Ratio, TIL Coverage, Mitotic Coverage

4. **Risk Model** (trained on HELIOS-2K)
   - Simple ML classifier (e.g., Logistic Regression or Random Forest)
   - Input: the 4 cellular features
   - Output: **Cellular Risk Score** (scalar, e.g., 0.232)

---

## Analysis Module 2: Pathological Concept Analysis

This module implements a **Post-hoc Concept Bottleneck Model** approach. It has a training phase and an inference phase.

### Training Phase

**Training data:** MGB H&E images paired with Tumor Synoptic Features (pathologist annotations), *without* recurrence labels

**Pathological concepts:**

| Concept | Type | Example Values |
|---|---|---|
| Ulceration | Categorical (Absent/Present) | binary |
| Radial Growth | Categorical (Absent/Present) | binary |
| Breslow Thickness | Numerical [mm] | 0.5–3.2 mm |
| Mitotic Rate | Numerical [per mm²] | 0–13 |

**Feature encoding:**
- Categorical features → **One-hot Encoding** → binary vectors
- Numerical features → **Uniform [0,1] Normalization** → normalized scalars

**Concept model training:**
- Source representations: **Aligned Slide Embeddings** from the A-MIL model (trained separately on HELIOS-2K)
- For each concept, a linear probe is trained on top of aligned slide embeddings:
  - Categorical concepts → **Logistic Regression** → binary Concept Model (produces Concept Activation Vector)
  - Numerical concepts → **Ridge Regression** → scalar Concept Model (produces Concept Activation Vector)
- This yields one trained Concept Model per concept

**Concept model validation:**
- MGB: cross-validation; MRV + Mayo: external validation
- Categorical AUROC (e.g., Ulceration: 0.638/0.724/0.773/0.867 across cohorts)
- Numerical 1-RMSE (e.g., Breslow Thickness: 0.79/0.811/0.812; Mitotic Rate: 0.738/0.716/0.709)

**Cosine similarity analysis (interpretability):**
- Computes cosine similarity between each Concept Activation Vector and the Recurrence Activation Vector
- Quantifies whether a concept is high-risk or low-risk:
  - Ulceration: +0.125 (high-risk), Radial Growth: −0.0674 (low-risk), Breslow Thickness: +0.0667, Mitotic Rate: +0.0419

### Inference Phase

**Input:** H&E-stained Whole Slide Image

1. A-MIL model produces → **Aligned Slide Embedding** and **Whole Image Risk Score**

2. Each Concept Model is applied to the Aligned Slide Embedding → **Concept Presence Predictions**:
   - Ulceration: 16.2%, Radial Growth: 81.8%, Breslow Thickness: 1.3 mm, ...
   - Optional: **Pathologist Corrections** can manually override any concept prediction

3. **Interpretable Concept Based Risk Model** (simple linear model, trained on HELIOS-2K)
   - Input: Concept Presence Predictions
   - Output: **Concept Based Risk Score**

4. **Residual Based Error Correction Model** (simple linear model, trained on HELIOS-2K)
   - Input: Aligned Slide Embedding
   - Captures information not encoded by the concept set
   - Output: residual correction term

5. Three output variants:
   - **Concept Based Risk Score** — interpretable only
   - **Concept Based Risk Score w/ Residual Error Correction** = concept score + residual correction
   - **Concept Based Risk Score w/ Pathologist Corrections** — human-in-the-loop override

**Output used downstream:** **Pathological Concept Risk Score** (e.g., 0.405) — which variant is used is configurable

---

## Analysis Module 3: Patch Morphology Analysis

**Inputs:**
- Aligned Patch Embeddings (from A-MIL pipeline)
- Patch Attention Scores (from A-MIL pipeline)

### Steps

1. **Attention Percentile Filtering**
   - Filters out low-attention patches (below a percentile threshold)
   - Retains only the most diagnostically relevant patches
   - Output: Filtered high-attention patch embeddings

2. **PCA**
   - Reduces dimensionality of filtered patch embeddings
   - Output: PCA-reduced patch embeddings

3. **Leiden Clustering** *(trained/fit on HELIOS-2K)*
   - Unsupervised community detection clustering algorithm
   - Groups patches by morphological similarity in PCA space
   - Output: Cluster assignments across all patches (UMAP-visualizable)
   - Clusters correspond to interpretable morphological patterns (e.g., Invasive melanocytes, Superficial melanocytes, Connective + lymphocytes)

4. **Cluster Exclusion**
   - Removes uninformative or artifact-driven clusters
   - Uses **Marginal Contribution Analysis** (see below) to identify non-contributing clusters
   - Output: Retained cluster set

5. **Cluster Presence Scoring**
   - Computes a per-slide presence score for each retained cluster (fraction of high-attention patches assigned to that cluster)
   - Example scores: Invasive melanocytes 56.7%, Superficial melanocytes 13.0%, Connective + lymphocytes 22.6%, ...
   - Output: **Cluster Presence Scores** (one vector of scores per slide)

6. **Risk Model** (simple ML model, e.g., Logistic Regression or Random Forest; trained on HELIOS-2K)
   - Input: Cluster Presence Scores vector
   - Output: **Patch Morphology Risk Score** (e.g., 0.271)

### Analysis / Interpretability Tools

- **Marginal Contribution Analysis**
  - Uses A-MIL Risk Model as reference
  - Measures how removing each cluster changes the whole image risk score
  - Answers: *How does the cluster affect the whole image risk score?*

- **Association with Recurrence**
  - Tests if each cluster's presence score independently associates with recurrence (survival/Cox analysis)
  - Answers: *Is the cluster independently associated with recurrence?*

- **SHAP Feature Attribution**
  - Applies SHAP to the Risk Model
  - Answers: *How is each cluster associated with recurrence in combination with other clusters?*

---

## Final Fusion: Weighted Risk Score Fusion Model

**Inputs (all scalar risk scores):**

| Score | Source | Example |
|---|---|---|
| Staging Risk Score | Pathologist-reported stage (optional) | 0.105 |
| Cellular Risk Score | Cellular Composition Analysis | 0.232 |
| Pathological Concept Risk Score | Pathological Concept Analysis | 0.405 |
| Patch Morphology Risk Score | Patch Morphology Analysis | 0.271 |
| Clinical Risk Score | Patient EHR data | 0.155 |

**Model:** Simple weighted linear fusion (Logistic Regression or similar simple ML model)

**Outputs:**
- **HELIOS Risk Score** — image-based fusion (e.g., 0.265)
- **HELIOS-S Risk Score** — adds pathologist staging to fusion (e.g., 0.265)
- **HELIOS Risk Groups:** Low / Intermediate / High (thresholded from score distribution)
- **Individualized Risk Profile:** Recurrence probability over time (survival curve, 0–5 years)
- **HELIOS Tumor Synoptic Report:** structured report with all interpretable risk factors and scores

---

## Datasets

| Dataset | Role |
|---|---|
| **HELIOS-2K** | Primary training and cross-validation dataset for all HELIOS models (MIL, clustering, risk models) |
| **HELIOS-2K (augmented)** | CycleGAN stain-augmented version of HELIOS-2K, used during MIL training |
| **MGB** | MGB H&E images + tumor synoptic features; used for concept model training (cross-validation) |
| **MRV** | External validation cohort for concept models and risk models |
| **Mayo** | External validation cohort for concept models |
| **MGB-MF** | External validation for OMG-Net mitotic figure classifier |
| **MIDOG++** | Public mitotic figure dataset; used to train and test OMG-Net |

---

## Key Models Summary

| Model | Type | Trained On | Output |
|---|---|---|---|
| CellViT++ | Deep learning segmentation | (pretrained) | Cell types & coordinates |
| OMG-Net | CNN binary classifier | MIDOG++ | Mitotic figure predictions |
| CycleGAN | Generative model | HELIOS-2K | Stain-augmented patches |
| Pathology Foundation Model (Virchow 2) | Transformer (pretrained) | (pretrained) | Patch embeddings |
| A-MIL (Alignment + Attention + Pooling + Classifier) | Trainable deep learning | HELIOS-2K | Aligned embeddings, attention scores, risk score |
| Concept Models (per concept) | Logistic Regression / Ridge Regression | MGB (no recurrence labels) | Concept presence predictions |
| Interpretable Concept Based Risk Model | Simple linear model | HELIOS-2K | Concept-based risk score |
| Residual Error Correction Model | Simple linear model | HELIOS-2K | Residual correction |
| Cellular Risk Model | Logistic Regression / Random Forest | HELIOS-2K | Cellular risk score |
| Patch Morphology Risk Model | Logistic Regression / Random Forest | HELIOS-2K | Patch morphology risk score |
| Weighted Risk Score Fusion Model | Logistic Regression / simple weighted model | HELIOS-2K | Final HELIOS risk score |

---

## Data Flow Summary

```
WSI
├── Cell Detection & Classification Pipeline
│   ├── CellViT++ → Cell Types & Coordinates
│   └── OMG-Net (on tumor cell patches) → Mitotic Figure Labels
│       └── [output] Tumor + Immune + Mitotic Cell Coordinates
│
└── Whole-Slide MIL Pipeline
    ├── Background Removal → Patch Extraction
    ├── [pretrained, inference only] CycleGAN Stain Augmentation
    ├── Foundation Model → Patch Embeddings
    └── A-MIL Model
        ├── [output] Aligned Patch Embeddings
        ├── [output] Patch Attention Scores
        ├── [output] Aligned Slide Embedding
        └── [output] Whole Image Risk Score

Cell Coordinates → Cellular Composition Analysis
    └── [output] Cellular Risk Score

Aligned Slide Embedding → Pathological Concept Analysis
    ├── Concept Models (Logistic Regression / Ridge Regression per concept)
    ├── Interpretable Concept Based Risk Model
    └── Residual Error Correction Model
        └── [output] Pathological Concept Risk Score

(Aligned Patch Embeddings + Patch Attention Scores) → Patch Morphology Analysis
    ├── Attention Filtering → PCA → Leiden Clustering
    ├── Cluster Presence Scoring
    └── Risk Model
        └── [output] Patch Morphology Risk Score

(Optional) Pathologist Stage → Staging Risk Score
(Optional) Patient EHR → Clinical Risk Score

All Risk Scores → Weighted Risk Score Fusion Model
    ├── HELIOS Risk Score
    ├── HELIOS-S Risk Score
    ├── HELIOS Risk Groups
    └── Individualized Risk Profile
```
