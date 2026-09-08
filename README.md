# HELIOS

End-to-end computational-pathology pipeline that predicts melanoma recurrence
risk from H&E whole-slide images (WSIs).

![HELIOS workflow overview](dev/assets/images/workflow-diagram-overview-expanded.png)

HELIOS (**H**ematoxylin & **E**osin-based **L**esion **I**nformed **O**utcome
**S**core) turns a routine H&E slide into a transparent, biologically grounded
recurrence-risk assessment. Instead of a single black-box number, it routes each
slide through parallel, individually interpretable branches and fuses them into
one continuous risk score that a pathologist can audit factor by factor.
Developed and validated across 2,860 early-stage patients on three continents, it
outperformed AJCC staging in every external cohort while staying interpretable by
design.

This repository is the open-source framework: a modular, config-driven,
reproducible pipeline so the community can extend each branch, swap in new
foundation models, validate on new cohorts, and build clinically deployable,
explainable risk tools on a shared foundation.

> **Status: scaffold.** The runtime seams — config, path resolution, IO, model
> base classes, stage/component wiring, and the CLI — are implemented and
> runnable on synthetic fixtures. The per-component **domain logic is stubbed**
> (`raise NotImplementedError`); this is the surface to port existing
> script/notebook code into. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Why HELIOS

One score, decomposable into the evidence behind it. Each slide is scored by five
parallel branches, each interpretable on its own:

- **Whole-slide representation** — attention-based multiple-instance learning
  (A-MIL) over foundation-model tile features, with the attention map showing
  *where* the model looked.
- **Cellular composition** — quantitative cell detection, classification, and
  mitosis counting aggregated into per-slide features.
- **Pathological concepts** — linear probes that decode human-readable concepts
  (thickness, ulceration, mitoses, growth pattern, …) from the slide embedding.
- **Patch morphology** — unsupervised clustering of high-attention regions into
  recurring morphological phenotypes whose presence carries risk.
- **Clinical staging** — the structured clinical and AJCC-staging signal.

A small, transparent fusion model (logistic regression / random forest) combines
the branch scores into the final HELIOS risk, so every prediction can be traced
back to the factors that drove it.

## Quickstart

```bash
make install        # uv sync: creates .venv, installs helios + dev tools
make shell          # open a sub-shell with the venv activated
make test           # end-to-end smoke run of the pipeline on synthetic fixtures
make lint           # ruff + mypy
helios --help       # browse the full command tree
```

`make test` runs the whole prep → fit → predict → report flow on a synthetic
cohort (no real data required), so you can confirm the wiring works before
touching a slide.

### Foundation-model setup and Hugging Face access

The basic installation is sufficient for development and the synthetic test
suite. To run every tile encoder—particularly CONCH and MUSK, whose official
Python packages are installed from their source repositories—install the
foundation-model extra:

```bash
uv sync --extra dev --extra foundation-models
```

Several model repositories require accepting model-specific terms before their
weights can be downloaded. Sign in to your personal Hugging Face account in a
browser and request access on the page for each model you intend to use:

- [Virchow2](https://huggingface.co/paige-ai/Virchow2)
- [UNI](https://huggingface.co/MahmoodLab/UNI) and
  [UNI2](https://huggingface.co/MahmoodLab/UNI2-h)
- [CONCH](https://huggingface.co/MahmoodLab/CONCH) and
  [TITAN/CONCH 1.5](https://huggingface.co/MahmoodLab/TITAN)
- [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)
- [MUSK](https://huggingface.co/xiangjx/musk)

Access belongs to an individual Hugging Face user, not merely to an
organization, and some repositories require manual approval. Complete this
step before starting a large extraction job.

Do **not** put a Hugging Face username, password, or token in this repository,
a YAML config, notebook, or committed `.env` file. Authenticate the machine
once with a read-only user token:

```bash
uv run hf auth login
uv run hf auth whoami                 # verify the active account
```

The Hugging Face client stores that token outside the repository at
`$HF_HOME/token` (by default `~/.cache/huggingface/token`) and HELIOS model
loaders discover it automatically. On a cluster, container, or CI runner where
the filesystem is ephemeral, inject a read-only or fine-grained token through
the job's secret manager instead:

```bash
export HF_TOKEN="<token supplied by your secret manager>"
helios prep tile-features \
  --dataset /data/my_cohort \
  --model virchow2 \
  --size-mm 0.25
```

`HF_TOKEN` takes precedence over a token saved on disk. Never paste a real token
into a shell script or scheduler file that will be committed. Model weights are
cached under `$HF_HUB_CACHE` (by default `$HF_HOME/hub`); on compute clusters,
point `HF_HOME` or `HF_HUB_CACHE` at persistent storage before running the
login or extraction command. CTransPath and CHIEF use their authors' separate
Google Drive checkpoints and therefore do not use Hugging Face credentials.

### User-trained MultiStain-CycleGAN normalization

HELIOS includes an inference-compatible copy of the U-Net generator from
[DBO-DKFZ MultiStain-CycleGAN](https://github.com/DBO-DKFZ/multistain_cyclegan_normalization),
but it intentionally does not ship stain-normalization weights. Train the model
in the upstream repository using images representative of your own scanners and
laboratories. In its standard `AtoB` setup, place source-domain examples in
`trainA/` and examples of the desired target stain distribution in `trainB/`:

```text
my_stain_training_data/
  trainA/    # source stains to normalize
  trainB/    # desired target stain distribution
```

Follow the upstream environment instructions, then train there—not inside the
HELIOS environment:

```bash
python train.py \
  --dataroot /data/my_stain_training_data \
  --name normalize-to-lab-b \
  --display_id 0
```

Training writes generator checkpoints beneath the upstream run's results
directory. HELIOS expects the `A → B` generator file named
`latest_net_G_A.pth`. Copy it into this conventional location relative to the
HELIOS repository root:

```text
src/
  helios/
    models/
      multistain_cyclegan/
        lab-b/                       # same name passed to --target
          latest_net_G_A.pth
```

The repository's `.gitignore` excludes `.pth` files under this directory, so
this private checkpoint will not be committed. With that layout, only the
target name is needed:

```bash
helios prep augment \
  --dataset /data/my_cohort \
  --target lab-b \
  --model virchow2 \
  --size-mm 0.25
```

HELIOS resolves that target to
`src/helios/models/multistain_cyclegan/lab-b/latest_net_G_A.pth`. Use
`--checkpoint /absolute/path.pth` to keep weights elsewhere, or
`--stain-model-root /shared/models/stain` to move the whole convention to
persistent/shared storage. Multiple targets and
explicit checkpoints are comma-separated and positional: the first checkpoint
belongs to the first target. Target names are user-defined, not built into
HELIOS. To use one target's augmented embeddings during MIL training:

```bash
helios fit mil \
  --dataset /data/my_cohort \
  --model virchow2 \
  --augmentation-target lab-b \
  --aug-swap-prob 0.5
```

The checkpoint is loaded with the upstream 8-level `unet_256` architecture,
instance normalization, RGB input/output, and strict state-dict validation.
HELIOS uses it for inference only; the discriminator and training code remain
upstream. The imported architecture's BSD terms are preserved in
[`LICENSES/multistain-cyclegan.txt`](LICENSES/multistain-cyclegan.txt).

### The four verbs

HELIOS is driven by a small CLI with four verbs you run in order:

```
helios prep    <noun>   # derive data artifacts (tiles, features, cells, splits)
helios fit     <noun>   # train models (loops all CV folds)
helios predict <noun>   # score cohorts with the fitted models
helios report           # assemble per-case + cohort reports
```

A first end-to-end run against a single cohort looks like:

```bash
helios prep --dataset /data/my_cohort        # run every prep stage in order
helios fit  mil       --dataset /data/my_cohort
helios predict mil    --dataset /data/my_cohort
helios fit  concept   --dataset /data/my_cohort
helios predict concept --dataset /data/my_cohort
helios fit  risk      --dataset /data/my_cohort
helios predict risk   --dataset /data/my_cohort
helios report --dataset /data/my_cohort
```

### Assembling a cohort

A cohort is just a directory with the slides and a metadata table:

```
my_cohort/
  wsi/                                  # one slide per image_id
    DEMO-SKCM-P00001-S01-B1-I1.svs      # .svs / .tiff / .tif / .ndpi
    DEMO-SKCM-P00002-S01-B1-I1.svs
    ...
  metadata/
    image_metadata.csv                  # one row per image_id
```

`image_metadata.csv` needs, at minimum:

- **`image_id`** — matches the WSI filename stem (e.g.
  `DEMO-SKCM-P00001-S01-B1-I1`).
- **Tiling fields** — `image_mpp`, `image_width`, `image_height`,
  `image_magnification`.
- **`patient_id`** — the cross-validation grouping key (splits are patient-level
  to prevent leakage).
- **Branch labels** — only for the branches you actually train: recurrence
  targets (`disease_pfs_recurred*`), pathological concepts (`path_thickness`,
  `path_ulceration_present`, …), staging (`path_stage*`), and clinical
  (`patient_sex`, `disease_age_at_diagnosis`, …).

You don't hard-code these names. They're **bound** in
[`configs/column_mapping/default.yaml`](configs/column_mapping/default.yaml) and
validated against the versioned data dictionary
[`configs/schema/image_metadata.datadict.yaml`](configs/schema/image_metadata.datadict.yaml),
which is the full column reference. A cohort whose columns are named differently
just needs a remapped YAML — no code changes. For a minimal, runnable example,
see [`tests/fixtures.py`](tests/fixtures.py) (`make_cohort`).

## The pipeline at a glance

The CLI surface is 16 commands across the four verbs. The whole tree is
browsable live with `helios --help` (and `helios <verb> --help`,
`helios <verb> <noun> --help`):

| Verb | Nouns |
| --- | --- |
| `prep` | `splits`, `thumbnails`, `tiles`, `augment`, `tile-features`, `cells`, `cell-features` |
| `fit` | `mil`, `concept`, `morphology`, `risk` |
| `predict` | `mil`, `concept`, `morphology`, `risk` |
| `report` | *(single command)* |

Bare `helios prep --dataset ...` runs every prep noun in order — handy for
prepping a fresh cohort end-to-end.

Under the hood the package is organized in three layers:

- **`helios.components`** — pure functions, one per unit of domain logic (the
  code you fill in).
- **`helios.stages`** — CLI-callable orchestrators that own all store IO and the
  dataset / partition / fold loops (one `verb noun` cell each).
- **`helios.cli`** — thin Typer wrappers that resolve config and wire progress.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full module tour and how the
artifact spec drives the graph.

## Working with HELIOS

### Workspace & output layout

By default, derived artifacts (tiles, features, models, predictions, reports) are
written back under the cohort root, so a cohort directory becomes the
self-contained record of everything computed from it. To keep source data
read-only, redirect derived outputs elsewhere with `--output-root`:

```bash
helios prep --dataset /data/my_cohort --output-root /scratch/runs/my_cohort
```

The source cohort is never written to; everything derived lands under the run
root instead. (`--output-root` targets a single `--dataset` at a time.)

### Named, versioned cohorts

Every command accepts one or more `--dataset` values. A value can be a plain
cohort root, or a **cohort-definition file** — a small, version-controllable YAML
that names a union of roots plus an optional row filter:

```yaml
# cohorts/train.yaml
name: helios_train
datasets:
  - /data/MGB
  - ../cohorts/MRV                      # relative paths resolve against this file
filter: "path_stage in ['I', 'II']"     # optional pandas-query over image_metadata
```

```bash
helios fit mil --dataset cohorts/train.yaml
```

This makes training and evaluation populations reproducible and reviewable. You
can also:

- **Union on the fly** — repeat `--dataset` (or comma-separate) to pool several
  cohorts: `--dataset /data/MGB --dataset /data/MRV`.
- **Filter ad hoc** — add `--filter "patient_age_at_diagnosis > 60"` on `fit` /
  `predict`; it's AND-combined with any cohort-file filter.
- **Trace provenance** — result tables carry a `dataset` column recording which
  physical cohort each row came from, so pooled outputs stay lossless.

### Cross-validation folds

Cross-validation is optional and driven by a `cv_splits` table (produced by
`helios prep splits`), which assigns each `image_id` to a fold and train/test
split. With splits present, `fit` trains **one model per fold** in a single
invocation, and `predict` reduces the per-fold scores out-of-fold on the CV
cohort. With no `cv_splits`, the same code path trains a single fold-free model
and ensembles at inference — exactly what you want when deploying to a fresh
cohort. Splits are grouped by `patient_id` so a patient's slides never straddle
the train/test boundary.

### Pathological concept modeling

Concept modeling runs after MIL inference because it consumes the per-fold
`aligned_slide_embedding` and `whole_image_risk_score` artifacts. Training uses
the `path_*` fields in `image_metadata.csv`: bootstrapped logistic probes learn
binary and categorical concept activation vectors, while bootstrapped ridge
probes learn numerical concepts. Categorical fields with more than two values
are represented as one-vs-rest concepts and normalized as a probability
distribution. The bootstrap ensemble also emits an uncertainty column named
`<concept>__std` for each prediction.

The predicted concepts feed a sparse Lasso concept-bottleneck model. A separate
ridge model predicts the remaining error between the MIL risk and the
cross-fitted concept-only risk from the slide embedding; the shipped
`risk_score` is the concept-only score plus this residual correction. This
preserves an explicitly inspectable concept score while allowing the corrected
score to retain information not captured by the available pathology labels.

```bash
helios fit concept \
  --dataset /data/my_cohort \
  --output-root /scratch/runs/my_cohort \
  --n-bootstrap 100

helios predict concept \
  --dataset /data/my_cohort \
  --output-root /scratch/runs/my_cohort
```

Use the same `--output-root` used for `helios predict mil`. The prediction stage
reduces fold outputs out-of-fold when `cv_splits` exists and averages the fold
models for deployment cohorts without splits. Programmatic callers may pass a
pathologist correction table to `infer_concepts`; non-null values keyed by
`image_id` replace the corresponding predicted concepts before risk scoring.

### Configuration

Every tunable parameter has a default declared in code (the stage's keyword
arguments), so the configuration and the CLI can never drift. Those defaults are
introspected into [`configs/default.yaml`](configs/default.yaml); regenerate it
with `make configs` after changing a default. At run time, values resolve with
clear precedence:

```
signature defaults  <  --config file  <  explicit CLI flags
```

Pass a partial override file with `--config my_run.yaml` to change a handful of
parameters without restating the rest. Configs hold parameters only — never data
paths or PHI.

## Contributing

HELIOS is built to be extended: each branch is a pluggable, independently
testable component, and most contributions are filling a stubbed component with
real code.

```bash
git clone <repo-url> && cd helios
make install
make lint && make test          # confirm a green baseline
grep -rn "NotImplementedError" src/helios/components   # find a stub to implement
```

Then read [CONTRIBUTING.md](CONTRIBUTING.md) for the component contract, the
worked thumbnail example, and how to add a brand-new stage or artifact.
