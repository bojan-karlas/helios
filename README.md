# HELIOS

End-to-end computational-pathology pipeline that predicts melanoma recurrence
risk from H&E whole-slide images (WSIs).

> **Status: scaffold.** The runtime seams (config, path resolution, IO, model
> base classes, pipeline wiring, CLI) are implemented and runnable on synthetic
> fixtures. The per-component **domain logic is stubbed** (`raise
> NotImplementedError`) — this is the surface to port existing
> script/notebook code into.

## Quickstart

```bash
make install        # uv sync (creates .venv, installs helios + dev tools)
make test           # smoke test: train + infer DAG run end-to-end on synthetic data
make lint           # ruff + mypy
helios --help       # CLI
```

## Layout

```
configs/            shipped config: schemas, role bindings, hyperparams (NO paths/data)
src/helios/
  cli/              thin Typer wrappers -> module functions
  config/           pydantic models, loader, role/datadict contracts
  data/             spec (input contract), resolver (path seam), manifest, io (per-format r/w)
  processing/       preprocessing stages (tiling, background, features, normalization, thumbnails)
  models/           pluggable model stubs (base + registry; cell/mil/concepts/cellular/morphology/fusion/...)
  analysis/         cellular composition, pathological concept, patch morphology, survival
  pipeline/         train.py (fit DAG) + infer.py (inference DAG)  [hand-wired]
  report/           case_report.json, markdown, cohort summary, assets
  progress/         status.json + events.jsonl + logging
  registry/         versioned model-bundle load/save
  utils/            digests, partition-key helpers
tests/              synthetic fixtures + smoke test
dev/                non-shipped: design notes, images, legacy makefile
```

## How a component maps to the spec

Every stub function corresponds 1:1 to a component in
`dev/notes/agent/pipeline-spec.yaml`. Its docstring records the spec contract
(`consumes` / `produces` / `key` / `fold`). To port code, fill in the function
body so it reads its declared inputs and writes its declared outputs via
`helios.data.io`.

## Cross-validation

K-fold CV is optional and driven by a `cv_splits.csv` (one row per
`(image_id, fold)`). Fold-aware components are tagged `map` (run per fold) or
`reduce` (collapse the K folds, out-of-fold on the CV cohort / ensemble on
deploy). See the "Cross-validation & folds" section of the planning doc.
