# HELIOS Repository -- Planning (workshopping, no code yet)

## 2026-10-26: Deterministic splits and overwrite policy

This amendment supersedes earlier split-path and force-flag descriptions.

- Use the artifact ID `splits`, dataset-local `preprocessing/splits.csv`, and
  YAML-adjacent `<stem>.splits.csv` for every cohort YAML, irrespective of entry
  count or existing files.
- Explicit YAML `splits` and CLI `--splits-file` references select exact files;
  CLI wins. Missing explicit selections error without fallback. Missing implicit
  defaults preserve fold-free fitting on the entire filtered cohort.
- Split generation settings never modify the destination. Only preparation
  accepts a fold-count option; consumers derive folds from file assignments.
- `--overwrite` / Python `overwrite` replace HELIOS `--force` / `force`.
  Completed batch units skip by default; incomplete units may be recomputed and
  aggregate tables rebuilt. Overwrite regenerates all units in invocation scope,
  not upstream prerequisites or unrelated artifacts.
- Split preparation errors on an existing destination unless overwrite is
  enabled, and checks all destinations before writing any. Distinct independent
  preparations cannot silently target the same file.
- Other pooling behavior is unchanged; this is not a pooling redesign.

## Goal
Build a repo that: (1) trains all HELIOS pipeline models from datasets placed in folders,
(2) runs inference producing web-app-consumable predictions + supplementary files,
(3) deploys inference as a Docker container, (4) provides progress/log files for a web app.

## Locked decisions
1. Stack: **uv** (activate paradigm via `make shell`, in-repo `.venv`; uv only for dep mgmt).
   Library-first; thin CLI mirrors importable module functions.
2. Task runner: **Make** (port `dev/makefiles/show-help.mk` self-documenting `##` convention).
   Verbs: shell, install, fit (+ fit-cell/mil/concepts/cellular/morphology/fusion),
   predict, build (Docker, gated on fitted models), test, lint, fmt, clean.
3. Fit steps use **file-target dependencies** -> auto "skip if done" + enforced training DAG.
   Fusion trains last on out-of-fold sub-scores.
4. Packaging: **PyPI-compatible** (pyproject.toml, hatchling). Native libs via bundled wheels
   (openslide-bin, pyvips[binary]); Docker/apt as backstop.
5. Models: **pluggable stubs/interfaces** now (fit/save/load/predict), real models wired later.
6. Progress: **status.json (snapshot) + events.jsonl (append-only)** in output dir;
   swappable ProgressReporter; optional webhook later.
7. Inference Docker: **models baked in** (image tag == model version);
   mount input/output/status/logs volumes only.
8. **Data-agnostic by contract**: ship code + schema + DatasetSpec; NEVER datasets, names, or real paths.
   Real locations via CLI/env/gitignored config. Tests use synthetic fixtures.
9. Config: **pydantic v2** validation (also defines DatasetSpec).

## Dataset contract (from sample cohorts + data dictionary) [AMENDED 2026-10-22 -- see "2026-10-22: Coherent 5-role layout, logical cohorts & Hive paths" below]
- USER PROVIDES (input):
  - `wsi/**/*.svs` -- FLAT set (subfolders like Cases/Controls are cosmetic, ignored).
  - `metadata/image_metadata.csv` -- authoritative per-image table
    (join of sample_metadata [labels+patient] + image_file_metadata [tech]).
  - Ignore `bag_metadata.csv`. `sensitive/` PHI variant never required/committed.
- PIPELINE PRODUCES (Hive-partitioned `key=value` dirs):
  - metadata/tile_metadata/size_mm=<MPP>/<image_id>.parquet (tile_row,col,x,y,width,height)
  - tiles/size_mm=<MPP>/<image_id>/<image_id>.hdf5 (+ debug pngs: background/histograms/thumbnail)
  - preprocessing/tile_background/size_mm=<MPP>/<image_id>.parquet
  - preprocessing/tile_features/size_mm=<MPP>/<model>/<image_id>/<image_id>.hdf5
    (12 foundation models: virchow2,uni,uni2,conch,conch1.5,chief,ctranspath,dino,gigapath,musk,phikon2,resnet-50)
  - preprocessing/tile_augmentation[...]/target=<STAIN_TARGET>/size_mm=<MPP>/... (e.g. COHORT1+COHORT2, COHORT3)
  - thumbnails/<image_id>.png
- PARTITION DIMENSIONS: size_mm (resolution), foundation model, stain target, image_id (opaque string).
- Path resolution layer: (stage, image_id, size_mm, model, target) -> path. Keeps code dataset-agnostic.
- Hybrid produce/consume: consume staged artifacts if present, else produce missing stages.

## Schema = the shippable data dictionary
- `image_metadata.datadict.yaml` (v0.1) is the CANONICAL, versioned column contract.
- Validate image_metadata.csv against it (types/units/categorical allowed-values).
- Column ROLES bound via CONFIG referencing dict field names; optional remap for other sites.
- Field namespaces: image_*, patient_*, disease_* (targets), sample_*, path_stage_* (staging),
  path_* (tumor pathology = concept ground-truth), omics_*.
- `caption`/`unit` fields = ready-made human labels/units for web-app report JSON.

### Role -> module mapping
- Targets: disease_pfs_recurred(+days/years), _5yfu (censor), locoregional/distant, disease_os_* -> risk models, fusion, survival curve.
- Concepts (Pathological Concept Analysis): path_thickness(Breslow), path_ulceration_present,
  path_mitoses_rate, path_radial_growth_present (+ optional: TIL pattern, vertical growth, Clark level...).
- Staging Risk Score: path_stage, path_stage_substage, path_stage_tumor(_substage), path_stage_lymph, path_stage_metastasis.
- Clinical Risk Score: patient_sex, disease_age_at_diagnosis, sample_tissue_site(_category), etc.
- Split key: patient_id (patient-level CV to avoid leakage).
- Tiling resolver: image_mpp, image_width/height, image_magnification.


## Path resolution & split roots (CONFIRMED) [SUPERSEDED 2026-10-22 -- see "2026-10-22: Coherent 5-role layout, logical cohorts & Hive paths" below]
- Hybrid: each stage checks artifact; present -> consume; missing -> produce. (Make file-target/DAG.)
- Every artifact explicitly declares one of five storage roles in pipeline-spec.yaml:
  - source: fixed wsi/ and metadata/image_metadata.csv.
  - preprocessing: tiles, augmented tiles, tile metadata, cv_splits, masks, thumbnails and cell detections.
  - features: original and augmented foundation embeddings.
  - models: fitted bundles.
  - results: MIL-aligned embeddings/attention, analysis tables (including cellular_features), predictions and reports.
- Role settings use source_root, preprocessing_root, features_root, models_root and results_root.
  Roles group storage, not execution order; results can feed downstream model training.
- Resolver: artifact -> declared role root + existing relative path template.
  No dataset-specific paths are embedded in the registry.
- Omitted derived roots default to the physical dataset directory. Source artifacts are never modified,
  but generated artifacts may share that directory. To mount it read-only, override all derived roles.
- Cohort manifests accept path strings or mappings with source_root and optional preprocessing_root/features_root.
  Relative paths resolve against the manifest. Existing datasets/roots path lists and cohort filters are supported.
- Cohort unions preserve per-constituent preprocessing/feature roots for both reads and writes.
  They create no union-level preprocessing/feature copy. Single-dataset CLI overrides win over manifest values;
  multi-dataset calls must specify preprocessing/features per constituent, not through a shared CLI override.
- Pooled fitting requires an explicit models_root, shared by all constituents and reused for prediction.
- An explicit results_root places each constituent under its directory name, even in single-dataset calls:
  <results_root>/<dataset-name>/<artifact-relative-path>. Names must be distinct within a shared run.
  Without an override, results stay in each dataset directory. This mapping is stable across union/single calls.
- --output-root is retired with an actionable error; it is not silently reinterpreted. No automatic data moves.
- Python stages accept a shared StorageRoots override object; resolved constituents retain dataset storage.
- Run status/events/logs use results_root, otherwise models_root, otherwise the first dataset directory.
- Resolver is the dataset-agnostic seam; hybrid skip logic lives here (exists(resolve(...))).

## Per-training-stage dataset binding + precondition resolution (CONFIRMED)
- WHY: a training stage may fit on a DIFFERENT dataset than the rest of the pipeline (e.g. concept models
    trained on a separate concept-annotated cohort; OMG-Net is already fully external). So "the training
    dataset" is NOT global -- it is PER FIT STAGE.
- CONFIG: each fit stage (cell*, mil, concepts, cellular, morphology, fusion, ...) declares its OWN
    `dataset` binding in the config files (configs/train/<stage>.yaml). A binding = the set of roots
    (source/preprocessing/features/results) for that stage. Stages default to the GLOBAL/primary dataset when unset, so the
    common single-cohort case needs zero extra config.
- DAG IMPLICATION: a stage's input artifacts resolve UNDER ITS OWN dataset binding, not a single global root.
    Cross-dataset deps are allowed and explicit, e.g. concept_models_training reads aligned_slide_embedding
    for cohort B, which must be produced by running cohort B through preprocessing -> feature_extraction ->
    mil_inference USING the (primary-dataset-trained) model_mil. A model fit on dataset A can be an input to
    generating artifacts on dataset B.
- PRECONDITION CHECK (the new contract): before a fit stage runs, the system RESOLVES every required input
    artifact under that stage's dataset binding and verifies it EXISTS. On a miss it either:
    (a) FAIL FAST with an actionable message (what's missing, which dataset, which step would produce it), OR
    (b) OFFER/auto-run the upstream steps needed to materialize it (the existing hybrid "missing -> produce"
        logic, extended to honor per-stage dataset bindings). Toggle via flag (e.g. --ensure-inputs /
        --strict). Default lean: interactive offer in CLI, FAIL in non-interactive/Docker.
- This is just the existing hybrid skip-or-produce rule made dataset-aware: exists(resolve(stage, artifact))
    is evaluated against the STAGE'S roots. No change to keys/paths/partitions; dataset stays a ROOT, never a key.
- concept_models_training note: the concept cohort's image_metadata.csv only needs the concept ground-truth
    (path_* columns); it may lack recurrence/survival labels (concept training never reads those).

## Cross-validation & folds (CONFIRMED)
- WHAT: K-fold CV (K=5 typically) on the training cohort. Produces K of everything model-shaped (K A-MIL models,
    K sets of embeddings/attention, K of each downstream risk model). We ALWAYS ensemble the K models at deploy --
    there is NEVER a final "train on all data" model. So the K models ARE the shipped deliverable.
- DRIVER: `cv_splits` artifact -- OPTIONAL tidy long CSV `metadata/cv_splits.csv`, columns `image_id, fold,
    split in {train, val?, test}` (val optional), one row per (image_id, fold). It is a new optional INPUT ROOT.
    On the training cohort each image is in exactly one fold's `test` split (its OOF / provenance fold).
- FOLD CARDINALITY is context-dependent (do NOT conflate):
    - TRAINING cohort: number of folds comes from `cv_splits`. ABSENT => trained without CV (K=1, single model).
    - INFERENCE / deploy: number of folds comes from the set of SHIPPED per-fold `model_*` artifacts (the K
        models baked in), NOT from cv_splits. `cv_splits` absent at inference simply means the reducer uses
        ENSEMBLE-average (OOF needs test-fold membership, which only exists on the labelled CV cohort).
    - This is why `mil_inference` does NOT consume cv_splits (it enumerates folds from model_mil), while every
        `fold: reduce` step DOES (to choose OOF-on-CV-cohort vs ensemble-on-deploy).
- `fold` IS AN OPTIONAL KEY DIMENSION (decided after workshopping K-vs-R-vs-hybrid). When present it is ALWAYS
    the OUTERMOST coordinate and materializes as a `fold={fold}/` PATH SEGMENT -- never a column. The
    "empty optional key segment drops its path segment" rule gives us ONE artifact identity at two arities.
- THREE artifact shapes:
    - per-fold  : `fold` in key + `fold={fold}/` in path => K physical instances. These are the K models and the
        MIL outputs consumed by per-fold training: `model_*` (mil/concepts/cellular/morphology/fusion/staging/
        clinical), `aligned_slide_embedding`, `aligned_patch_embeddings`, `patch_attention`, and the per-fold
        `whole_image_risk_score` (one CSV per fold, one row per image).
    - reduced   : NO fold in key/path; one instance per image, produced by a fold-aware REDUCE component that
        collapses the K inputs -- OUT-OF-FOLD (each image scored by its TEST-fold model) on the CV cohort /
        ENSEMBLE-average on deploy. All shipped result CSVs (the 5 sub-scores, helios_risk_score,
        whole_image_risk_score_ensemble, patch_morphology_presence, tile_clusters, survival_curve).
    - fold-free : computed before/independently of CV (preprocessing, cells, cellular_features). No fold at all.
- EVALUATION provenance: every reduced SCORE table additionally carries a `fold` PROVENANCE COLUMN = the test
    fold that produced each image's OOF value (null on deploy/ensemble). This is what makes our metric protocol
    work directly off the score files: per-fold bootstrap (e.g. 100 resamples/fold) -> average across folds ->
    CI / median. It is provenance metadata, NOT a key dimension (still one row per image).
- TWO-ARTIFACT RULE: when BOTH the per-fold and the reduced value have real consumers, they are TWO artifacts
    joined by an explicit reduce component (a reduce node consuming and producing the SAME id would be a DAG
    self-loop). The ONLY case today: `whole_image_risk_score` (per-fold; consumed by concept train/infer per
    fold) -> `whole_image_risk_score_ensemble` (reduced; shipped, joined into cohort_summary) via
    `whole_image_risk_reduce`. Everywhere else reduction is PRIVATE inside the producing inference component, so
    one fold-free artifact suffices.
- COMPONENT `fold:` TAG (none | map | reduce; default none):
    - map    = runs ONCE PER FOLD (parallel work unit) on that fold's train split, emits per-fold artifacts.
        All `*_training`, plus `mil_inference` (scores all images with each fold model).
    - reduce = consumes the K per-fold artifacts (+ cv_splits) and emits reduced fold-free outputs. All
        `*_infer`, `fusion_infer`, `whole_image_risk_reduce`, and `assets_render` (OOF-selects test-fold
        attention for the heatmap).
    - none   = pre-CV / fold-agnostic (preprocessing incl. pretrained CycleGAN stain augmentation, cell pipeline,
        cellular_features_compute, case_report_build, markdown_render, cohort_summarize -- all consume only
        reduced/fold-free inputs).
- PARALLELISM: fold is embarrassingly parallel -- each `fold: map` run is an independent job writing its own
    `fold={fold}` files (no shared-writer hazard). The per-fold whole-image score is a per-fold CSV
    (`mil/fold={fold}/whole_image_risk_score.csv`) for the same reason (avoids one process holding all K MIL
    models to write a single file).
- CV SPLITS GENERATION: `cv_splits_generation` (helios.data.cv_splits, train-only) builds cv_splits.csv from a
    cohort's image_metadata. Params: n_folds (5), val_fraction (0 => no val rows), stratify_by (label cols),
    group_by (e.g. patient_id, to keep a patient's images in one fold -- leakage guard), seed. Run once per
    training cohort; a separate concept cohort gets its own run if it needs folds.
- VAL SPLIT is OPTIONAL: cv_splits.split may include `val`, but if a model needs validation and no `val` rows
    exist, it carves a random train/val split out of that fold's `train` rows. So `val` is provisioned but never
    required.
- CONCEPT BOOTSTRAPPING: concept_models_training fits, per fold, a SET of probes PER CONCEPT by bootstrapping the
    concept-training set (n_bootstrap, default ~100; n_bootstrap=1 disables). This yields a concept-prediction
    DISTRIBUTION / uncertainty rather than a point estimate. model_concepts (per fold) therefore holds
    n_bootstrap x n_concepts probes + the concept risk + residual error-correction models. Bootstrap is an
    INTERNAL replication inside the per-fold bundle (NOT a key/path dimension) -- folds stay the only fold axis.
- HIVE / PARTITIONS-AS-COLUMNS (future convenience, not a dependency): the `field={field}/` path convention is
    Hive-compatible by design, so a future `helios.data.dataset` reader can present a partitioned tree as ONE
    logical table with fold/size_mm/model/target surfaced as COLUMNS, with predicate pushdown -- via PyArrow
    Datasets (`partitioning="hive"`), DuckDB (`hive_partitioning=true`), or Polars (`hive_partitioning=True`).
    Applies to the columnar/csv artifacts (scores, metadata, attention, clusters); hdf5/npy blobs stay file-addressed.

## Output/report contract (web-app boundary) -- workshopped
PRINCIPLE: output is a DATA PACKAGE, not a rendered report. Structured arrays
(parquet/hdf5/npy/json) are authoritative; web app can self-render overlays/heatmaps/
cell-maps at any zoom. Pre-rendered PNGs optional. Schemas LIGHT + versioned
(schema_version), meant to evolve. NO heavy validators now (permissive parsing;
dataclasses/TypedDict; pydantic only sparingly for config). Self-describing formats.

Deployment packaging sketch (separate from current artifact-relative paths):
  <results_root>/<job_id>/
    status.json (job snapshot; poll target), events.jsonl (append-only),
    job.json (manifest+digests), logs/run.log, cases/<image_id>/...

Per-case dir (cases/<image_id>/):
  report.json    -- small headline + refs (web-app entry); labels/units from data dict caption/unit
  tiles/tile_metadata.parquet (row,col,x,y,w,h,size_mm)
  tiles/tile_clusters.parquet (tile_id->cluster_id, attention_score)
  tiles/tile_attention.parquet (OPTIONAL), tiles/tile_embeddings.hdf5 (OPTIONAL patch embeddings)
  clusters/cluster_summary.json (per-cluster: presence score, risk score, attribution/SHAP)
  cells/cells.parquet (ALL CellViT++ cells: x,y,type tumor/lymph/other, mitotic flag)
  cells/tile_cell_counts.parquet (per-tile n_tumor,n_lymph,n_mitosis)
  cells/tile_cell_class.parquet (per-tile TIL-dominant/mitosis-dominant/...)
  embeddings/slide_embedding.npy (OPTIONAL aligned slide embedding)
  survival_curve.csv (recurrence-free prob over time; per-image rows=timepoints) -- mirrors results/survival_curve/<image_id>.csv
  report.md      -- human-readable synoptic report RENDERED from report.json (single source of truth);
                    embeds assets via RELATIVE image links; whole case dir = portable report bundle.
                    Template-driven (e.g. Jinja2; format TBD). Labels/units from data dict caption/unit.
                    PDF = downstream optional step (engine TBD: pandoc/weasyprint/md-to-pdf).
                    Requires pre-rendered assets (MD can't self-render). Generation behind a flag.
  assets/*.png (OPTIONAL pre-rendered: thumbnail, attention_heatmap, cluster_map,
    til_density, mitosis_overlay)
  NOTE: per-image SCALARS/feature-vectors are NOT stored per-case anymore (no concepts.json/cellular.json/
    risk.json). They live in batch CSVs at image_metadata.csv level (see "Batch tabular aggregation").
    report.json/report.md are per-case VIEWS that READ from those batch CSVs.

Web app rendering = two paths: self-render from tile/cell parquet (+attention/clusters)
  OR fallback PNGs.

## Batch tabular aggregation (cohort tables) [AMENDED 2026-10-22 -- see "2026-10-22: Atomic publication, .parts/ tables & completion" below]
- DECISION: per-image SCALARS + short feature-vectors are PRODUCED DIRECTLY as batch CSVs, one CSV per
    producing component, ONE ROW PER image_id, at the SAME level as image_metadata.csv (joinable on image_id).
    Replaces the old "1000s of tiny per-case JSONs + view-builder scan" model. (Resolves prior OPEN question:
    no per-case report.row.json / concepts.json / cellular.json / risk.json.)
- CANONICAL result CSVs (under <job_dir>/results/, key=image_id):
    cellular_features.csv, cellular_risk_score.csv,
    path_concept_predictions.csv, path_concept_risk_score.csv,
    patch_morphology_presence.csv, patch_morphology_risk_score.csv,
    whole_image_risk_score.csv, staging_risk_score.csv, clinical_risk_score.csv,
    helios_risk_score.csv (cols risk_helios + risk_helios_s).
    survival_curve = per-image CSV results/survival_curve/<image_id>.csv (rows=timepoints; it's a curve, not a scalar).
- FORMAT POLICY: CSV by DEFAULT. Parquet ONLY when size/load-speed demands.
    CSV: result tables (above), cohort summary, small/low-frequency/human-inspectable tables.
    Parquet/hdf5/npy: per-tile (tile_metadata, tile_clusters, tile_cell_counts),
      per-cell (cells: 1e4-1e6/slide), embeddings.
    Rule: human-inspectable/small/low-freq -> CSV; per-tile/per-cell/bulk arrays -> parquet/hdf5/npy.
- WRITE model: each result CSV written by its component. Parallel per-image inference appends rows
    (or writes shards merged by a trivial concat); UNION of columns (missing=null, never error);
    incremental/resumable (a live table that fills as cases complete).
- cohort_summary.csv = OPTIONAL convenience WIDE view = JOIN of the result CSVs (+ image_metadata) on
    image_id. Dotted/namespaced columns mirroring the data dictionary (stable, groupable, Excel-filterable):
      image_id, sample_id, patient_id, model_version, state,
      risk_helios, risk_helios_s, risk_group, risk.{cellular,concept,morphology,clinical,staging},
      concept.<field>.pred, concept.<field>.prob,
      cellular.{etil,mitosis_tumor_ratio,til_coverage,mitotic_coverage},
      cells.{n_tumor,n_lymph,n_mitosis},
      morphology.cluster.<name>.{presence,risk},
      survival.{prob_1y,prob_3y,prob_5y},   # sampled scalars; full curve stays in survival_curve CSV
      provenance.{foundation_model,size_mm,config_digest}
- summary builder = one importable fn build_summary(job_dir)->cohort_summary.csv, exposed TWO ways:
    standalone: `helios summarize --output <job_dir>` / `make summarize` (rebuild anytime), AND
    inline: `helios predict ... --summarize` (auto-build as final step). Default ON for predict (--no-summarize to skip).
    Now a JOIN of result CSVs (NOT a scan of per-case report.json). Decoupled, re-runnable, idempotent.

Report file family per case:
  report.json (per-case web-app entry; READS sub-scores/features from the batch result CSVs) |
  report.md (rendered prose+images, PDF source).
  Wide cohort view (cohort_summary.csv) = JOIN of result CSVs (+ image_metadata), built by summarize.

status.json: schema_version, job_id, state(queued/running/completed/failed/partial),
  model_version, progress{percent,stage,stage_index,stage_count},
  counts{total,completed,failed,running,queued}, started_at, updated_at, eta_seconds, error.

report.json: schema_version, image_id, model_version, generated_at, state,
  headline{risk_helios, risk_helios_s, risk_group, legend},
  risk_factors[]{key,label,score, weight OPTIONAL/nullable -- fusion weighting UNDECIDED,
    web app must NOT depend on it},
  concepts[]{key,caption,unit,type,predicted,probability,direction,cosine_to_recurrence,override},
  survival_curve_ref, assets{...},
  provenance{dataset_digest,config_digest,foundation_model,size_mm}.
  -> override slot per concept = pathologist-in-the-loop; re-run recomputes concept/fusion.
  -> provenance uses DIGESTS not paths/cohort names (honors data-agnostic rule).

CELL-LEVEL outputs are first-class (Cell Detection pipeline): all CellViT++ cell locations,
  per-tile tumor/lymph/mitotic counts, per-tile threshold classifications, derived cellular feats.

GRANULARITY: per-image primary (image_id). Patient/sample aggregation = OPTIONAL later layer. [CONFIRMED]

## Open questions (next)
- [RESOLVED] Hybrid + split roots.
- [RESOLVED] Output/report contract (data-package, light+versioned, dual-render).
- [RESOLVED] Cross-validation & folds (fold = optional outermost key/path segment; reduce nodes; provenance column).
- Model-stub interface base classes + artifact bundling/versioning.
- CLI command tree (maps to module fns + Make targets).
- [RESOLVED] Final repository skeleton / directory layout (see below).

## Repository skeleton (CONFIRMED: src-layout)
LAYOUT DECISION: src/helios/ (src-layout) -- guarantees tests exercise the INSTALLED artifact
(same as users/Docker get), catches packaging errors early, no import shadowing, clean root.
Requires install-first (uv sync / make install) -- already step one.

helios/
  Makefile, pyproject.toml (hatchling), uv.lock, README.md, .gitignore,
  .vscode/settings.json (pins .venv)
  dev/                         NON-shipped: assets/images, makefiles/show-help.mk, notes/agent/
  configs/                     SHIPPED: schemas/hyperparams ONLY (no paths/datasets)
    base.yaml
    schema/image_metadata.datadict.yaml   (canonical column contract)
    roles/default.yaml          (column ROLE bindings)
    train/{cell,mil,concepts,cellular,morphology,fusion,full}.yaml
    infer/default.yaml
    examples/{paths.example.yaml, roles.custom.example.yaml}  (templates; filled = gitignored)
  src/helios/                  importable library (library-first)
    cli/        thin Typer wrappers -> module fns (main, fit, predict, summarize, report)
                CLI LIB = Typer (type-hint-driven). Keep CLI layer THIN + isolated (only
                Typer-aware code) so a future swap to cyclopts (same typed-function logic)
                = rewrite only cli/ wrappers, zero changes to logic/config/tests.
                CLI surface is thin (mostly --config/--input/--output/flags); real params in YAML/pydantic.
                Prefer CLI-generated commands; custom scripts/ ONLY when genuinely non-CLI.
    config/     pydantic models, loader, roles, datadict
    data/       spec.py (DatasetSpec = lightweight INPUT CONTRACT: flat wsi/ + image_metadata.csv
                  conforming to data dictionary; light checks, not heavy validation),
                resolver.py (PATH RESOLVER = dataset-agnostic seam:
                  (stage,image_id,size_mm,model,target)->path under correct root; the ONLY place
                  that knows on-disk layout),
                manifest.py (builds in-memory IMAGE INDEX: scan wsi/ flattened + join image_metadata.csv),
                io.py (read+write together, per-format side by side: wsi/parquet/hdf5/csv/npy)
    processing/ hybrid produce/consume stages: tiling, background, features, augmentation, thumbnails
                  (RENAMED from "stages")
    models/     PLUGGABLE STUBS (fit/save/load/predict): base.py(+registry),
                cell/, mil/, foundation/, concepts/, cellular/, morphology/, fusion/
    pipeline/   train.py (fit DAG), infer.py (inference DAG -> per-case outputs)
    analysis/   cellular_composition, pathological_concept, patch_morphology, survival
    report/     case_report.py(report.json), markdown.py(report.md Jinja2),
                summary.py(build_summary->cohort_summary.csv), assets.py(optional PNGs),
                templates/report.md.j2
    progress/   reporter.py(status.json+events.jsonl), logging.py(run.log)
    registry/   bundle.py (artifacts/models/<version>/ load/save + manifest)
    # NOTE: remote SYNC is NOT a library module. It is an OPS layer (Make + rclone + .env).
    #   Package stays pure: no remote/path knowledge, no rclone dep in helios.*.
    #   See "Remote sync (ops layer)" section below.
    utils/      digests, partition keys, helpers
  artifacts/                   GITIGNORED: produced model bundles
    models/<version>/{cell,mil,concepts,cellular,morphology,fusion,metrics}/
  data/                        GITIGNORED: user data (or elsewhere via config)
  docker/                      Dockerfile.infer (bakes bundle; mounts in/out/status/logs), entrypoint.sh
  scripts/                     thin CLI-mirroring scripts (fit, predict, summarize)
  tests/                       fixtures/ = SYNTHETIC generators (no real datasets)

## Remote sync (ops layer) -- CONFIRMED: scripts/sync.sh (smart) + Make shortcuts + rclone + .env (NOT library code)
- Sync is OPS/plumbing, not domain logic -> lives in scripts/sync.sh + Make, not helios.*.
- NO hardcoded paths anywhere in code OR Makefile. All locations from .env (gitignored).
- .env (gitignored, user-created from committed .env.example):
    HELIOS_REMOTE=dropbox:helios        # rclone remote:base-path
    HELIOS_MODELS_DIR=artifacts/models
    HELIOS_CACHE_DIR=...                 # intermediate preprocessing root
    HELIOS_OUTPUTS_DIR=...               # job outputs root
  .env.example committed with blanks/placeholders only.
- Makefile: `-include .env; export`. Make targets are THIN SHORTCUTS that call the script
    with DEFAULT behavior (not rclone directly):
    push-models/pull-models, push-cache/pull-cache, push-outputs/pull-outputs.
    e.g.  pull-models: ./scripts/sync.sh pull models --latest $(ARGS)
- PRIMARY interface = scripts/sync.sh (smart push/pull, full control). Reads .env; builds
    rclone filters from FRIENDLY flags. "Smart" logic lives here (not feasible in Make):
      scripts/sync.sh <push|pull> <models|cache|outputs> [selectors] [--dry-run]
      selectors: --version vNNN | --latest | --component mil | --size-mm 0.250 | --model virchow2
                 | --image-id <id> (repeatable) | --job <id>
      smart features: latest-version resolution, friendly->rclone --include translation,
                      skip-if-present checks, size estimates, confirm prompts, multi-filter compose.
- TWO ergonomic levels: `make pull-models` (sensible default = latest bundle) for the common case;
    `./scripts/sync.sh pull models --version v002 --component mil` for full control.
- Both Make and humans go THROUGH the script -> single source of sync truth, consistent behavior.
- SELECTIVE examples:
    ./scripts/sync.sh pull models --version helios-v003 --component mil
    ./scripts/sync.sh pull cache  --size-mm 0.250 --model virchow2
    ./scripts/sync.sh push outputs --job job_<id>
    ./scripts/sync.sh pull models --latest --dry-run
- Backend rclone (Dropbox first-class); versioning via IMMUTABLE versioned paths (models/<version>/).
- Script reads .env, ZERO paths in Python; package stays pure (no rclone dep). Ops layer only.
- Dataset storage is configured through cohort manifests and explicit root flags.
  Do not assume .env root variables are read by the CLI.
  -> single uncommitted config point, consistent pattern.
- SUPERSEDES earlier src/helios/sync/ module idea (package stays pure, no rclone dep).

## Docs split -- CONFIRMED
- README.md = LIGHT public overview (ships with paper): what HELIOS does, workflow, how to cite.
- CONTRIBUTING.md = DETAILED contributor guide: directory structure, what each file/module is for,
  where to place new code, conventions (CLI-first, library-first, data-agnostic rule, sync ops layer),
  dev workflow (make shell/install/fit/predict, tests).

## Renames/clarifications
- src/helios/stages/ -> src/helios/processing/
- data/ readers.py + writers.py -> data/io.py (read+write together per format)
- data/spec.py = DatasetSpec input contract; data/resolver.py = path resolver (only layout-aware code);
  data/manifest.py = image index (scan wsi flattened + join image_metadata.csv).

---

## 2026-10-22: Coherent 5-role layout, logical cohorts & Hive paths (SUPERSEDES "Path resolution & split roots"; AMENDS "Dataset contract" and "Cross-validation & folds")

### 1. Five Storage Roles & Direct Role Roots
Every artifact in [pipeline-spec.yaml](./pipeline-spec.yaml) declares one of five storage roles (`source`, `preprocessing`, `features`, `models`, `results`):

| Role | Default root (when CLI flag omitted) | Explicit root (when `--<role>-root PATH` passed) | Scope in cohort unions |
|---|---|---|---|
| `source` | `<source_root>` | `<source_root>` (from `--dataset` or cohort YAML) | Per physical dataset |
| `preprocessing` | `<source_root>/preprocessing` | `PATH` (single dataset) or per-entry in cohort YAML | Per physical dataset |
| `features` | `<source_root>/features` | `PATH` (single dataset) or per-entry in cohort YAML | Per physical dataset |
| `models` | `<source_root>/models` | `PATH` | Shared across pooled training cohort |
| `results` | `<source_root>/results` | `PATH/<cohort>` | Per logical cohort |

- **Scenario 1 (single-cohort self-contained run, no root flags):** all derived roles resolve under `<source_root>/{preprocessing,features,models,results}/...` with no extra `<cohort>` directory under `results/`.
- **Scenario 2 (experiments / multi-cohort evaluation with `--models-root /exp1/models --results-root /exp1/results`):** models resolve under `/exp1/models/<stage>/...` and each evaluated cohort resolves side-by-side under `/exp1/results/<cohort>/<stage>/...`.

### 2. Directory Tree Across Roles
```text
# 1. source_root (e.g. /data/COHORT1)
wsi/{image_id}.svs
metadata/image_metadata.csv

# 2. preprocessing_root (default: <source_root>/preprocessing)
cv_splits.csv
thumbnails/{image_id}.png
background_mask/size_mm={size_mm}/{image_id}.png
tiles/size_mm={size_mm}/{image_id}/{image_id}.hdf5
tile_metadata/size_mm={size_mm}/{image_id}.parquet
tile_background/size_mm={size_mm}/{image_id}.parquet
tile_augmentation/method={method}/{aug_params}/size_mm={size_mm}/{image_id}/{image_id}.hdf5
cells/
  cell_coordinates/{image_id}.parquet
  tumor_cell_patches/{image_id}.hdf5
  mitotic/{image_id}.parquet
  tile_counts/size_mm={size_mm}/{image_id}.parquet
  cellular_features/size_mm={size_mm}/
    cellular_features.csv
    cellular_features.csv.parts/{image_id}.csv

# 3. features_root (default: <source_root>/features)
tile_features/size_mm={size_mm}/model={model}/{image_id}/{image_id}.hdf5
tile_augmentation_features/method={method}/{aug_params}/size_mm={size_mm}/model={model}/{image_id}/{image_id}.hdf5

# 4. models_root (default: <source_root>/models, or --models-root /exp1/models)
mil/fold={fold}
cellular/fold={fold}
concepts/fold={fold}
morphology/fold={fold}
staging/fold={fold}
clinical/fold={fold}
fusion/fold={fold}

# 5. results_root (default: <source_root>/results, or --results-root /exp1/results -> /exp1/results/<cohort>)
mil/
  patch_embeddings/fold={fold}/{image_id}.hdf5
  attention/fold={fold}/{image_id}.parquet
  slide_embedding/fold={fold}/{image_id}.npy
  slide_embedding_grad/fold={fold}/{image_id}.npy
  risk_score/
    fold={fold}/
      whole_image_risk_score.csv
      whole_image_risk_score.csv.parts/{image_id}.csv
    whole_image_risk_score.csv
cellular/
  cellular_risk_score.csv
concepts/
  path_concept_predictions.csv
  path_concept_risk_score.csv
morphology/
  clusters/{image_id}.parquet
  patch_morphology_presence.csv
  patch_morphology_risk_score.csv
staging/
  staging_risk_score.csv
clinical/
  clinical_risk_score.csv
fusion/
  helios_risk_score.csv
  survival_curve/{image_id}.csv
reports/
  cohort_summary.csv
  {image_id}/
    report.json
    report.md
    assets/{asset_id}.png
```

### 3. Logical Cohorts, `cv_splits`, and Hive Partitioning
- **Logical cohorts:** A cohort is either a single dataset directory (`/data/COHORT1` -> name `COHORT1`) or a cohort YAML (`cohorts/train.yaml` -> name `TRAIN_COHORT`). Multi-dataset union cohorts validate that `image_id`s are unique across constituents.
- **`cv_splits` resolution:**
  - Default location for a single dataset: `<preprocessing_root>/cv_splits.csv`.
  - Default location for a cohort YAML (`cohorts/train.yaml`): `cohorts/train.splits.csv` (or `splits:` in the YAML).
  - Overridable via `--splits-file`. `fit` and `predict` log the resolved splits file path (or log ensemble fallback when absent on `predict`).
- **Hive `key=value` directories & `scan_hive()`:**
  - All partition segments in paths use `key=value` (`size_mm={size_mm}`, `model={model}`, `method={method}`, `fold={fold}`).
  - Numbers are canonicalized (`0.250` -> `0.25`); list values join with `+`.
  - Non-default parameter variants on `prep splits` or `prep cell-features` nest sorted `key=value/` segments before the filename.
  - `scan_hive()` collects matching CSV/Parquet files (skipping `.parts/`), parses all `key=value` path segments into columns, and returns a unified DataFrame.

### 4. Pluggable Tile Augmentation & Per-Dataset Cohort Attachment
- **Pluggable transform interface (`helios.components.augmentation`):**
  - Uses a general `method={method}/{aug_params}` Hive path interface so augmentation is not locked to a single `target` string. Currently only `cyclegan` (`method=cyclegan` with optional `source` and required `target`) is implemented; other methods (e.g. combined HSV/blur) are deferred until needed.
- **Storage layout:**
  - Augmented tiles: `<preprocessing_root>/tile_augmentation/method={method}/{aug_params}/size_mm={size_mm}/{image_id}/{image_id}.hdf5`
  - Augmented features: `<features_root>/tile_augmentation_features/method={method}/{aug_params}/size_mm={size_mm}/model={model}/{image_id}/{image_id}.hdf5`
- **Single specification in Cohort YAML (materialized by `prep augment`, consumed by `fit mil`):**
  - Each dataset entry in a cohort YAML may declare an `augmentations:` list of augmentation specs:
    ```yaml
    name: TRAIN_COHORT
    datasets:
      - source_root: /data/COHORT1
        augmentations:
          - method: cyclegan
            source: COHORT1
            target: COHORT2
            checkpoint: /path/to/c1_to_c2/latest_net_G_A.pth
      - source_root: /data/COHORT2
        augmentations:
          - method: cyclegan
            source: COHORT2
            target: COHORT1
            checkpoint: /path/to/c2_to_c1/latest_net_G_A.pth
    ```
  - Stages that do not use augmentations (`prep thumbnails`, `prep tiles`, `prep tile-features`, `prep cells`) ignore the `augmentations:` field.
  - `helios prep augment --dataset cohorts/train.yaml` reads each constituent's `augmentations:` list and generates the corresponding `tile_augmentation` and `tile_augmentation_features` artifacts under that constituent's `preprocessing_root` and `features_root`.
  - `helios fit mil --dataset cohorts/train.yaml` resolves all configured `augmentations:` for each slide into `SlideFeatureRef.aug_features_paths: tuple[Path, ...]`, verifies they exist on disk, and merges them with the original slide features via per-tile swapping (`aug_mode="replace"`, sampling across the slide's attached augmentations with probability `aug_swap_prob`) or bag pooling (`aug_mode="augment"`).

## 2026-10-22: Atomic publication, `.parts/` tables & completion (AMENDS "Batch tabular aggregation")

- **Atomic writes:** Every file write uses a sibling temp file (`.<name>.<uuid>.tmp`) + `os.replace()`. Model bundles write `model.pkl` via a temp file and atomically replace `manifest.json` last (removing any prior `manifest.json` first when overwriting).
- **Re-runnable model provenance:** Alongside `model.pkl` and `manifest.json`, each model bundle writes `config.yaml` (the exact `fit.<noun>` parameter block accepted by `--config`) and `cohort.yaml` (the resolved cohort definition accepted by `--dataset`), so training can be reproduced directly via `helios fit <noun> --dataset <bundle>/cohort.yaml --config <bundle>/config.yaml`.
- **`.parts/` tables:** Slide-by-slide tabular stages (`prep cell-features` with single `size_mm: float = 0.25`, and per-fold `predict mil`) write `<table.csv>.parts/{image_id}.csv` per slide and deterministically reduce the cohort's parts into `<table.csv>`. Vectorized cohort-wide stages (`predict concept`, `predict morphology`, `predict risk`, `report` summary) write their CSVs atomically without `.parts/`.
- **Completion & input guards:** Multi-output work units (`prep tiles`, `prep cells`, `predict mil`) are skipped only if all outputs of the unit exist. `segment_cells` uses a unique temp directory per call (`tempfile.mkdtemp(prefix=f"cellvit_{slide_id}_")`). `fit` stages verify all training slides have complete input artifacts before training.

## 2026-10-22: Run-scoped logging & worker status aggregation (AMENDS "Output/report contract")

- Each CLI run creates `<log_dir>/<run_id>/` where `<run_id>` is `YYYY-MM-DD-HH-MM-SS.ffffff-utc[-<kebab-tag>]` (`--log-dir`, `--run-tag`).
- Contains `run.log` (combined lifecycle + error tracebacks), `status.json` (atomic snapshot), `events.jsonl` (append-only events), and `workers/<worker-id>.log` (per-worker detailed logs).
- Workers send throttled status events to a Ray reporting actor; the CLI process drains events, treats Ray task results/exceptions as authoritative for completion/failure, and renders one console progress display.

## 2026-10-22: Parallel execution, Ray workers & SLURM (NEW)

- **Serial by default (CONFIRMED 2026-10-23):** Without `--ray-address` or `--max-cpus > 1`, every stage runs single-process in the CLI process; Ray is never started. Execution settings travel as a separate `ParallelConfig` runtime seam (`parallel=`), not inside `StorageRoots` (which stays a pure location object: role roots + `splits_file`).
- **Ray lifecycle (opt-in):** With `--max-cpus N` (N > 1) and no `--ray-address`, starts a local Ray instance respecting `CUDA_VISIBLE_DEVICES` and `--max-cpus`, shutting it down on exit. With `--ray-address`, attaches to the existing cluster and leaves it running on exit so bash scripts (or a future GUI) can run multiple `helios` commands sequentially or in parallel (`&` + `wait`) on the shared pool. Dataset roots, role roots and the run log dir are made absolute at the CLI boundary so workers with a different cwd/node resolve the same paths. `RAY_ENABLE_UV_RUN_RUNTIME_ENV` defaults to off (set in `helios/__init__.py`) so `uv run` never ships the working directory to workers.
- **Work units:** `run_work_units(units, fn, ...)` calls `fn(unit, ctx: WorkContext)`; `ctx.progress` is a worker-local `Progress` whose events relay to the coordinator's status actor (workers never write `status.json`/`events.jsonl` directly), and `ctx.log_path` is that worker's `workers/<worker-id>.log`. Workers return a structured `WorkResult` (status, worker id, log path, error, traceback) rather than raising; the coordinator keeps a per-unit state machine (`queued -> running -> completed|skipped|failed`) from which `status.json` counts are derived.
- **`.parts/` reduce:** The reduce step always reads the parts of the whole cohort population (not just the current `--image-id`/filter selection) so partial re-runs never shrink the reduced table.
- **Cellular table selection:** `fit risk`, `predict risk` and `report` read `cellular_features` for exactly one `cellular_size_mm` (default `0.25`, shared with `prep cell-features`); no cross-partition globbing.
- **Workers:**
  - CPU tasks (`thumbnails`, `tiles`, `cell-features`) with optional `(start, stop)` slice bounds for future single-slide chunking.
  - Reusable 1-per-GPU actors (`tile-features`, `augment`, `predict mil`) with slice-ready component signatures.
    Pools are scoped to each scheduling group and explicitly released on completion/failure, including on an
    attached cluster. They are not named lookup-or-create actors shared between commands. This bounds GPU
    ownership and prevents stale model/settings reuse across independent runs. CPU-only clusters use CPU
    actors. Resource detection uses the connected cluster, not the coordinator's GPU visibility.
  - Isolated per-GPU subprocess worker for `prep cells` (CellViT) with `CUDA_VISIBLE_DEVICES` pinned to that GPU and `SystemConfiguration` CPU/memory bounded to `cpus_per_slide`; the same CPU share is reserved from Ray (`num_cpus`) so slides never oversubscribe, and CellViT stdout/stderr are appended to the worker log. In serial mode CellViT runs in-process with the same CPU bound (CellViT's internal Ray pool is its own and cannot be disabled).
  - Parallel fold tasks for `fit` stages (1 GPU per fold for `fit mil`, CPU fold tasks for `concept`, `morphology`, `risk`).
- **Model reuse:** Extraction and augmentation caches retain at most one encoder/generator configuration per
  process. Augmentation tile generation and embedding extraction have separate worker groups. MIL prediction
  caches the loaded bundle per fold and the built inference network per model/device; refitting invalidates
  the network cache, which is not included in serialized bundles.
- **Scheduling bounds:** At most one active call per actor. Both tasks and actors are bounded by cluster
  CPU/GPU capacity and the invocation's CPU cap; work is replenished as results arrive. Single-unit
  invocations still honor explicit Ray execution. Risk kinds finish their folds before dependent kinds start.
- **Failures:** Slide tasks continue on remaining slides when one fails, log tracebacks, and exit nonzero at stage end.
- **SLURM scripts:** Provide both fixed multi-node allocation (`ray-cluster.sbatch`) and dynamic job-array worker pool (`ray-array-workers.sbatch`).

## 2026-10-26: Dataset descriptors and prepared augmentation directories

This amendment supersedes the earlier features storage role and augmentation
production/selection contract. Other runtime/design concerns remain separate.

- A physical dataset has `dataset.yaml` with required nonempty `id` and `name`
  strings and optional `description`. IDs have no hierarchy or path semantics.
  Directory moves do not change identity; logical cohort names remain separate.
  Original-only stage calls remain compatible without descriptors, but dataset
  checks and augmentation preparation/selection require them.
- Four storage roles remain: source, preprocessing, models, results. Embeddings
  belong to preprocessing under `features/size_mm=.../model=.../`. The removed
  features-root CLI/YAML options report migration errors; files are not moved
  or selected from the legacy layout automatically.
- Augmentations are prepared directories containing `augmentation.yaml`, tiles,
  and features side by side. The definition records `source_dataset_id`, method,
  parameters, and checkpoint location. Readable partition names organize output;
  fit does not parse them or reconstruct transformation settings.
- Prep retains transformation-specific options and supports an explicit output
  directory for one dataset/target. Without explicit targets it can resume the
  saved definitions selected per dataset in cohort YAML.
- Fit takes repeatable augmentation directory paths, pairs them by dataset ID
  even for CLI pooling, and rejects absent/ambiguous ownership. YAML can attach
  paths directly to constituents. Relative YAML paths are YAML-file-relative.
  CLI paths replace YAML selections; an explicit disable selects original-only
  training. Effective selections are preserved in bundle cohort provenance.
- Fit requires prepared features, not retained augmented tiles. It validates
  shapes, embedding widths, tile IDs/order, and coordinates before training.
  It does not generate missing prerequisites; force means retrain, not prepare.
  Prep retains alignment metadata with both augmented tiles and embeddings.
- Discovery lists saved directories and available tile/feature file counts,
  without claiming that every required artifact exists.
- Checkpoint/content hashing and mandatory settings-conflict enforcement are
  intentionally absent. Reuse with changed settings warns but is permitted.
  Existing files skip unless forced; force affects only the requested units.
  Preventing mixed outputs when reusing directories is the user's responsibility.
