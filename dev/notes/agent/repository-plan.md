# HELIOS Repository — Planning (workshopping, no code yet)

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

## Dataset contract (from WASHU/MRV samples + data dictionary)
- USER PROVIDES (input):
  - `wsi/**/*.svs` — FLAT set (subfolders like Cases/Controls are cosmetic, ignored).
  - `metadata/image_metadata.csv` — authoritative per-image table
    (join of sample_metadata [labels+patient] + image_file_metadata [tech]).
  - Ignore `bag_metadata.csv`. `sensitive/` PHI variant never required/committed.
- PIPELINE PRODUCES (Hive-partitioned `key=value` dirs):
  - metadata/tile_metadata/size_mm=<MPP>/<image_id>.parquet (tile_row,col,x,y,width,height)
  - tiles/size_mm=<MPP>/<image_id>/<image_id>.hdf5 (+ debug pngs: background/histograms/thumbnail)
  - preprocessing/tile_background/size_mm=<MPP>/<image_id>.parquet
  - preprocessing/tile_features/size_mm=<MPP>/<model>/<image_id>/<image_id>.hdf5
    (12 foundation models: virchow2,uni,uni2,conch,conch1.5,chief,ctranspath,dino,gigapath,musk,phikon2,resnet-50)
  - preprocessing/tile_normalization[...]/target=<STAIN_TARGET>/size_mm=<MPP>/... (e.g. MGB+MRV, VISIOMEL)
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


## Path resolution & split roots (CONFIRMED)
- Hybrid: each stage checks artifact; present -> consume; missing -> produce. (Make file-target/DAG.)
- Stages live under INDEPENDENT, configurable roots (NOT one monolithic dataset dir):
  - source root (READ-ONLY): wsi/**/*.svs, metadata/image_metadata.csv
  - work/intermediate root (RW): tiles/, preprocessing/, tile_metadata/, thumbnails/
  - features root (RW, optional): large feature stores (WASHU symlinks tile_features to pipelinecache)
- Per-stage resolver: stage -> (root, partition-template). Templates use partition keys only
  (image_id, size_mm, model, target); NO cohort names / absolute paths in repo.
- Defaults to single root if only one given (WASHU-as-one-tree still works).
- Pipeline NEVER writes into source root (safe for ro mounts; matches Docker mount contract:
  input ro, output rw with status.json/events.jsonl).
- Resolver is the dataset-agnostic seam; hybrid skip logic lives here (exists(resolve(...))).

## Per-training-stage dataset binding + precondition resolution (CONFIRMED)
- WHY: a training stage may fit on a DIFFERENT dataset than the rest of the pipeline (e.g. concept models
    trained on a separate concept-annotated cohort; OMG-Net is already fully external). So "the training
    dataset" is NOT global — it is PER FIT STAGE.
- CONFIG: each fit stage (cell*, mil, concepts, cellular, morphology, fusion, ...) declares its OWN
    `dataset` binding in the config files (configs/train/<stage>.yaml). A binding = the set of roots
    (source/work/features) for that stage. Stages default to the GLOBAL/primary dataset when unset, so the
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
    K sets of embeddings/attention, K of each downstream risk model). We ALWAYS ensemble the K models at deploy —
    there is NEVER a final "train on all data" model. So the K models ARE the shipped deliverable.
- DRIVER: `cv_splits` artifact — OPTIONAL tidy long CSV `metadata/cv_splits.csv`, columns `image_id, fold,
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
    the OUTERMOST coordinate and materializes as a `fold={fold}/` PATH SEGMENT — never a column. The
    "empty optional key segment drops its path segment" rule gives us ONE artifact identity at two arities.
- THREE artifact shapes:
    - per-fold  : `fold` in key + `fold={fold}/` in path => K physical instances. These are the K models and the
        MIL outputs consumed by per-fold training: `model_*` (mil/concepts/cellular/morphology/fusion/staging/
        clinical), `aligned_slide_embedding`, `aligned_patch_embeddings`, `patch_attention`, and the per-fold
        `whole_image_risk_score` (one CSV per fold, one row per image). `model_cyclegan` stays fold-free (pre-CV).
    - reduced   : NO fold in key/path; one instance per image, produced by a fold-aware REDUCE component that
        collapses the K inputs — OUT-OF-FOLD (each image scored by its TEST-fold model) on the CV cohort /
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
    - none   = pre-CV / fold-agnostic (preprocessing, cell pipeline, cellular_features_compute, cyclegan,
        case_report_build, markdown_render, cohort_summarize — all consume only reduced/fold-free inputs).
- PARALLELISM: fold is embarrassingly parallel — each `fold: map` run is an independent job writing its own
    `fold={fold}` files (no shared-writer hazard). The per-fold whole-image score is a per-fold CSV
    (`mil/fold={fold}/whole_image_risk_score.csv`) for the same reason (avoids one process holding all K MIL
    models to write a single file).
- CV SPLITS GENERATION: `cv_splits_generation` (helios.data.cv_splits, train-only) builds cv_splits.csv from a
    cohort's image_metadata. Params: n_folds (5), val_fraction (0 => no val rows), stratify_by (label cols),
    group_by (e.g. patient_id, to keep a patient's images in one fold — leakage guard), seed. Run once per
    training cohort; a separate concept cohort gets its own run if it needs folds.
- VAL SPLIT is OPTIONAL: cv_splits.split may include `val`, but if a model needs validation and no `val` rows
    exist, it carves a random train/val split out of that fold's `train` rows. So `val` is provisioned but never
    required.
- CONCEPT BOOTSTRAPPING: concept_models_training fits, per fold, a SET of probes PER CONCEPT by bootstrapping the
    concept-training set (n_bootstrap, default ~100; n_bootstrap=1 disables). This yields a concept-prediction
    DISTRIBUTION / uncertainty rather than a point estimate. model_concepts (per fold) therefore holds
    n_bootstrap x n_concepts probes + the concept risk + residual error-correction models. Bootstrap is an
    INTERNAL replication inside the per-fold bundle (NOT a key/path dimension) — folds stay the only fold axis.
- HIVE / PARTITIONS-AS-COLUMNS (future convenience, not a dependency): the `field={field}/` path convention is
    Hive-compatible by design, so a future `helios.data.dataset` reader can present a partitioned tree as ONE
    logical table with fold/size_mm/model/target surfaced as COLUMNS, with predicate pushdown — via PyArrow
    Datasets (`partitioning="hive"`), DuckDB (`hive_partitioning=true`), or Polars (`hive_partitioning=True`).
    Applies to the columnar/csv artifacts (scores, metadata, attention, clusters); hdf5/npy blobs stay file-addressed.

## Output/report contract (web-app boundary) — workshopped
PRINCIPLE: output is a DATA PACKAGE, not a rendered report. Structured arrays
(parquet/hdf5/npy/json) are authoritative; web app can self-render overlays/heatmaps/
cell-maps at any zoom. Pre-rendered PNGs optional. Schemas LIGHT + versioned
(schema_version), meant to evolve. NO heavy validators now (permissive parsing;
dataclasses/TypedDict; pydantic only sparingly for config). Self-describing formats.

Job dir:
  <output_root>/<job_id>/
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

## Batch tabular aggregation (cohort tables)
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
LAYOUT DECISION: src/helios/ (src-layout) — guarantees tests exercise the INSTALLED artifact
(same as users/Docker get), catches packaging errors early, no import shadowing, clean root.
Requires install-first (uv sync / make install) — already step one.

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
    processing/ hybrid produce/consume stages: tiling, background, features, normalization, thumbnails
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

## Remote sync (ops layer) — CONFIRMED: scripts/sync.sh (smart) + Make shortcuts + rclone + .env (NOT library code)
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
- .env can ALSO hold other runtime roots (source/work/output) the CLI reads via env vars
  -> single uncommitted config point, consistent pattern.
- SUPERSEDES earlier src/helios/sync/ module idea (package stays pure, no rclone dep).

## Docs split — CONFIRMED
- README.md = LIGHT public overview (ships with paper): what HELIOS does, workflow, how to cite.
- CONTRIBUTING.md = DETAILED contributor guide: directory structure, what each file/module is for,
  where to place new code, conventions (CLI-first, library-first, data-agnostic rule, sync ops layer),
  dev workflow (make shell/install/fit/predict, tests).

## Renames/clarifications
- src/helios/stages/ -> src/helios/processing/
- data/ readers.py + writers.py -> data/io.py (read+write together per format)
- data/spec.py = DatasetSpec input contract; data/resolver.py = path resolver (only layout-aware code);
  data/manifest.py = image index (scan wsi flattened + join image_metadata.csv).
