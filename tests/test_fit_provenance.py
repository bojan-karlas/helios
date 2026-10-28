"""Fit configuration matches both component arguments and bundle provenance."""
from __future__ import annotations

from inspect import signature

import numpy as np
import pandas as pd
import pytest
import yaml

from helios.config.introspect import build_default_config
from helios.data.io import ArtifactStore
from helios.data.resolver import Resolver, Roots
from helios.stages import concept, mil, morphology, risk
from tests.fixtures import make_cohort


@pytest.mark.parametrize(
    ("module", "noun", "component_name", "artifact", "overrides"),
    [
        (mil, "mil", "train_mil", "model_mil", {
            "size_mm": 0.5, "model": "uni", "target": "disease_pfs_recurred",
            "aug_swap_prob": 0.3, "aug_mode": "augment", "n_branches": 2, "num_heads": 2,
            "dropout_rate": 0.2, "lambda_div": 0.02, "lr": 0.002, "weight_decay": 0.003,
            "max_epochs": 3, "patience": 2, "val_frac": 0.2, "seed": 7, "device": "cpu",
        }),
        (concept, "concept", "train_concepts", "model_concepts", {"n_bootstrap": 3}),
        (morphology, "morphology", "train_morphology", "model_morphology", {
            "target": "disease_pfs_recurred", "attention_cutoff": 0.6, "n_pca": 2,
            "k_micro": 4, "k_neighbors": 3, "resolution": 0.7,
            "significance_alpha": 0.1, "estimator": "logreg_l2", "seed": 7,
        }),
        (risk, "risk", "train_risk_model", "model_clinical", {
            "risk": ["clinical"], "target": "disease_pfs_recurred", "staging_feature": "path_thickness",
            "clinical_features": ["disease_age_at_diagnosis"], "estimator": "logreg", "cellular_size_mm": 0.5,
        }),
    ],
)
def test_fit_saves_effective_parameters(monkeypatch, tmp_path, module, noun, component_name, artifact, overrides):
    root = make_cohort(tmp_path / "example", n_images=2, n_patients=2)
    store = ArtifactStore(Resolver(Roots.single(root)))
    ids = store.read("image_metadata")["image_id"].tolist()
    for image_id in ids:
        store.write(
            "tile_features", {"features": np.ones((2, 4), dtype=np.float32)},
            size_mm=0.5, model="uni", image_id=image_id,
        )
        store.write(
            "aligned_patch_embeddings", {"embeddings": np.ones((2, 4), dtype=np.float32)}, image_id=image_id,
        )
        store.write(
            "patch_attention", pd.DataFrame({"tile_id": [0, 1], "attention": [0.4, 0.6]}), image_id=image_id,
        )
        store.write("aligned_slide_embedding", np.ones(4, dtype=np.float32), image_id=image_id)
    store.write("whole_image_risk_score", pd.DataFrame({"image_id": ids, "risk_score": [0.2, 0.4]}))

    component = getattr(module, component_name)
    training_keys = {
        key for key, param in signature(component).parameters.items()
        if param.kind == param.KEYWORD_ONLY and key not in {"progress", "fold"}
    }
    calls = []

    def train(*args, **kwargs):
        calls.append(kwargs)
        return {"trained": True}

    monkeypatch.setattr(module, component_name, train)
    getattr(module, f"fit_{noun}")(datasets=[root], **overrides)

    expected = build_default_config()["fit"][noun] | overrides
    config = yaml.safe_load((store.path(artifact, fold=None) / "config.yaml").read_text())
    assert config == {"fit": {noun: expected}}
    assert len(calls) == 1
    assert {key: value for key, value in calls[0].items() if key not in {"progress", "fold"}} == {
        key: expected[key] for key in training_keys
    }
