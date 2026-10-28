"""Scalar tile-size contract for cellular-feature preparation."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from helios.stages import cell_features


@pytest.mark.parametrize("size_mm", [0.25, 0.5])
def test_scalar_size_is_used_for_work_units_and_reduction(monkeypatch, size_mm: float) -> None:
    dataset = MagicMock(name="dataset")
    store = MagicMock()
    store.read_parts.return_value = pd.DataFrame()
    dispatch = MagicMock()
    monkeypatch.setattr(cell_features, "resolve_datasets", lambda *args, **kwargs: [dataset])
    monkeypatch.setattr(cell_features, "cohort_store", lambda *args: store)
    monkeypatch.setattr(cell_features, "select_image_ids", lambda *args: [])
    monkeypatch.setattr(cell_features, "run_work_units", dispatch)

    cell_features.prep_cell_features(datasets=[], size_mm=size_mm)

    worker = dispatch.call_args.args[1]
    assert worker.keywords["table_keys"]["size_mm"] == size_mm
    assert store.read_parts.call_args.kwargs["size_mm"] == size_mm


@pytest.mark.parametrize("size_mm", [[], [0.25], [0.25, 0.5]])
def test_list_size_is_rejected_before_dataset_resolution(monkeypatch, size_mm) -> None:
    resolve = MagicMock()
    monkeypatch.setattr(cell_features, "resolve_datasets", resolve)
    with pytest.raises(TypeError):
        cell_features.prep_cell_features(datasets=[], size_mm=size_mm)
    resolve.assert_not_called()
