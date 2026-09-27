"""Focused tests for the leakage-safe Stage 1 five-class regime baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.features.feature_store import FeatureStore
from src.features.preprocessing import FEATURE_CHANNELS
from src.features.feature_store import SYNOPTIC_FEATURE_NAMES
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS
from src.models.stage1_regime.classifier import (
    CLASS_ID_TO_REGIME_CODE,
    ML_CLASS_IDS,
    ML_REGIME_CODES,
    WESTERN_DISTURBANCE_POLICY,
    balanced_training_weights,
    evaluate_predictions,
    lightgbm_parameters,
    list_feature_store_dates,
    load_stage1_datasets,
    load_stage1_features_for_date,
    predict_regime_for_date,
    train_stage1_model,
)


@pytest.fixture(scope="module")
def synthetic_feature_store(tmp_path_factory: pytest.TempPathFactory) -> FeatureStore:
    store_dir = tmp_path_factory.mktemp("manual_test_store") / "store"
    store_dir.mkdir()
    train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    validation_labels = [0, 1, 2, 3, 4, 5]
    test_labels = [0, 1, 2, 3, 4]
    label_groups = [(2021, train_labels[:6]), (2022, train_labels[6:]), (2023, validation_labels), (2024, test_labels)]
    records = []
    dates = []
    for year, labels in label_groups:
        for day_index, regime_id in enumerate(labels, start=1):
            date = pd.Timestamp(year=year, month=6, day=day_index)
            dates.append(date)
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "year": year,
                    "regime_id": regime_id,
                    "regime_code": (
                        CLASS_ID_TO_REGIME_CODE[regime_id]
                        if regime_id in CLASS_ID_TO_REGIME_CODE
                        else "WESTERN_DISTURBANCE"
                    ),
                }
            )

    metadata = pd.DataFrame(records)
    metadata.to_parquet(store_dir / "metadata.parquet", index=False)
    sample_count = len(metadata)
    features = np.empty(
        (sample_count, len(FEATURE_CHANNELS), len(CANONICAL_LATS), len(CANONICAL_LONS)),
        dtype=np.float32,
    )
    for index in range(sample_count):
        features[index] = np.float32(index + 1)
    synoptic = np.arange(sample_count * len(SYNOPTIC_FEATURE_NAMES), dtype=np.float32).reshape(
        sample_count, len(SYNOPTIC_FEATURE_NAMES)
    )
    observed_target = np.full(
        (sample_count, len(CANONICAL_LATS), len(CANONICAL_LONS)), 1_000_000.0, dtype=np.float32
    )
    valid_mask = np.ones_like(observed_target, dtype=bool)
    dataset = xr.Dataset(
        data_vars={
            "features": (("date", "channel", "lat", "lon"), features),
            "target": (("date", "lat", "lon"), observed_target),
            "valid_mask": (("date", "lat", "lon"), valid_mask),
            "synoptic_vector": (("date", "synoptic_var"), synoptic),
        },
        coords={
            "date": dates,
            "channel": FEATURE_CHANNELS,
            "lat": CANONICAL_LATS,
            "lon": CANONICAL_LONS,
            "synoptic_var": SYNOPTIC_FEATURE_NAMES,
        },
    )
    dataset.to_netcdf(store_dir / "tensors.nc", engine="netcdf4")
    dataset.close()
    return FeatureStore(store_dir)


@pytest.fixture(scope="module")
def manual_test_model_artifact(synthetic_feature_store, tmp_path_factory: pytest.TempPathFactory) -> Path:
    partitions = load_stage1_datasets(synthetic_feature_store)
    artifact_dir = tmp_path_factory.mktemp("manual_test_model") / "stage1"
    train_stage1_model(
        partitions,
        artifact_dir=artifact_dir,
        random_seed=17,
        num_boost_round=6,
        early_stopping_rounds=2,
    )
    return artifact_dir


def test_feature_store_loading_split_and_five_class_mapping(synthetic_feature_store):
    partitions = load_stage1_datasets(synthetic_feature_store)
    assert set(partitions) == {"train", "validation", "test"}
    assert [partitions[key].labels.size for key in ("train", "validation", "test")] == [10, 5, 5]
    assert [partitions[key].excluded_western_disturbance for key in ("train", "validation", "test")] == [1, 1, 0]
    assert all(set(np.unique(part.labels)) == set(ML_CLASS_IDS) for part in partitions.values())
    assert len(partitions["train"].feature_names) == 7 * len(FEATURE_CHANNELS) + len(SYNOPTIC_FEATURE_NAMES)
    assert all(part.features.shape[1] == len(part.feature_names) for part in partitions.values())
    assert partitions["train"].dates[0].startswith("2021-")
    assert all(date.startswith("2023-") for date in partitions["validation"].dates)
    assert all(date.startswith("2024-") for date in partitions["test"].dates)
    assert "target" not in partitions["train"].feature_names
    assert "valid_mask" not in partitions["train"].feature_names
    assert "synoptic__trough_latitude_departure" in partitions["train"].feature_names


def test_observed_target_changes_do_not_change_predictor_matrix(synthetic_feature_store, tmp_path):
    original = load_stage1_datasets(synthetic_feature_store)
    copied_dir = tmp_path / "target_changed"
    copied_dir.mkdir()
    original_metadata = synthetic_feature_store.load_metadata()
    original_metadata.to_parquet(copied_dir / "metadata.parquet", index=False)
    source = synthetic_feature_store.load_tensors()
    changed = source.copy(deep=True)
    changed["target"] = xr.full_like(source["target"], -987654.0)
    changed.to_netcdf(copied_dir / "tensors.nc", engine="netcdf4")
    source.close()
    changed.close()

    modified_target = load_stage1_datasets(copied_dir)
    for split in original:
        np.testing.assert_array_equal(original[split].features, modified_target[split].features)
        np.testing.assert_array_equal(original[split].labels, modified_target[split].labels)


def test_training_weights_use_only_passed_training_labels():
    labels = np.asarray([0, 0, 0, 1, 1, 2, 3, 3, 4, 4], dtype=np.int32)
    weights, class_weights = balanced_training_weights(labels)
    expected = {0: 2 / 3, 1: 1.0, 2: 2.0, 3: 1.0, 4: 1.0}
    assert class_weights == pytest.approx(expected)
    np.testing.assert_allclose(weights, [expected[int(label)] for label in labels])
    assert 5 not in class_weights


def test_lightgbm_configuration_is_reproducible_and_five_class():
    first = lightgbm_parameters(42)
    second = lightgbm_parameters(42)
    assert first == second
    assert first["objective"] == "multiclass"
    assert first["num_class"] == 5
    assert first["seed"] == 42
    assert first["deterministic"] is True


def test_train_artifact_predictions_probabilities_and_metrics(synthetic_feature_store, tmp_path):
    partitions = load_stage1_datasets(synthetic_feature_store)
    result = train_stage1_model(
        partitions,
        artifact_dir=tmp_path / "model_artifact",
        random_seed=42,
        num_boost_round=8,
        early_stopping_rounds=3,
    )
    test = partitions["test"]
    probabilities = np.asarray(result.model.predict(test.features))
    predictions = np.argmax(probabilities, axis=1)
    assert predictions.shape == (test.labels.size,)
    assert probabilities.shape == (test.labels.size, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(predictions)).issubset(set(ML_CLASS_IDS))
    assert set(result.metrics) == {"train", "validation", "test"}
    assert set(result.metrics["test"]) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "confusion_matrix",
        "per_class",
    }
    assert result.metrics["test"]["confusion_matrix"]
    assert set(result.metrics["test"]["per_class"]) == set(ML_REGIME_CODES)
    assert result.metrics["test"]["per_class"]["COASTAL_REGIME"]["support"] == 1
    assert (result.artifact_dir / "model.txt").exists()
    assert (result.artifact_dir / "feature_names.json").exists()
    assert (result.artifact_dir / "metadata.json").exists()
    assert (result.artifact_dir / "evaluation_metrics.json").exists()
    assert (result.artifact_dir / "feature_importance.csv").exists()
    assert result.metadata["num_classes"] == 5
    assert result.metadata["western_disturbance_policy"] == WESTERN_DISTURBANCE_POLICY
    assert result.metadata["western_disturbance_excluded_by_split"] == {
        "train": 1,
        "validation": 1,
        "test": 0,
    }


def test_metrics_use_fixed_five_class_order():
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    probabilities = np.eye(5, dtype=np.float64)
    metrics = evaluate_predictions(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["confusion_matrix"] == np.eye(5, dtype=int).tolist()
    assert metrics["class_order"] == list(ML_REGIME_CODES)


def test_manual_date_prediction_and_candidate_label(
    synthetic_feature_store, manual_test_model_artifact
):
    prediction = predict_regime_for_date(
        "2024-06-01",
        artifact_dir=manual_test_model_artifact,
        store=synthetic_feature_store,
    )
    assert prediction.date == "2024-06-01"
    assert prediction.candidate_a_label == "ACTIVE_MONSOON"
    assert prediction.predicted_regime in ML_REGIME_CODES
    assert set(prediction.probabilities) == set(ML_REGIME_CODES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert prediction.confidence == pytest.approx(max(prediction.probabilities.values()))
    assert prediction.matches_candidate_a == (
        prediction.predicted_regime == prediction.candidate_a_label
    )


def test_manual_date_prediction_rejects_invalid_or_unavailable_dates(
    synthetic_feature_store, manual_test_model_artifact
):
    with pytest.raises(ValueError, match="expected YYYY-MM-DD"):
        predict_regime_for_date(
            "2024-02-30",
            artifact_dir=manual_test_model_artifact,
            store=synthetic_feature_store,
        )
    with pytest.raises(ValueError, match="not present in the feature store"):
        predict_regime_for_date(
            "2024-05-01",
            artifact_dir=manual_test_model_artifact,
            store=synthetic_feature_store,
        )


def test_manual_feature_transform_matches_training_transform(synthetic_feature_store):
    features, names, candidate_label = load_stage1_features_for_date(
        "2024-06-01", store=synthetic_feature_store
    )
    test_partition = load_stage1_datasets(synthetic_feature_store)["test"]
    row_index = test_partition.dates.index("2024-06-01")
    np.testing.assert_array_equal(features[0], test_partition.features[row_index])
    assert names == test_partition.feature_names
    assert candidate_label == "ACTIVE_MONSOON"


def test_manual_tester_lists_available_dates(synthetic_feature_store):
    dates = list_feature_store_dates(synthetic_feature_store)
    assert dates == tuple(sorted(dates))
    assert "2024-06-01" in dates
    assert "2021-06-01" in dates
