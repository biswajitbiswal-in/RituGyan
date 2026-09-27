"""Stage 1 five-class weather-regime classifier.

The model uses operational predictor fields and synoptic scalars only. The
observed rainfall target and its validity mask are never read by this module.
Western Disturbance remains a rule-based fallback because the available
training data do not support learning or testing that class.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.features.feature_store import FeatureStore


ML_REGIME_CODES: Tuple[str, ...] = (
    "ACTIVE_MONSOON",
    "BREAK_MONSOON",
    "MONSOON_DEPRESSION",
    "OROGRAPHIC_MONSOON",
    "COASTAL_REGIME",
)
ML_CLASS_IDS: Tuple[int, ...] = tuple(range(len(ML_REGIME_CODES)))
REGIME_ID_TO_CLASS_ID: Dict[int, int] = dict(zip(range(5), ML_CLASS_IDS))
CLASS_ID_TO_REGIME_CODE: Dict[int, str] = dict(enumerate(ML_REGIME_CODES))
WESTERN_DISTURBANCE_ID = 5

SPATIAL_REGIONS: Mapping[str, Tuple[float, float, float, float] | None] = {
    "domain": None,
    "depression_track": (16.0, 24.0, 76.0, 90.0),
    "central_india": (18.0, 25.0, 75.0, 85.0),
    "arabian_sea_gateway": (6.5, 12.0, 68.0, 78.0),
    "western_ghats": (10.0, 18.0, 72.5, 76.0),
    "northwest_india": (28.0, 36.0, 70.0, 80.0),
    "himalayan_foothills": (26.0, 30.0, 78.0, 90.0),
}

WESTERN_DISTURBANCE_POLICY = (
    "WESTERN_DISTURBANCE is not an ML-trained class because the available "
    "dataset contains only 4 total samples and zero 2024 test samples. It "
    "remains a documented rule-based/fallback regime."
)


@dataclass(frozen=True)
class Stage1Dataset:
    """One temporally isolated predictor/label partition."""

    features: np.ndarray
    labels: np.ndarray
    dates: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    excluded_western_disturbance: int


@dataclass(frozen=True)
class Stage1TrainingResult:
    """Trained LightGBM booster and its saved evaluation summary."""

    model: lgb.Booster
    artifact_dir: Path
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Stage1Prediction:
    """Manual date prediction and comparison with its Candidate A label."""

    date: str
    candidate_a_label: str
    predicted_regime: str
    confidence: float
    probabilities: Mapping[str, float]
    matches_candidate_a: bool


def _date_key(value: Any) -> str:
    return pd.Timestamp(str(value)).strftime("%Y-%m-%d")


def _feature_names(channel_names: Sequence[str], synoptic_names: Sequence[str]) -> Tuple[str, ...]:
    spatial_names = tuple(
        f"spatial__{channel}__{region}__area_weighted_mean"
        for region in SPATIAL_REGIONS
        for channel in channel_names
    )
    return spatial_names + tuple(f"synoptic__{name}" for name in synoptic_names)


def _sample_predictors(
    tensors: Any,
    sample_indices: Sequence[int],
    channel_names: Sequence[str],
    synoptic_names: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    feature_array = tensors["features"]
    synoptic_array = tensors["synoptic_vector"]
    if feature_array.dims != ("date", "channel", "lat", "lon"):
        raise ValueError(f"Unexpected predictor dimensions: {feature_array.dims}")
    if synoptic_array.dims != ("date", "synoptic_var"):
        raise ValueError(f"Unexpected synoptic dimensions: {synoptic_array.dims}")

    lats = np.asarray(tensors["lat"].values, dtype=np.float64)
    lons = np.asarray(tensors["lon"].values, dtype=np.float64)
    if np.any(np.diff(lats) <= 0) or np.any(np.diff(lons) <= 0):
        raise ValueError("Feature-store coordinates must be strictly ascending")
    lat_radians = np.deg2rad(lats)
    area_weights = np.broadcast_to(np.cos(lat_radians)[:, None], (len(lats), len(lons)))

    region_masks: Dict[str, np.ndarray] = {}
    for region, bounds in SPATIAL_REGIONS.items():
        if bounds is None:
            mask = np.ones((len(lats), len(lons)), dtype=bool)
        else:
            lat_min, lat_max, lon_min, lon_max = bounds
            mask = (
                (lats[:, None] >= lat_min)
                & (lats[:, None] <= lat_max)
                & (lons[None, :] >= lon_min)
                & (lons[None, :] <= lon_max)
            )
        if not np.any(mask):
            raise ValueError(f"Spatial region {region!r} has no grid cells")
        region_masks[region] = mask

    names = _feature_names(channel_names, synoptic_names)
    feature_rows: List[np.ndarray] = []
    for sample_index in sample_indices:
        spatial_grid = np.asarray(feature_array.isel(date=int(sample_index)).values, dtype=np.float64)
        synoptic_vector = np.asarray(synoptic_array.isel(date=int(sample_index)).values, dtype=np.float64)
        if spatial_grid.shape != (len(channel_names), len(lats), len(lons)):
            raise ValueError(f"Unexpected predictor shape: {spatial_grid.shape}")
        if synoptic_vector.shape != (len(synoptic_names),):
            raise ValueError(f"Unexpected synoptic feature shape: {synoptic_vector.shape}")

        spatial_values: List[float] = []
        for region in SPATIAL_REGIONS:
            mask = region_masks[region]
            weights = area_weights[mask]
            denominator = float(weights.sum())
            for channel_index in range(len(channel_names)):
                values = spatial_grid[channel_index][mask]
                spatial_values.append(float(np.dot(values, weights) / denominator))
        feature_rows.append(np.concatenate((np.asarray(spatial_values), synoptic_vector)))

    matrix = np.asarray(feature_rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"Unexpected assembled feature matrix shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Stage 1 predictor matrix contains NaN or Inf")
    return matrix, names


def _prepare_partition(
    metadata: pd.DataFrame,
    tensors: Any,
    date_to_index: Mapping[str, int],
    channel_names: Sequence[str],
    synoptic_names: Sequence[str],
) -> Stage1Dataset:
    unknown_ids = set(metadata["regime_id"].astype(int).unique()) - set(range(6))
    if unknown_ids:
        raise ValueError(f"Unsupported regime IDs in feature store: {sorted(unknown_ids)}")
    expected_codes = {**dict(enumerate(ML_REGIME_CODES)), WESTERN_DISTURBANCE_ID: "WESTERN_DISTURBANCE"}
    inconsistent = metadata[
        metadata.apply(lambda row: expected_codes[int(row["regime_id"])] != row["regime_code"], axis=1)
    ]
    if not inconsistent.empty:
        raise ValueError("Feature-store regime IDs do not match their regime codes")

    excluded = metadata[metadata["regime_id"].astype(int) == WESTERN_DISTURBANCE_ID]
    supported = metadata[metadata["regime_id"].astype(int).isin(REGIME_ID_TO_CLASS_ID)].copy()
    dates = tuple(_date_key(value) for value in supported["date"])
    indices = [date_to_index[date] for date in dates]
    features, names = _sample_predictors(tensors, indices, channel_names, synoptic_names)
    labels = np.asarray(
        [REGIME_ID_TO_CLASS_ID[int(value)] for value in supported["regime_id"]],
        dtype=np.int32,
    )
    return Stage1Dataset(
        features=features,
        labels=labels,
        dates=dates,
        feature_names=names,
        excluded_western_disturbance=len(excluded),
    )


def load_stage1_datasets(
    store: FeatureStore | str | Path = "data/processed/feature_store",
) -> Dict[str, Stage1Dataset]:
    """Load leakage-safe train/validation/test matrices from the canonical store.

    Only metadata labels, predictor channels, and synoptic vectors are accessed.
    The target rainfall and valid-land mask are not read.
    """
    feature_store = store if isinstance(store, FeatureStore) else FeatureStore(store)
    train_meta, validation_meta, test_meta = feature_store.get_temporal_split(
        train_years=[2021, 2022], val_years=[2023], test_years=[2024]
    )
    with feature_store.load_tensors() as tensors:
        tensor_dates = tuple(_date_key(value) for value in tensors["date"].values)
        if len(set(tensor_dates)) != len(tensor_dates):
            raise ValueError("Feature-store tensor dates contain duplicates")
        date_to_index = {date: index for index, date in enumerate(tensor_dates)}
        channel_names = tuple(str(value) for value in tensors["channel"].values)
        synoptic_names = tuple(str(value) for value in tensors["synoptic_var"].values)
        if len(channel_names) != tensors.sizes["channel"]:
            raise ValueError("Feature-store channel coordinates are inconsistent")
        if len(synoptic_names) != tensors.sizes["synoptic_var"]:
            raise ValueError("Feature-store synoptic coordinates are inconsistent")
        partitions = {
            "train": _prepare_partition(train_meta, tensors, date_to_index, channel_names, synoptic_names),
            "validation": _prepare_partition(
                validation_meta, tensors, date_to_index, channel_names, synoptic_names
            ),
            "test": _prepare_partition(test_meta, tensors, date_to_index, channel_names, synoptic_names),
        }
    if not partitions["train"].feature_names:
        raise ValueError("No supported Stage 1 training samples found")
    if any(part.features.shape[1] != len(partitions["train"].feature_names) for part in partitions.values()):
        raise ValueError("Feature names differ across temporal partitions")
    return partitions


def list_feature_store_dates(
    store: FeatureStore | str | Path = "data/processed/feature_store",
) -> Tuple[str, ...]:
    """Return the available sample dates without reading tensor targets."""
    feature_store = store if isinstance(store, FeatureStore) else FeatureStore(store)
    dates = pd.read_parquet(feature_store.metadata_path, columns=["date"])["date"]
    normalized = tuple(sorted(_date_key(value) for value in dates))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Feature-store metadata dates contain duplicates")
    return normalized


def load_stage1_features_for_date(
    date: str,
    store: FeatureStore | str | Path = "data/processed/feature_store",
) -> Tuple[np.ndarray, Tuple[str, ...], str]:
    """Build the exact training feature row for a date and return its rule label.

    Reads only date/regime metadata plus predictor and synoptic variables. The
    observed-rainfall target and validity mask are not read.
    """
    try:
        requested_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date {date!r}; expected YYYY-MM-DD") from exc

    feature_store = store if isinstance(store, FeatureStore) else FeatureStore(store)
    metadata = pd.read_parquet(
        feature_store.metadata_path,
        columns=["date", "regime_id", "regime_code"],
    )
    metadata["date"] = metadata["date"].map(_date_key)
    match = metadata.loc[metadata["date"] == requested_date]
    if match.empty:
        raise ValueError(f"Date {requested_date} is not present in the feature store")
    if len(match) != 1:
        raise ValueError(f"Date {requested_date} occurs more than once in the feature store")
    candidate_row = match.iloc[0]
    regime_id = int(candidate_row["regime_id"])
    if regime_id not in range(6):
        raise ValueError(f"Unsupported Candidate A regime ID {regime_id} for {requested_date}")
    expected_code = (
        CLASS_ID_TO_REGIME_CODE[regime_id]
        if regime_id in CLASS_ID_TO_REGIME_CODE
        else "WESTERN_DISTURBANCE"
    )
    candidate_label = str(candidate_row["regime_code"])
    if candidate_label != expected_code:
        raise ValueError(f"Regime ID/code mismatch for {requested_date}: {regime_id}/{candidate_label}")

    with feature_store.load_tensors() as tensors:
        tensor_dates = tuple(_date_key(value) for value in tensors["date"].values)
        if tensor_dates.count(requested_date) != 1:
            raise ValueError(f"Date {requested_date} is missing or duplicated in tensor coordinates")
        sample_index = tensor_dates.index(requested_date)
        channel_names = tuple(str(value) for value in tensors["channel"].values)
        synoptic_names = tuple(str(value) for value in tensors["synoptic_var"].values)
        features, feature_names = _sample_predictors(
            tensors,
            [sample_index],
            channel_names,
            synoptic_names,
        )
    return features, feature_names, candidate_label


def predict_regime_for_date(
    date: str,
    artifact_dir: str | Path = "models/regime_classifier/stage1_lightgbm_5class_candidate_a_v1",
    store: FeatureStore | str | Path = "data/processed/feature_store",
) -> Stage1Prediction:
    """Load the saved model and predict one feature-store date."""
    features, feature_names, candidate_label = load_stage1_features_for_date(date, store)
    destination = Path(artifact_dir)
    model_path = destination / "model.txt"
    metadata_path = destination / "metadata.json"
    feature_names_path = destination / "feature_names.json"
    if not model_path.is_file() or not metadata_path.is_file() or not feature_names_path.is_file():
        raise FileNotFoundError(f"Incomplete Stage 1 model artifact at {destination}")

    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored_feature_names = tuple(json.loads(feature_names_path.read_text(encoding="utf-8")))
    if stored_feature_names != feature_names:
        raise ValueError("Manual tester features do not match the features used to train the saved model")
    expected_mapping = {str(class_id): code for class_id, code in CLASS_ID_TO_REGIME_CODE.items()}
    if artifact_metadata.get("class_mapping") != expected_mapping:
        raise ValueError("Saved model class mapping does not match the Stage 1 five-class mapping")

    model = lgb.Booster(model_file=str(model_path))
    if model.num_model_per_iteration() != len(ML_REGIME_CODES):
        raise ValueError("Saved model does not contain exactly five trained classes")
    probabilities_array = np.asarray(model.predict(features), dtype=np.float64).reshape(-1)
    if probabilities_array.shape != (len(ML_REGIME_CODES),):
        raise ValueError(f"Unexpected model probability shape: {probabilities_array.shape}")
    if not np.all(np.isfinite(probabilities_array)) or not np.isclose(
        probabilities_array.sum(), 1.0, atol=1e-6
    ):
        raise ValueError("Model probabilities are invalid or do not sum to one")

    class_id = int(np.argmax(probabilities_array))
    predicted_regime = CLASS_ID_TO_REGIME_CODE[class_id]
    probabilities = {
        CLASS_ID_TO_REGIME_CODE[index]: float(probabilities_array[index])
        for index in ML_CLASS_IDS
    }
    return Stage1Prediction(
        date=_date_key(date),
        candidate_a_label=candidate_label,
        predicted_regime=predicted_regime,
        confidence=float(probabilities_array[class_id]),
        probabilities=probabilities,
        matches_candidate_a=predicted_regime == candidate_label,
    )


def balanced_training_weights(labels: Sequence[int]) -> Tuple[np.ndarray, Dict[int, float]]:
    """Calculate inverse-frequency sample weights from training labels only."""
    y = np.asarray(labels, dtype=np.int32)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Training labels must be a non-empty one-dimensional array")
    if not set(np.unique(y)).issubset(ML_CLASS_IDS):
        raise ValueError("Training labels include a class outside the five-class ML set")
    counts = np.bincount(y, minlength=len(ML_REGIME_CODES))
    if np.any(counts == 0):
        raise ValueError("Every supported ML class must be represented in the training split")
    class_weights = {
        class_id: float(len(y) / (len(ML_REGIME_CODES) * counts[class_id]))
        for class_id in ML_CLASS_IDS
    }
    sample_weights = np.asarray([class_weights[int(label)] for label in y], dtype=np.float32)
    return sample_weights, class_weights


def lightgbm_parameters(random_seed: int) -> Dict[str, Any]:
    """Return the deterministic multiclass baseline configuration."""
    return {
        "objective": "multiclass",
        "num_class": len(ML_REGIME_CODES),
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 4,
        "min_data_in_leaf": 8,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbosity": -1,
        "seed": int(random_seed),
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
    }


def evaluate_predictions(y_true: Sequence[int], probabilities: np.ndarray) -> Dict[str, Any]:
    """Calculate aggregate and fixed-order per-class multiclass metrics."""
    labels = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (labels.size, len(ML_CLASS_IDS)):
        raise ValueError(f"Expected probability shape {(labels.size, 5)}, got {probs.shape}")
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty split")
    predictions = np.argmax(probs, axis=1).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=ML_CLASS_IDS, zero_division=0
    )
    per_class = {
        CLASS_ID_TO_REGIME_CODE[class_id]: {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in ML_CLASS_IDS
    }
    return {
        "sample_count": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=ML_CLASS_IDS).tolist(),
        "class_order": [CLASS_ID_TO_REGIME_CODE[class_id] for class_id in ML_CLASS_IDS],
        "per_class": per_class,
    }


def train_stage1_model(
    partitions: Mapping[str, Stage1Dataset],
    artifact_dir: str | Path = "models/regime_classifier/stage1_lightgbm_5class_candidate_a_v1",
    random_seed: int = 42,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> Stage1TrainingResult:
    """Fit, evaluate, and save the five-class Stage 1 LightGBM model."""
    required = {"train", "validation", "test"}
    if set(partitions) != required:
        raise ValueError(f"Expected exactly these partitions: {sorted(required)}")
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    if any(part.feature_names != train.feature_names for part in partitions.values()):
        raise ValueError("Feature names must match across train/validation/test")
    if validation.labels.size == 0 or test.labels.size == 0:
        raise ValueError("Validation and test partitions must contain supported ML classes")
    for split_name, partition in partitions.items():
        if not set(np.unique(partition.labels)).issubset(ML_CLASS_IDS):
            raise ValueError(f"{split_name} labels contain unsupported ML class IDs")

    sample_weights, class_weights = balanced_training_weights(train.labels)
    train_set = lgb.Dataset(
        train.features,
        label=train.labels,
        weight=sample_weights,
        feature_name=list(train.feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation.features,
        label=validation.labels,
        reference=train_set,
        feature_name=list(validation.feature_names),
        free_raw_data=False,
    )
    model = lgb.train(
        params=lightgbm_parameters(random_seed),
        train_set=train_set,
        num_boost_round=int(num_boost_round),
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    iteration = model.best_iteration or model.current_iteration()

    split_metrics: Dict[str, Any] = {}
    for split_name, partition in partitions.items():
        probabilities = np.asarray(model.predict(partition.features, num_iteration=iteration))
        split_metrics[split_name] = evaluate_predictions(partition.labels, probabilities)

    destination = Path(artifact_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing model artifact: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    model_path = destination / "model.txt"
    model.save_model(str(model_path), num_iteration=iteration)

    feature_importance = pd.DataFrame(
        {
            "feature": list(train.feature_names),
            "gain": model.feature_importance(importance_type="gain", iteration=iteration),
            "split": model.feature_importance(importance_type="split", iteration=iteration),
        }
    ).sort_values("gain", ascending=False)
    feature_importance.to_csv(destination / "feature_importance.csv", index=False)

    excluded_by_split = {
        name: int(partition.excluded_western_disturbance)
        for name, partition in partitions.items()
    }
    metadata = {
        "model": "LightGBM multiclass",
        "objective": "multiclass",
        "num_classes": len(ML_REGIME_CODES),
        "class_mapping": {str(class_id): code for class_id, code in CLASS_ID_TO_REGIME_CODE.items()},
        "western_disturbance_policy": WESTERN_DISTURBANCE_POLICY,
        "western_disturbance_excluded_by_split": excluded_by_split,
        "feature_representation": (
            "Area-weighted means for 15 predictor channels over seven fixed regions, "
            "concatenated with the 17 stored synoptic scalars. Target rainfall and "
            "valid-land mask are not read."
        ),
        "feature_count": len(train.feature_names),
        "feature_names_file": "feature_names.json",
        "random_seed": int(random_seed),
        "training_class_weights": {
            CLASS_ID_TO_REGIME_CODE[class_id]: weight for class_id, weight in class_weights.items()
        },
        "sample_counts_after_western_disturbance_exclusion": {
            name: int(partition.labels.size) for name, partition in partitions.items()
        },
        "temporal_ranges": {
            name: {
                "start": min(partition.dates),
                "end": max(partition.dates),
            }
            for name, partition in partitions.items()
        },
        "best_iteration": int(iteration),
        "training_seed_and_parameters": lightgbm_parameters(random_seed),
        "software_versions": {
            "python": platform.python_version(),
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "feature_names.json").write_text(
        json.dumps(list(train.feature_names), indent=2), encoding="utf-8"
    )
    (destination / "evaluation_metrics.json").write_text(
        json.dumps(split_metrics, indent=2), encoding="utf-8"
    )
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return Stage1TrainingResult(
        model=model,
        artifact_dir=destination,
        metrics=split_metrics,
        metadata=metadata,
    )


def train_stage1_from_store(
    feature_store_dir: str | Path = "data/processed/feature_store",
    artifact_dir: str | Path = "models/regime_classifier/stage1_lightgbm_5class_candidate_a_v1",
    random_seed: int = 42,
) -> Stage1TrainingResult:
    """Load the canonical feature store and train the Stage 1 baseline."""
    partitions = load_stage1_datasets(feature_store_dir)
    return train_stage1_model(partitions, artifact_dir=artifact_dir, random_seed=random_seed)
