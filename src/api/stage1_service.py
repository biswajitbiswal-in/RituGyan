"""Loading and prediction service for the trained operational Stage 1 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from src.features.operational_gfs import operational_feature_names
from src.models.stage1_regime.classifier import (
    CLASS_ID_TO_REGIME_CODE,
    ML_CLASS_IDS,
    ML_REGIME_CODES,
    load_stage1_features_for_date,
)


MODEL_NAME = "RituGyan Stage 1 Operational GFS Regime Classifier"
MODEL_VERSION = "stage1_lightgbm_5class_operational_gfs_v1"
SERVICE_NAME = "RituGyan Historical Stage 1 API"
API_VERSION = "1.0.0"
FEATURE_SOURCE = "NOAA GFS 00Z forecast fields via the operational GFS feature store"
EXPECTED_CLASS_MAPPING = {str(index): code for index, code in CLASS_ID_TO_REGIME_CODE.items()}


class ModelUnavailableError(RuntimeError):
    """Raised when the saved operational model cannot be loaded or validated."""


class OperationalFeatureError(RuntimeError):
    """Raised when operational feature data are missing or inconsistent."""


class OperationalStage1Predictor:
    """Lazy, cached model wrapper that serves exact saved feature vectors."""

    def __init__(self, artifact_dir: Path, feature_store_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.feature_store_dir = feature_store_dir
        self._model: lgb.Booster | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._metadata: dict[str, Any] | None = None
        self._store_snapshot: dict[str, Any] | None = None
        self._evaluation_metrics: dict[str, Any] | None = None

    def _load_model(self) -> lgb.Booster:
        if self._model is not None:
            return self._model
        try:
            metadata_path = self.artifact_dir / "metadata.json"
            names_path = self.artifact_dir / "feature_names.json"
            model_path = self.artifact_dir / "model.txt"
            if not all(path.is_file() for path in (metadata_path, names_path, model_path)):
                raise FileNotFoundError("model.txt, metadata.json, or feature_names.json is missing")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = tuple(json.loads(names_path.read_text(encoding="utf-8")))
            if metadata.get("model_version") != MODEL_VERSION:
                raise ValueError("Unexpected operational model version")
            if metadata.get("class_mapping") != EXPECTED_CLASS_MAPPING:
                raise ValueError("Saved class mapping does not match the five-class Stage 1 mapping")
            if int(metadata.get("feature_count", -1)) != len(operational_feature_names()):
                raise ValueError("Saved feature count does not match the operational feature contract")
            if feature_names != operational_feature_names():
                raise ValueError("Saved model feature names do not match the operational feature order")
            model = lgb.Booster(model_file=str(model_path))
            if model.num_model_per_iteration() != len(ML_REGIME_CODES):
                raise ValueError("Saved LightGBM model is not a five-class classifier")
            if model.num_feature() != len(feature_names):
                raise ValueError("Saved LightGBM feature count differs from its manifest")
        except Exception as exc:
            raise ModelUnavailableError(f"Operational Stage 1 model is unavailable: {exc}") from exc
        self._metadata = metadata
        self._feature_names = feature_names
        self._model = model
        return model

    def health(self) -> dict[str, Any]:
        store_available = True
        try:
            snapshot = self._load_store_snapshot()
        except OperationalFeatureError:
            store_available = False
            snapshot = None
        model_available = True
        try:
            self._load_model()
        except ModelUnavailableError:
            model_available = False
        return {
            "service_status": "ok" if store_available and model_available else "degraded",
            "service_name": SERVICE_NAME,
            "api_version": API_VERSION,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "feature_count": (
                len(self._feature_names)
                if self._feature_names is not None
                else int((snapshot or {}).get("config", {}).get("feature_count", len(operational_feature_names())))
            ),
            "feature_source": FEATURE_SOURCE,
            "supported_date_range": (
                {"start_date": snapshot["dates"][0], "end_date": snapshot["dates"][-1]}
                if snapshot and snapshot["dates"]
                else None
            ),
            "available_sample_count": len(snapshot["dates"]) if snapshot else 0,
            "stage1_status": "available" if model_available and store_available else "unavailable",
            "stage2_status": "not_implemented",
            "operational_gfs_status": "available" if store_available else "unavailable",
        }

    def _load_store_snapshot(self) -> dict[str, Any]:
        if self._store_snapshot is not None:
            return self._store_snapshot
        config_path = self.feature_store_dir / "store_config.json"
        metadata_path = self.feature_store_dir / "metadata.parquet"
        tensors_path = self.feature_store_dir / "tensors.nc"
        names_path = self.feature_store_dir / "feature_names.json"
        if not all(path.is_file() for path in (config_path, metadata_path, tensors_path, names_path)):
            raise OperationalFeatureError("Operational feature-store files are missing")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("feature_version") != "stage1_operational_gfs_v1":
                raise ValueError("Unexpected operational feature-store version")
            if config.get("provenance", {}).get("era5_used_for_predictors") is not False:
                raise ValueError("Feature store does not certify GFS-only predictors")
            metadata = pd.read_parquet(metadata_path, columns=["date"])
            metadata_dates = set(pd.to_datetime(metadata["date"]).dt.strftime("%Y-%m-%d"))
            with xr.open_dataset(tensors_path) as tensors:
                tensor_dates = tuple(
                    pd.to_datetime(tensors["date"].values).strftime("%Y-%m-%d")
                )
                if not tensor_dates or len(tensor_dates) != len(set(tensor_dates)):
                    raise ValueError("Feature tensor dates are empty or duplicated")
                if set(tensor_dates) != metadata_dates:
                    raise ValueError("Feature tensor and metadata dates do not match")
                lats = np.asarray(tensors["lat"].values, dtype=np.float64)
                lons = np.asarray(tensors["lon"].values, dtype=np.float64)
                channels = [str(value) for value in tensors["channel"].values]
                grid = {
                    "latitude_count": int(len(lats)),
                    "longitude_count": int(len(lons)),
                    "latitude_min": float(lats.min()),
                    "latitude_max": float(lats.max()),
                    "longitude_min": float(lons.min()),
                    "longitude_max": float(lons.max()),
                    "resolution_degrees": float(np.median(np.diff(lats))),
                }
            dates = tuple(sorted(tensor_dates))
            metadata_frame = pd.to_datetime(metadata["date"])
            years = sorted(int(year) for year in metadata_frame.dt.year.unique())
            channel_names = json.loads(names_path.read_text(encoding="utf-8"))
            if len(channel_names) != int(config.get("feature_count", -1)):
                raise ValueError("Feature-name manifest count does not match store metadata")
        except Exception as exc:
            raise OperationalFeatureError(f"Operational feature store is invalid: {exc}") from exc
        self._store_snapshot = {
            "config": config,
            "dates": dates,
            "years": years,
            "spatial_grid": grid,
            "channels": channels,
            "channel_source_map": config.get("channel_source_map", {}),
        }
        return self._store_snapshot

    def available_dates(self) -> dict[str, Any]:
        snapshot = self._load_store_snapshot()
        dates = list(snapshot["dates"])
        return {
            "start_date": dates[0],
            "end_date": dates[-1],
            "total_dates": len(dates),
            "dates": dates,
        }

    def dataset_info(self) -> dict[str, Any]:
        snapshot = self._load_store_snapshot()
        config = snapshot["config"]
        return {
            "dataset_name": "RituGyan Operational GFS Stage 1 Feature Store",
            "source": FEATURE_SOURCE,
            "date_range": {
                "start_date": snapshot["dates"][0],
                "end_date": snapshot["dates"][-1],
            },
            "total_samples": len(snapshot["dates"]),
            "years_available": snapshot["years"],
            "feature_count": int(config["feature_count"]),
            "spatial_grid": snapshot["spatial_grid"],
            "predictor_channels": snapshot["channels"],
            "channel_source_map": snapshot["channel_source_map"],
            "operational_status": "historical_gfs_operational_predictors",
            "limitations": config.get("limitations", []),
        }

    def model_info(self) -> dict[str, Any]:
        self._load_model()
        if self._metadata is None:
            raise ModelUnavailableError("Operational model metadata is unavailable")
        if self._evaluation_metrics is None:
            try:
                metrics_path = self.artifact_dir / "evaluation_metrics.json"
                self._evaluation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ModelUnavailableError(f"Saved evaluation metrics are unavailable: {exc}") from exc
        metadata = self._metadata
        try:
            test_metrics = self._evaluation_metrics["test"]
            periods = metadata["temporal_ranges"]
            return {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "algorithm": metadata["model"],
                "number_of_classes": int(metadata["num_classes"]),
                "class_mapping": metadata["class_mapping"],
                "feature_count": int(metadata["feature_count"]),
                "train_period": periods["train"],
                "validation_period": periods["validation"],
                "test_period": periods["test"],
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_weighted_f1": float(test_metrics["weighted_f1"]),
            }
        except Exception as exc:
            raise ModelUnavailableError(f"Saved model information is incomplete: {exc}") from exc

    def predict(self, date: str) -> dict[str, Any]:
        self._load_store_snapshot()
        model = self._load_model()
        try:
            features, feature_names, _label = load_stage1_features_for_date(
                date,
                store=self.feature_store_dir,
            )
        except ValueError as exc:
            if "not present in the feature store" in str(exc):
                raise LookupError(str(exc)) from exc
            raise OperationalFeatureError(str(exc)) from exc
        except (FileNotFoundError, KeyError, OSError) as exc:
            raise OperationalFeatureError(f"Operational feature data unavailable: {exc}") from exc
        if self._feature_names is None or tuple(feature_names) != self._feature_names:
            raise OperationalFeatureError("Requested date features do not match the model feature manifest")
        try:
            probabilities_array = np.asarray(model.predict(features), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ModelUnavailableError(f"LightGBM prediction failed: {exc}") from exc
        if probabilities_array.shape != (len(ML_CLASS_IDS),):
            raise ModelUnavailableError("Model returned an invalid probability vector")
        if not np.all(np.isfinite(probabilities_array)) or not np.isclose(
            probabilities_array.sum(), 1.0, atol=1e-6
        ):
            raise ModelUnavailableError("Model returned invalid probabilities")
        class_id = int(np.argmax(probabilities_array))
        probabilities = {
            CLASS_ID_TO_REGIME_CODE[index]: float(probabilities_array[index])
            for index in ML_CLASS_IDS
        }
        return {
            "date": date,
            "predicted_regime": CLASS_ID_TO_REGIME_CODE[class_id],
            "confidence": float(probabilities_array[class_id]),
            "probabilities": probabilities,
            "model_version": MODEL_VERSION,
            "feature_source": FEATURE_SOURCE,
        }
