"""API tests for the saved operational GFS-only Stage 1 model."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.api.stage1_service import MODEL_VERSION
from src.models.stage1_regime.classifier import ML_REGIME_CODES


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models/regime_classifier/stage1_lightgbm_5class_operational_gfs_v1"
STORE_DIR = REPO_ROOT / "data/processed/feature_store_operational_gfs"


def _client(model_dir: Path = MODEL_DIR, store_dir: Path = STORE_DIR) -> TestClient:
    return TestClient(create_app(model_dir=model_dir, store_dir=store_dir))


def test_health_reports_operational_model_status():
    response = _client().get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service_name"] == "RituGyan Historical Stage 1 API"
    assert payload["api_version"] == "1.0.0"
    assert payload["service_status"] == "ok"
    assert payload["model_name"] == "RituGyan Stage 1 Operational GFS Regime Classifier"
    assert payload["model_version"] == MODEL_VERSION
    assert payload["feature_count"] == 122
    assert payload["feature_source"].startswith("NOAA GFS")
    assert payload["supported_date_range"] == {
        "start_date": "2021-06-01",
        "end_date": "2024-09-30",
    }
    assert payload["available_sample_count"] == 488
    assert payload["stage1_status"] == "available"
    assert payload["stage2_status"] == "not_implemented"
    assert payload["operational_gfs_status"] == "available"


def test_valid_historical_prediction_returns_five_class_probabilities():
    response = _client().post("/api/v1/predict", json={"date": "2024-09-15"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["date"] == "2024-09-15"
    assert payload["predicted_regime"] in ML_REGIME_CODES
    assert payload["model_version"] == MODEL_VERSION
    assert payload["feature_source"].startswith("NOAA GFS")
    assert list(payload["probabilities"]) == list(ML_REGIME_CODES)
    assert all(0.0 <= probability <= 1.0 for probability in payload["probabilities"].values())
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert payload["confidence"] == max(payload["probabilities"].values())


def test_invalid_date_is_rejected():
    response = _client().post("/api/v1/predict", json={"date": "2024-02-30"})

    assert response.status_code == 422


def test_non_iso_date_is_rejected():
    response = _client().post("/api/v1/predict", json={"date": "2024-9-15"})

    assert response.status_code == 422


def test_date_outside_operational_store_returns_not_found():
    response = _client().post("/api/v1/predict", json={"date": "2025-09-15"})

    assert response.status_code == 404


def test_malformed_prediction_request_is_rejected():
    response = _client().post("/api/v1/predict", json={"date": "2024-09-15", "rainfall": 10})

    assert response.status_code == 422


def test_dates_endpoint_returns_real_feature_store_dates():
    response = _client().get("/api/v1/dates")

    assert response.status_code == 200
    payload = response.json()
    assert payload["start_date"] == "2021-06-01"
    assert payload["end_date"] == "2024-09-30"
    assert payload["total_dates"] == 488
    assert len(payload["dates"]) == 488
    assert len(set(payload["dates"])) == 488
    assert payload["dates"][0] == payload["start_date"]
    assert payload["dates"][-1] == payload["end_date"]
    assert "2024-09-15" in payload["dates"]


def test_dataset_endpoint_reports_store_metadata():
    response = _client().get("/api/v1/dataset")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_samples"] == 488
    assert payload["years_available"] == [2021, 2022, 2023, 2024]
    assert payload["feature_count"] == 122
    assert payload["spatial_grid"]["latitude_count"] == 127
    assert payload["spatial_grid"]["longitude_count"] == 121
    assert payload["spatial_grid"]["resolution_degrees"] == pytest.approx(0.25)
    assert payload["operational_status"] == "historical_gfs_operational_predictors"
    assert payload["channel_source_map"]["era5_msl_mean"] == "gfs_prmsl_mean"


def test_model_endpoint_reports_saved_mapping_periods_and_metrics():
    response = _client().get("/api/v1/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == MODEL_VERSION
    assert payload["number_of_classes"] == 5
    assert payload["class_mapping"] == {
        "0": "ACTIVE_MONSOON",
        "1": "BREAK_MONSOON",
        "2": "MONSOON_DEPRESSION",
        "3": "OROGRAPHIC_MONSOON",
        "4": "COASTAL_REGIME",
    }
    assert payload["feature_count"] == 122
    assert payload["train_period"] == {"start": "2021-06-01", "end": "2022-09-30"}
    assert payload["validation_period"] == {"start": "2023-06-01", "end": "2023-09-30"}
    assert payload["test_period"] == {"start": "2024-06-01", "end": "2024-09-30"}
    assert payload["test_accuracy"] == pytest.approx(0.8114754098360656)
    assert payload["test_balanced_accuracy"] == pytest.approx(0.6611230981093994)
    assert payload["test_macro_f1"] == pytest.approx(0.6491419093036344)
    assert payload["test_weighted_f1"] == pytest.approx(0.810985903319428)


def test_missing_model_artifact_returns_service_unavailable(tmp_path):
    client = _client(model_dir=tmp_path)

    health_response = client.get("/api/v1/health")
    predict_response = client.post("/api/v1/predict", json={"date": "2024-09-15"})

    assert health_response.status_code == 200
    assert health_response.json()["service_status"] == "degraded"
    assert health_response.json()["stage1_status"] == "unavailable"
    assert health_response.json()["operational_gfs_status"] == "available"
    assert predict_response.status_code == 503
    assert client.get("/api/v1/model").status_code == 503


def test_missing_feature_store_returns_service_unavailable(tmp_path):
    response = _client(store_dir=tmp_path).post("/api/v1/predict", json={"date": "2024-09-15"})

    assert response.status_code == 503

