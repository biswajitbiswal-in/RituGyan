"""Stage 1 operational regime prediction endpoints."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, field_validator

from src.api.stage1_service import (
    MODEL_NAME,
    MODEL_VERSION,
    ModelUnavailableError,
    OperationalFeatureError,
    OperationalStage1Predictor,
)
from src.models.stage1_regime.classifier import ML_REGIME_CODES


router = APIRouter(prefix="/api/v1", tags=["Stage 1"])


class SupportedDateRange(BaseModel):
    start_date: str
    end_date: str


class HealthResponse(BaseModel):
    status: str
    service_name: str
    api_version: str
    service_status: str
    model_name: str
    model_version: str
    feature_count: int
    feature_source: str
    supported_date_range: SupportedDateRange | None
    available_sample_count: int
    stage1_status: str
    stage2_status: Literal["not_implemented"]
    operational_gfs_status: str


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date

    @field_validator("date", mode="before")
    @classmethod
    def require_iso_date(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("date must use YYYY-MM-DD format") from exc
        if parsed.isoformat() != value:
            raise ValueError("date must use YYYY-MM-DD format")
        return value


class PredictionResponse(BaseModel):
    date: str
    predicted_regime: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    feature_source: str


class AvailableDatesResponse(BaseModel):
    start_date: str
    end_date: str
    total_dates: int
    dates: List[str]


class SpatialGridResponse(BaseModel):
    latitude_count: int
    longitude_count: int
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float
    resolution_degrees: float


class DatasetResponse(BaseModel):
    dataset_name: str
    source: str
    date_range: SupportedDateRange
    total_samples: int
    years_available: List[int]
    feature_count: int
    spatial_grid: SpatialGridResponse
    predictor_channels: List[str]
    channel_source_map: Dict[str, str]
    operational_status: str
    limitations: List[str]


class PeriodResponse(BaseModel):
    start: str
    end: str


class ModelResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    number_of_classes: int
    class_mapping: Dict[str, str]
    feature_count: int
    train_period: PeriodResponse
    validation_period: PeriodResponse
    test_period: PeriodResponse
    test_accuracy: float
    test_balanced_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float


@router.get("/health", response_model=HealthResponse, summary="Check API and operational model health")
def health(request: Request) -> HealthResponse:
    predictor: OperationalStage1Predictor = request.app.state.predictor
    values = predictor.health()
    return HealthResponse(
        status=values["service_status"],
        **values,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict the weather regime for a stored historical date",
    description=(
        f"Returns one of the five trained regimes: {', '.join(ML_REGIME_CODES)}. "
        "Western Disturbance is not an ML-trained class. Stage 2 rainfall correction is not implemented."
    ),
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: OperationalStage1Predictor = request.app.state.predictor
    try:
        result = predictor.predict(payload.date.isoformat())
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OperationalFeatureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.get(
    "/dates",
    response_model=AvailableDatesResponse,
    summary="List the historical dates present in the operational feature store",
)
def available_dates(request: Request) -> AvailableDatesResponse:
    predictor: OperationalStage1Predictor = request.app.state.predictor
    try:
        return AvailableDatesResponse(**predictor.available_dates())
    except OperationalFeatureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/dataset",
    response_model=DatasetResponse,
    summary="Describe the operational historical GFS feature dataset",
)
def dataset_info(request: Request) -> DatasetResponse:
    predictor: OperationalStage1Predictor = request.app.state.predictor
    try:
        return DatasetResponse(**predictor.dataset_info())
    except OperationalFeatureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/model",
    response_model=ModelResponse,
    summary="Describe the saved operational Stage 1 model and test metrics",
)
def model_info(request: Request) -> ModelResponse:
    predictor: OperationalStage1Predictor = request.app.state.predictor
    try:
        return ModelResponse(**predictor.model_info())
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
