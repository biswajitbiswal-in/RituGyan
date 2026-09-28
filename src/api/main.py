"""FastAPI application for the RituGyan operational Stage 1 prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.stage1 import router as stage1_router
from src.api.stage1_service import OperationalStage1Predictor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models/regime_classifier/stage1_lightgbm_5class_operational_gfs_v1"
DEFAULT_STORE_DIR = REPO_ROOT / "data/processed/feature_store_operational_gfs"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def create_app(
    model_dir: Path = DEFAULT_MODEL_DIR,
    store_dir: Path = DEFAULT_STORE_DIR,
    cors_origins: Sequence[str] = DEFAULT_CORS_ORIGINS,
) -> FastAPI:
    application = FastAPI(
        title="RituGyan Stage 1 Operational API",
        description=(
            "Historical-date regime predictions from the saved GFS-only Stage 1 model. "
            "Stage 2 rainfall correction is not implemented."
        ),
        version="1.0.0",
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
    )
    application.state.predictor = OperationalStage1Predictor(model_dir, store_dir)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    application.include_router(stage1_router)
    return application


app = create_app()
