"""Synoptic index calculators, dynamic feature pipelines, and regime labelers for RituGyan."""

from src.features.batch_pipeline import (
    DataAvailabilityAudit,
    MultiYearDatasetBuilder,
)
from src.features.feature_store import (
    DatasetSplitSummary,
    FeatureStore,
    SYNOPTIC_FEATURE_NAMES,
)
from src.features.preprocessing import (
    CHANNEL_UNITS,
    FEATURE_CHANNELS,
    AlignedDaySample,
    build_aligned_day_sample,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
    load_imd_target_day,
)
from src.features.regimes import (
    REGIME_CATALOG,
    RegimeClassificationResult,
    RegimeDefinition,
    SynopticRegimeClassifier,
)
from src.features.synoptic import (
    CLIMATOLOGICAL_TROUGH_LAT_NORMAL,
    IndexCategory,
    MissingDependencyError,
    SynopticFeatureSet,
    SynopticIndexResult,
    compute_cross_equatorial_flow_index,
    compute_llj_strength_index,
    compute_moisture_flux_convergence,
    compute_monsoon_depression_index,
    compute_monsoon_trough_index,
    compute_orographic_interception_index,
    compute_relative_vorticity_2d,
    compute_western_disturbance_index,
    extract_all_synoptic_features,
)

__all__ = [
    # Preprocessing
    "FEATURE_CHANNELS",
    "CHANNEL_UNITS",
    "AlignedDaySample",
    "build_aligned_day_sample",
    "load_imd_target_day",
    "load_era5_in_window_predictors",
    "load_gfs_forecast_predictors",
    # Synoptic Indices
    "IndexCategory",
    "MissingDependencyError",
    "SynopticIndexResult",
    "SynopticFeatureSet",
    "CLIMATOLOGICAL_TROUGH_LAT_NORMAL",
    "compute_monsoon_trough_index",
    "compute_llj_strength_index",
    "compute_cross_equatorial_flow_index",
    "compute_moisture_flux_convergence",
    "compute_relative_vorticity_2d",
    "compute_monsoon_depression_index",
    "compute_orographic_interception_index",
    "compute_western_disturbance_index",
    "extract_all_synoptic_features",
    # Regimes
    "REGIME_CATALOG",
    "RegimeDefinition",
    "RegimeClassificationResult",
    "SynopticRegimeClassifier",
    # Feature Store & Multi-Year Ingestion
    "FeatureStore",
    "DatasetSplitSummary",
    "SYNOPTIC_FEATURE_NAMES",
    "DataAvailabilityAudit",
    "MultiYearDatasetBuilder",
]
