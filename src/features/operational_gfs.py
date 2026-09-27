"""Versioned GFS-only predictor construction for Stage 1 operationalization.

This module is deliberately separate from the canonical multi-source pipeline.
It reads GFS forecasts and copies historical targets/labels only when building
an evaluation store; neither is used to construct predictors or synoptic fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd
import xarray as xr

from src.features.feature_store import SYNOPTIC_FEATURE_NAMES
from src.features.preprocessing import (
    CHANNEL_UNITS,
    FEATURE_CHANNELS,
    load_gfs_forecast_predictors,
)
from src.features.synoptic import SynopticFeatureSet, extract_all_synoptic_features
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS


OPERATIONAL_FEATURE_VERSION = "stage1_operational_gfs_v1"
OPERATIONAL_STORE_DEFAULT = Path("data/processed/feature_store_operational_gfs")
REQUIRED_GFS_VARS = ("prmsl", "2t", "2d", "10u", "10v", "pwat", "tp")
JJAS_YEARS = (2021, 2022, 2023, 2024)

# The 15 positions are retained so a future model can compare representations
# without an accidental region/channel reordering. Former ERA5 slots now carry
# the corresponding GFS quantity and are documented in CHANNEL_SOURCE_MAP.
OPERATIONAL_CHANNELS: Tuple[str, ...] = tuple(FEATURE_CHANNELS)
CHANNEL_SOURCE_MAP: Mapping[str, str] = {
    "gfs_tp_24h": "gfs_tp_24h",
    "gfs_prmsl_mean": "gfs_prmsl_mean",
    "gfs_2t_mean": "gfs_2t_mean",
    "gfs_2d_mean": "gfs_2d_mean",
    "gfs_10u_mean": "gfs_10u_mean",
    "gfs_10v_mean": "gfs_10v_mean",
    "gfs_pwat_mean": "gfs_pwat_mean",
    "gfs_wind_speed": "gfs_wind_speed",
    "era5_msl_mean": "gfs_prmsl_mean",
    "era5_t2m_mean": "gfs_2t_mean",
    "era5_d2m_mean": "gfs_2d_mean",
    "era5_u10_mean": "gfs_10u_mean",
    "era5_v10_mean": "gfs_10v_mean",
    "era5_tcwv_mean": "gfs_pwat_mean",
    "era5_wind_speed": "gfs_wind_speed",
}


def build_operational_channel_tensor(gfs_features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Map the eight GFS fields into the stable 15-channel Stage 1 layout."""
    missing = sorted(set(CHANNEL_SOURCE_MAP.values()) - set(gfs_features))
    if missing:
        raise KeyError(f"Missing GFS feature fields: {missing}")
    planes = [np.asarray(gfs_features[CHANNEL_SOURCE_MAP[name]], dtype=np.float32) for name in OPERATIONAL_CHANNELS]
    expected_shape = (len(CANONICAL_LATS), len(CANONICAL_LONS))
    if any(plane.shape != expected_shape for plane in planes):
        raise ValueError(f"GFS fields must have shape {expected_shape}")
    tensor = np.stack(planes, axis=0).astype(np.float32)
    if not np.all(np.isfinite(tensor)):
        raise ValueError("Operational GFS predictor tensor contains NaN or Inf")
    return tensor


def build_operational_synoptic_features(
    gfs_features: Mapping[str, np.ndarray],
    target_date: str,
) -> SynopticFeatureSet:
    """Calculate the existing 17 synoptic concepts exclusively from GFS fields."""
    return extract_all_synoptic_features(
        mslp_hpa=np.asarray(gfs_features["gfs_prmsl_mean"], dtype=np.float32),
        pwat_kg_m2=np.asarray(gfs_features["gfs_pwat_mean"], dtype=np.float32),
        u10_ms=np.asarray(gfs_features["gfs_10u_mean"], dtype=np.float32),
        v10_ms=np.asarray(gfs_features["gfs_10v_mean"], dtype=np.float32),
        target_date=target_date,
        lats=CANONICAL_LATS,
        lons=CANONICAL_LONS,
    )


def synoptic_vector(synoptic: SynopticFeatureSet) -> np.ndarray:
    """Return the deterministic 17-value vector in canonical order."""
    values = [
        synoptic.trough_mean_latitude,
        synoptic.trough_latitude_departure,
        synoptic.trough_min_pressure_hpa,
        synoptic.trough_pressure_gradient_hpa,
        synoptic.low_latitude_inflow_speed_ms,
        synoptic.low_latitude_inflow_zonal_ms,
        synoptic.low_latitude_inflow_meridional_ms,
        synoptic.low_latitude_kinetic_energy,
        synoptic.mfc_domain_mean_mm_day,
        synoptic.mfc_central_india_mean_mm_day,
        synoptic.depression_mslp_anomaly_hpa,
        synoptic.depression_max_vorticity_s1,
        synoptic.orographic_ghats_zonal_flux,
        synoptic.nw_india_min_mslp_hpa,
        synoptic.nw_india_mean_pwat_mm,
        synoptic.domain_mean_pwat_mm,
        synoptic.domain_mean_mslp_hpa,
    ]
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(SYNOPTIC_FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError("Operational synoptic vector is malformed or non-finite")
    return vector


def operational_feature_names() -> Tuple[str, ...]:
    """Return the deterministic 122-name operational feature list."""
    regions = (
        "domain",
        "depression_track",
        "central_india",
        "arabian_sea_gateway",
        "western_ghats",
        "northwest_india",
        "himalayan_foothills",
    )
    return tuple(
        [
            f"spatial__{channel}__{region}__area_weighted_mean"
            for region in regions
            for channel in OPERATIONAL_CHANNELS
        ]
        + [f"synoptic__{name}" for name in SYNOPTIC_FEATURE_NAMES]
    )


def build_operational_day_features(gfs_dir: str | Path, target_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build one GFS-only 15-channel tensor and 17-value synoptic vector."""
    compact_date = target_date.replace("-", "")
    gfs_features = load_gfs_forecast_predictors(gfs_dir, date_str=compact_date, cycle_str="00")
    return build_operational_channel_tensor(gfs_features), synoptic_vector(
        build_operational_synoptic_features(gfs_features, target_date)
    )


def _build_operational_day_task(arguments: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    gfs_dir, target_date = arguments
    return build_operational_day_features(gfs_dir, target_date)


def _store_dates(metadata_path: Path) -> Tuple[str, ...]:
    metadata = pd.read_parquet(metadata_path, columns=["date", "year"])
    dates = tuple(sorted(pd.to_datetime(metadata["date"]).dt.strftime("%Y-%m-%d")))
    expected = tuple(
        sorted(
            date.strftime("%Y-%m-%d")
            for year in JJAS_YEARS
            for date in pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")
        )
    )
    if dates != expected:
        raise ValueError(f"Canonical store does not contain the expected 488 JJAS dates: {len(dates)}")
    return dates


def _synoptic_metadata(vector: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(SYNOPTIC_FEATURE_NAMES, vector)}


def build_operational_feature_store(
    canonical_store_dir: str | Path = "data/processed/feature_store",
    gfs_root_dir: str | Path = "data/raw/gfs_historical",
    output_dir: str | Path = OPERATIONAL_STORE_DEFAULT,
    dates: Sequence[str] | None = None,
    workers: int = 1,
    reuse_canonical_gfs: bool = False,
) -> Path:
    """Build an isolated operational store while preserving canonical labels.

    The canonical target, mask, and Candidate A labels are copied read-only for
    retrospective validation. They never enter the predictor or synoptic math.
    Existing output is rejected to prevent accidental overwrite.
    """
    canonical_dir = Path(canonical_store_dir)
    gfs_root = Path(gfs_root_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite operational store: {destination}")

    canonical_metadata_path = canonical_dir / "metadata.parquet"
    canonical_tensors_path = canonical_dir / "tensors.nc"
    if not canonical_metadata_path.is_file() or not canonical_tensors_path.is_file():
        raise FileNotFoundError("Canonical feature store metadata.parquet and tensors.nc are required")

    all_dates = _store_dates(canonical_metadata_path)
    selected_dates = tuple(sorted(dates or all_dates))
    if not set(selected_dates).issubset(set(all_dates)):
        raise ValueError("Requested operational dates are not in the canonical JJAS date set")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    canonical_metadata = pd.read_parquet(canonical_metadata_path)
    canonical_metadata["date"] = pd.to_datetime(canonical_metadata["date"]).dt.strftime("%Y-%m-%d")
    metadata_by_date = canonical_metadata.set_index("date")

    with xr.open_dataset(canonical_tensors_path) as canonical_tensors:
        canonical_dates = tuple(pd.to_datetime(canonical_tensors["date"].values).strftime("%Y-%m-%d"))
        date_indices = {date: index for index, date in enumerate(canonical_dates)}
        feature_rows: List[np.ndarray] = []
        synoptic_rows: List[np.ndarray] = []
        target_rows: List[np.ndarray] = []
        mask_rows: List[np.ndarray] = []
        day_tasks = [(str(gfs_root / target_date[:4]), target_date) for target_date in selected_dates]
        if workers > 1 and not reuse_canonical_gfs:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                day_results = executor.map(_build_operational_day_task, day_tasks)
                for target_date, (feature_row, synoptic_row) in zip(selected_dates, day_results):
                    feature_rows.append(feature_row)
                    synoptic_rows.append(synoptic_row)
                    source_index = date_indices[target_date]
                    target_rows.append(np.asarray(canonical_tensors["target"].isel(date=source_index).values, dtype=np.float32))
                    mask_rows.append(np.asarray(canonical_tensors["valid_mask"].isel(date=source_index).values, dtype=bool))
        elif not reuse_canonical_gfs:
            for target_date, (feature_row, synoptic_row) in zip(
                selected_dates, map(_build_operational_day_task, day_tasks)
            ):
                feature_rows.append(feature_row)
                synoptic_rows.append(synoptic_row)
                source_index = date_indices[target_date]
                target_rows.append(np.asarray(canonical_tensors["target"].isel(date=source_index).values, dtype=np.float32))
                mask_rows.append(np.asarray(canonical_tensors["valid_mask"].isel(date=source_index).values, dtype=bool))
        else:
            gfs_indices = {name: FEATURE_CHANNELS.index(name) for name in CHANNEL_SOURCE_MAP.values()}
            for target_date in selected_dates:
                source_index = date_indices[target_date]
                gfs_features = {
                    name: np.asarray(
                        canonical_tensors["features"].isel(
                            date=source_index, channel=index
                        ).values,
                        dtype=np.float32,
                    )
                    for name, index in gfs_indices.items()
                }
                feature_row = build_operational_channel_tensor(gfs_features)
                synoptic = build_operational_synoptic_features(gfs_features, target_date)
                feature_rows.append(feature_row)
                synoptic_rows.append(synoptic_vector(synoptic))
                target_rows.append(np.asarray(canonical_tensors["target"].isel(date=source_index).values, dtype=np.float32))
                mask_rows.append(np.asarray(canonical_tensors["valid_mask"].isel(date=source_index).values, dtype=bool))

    features = np.stack(feature_rows, axis=0).astype(np.float32)
    synoptic_vectors = np.stack(synoptic_rows, axis=0).astype(np.float32)
    targets = np.stack(target_rows, axis=0).astype(np.float32)
    masks = np.stack(mask_rows, axis=0).astype(bool)
    destination.mkdir(parents=True, exist_ok=False)

    dates_array = np.asarray(selected_dates, dtype="datetime64[ns]")
    dataset = xr.Dataset(
        data_vars={
            "features": (("date", "channel", "lat", "lon"), features),
            "target": (("date", "lat", "lon"), targets),
            "valid_mask": (("date", "lat", "lon"), masks),
            "synoptic_vector": (("date", "synoptic_var"), synoptic_vectors),
            "regime_id": (("date",), np.asarray([int(metadata_by_date.loc[d, "regime_id"]) for d in selected_dates], dtype=np.int32)),
            "regime_confidence": (("date",), np.asarray([float(metadata_by_date.loc[d, "regime_confidence"]) for d in selected_dates], dtype=np.float32)),
        },
        coords={
            "date": dates_array,
            "channel": list(OPERATIONAL_CHANNELS),
            "lat": CANONICAL_LATS,
            "lon": CANONICAL_LONS,
            "synoptic_var": SYNOPTIC_FEATURE_NAMES,
        },
        attrs={
            "feature_version": OPERATIONAL_FEATURE_VERSION,
            "predictor_source": "NOAA GFS 0.25 degree 00Z f003-f027 only",
            "historical_build_source_mode": "canonical serialized GFS planes" if reuse_canonical_gfs else "raw GFS GRIB2",
            "era5_predictors_used": "false",
            "candidate_a_labels_modified": "false",
            "gfs_tp_24h_definition": "A(0->27)-A(0->3), 03Z-03Z forecast window",
        },
    )
    dataset.to_netcdf(destination / "tensors.nc", engine="netcdf4")
    dataset.close()

    output_metadata = canonical_metadata[canonical_metadata["date"].isin(selected_dates)].copy()
    output_metadata["date"] = pd.Categorical(output_metadata["date"], categories=list(selected_dates), ordered=True)
    output_metadata = output_metadata.sort_values("date")
    output_metadata["date"] = output_metadata["date"].astype(str)
    output_metadata.to_parquet(destination / "metadata.parquet", index=False)

    feature_names = list(operational_feature_names())
    (destination / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    config = {
        "feature_version": OPERATIONAL_FEATURE_VERSION,
        "feature_count": len(feature_names),
        "sample_count": len(selected_dates),
        "date_range": {"start": selected_dates[0], "end": selected_dates[-1]},
        "split_definitions": {"train": [2021, 2022], "validation": [2023], "test": [2024]},
        "channels": list(OPERATIONAL_CHANNELS),
        "channel_units": CHANNEL_UNITS,
        "channel_source_map": dict(CHANNEL_SOURCE_MAP),
        "required_gfs_variables": list(REQUIRED_GFS_VARS),
        "synoptic_variables": list(SYNOPTIC_FEATURE_NAMES),
        "spatial_shape": [len(CANONICAL_LATS), len(CANONICAL_LONS)],
        "spatial_regions": ["domain", "depression_track", "central_india", "arabian_sea_gateway", "western_ghats", "northwest_india", "himalayan_foothills"],
        "provenance": {
            "predictors": "GFS forecast GRIB2 files under data/raw/gfs_historical/{year}",
            "labels": "Copied unchanged from canonical feature store metadata.parquet",
            "targets": "Copied unchanged from canonical feature store for retrospective validation only",
            "era5_used_for_predictors": False,
            "candidate_a_regenerated": False,
        },
        "limitations": [
            "Candidate A labels remain ERA5-conditioned research proxy labels.",
            "Depression and MFC proxy thresholds require GFS-specific recalibration.",
            "gfs_tp_24h is retained pending ablation and validation.",
        ],
    }
    (destination / "store_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return destination
