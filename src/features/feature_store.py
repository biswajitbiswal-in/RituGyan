"""Feature Store serialization, indexing, and temporal split management for RituGyan.

Supports:
1. Multi-source spatial tensor serialization:
   - Feature tensors X: (N, 15, 127, 121) float32
   - Ground truth target y: (N, 127, 121) float32 (NaNs preserved for ocean)
   - Valid land masks: (N, 127, 121) bool
   - Synoptic feature matrices: (N, K) float32
   - Regime classification labels: (N,) int32 / categorical
2. Tabular metadata index in Apache Parquet format (via pyarrow/pandas).
3. Temporal train / validation / test partitioning (preventing spatial/temporal leakage).
4. Chunked, low-memory access without loading all multi-year data into RAM simultaneously.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import xarray as xr

# ── Pre-load pyarrow DLL at import time, before tensor data fills RAM ─────────
# On Windows, pyarrow._parquet.pyd fails to load once address space is
# fragmented by hundreds of MB of NumPy arrays.  Importing here guarantees
# the DLL is resident long before FeatureStore.save() is invoked.
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass  # pandas will fall back to fastparquet if available


from src.features.preprocessing import (
    CHANNEL_UNITS,
    FEATURE_CHANNELS,
    AlignedDaySample,
)
from src.features.regimes import REGIME_CATALOG, RegimeClassificationResult
from src.features.synoptic import SynopticFeatureSet
from src.ingestion.spatial import (
    CANONICAL_LATS,
    CANONICAL_LONS,
    CANONICAL_NUM_LATS,
    CANONICAL_NUM_LONS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYNOPTIC_FEATURE_NAMES: List[str] = [
    "trough_mean_latitude",
    "trough_latitude_departure",
    "trough_min_pressure_hpa",
    "trough_pressure_gradient_hpa",
    "low_latitude_inflow_speed_ms",
    "low_latitude_inflow_zonal_ms",
    "low_latitude_inflow_meridional_ms",
    "low_latitude_kinetic_energy",
    "mfc_domain_mean_mm_day",
    "mfc_central_india_mean_mm_day",
    "depression_mslp_anomaly_hpa",
    "depression_max_vorticity_s1",
    "orographic_ghats_zonal_flux",
    "nw_india_min_mslp_hpa",
    "nw_india_mean_pwat_mm",
    "domain_mean_pwat_mm",
    "domain_mean_mslp_hpa",
]


@dataclass
class DatasetSplitSummary:
    """Summary metrics of a dataset partition."""
    split_name: str
    num_days: int
    date_range: Tuple[str, str]
    valid_spatial_samples: int
    regime_distribution: Dict[str, int]
    missing_land_fraction: float


class FeatureStore:
    """Serializes, reads, and partitions multi-source meteorological tensors and metadata."""

    def __init__(self, store_dir: Union[str, Path]) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.store_dir / "metadata.parquet"
        self.tensors_path = self.store_dir / "tensors.nc"
        self.config_path = self.store_dir / "store_config.json"

    def save(
        self,
        samples: List[AlignedDaySample],
        synoptic_features: List[SynopticFeatureSet],
        regime_results: List[RegimeClassificationResult],
    ) -> Path:
        """Save aligned day samples, synoptic vectors, and regime metadata to store.
        
        Args:
            samples: List of AlignedDaySample objects.
            synoptic_features: List of SynopticFeatureSet objects.
            regime_results: List of RegimeClassificationResult objects.
            
        Returns:
            Path to the saved store directory.
        """
        if not samples:
            raise ValueError("Cannot save empty sample list to FeatureStore")
        assert len(samples) == len(synoptic_features) == len(regime_results), (
            "Mismatch in length of samples, synoptic features, and regime results"
        )

        n_samples = len(samples)
        dates = [s.target_date for s in samples]

        # 1. Stack arrays
        x_stack = np.stack([s.features for s in samples], axis=0).astype(np.float32)       # (N, C, 127, 121)
        y_stack = np.stack([s.target for s in samples], axis=0).astype(np.float32)         # (N, 127, 121)
        mask_stack = np.stack([s.valid_mask for s in samples], axis=0).astype(bool)        # (N, 127, 121)

        # Synoptic vector stack (N, K)
        syn_rows = []
        for syn in synoptic_features:
            syn_vec = [
                syn.trough_mean_latitude,
                syn.trough_latitude_departure,
                syn.trough_min_pressure_hpa,
                syn.trough_pressure_gradient_hpa,
                syn.low_latitude_inflow_speed_ms,
                syn.low_latitude_inflow_zonal_ms,
                syn.low_latitude_inflow_meridional_ms,
                syn.low_latitude_kinetic_energy,
                syn.mfc_domain_mean_mm_day,
                syn.mfc_central_india_mean_mm_day,
                syn.depression_mslp_anomaly_hpa,
                syn.depression_max_vorticity_s1,
                syn.orographic_ghats_zonal_flux,
                syn.nw_india_min_mslp_hpa,
                syn.nw_india_mean_pwat_mm,
                syn.domain_mean_pwat_mm,
                syn.domain_mean_mslp_hpa,
            ]
            syn_rows.append(syn_vec)
        syn_stack = np.array(syn_rows, dtype=np.float32)

        regime_ids = np.array([r.regime_id for r in regime_results], dtype=np.int32)
        regime_confidences = np.array([r.confidence for r in regime_results], dtype=np.float32)
        regime_codes = [r.regime_code for r in regime_results]

        # 2. Construct Xarray Dataset for NetCDF4 serialization
        ds = xr.Dataset(
            data_vars={
                "features": (["date", "channel", "lat", "lon"], x_stack),
                "target": (["date", "lat", "lon"], y_stack),
                "valid_mask": (["date", "lat", "lon"], mask_stack),
                "synoptic_vector": (["date", "synoptic_var"], syn_stack),
                "regime_id": (["date"], regime_ids),
                "regime_confidence": (["date"], regime_confidences),
            },
            coords={
                "date": dates,
                "channel": FEATURE_CHANNELS,
                "lat": CANONICAL_LATS,
                "lon": CANONICAL_LONS,
                "synoptic_var": SYNOPTIC_FEATURE_NAMES,
            },
            attrs={
                "project": "RituGyan",
                "grid_resolution": 0.25,
                "spatial_shape": [CANONICAL_NUM_LATS, CANONICAL_NUM_LONS],
                "channels": FEATURE_CHANNELS,
                "channel_units": json.dumps(CHANNEL_UNITS),
                "num_samples": n_samples,
            },
        )

        # Save NetCDF4
        ds.to_netcdf(self.tensors_path, engine="netcdf4")
        ds.close()

        # 3. Build and save Parquet metadata index table
        meta_records = []
        for i, (s, syn, reg) in enumerate(zip(samples, synoptic_features, regime_results)):
            rec = {
                "date": s.target_date,
                "year": int(s.target_date.split("-")[0]),
                "month": int(s.target_date.split("-")[1]),
                "day": int(s.target_date.split("-")[2]),
                "regime_id": reg.regime_id,
                "regime_code": reg.regime_code,
                "regime_name": reg.regime_name,
                "regime_confidence": float(reg.confidence),
                "fallback_applied": bool(reg.fallback_applied),
                "valid_land_cells": int(s.metadata.get("valid_land_cells", s.valid_mask.sum())),
                "rainfall_max_mm": float(s.metadata.get("rainfall_max_mm", np.nanmax(s.target))),
                "rainfall_mean_mm": float(s.metadata.get("rainfall_mean_mm", np.nanmean(s.target))),
                "gfs_tp_max_mm": float(s.metadata.get("gfs_tp_max_mm", 0.0)),
                "gfs_tp_mean_mm": float(s.metadata.get("gfs_tp_mean_mm", 0.0)),
                "trough_latitude_departure": float(syn.trough_latitude_departure),
                "inflow_speed_ms": float(syn.low_latitude_inflow_speed_ms),
                "domain_pwat_mm": float(syn.domain_mean_pwat_mm),
                "mfc_central_india_mm_day": float(syn.mfc_central_india_mean_mm_day),
                "depression_mslp_anomaly": float(syn.depression_mslp_anomaly_hpa),
            }
            meta_records.append(rec)

        df_meta = pd.DataFrame(meta_records)
        df_meta.to_parquet(self.metadata_path, index=False)

        # 4. Save Store Config JSON
        config_data = {
            "num_samples": n_samples,
            "dates": dates,
            "channels": FEATURE_CHANNELS,
            "channel_units": CHANNEL_UNITS,
            "synoptic_variables": SYNOPTIC_FEATURE_NAMES,
            "spatial_shape": [CANONICAL_NUM_LATS, CANONICAL_NUM_LONS],
            "lat_bounds": [float(CANONICAL_LATS[0]), float(CANONICAL_LATS[-1])],
            "lon_bounds": [float(CANONICAL_LONS[0]), float(CANONICAL_LONS[-1])],
            "storage_format": "NetCDF4 (Tensors) + Apache Parquet (Metadata)",
        }
        with open(self.config_path, "w", encoding="utf-8") as fp:
            json.dump(config_data, fp, indent=2)

        logger.info(f"Feature store successfully saved at {self.store_dir} ({n_samples} days)")
        return self.store_dir

    def load_metadata(self) -> pd.DataFrame:
        """Load tabular metadata index from Parquet."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata index not found: {self.metadata_path}")
        return pd.read_parquet(self.metadata_path)

    def load_tensors(self) -> xr.Dataset:
        """Open lazy NetCDF4 dataset for tensor access."""
        if not self.tensors_path.exists():
            raise FileNotFoundError(f"Tensors dataset not found: {self.tensors_path}")
        return xr.open_dataset(self.tensors_path)

    def get_temporal_split(
        self,
        train_years: List[int] = [2021, 2022],
        val_years: List[int] = [2023],
        test_years: List[int] = [2024],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partition metadata into chronological Train / Validation / Test splits.

        Strict chronological ordering prevents any temporal data leakage:
          - Training years feed model fitting.
          - Validation year provides a held-out signal for early stopping
            and hyperparameter tuning without touching the test year.
          - Test year (2024) is held entirely out until final evaluation.

        Default split:
          Train      : 2021, 2022  (244 JJAS days)
          Validation : 2023        (122 JJAS days)
          Test       : 2024        (122 JJAS days)

        Note: 2020 is excluded from defaults — GFS data for 2020 was not acquired.

        Args:
            train_years: List of training years. Default [2021, 2022].
            val_years:   List of validation years. Default [2023].
                         Pass [] to skip (returns empty DataFrame).
            test_years:  List of test years. Default [2024].

        Returns:
            Tuple of (train_df, val_df, test_df) metadata DataFrames.
        """
        df = self.load_metadata()
        train_df = df[df["year"].isin(train_years)].copy()
        val_df   = df[df["year"].isin(val_years)].copy()   if val_years else df.iloc[0:0].copy()
        test_df  = df[df["year"].isin(test_years)].copy()
        return train_df, val_df, test_df
