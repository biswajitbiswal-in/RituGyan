"""Batch ingestion and multi-year dataset assembly pipeline for RituGyan.

Audits data availability across multi-source historical archives (IMD, ERA5, GFS),
identifies overlapping aligned dates, and constructs serializable feature stores
using the validated Phase 2A-2D ingestion, synoptic, and regime pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import datetime
import json
import numpy as np
import pandas as pd
import xarray as xr

from src.features.feature_store import FeatureStore
from src.features.preprocessing import (
    AlignedDaySample,
    build_aligned_day_sample,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
    load_imd_target_day,
)
from src.features.regimes import RegimeClassificationResult, SynopticRegimeClassifier
from src.features.synoptic import SynopticFeatureSet, extract_all_synoptic_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataAvailabilityAudit:
    """Detailed audit of historical dataset availability across IMD, ERA5, and GFS."""
    target_period: str
    expected_jjas_days_total: int
    expected_days_per_year: Dict[int, int]
    imd_available_years: Dict[int, int]
    era5_available_years: Dict[int, int]
    gfs_available_dates: List[str]
    common_overlapping_dates: List[str]
    missing_dates_by_year: Dict[int, int]
    usable_aligned_days_count: int
    is_multiyear_complete: bool
    required_additional_downloads: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiYearDatasetBuilder:
    """Builds multi-year feature stores by streaming aligned day samples."""

    def __init__(
        self,
        raw_data_dir: Union[str, Path] = "data/raw",
        processed_data_dir: Union[str, Path] = "data/processed",
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.regime_classifier = SynopticRegimeClassifier()

    def audit_availability(
        self,
        target_years: List[int] = [2020, 2021, 2022, 2023, 2024],
    ) -> DataAvailabilityAudit:
        """Inspect all raw datasets and audit exact physical date availability for JJAS."""
        expected_days_per_year = {}
        all_expected_dates: Set[str] = set()

        for yr in target_years:
            start_dt = datetime.date(yr, 6, 1)
            end_dt = datetime.date(yr, 9, 30)
            days_in_jjas = (end_dt - start_dt).days + 1  # 122 days
            expected_days_per_year[yr] = days_in_jjas
            cur = start_dt
            while cur <= end_dt:
                all_expected_dates.add(cur.isoformat())
                cur += datetime.timedelta(days=1)

        total_expected = len(all_expected_dates)

        # 1. Audit IMD
        imd_dates_by_year: Dict[int, Set[str]] = {yr: set() for yr in target_years}
        for imd_file in self.raw_data_dir.glob("RF25_ind*.nc"):
            try:
                ds = xr.open_dataset(imd_file)
                time_coord = "TIME" if "TIME" in ds.coords or "TIME" in ds.dims else "time"
                times = pd.to_datetime(ds[time_coord].values)
                for t in times:
                    dt_str = t.strftime("%Y-%m-%d")
                    if dt_str in all_expected_dates:
                        imd_dates_by_year[t.year].add(dt_str)
                ds.close()
            except Exception as e:
                logger.warning(f"Failed to read IMD file {imd_file}: {e}")

        # 2. Audit ERA5
        era5_dates_by_year: Dict[int, Set[str]] = {yr: set() for yr in target_years}
        for era5_file in self.raw_data_dir.glob("era5*.nc"):
            try:
                ds = xr.open_dataset(era5_file)
                time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                times = pd.to_datetime(ds[time_coord].values)
                for t in times:
                    dt_str = t.strftime("%Y-%m-%d")
                    if dt_str in all_expected_dates:
                        era5_dates_by_year[t.year].add(dt_str)
                ds.close()
            except Exception as e:
                logger.warning(f"Failed to read ERA5 file {era5_file}: {e}")

        # 3. Audit GFS Forecast Leads Sequence
        gfs_dates: Set[str] = set()
        # Check subdirectories (e.g. gfs_exact_test, gfs_test) and raw_dir
        candidate_dirs = [self.raw_data_dir, self.raw_data_dir / "gfs_exact_test", self.raw_data_dir / "gfs_test"]
        for cdir in candidate_dirs:
            if not cdir.exists():
                continue
            # Look for f003 and f027 pairs for 00Z cycle
            for f003_file in cdir.glob("gfs.0p25.*00.f003.grib2"):
                # Extract date from name: gfs.0p25.YYYYMMDD00.f003.grib2
                stem_parts = f003_file.stem.split(".")
                cycle_code = stem_parts[2]  # YYYYMMDD00
                date_compact = cycle_code[:8]
                dt_iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                # Check if corresponding f027 exists
                f027_name = f003_file.name.replace("f003", "f027")
                if (cdir / f027_name).exists() and dt_iso in all_expected_dates:
                    gfs_dates.add(dt_iso)

        # 4. Compute intersection of usable aligned days
        all_imd_dates = set.union(*imd_dates_by_year.values()) if imd_dates_by_year else set()
        all_era5_dates = set.union(*era5_dates_by_year.values()) if era5_dates_by_year else set()
        common_dates = sorted(list(all_imd_dates & all_era5_dates & gfs_dates))

        # Missing counts
        missing_by_year = {}
        for yr in target_years:
            yr_common = [d for d in common_dates if d.startswith(str(yr))]
            missing_by_year[yr] = expected_days_per_year[yr] - len(yr_common)

        # Missing downloads required
        missing_downloads = []
        for yr in target_years:
            if len(imd_dates_by_year[yr]) == 0:
                missing_downloads.append(f"IMD daily rainfall NetCDF for year {yr} (RF25_ind{yr}_rfp25.nc)")
            if len(era5_dates_by_year[yr]) == 0:
                missing_downloads.append(f"ERA5 JJAS {yr} 6-hourly reanalysis (era5_{yr}.nc)")
            yr_gfs_count = len([d for d in gfs_dates if d.startswith(str(yr))])
            if yr_gfs_count < expected_days_per_year[yr]:
                missing_downloads.append(f"GFS 00Z forecast sequence (f003-f027) for JJAS {yr} ({expected_days_per_year[yr] - yr_gfs_count} missing cycles)")

        return DataAvailabilityAudit(
            target_period=f"JJAS {target_years[0]}–{target_years[-1]}",
            expected_jjas_days_total=total_expected,
            expected_days_per_year=expected_days_per_year,
            imd_available_years={yr: len(dates) for yr, dates in imd_dates_by_year.items()},
            era5_available_years={yr: len(dates) for yr, dates in era5_dates_by_year.items()},
            gfs_available_dates=sorted(list(gfs_dates)),
            common_overlapping_dates=common_dates,
            missing_dates_by_year=missing_by_year,
            usable_aligned_days_count=len(common_dates),
            is_multiyear_complete=len(common_dates) == total_expected,
            required_additional_downloads=missing_downloads,
            metadata={"target_years": target_years},
        )

    def build_and_serialize_feature_store(
        self,
        output_dir: Union[str, Path] = "data/processed/feature_store",
        dates_to_process: Optional[List[str]] = None,
    ) -> FeatureStore:
        """Run batch day-by-day ingestion for all available aligned dates and serialize to FeatureStore.
        
        Memory-safe: Streams and processes one day at a time without accumulating raw datasets in RAM.
        """
        audit = self.audit_availability()
        target_dates = dates_to_process or audit.common_overlapping_dates

        if not target_dates:
            raise RuntimeError(
                "No common aligned dates available across IMD, ERA5, and GFS to build feature store. "
                "See audit report for missing dataset dependencies."
            )

        store = FeatureStore(output_dir)

        # Locate raw files
        imd_file = self.raw_data_dir / "RF25_ind2024_rfp25.nc"
        era5_file = self.raw_data_dir / "era5.nc"
        gfs_dir = self.raw_data_dir / "gfs_exact_test"
        if not gfs_dir.exists():
            gfs_dir = self.raw_data_dir / "gfs_test"

        samples: List[AlignedDaySample] = []
        synoptic_list: List[SynopticFeatureSet] = []
        regime_list: List[RegimeClassificationResult] = []

        logger.info(f"Starting batch construction for {len(target_dates)} aligned days...")

        for dt_str in target_dates:
            logger.info(f"Processing day {dt_str}...")
            # 1. Build aligned day sample
            sample = build_aligned_day_sample(
                imd_file=imd_file,
                era5_file=era5_file,
                gfs_dir=gfs_dir,
                target_date=dt_str,
            )

            # 2. Extract synoptic features from ERA5 predictors
            era5_idx = {name: i for i, name in enumerate(sample.channel_names)}
            mslp = sample.features[era5_idx["era5_msl_mean"]]
            t2m = sample.features[era5_idx["era5_t2m_mean"]]
            u10 = sample.features[era5_idx["era5_u10_mean"]]
            v10 = sample.features[era5_idx["era5_v10_mean"]]
            tcwv = sample.features[era5_idx["era5_tcwv_mean"]]

            syn_feats = extract_all_synoptic_features(
                mslp_hpa=mslp,
                pwat_kg_m2=tcwv,
                u10_ms=u10,
                v10_ms=v10,
                target_date=dt_str,
                lats=sample.lats,
                lons=sample.lons,
            )

            # 3. Classify synoptic regime
            regime_res = self.regime_classifier.classify(
                synoptic_feats=syn_feats,
                rainfall_field=sample.target,
                valid_mask=sample.valid_mask,
                lats=sample.lats,
                lons=sample.lons,
            )

            samples.append(sample)
            synoptic_list.append(syn_feats)
            regime_list.append(regime_res)

        # 4. Serialize into FeatureStore (NetCDF4 tensors + Parquet metadata table)
        store.save(samples, synoptic_list, regime_list)
        return store
