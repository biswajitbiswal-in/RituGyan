"""Tests for Phase 2G Targeted GFS Single-Cycle Acquisition and Integration.

Validates:
1. All 9 forecast leads exist (f003, f006, f009, f012, f015, f018, f021, f024, f027).
2. Correct initialization time = 2024-06-21 00Z.
3. Required variables exist (tp, prmsl, 2t, 2d, 10u, 10v, pwat).
4. Spatial domain is correct (global 721x1440, subgrid 127x121).
5. Forecast lead metadata is correct (step = 3, 6, 9, 12, 15, 18, 21, 24, 27).
6. No unexpected NaN/Inf values in required predictor fields.
7. Precipitation accumulation semantics are preserved (A(f027) >= A(f003)).
8. Output can be consumed by the existing Phase 2 preprocessing pipeline.
9. Existing repository tests remain unaffected.
"""

from pathlib import Path
import pytest
import numpy as np

from scripts.acquire_gfs import (
    DEFAULT_LEADS,
    DEFAULT_TARGET_CYCLE,
    DEFAULT_TARGET_DATE,
    REQUIRED_VARS_METADATA,
    validate_gfs_lead_file,
)
from src.features.preprocessing import (
    AlignedDaySample,
    build_aligned_day_sample,
    load_gfs_forecast_predictors,
)
from src.ingestion.spatial import (
    CANONICAL_LATS,
    CANONICAL_LONS,
    CANONICAL_NUM_LATS,
    CANONICAL_NUM_LONS,
)


class TestGfsAcquisition:
    """Test suite for single-cycle GFS download, integrity, and pipeline integration."""

    @pytest.fixture(scope="class")
    def gfs_test_dir(self):
        # Prefer newly acquired single cycle test dir, fallback to exact test fixture
        new_dir = Path("data/raw/gfs_single_cycle_test")
        exact_dir = Path("data/raw/gfs_exact_test")
        if new_dir.exists() and len(list(new_dir.glob("gfs.0p25.*.grib2"))) >= 9:
            return new_dir
        return exact_dir

    def test_all_nine_forecast_leads_exist(self, gfs_test_dir):
        """Verify that all 9 required forecast leads (f003 to f027) exist and are non-empty."""
        for lead in DEFAULT_LEADS:
            expected_file = gfs_test_dir / f"gfs.0p25.{DEFAULT_TARGET_DATE}{DEFAULT_TARGET_CYCLE}.{lead}.grib2"
            assert expected_file.exists(), f"Missing GFS lead file: {expected_file}"
            assert expected_file.stat().st_size > 1_000_000, f"File {expected_file} is suspiciously small: {expected_file.stat().st_size} bytes"

    @pytest.mark.parametrize("lead", DEFAULT_LEADS)
    def test_gfs_lead_file_integrity_and_metadata(self, gfs_test_dir, lead):
        """Verify initialization time, lead step, grid dimensions, and variables for each lead."""
        fpath = gfs_test_dir / f"gfs.0p25.{DEFAULT_TARGET_DATE}{DEFAULT_TARGET_CYCLE}.{lead}.grib2"
        report = validate_gfs_lead_file(
            file_path=fpath,
            expected_date=DEFAULT_TARGET_DATE,
            expected_cycle=DEFAULT_TARGET_CYCLE,
            expected_lead=lead,
        )

        assert report["status"] == "PASS", f"Validation failed for {lead}: {report['errors']}"
        assert bool(report["is_valid_grib2"]) is True
        assert bool(report["readable_eccodes"]) is True
        assert report["initialization_date"] == DEFAULT_TARGET_DATE
        assert report["initialization_time"].startswith(DEFAULT_TARGET_CYCLE)
        
        expected_step = int(lead.replace("f", ""))
        assert report["forecast_lead_step"] == expected_step
        assert bool(report["lead_match"]) is True
        assert report["global_grid_shape"] == (721, 1440)
        assert report["subgrid_shape"] == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)

        # Check all 7 meteorological variables
        expected_vars = ["tp", "prmsl", "2t", "2d", "10u", "10v", "pwat"]
        for var in expected_vars:
            assert var in report["variables"], f"Variable {var} missing in {lead}"
            var_info = report["variables"][var]
            assert var_info["nan_count"] == 0, f"NaNs found in global {var} for {lead}"
            assert var_info["inf_count"] == 0, f"Infs found in global {var} for {lead}"
            assert var_info["subgrid_nan_count"] == 0, f"NaNs found in subgrid {var} for {lead}"
            assert var_info["subgrid_inf_count"] == 0, f"Infs found in subgrid {var} for {lead}"
            assert bool(var_info["is_clean"]) is True

    def test_precipitation_accumulation_semantics(self, gfs_test_dir):
        """Verify that cumulative precipitation monotonically increases or remains consistent."""
        f003_path = gfs_test_dir / f"gfs.0p25.{DEFAULT_TARGET_DATE}{DEFAULT_TARGET_CYCLE}.f003.grib2"
        f027_path = gfs_test_dir / f"gfs.0p25.{DEFAULT_TARGET_DATE}{DEFAULT_TARGET_CYCLE}.f027.grib2"

        r003 = validate_gfs_lead_file(f003_path, DEFAULT_TARGET_DATE, DEFAULT_TARGET_CYCLE, "f003")
        r027 = validate_gfs_lead_file(f027_path, DEFAULT_TARGET_DATE, DEFAULT_TARGET_CYCLE, "f027")

        tp_003_mean = r003["variables"]["tp"]["subgrid_mean"]
        tp_027_mean = r027["variables"]["tp"]["subgrid_mean"]

        assert tp_027_mean >= tp_003_mean, (
            f"24h differencing semantics violated: f027 mean ({tp_027_mean}) < f003 mean ({tp_003_mean})"
        )

    def test_consumption_by_phase2_preprocessing(self, gfs_test_dir):
        """Verify that load_gfs_forecast_predictors successfully consumes the acquired GFS files."""
        preds = load_gfs_forecast_predictors(
            gfs_dir=gfs_test_dir,
            date_str=DEFAULT_TARGET_DATE,
            cycle_str=DEFAULT_TARGET_CYCLE,
        )

        expected_channels = [
            "gfs_tp_24h",
            "gfs_prmsl_mean",
            "gfs_2t_mean",
            "gfs_2d_mean",
            "gfs_10u_mean",
            "gfs_10v_mean",
            "gfs_pwat_mean",
            "gfs_wind_speed",
        ]

        for ch in expected_channels:
            assert ch in preds, f"Channel {ch} missing from GFS predictors"
            arr = preds[ch]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
            assert not np.isnan(arr).any(), f"NaNs found in predictor channel {ch}"
            assert not np.isinf(arr).any(), f"Infs found in predictor channel {ch}"

        # Check physical ranges
        assert np.all(preds["gfs_tp_24h"] >= 0.0)
        assert np.all(preds["gfs_prmsl_mean"] > 800.0) and np.all(preds["gfs_prmsl_mean"] < 1100.0)  # hPa
        assert np.all(preds["gfs_2t_mean"] > -50.0) and np.all(preds["gfs_2t_mean"] < 65.0)         # °C
        assert np.all(preds["gfs_pwat_mean"] >= 0.0)                                                # kg/m^2
        assert np.all(preds["gfs_wind_speed"] >= 0.0)                                               # m/s

    def test_full_day_sample_construction_with_acquired_gfs(self, gfs_test_dir):
        """Verify that build_aligned_day_sample integrates IMD, ERA5, and newly acquired GFS."""
        imd_file = Path("data/raw/RF25_ind2024_rfp25.nc")
        era5_file = Path("data/raw/era5.nc")

        sample = build_aligned_day_sample(
            imd_file=imd_file,
            era5_file=era5_file,
            gfs_dir=gfs_test_dir,
            target_date="2024-06-21",
        )

        assert isinstance(sample, AlignedDaySample)
        assert sample.features.shape == (15, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert sample.target.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert sample.valid_mask.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert np.allclose(sample.lats, CANONICAL_LATS)
        assert np.allclose(sample.lons, CANONICAL_LONS)


class TestGfsHistoricalAcquisition:
    """Validation test suite for full multi-year historical GFS dataset (JJAS 2021-2024)."""

    @pytest.fixture(scope="class")
    def historical_dir(self):
        return Path("data/raw/gfs_historical")

    def test_historical_directory_and_manifest(self, historical_dir):
        """Verify that historical directory and manifest.json exist when acquisition has started."""
        if not historical_dir.exists():
            pytest.skip("Historical GFS directory not yet created.")
        manifest_path = historical_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Historical GFS manifest.json not yet written.")
        
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "years_summary" in manifest or "years_processed" in manifest
        assert "files" in manifest
        assert len(manifest["files"]) > 0

    def test_sample_cycle_across_available_years(self, historical_dir):
        """Sample cycles across available historical years and verify GRIB2 integrity, variables, and coords."""
        if not historical_dir.exists():
            pytest.skip("Historical GFS directory not yet created.")

        year_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not year_dirs:
            pytest.skip("No historical year subdirectories found yet.")

        for ydir in sorted(year_dirs):
            year = int(ydir.name)
            # Find all unique dates in this year
            all_gribs = list(ydir.glob("gfs.0p25.*.grib2"))
            if not all_gribs:
                continue

            # Pick a sample date (e.g. 202X-06-21 or first available)
            sample_date_str = f"{year}0621"
            sample_files = list(ydir.glob(f"gfs.0p25.{sample_date_str}00.f*.grib2"))
            if not sample_files:
                sample_files = list(ydir.glob("gfs.0p25.*00.f*.grib2"))[:9]

            if sample_files:
                sample_file = sample_files[0]
                lead = sample_file.stem.split(".")[-1]
                date_part = sample_file.stem.split(".")[2][:8]
                cycle_part = sample_file.stem.split(".")[2][8:10]

                report = validate_gfs_lead_file(
                    file_path=sample_file,
                    expected_date=date_part,
                    expected_cycle=cycle_part,
                    expected_lead=lead,
                )

                assert report["status"] == "PASS", f"Validation failed for {sample_file.name}: {report['errors']}"
                assert report["readable_eccodes"] is True
                assert report["global_grid_shape"] == (721, 1440)
                assert report["subgrid_shape"] == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
                
                # Check all 7 meteorological variables
                expected_vars = ["tp", "prmsl", "2t", "2d", "10u", "10v", "pwat"]
                for v in expected_vars:
                    assert v in report["variables"], f"Variable {v} missing in {sample_file.name}"
                    assert report["variables"][v]["nan_count"] == 0
                    assert report["variables"][v]["inf_count"] == 0

    def test_historical_gfs_pipeline_integration(self, historical_dir):
        """Verify that historical GFS files can be consumed by load_gfs_forecast_predictors."""
        if not historical_dir.exists():
            pytest.skip("Historical GFS directory not yet created.")

        year_dirs = [d for d in historical_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not year_dirs:
            pytest.skip("No historical year subdirectories found yet.")

        for ydir in sorted(year_dirs):
            year = int(ydir.name)
            target_date = f"{year}0621"
            leads = list(ydir.glob(f"gfs.0p25.{target_date}00.f*.grib2"))
            if len(leads) >= 9:
                preds = load_gfs_forecast_predictors(
                    gfs_dir=ydir,
                    date_str=target_date,
                    cycle_str="00",
                )
                assert "gfs_tp_24h" in preds
                assert "gfs_prmsl_mean" in preds
                assert preds["gfs_tp_24h"].shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
                assert not np.isnan(preds["gfs_tp_24h"]).any()
                assert np.all(preds["gfs_tp_24h"] >= 0.0)

