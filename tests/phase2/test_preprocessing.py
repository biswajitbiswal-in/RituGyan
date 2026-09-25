"""Automated validation tests for Phase 2C: Multi-Source Preprocessing & Aligned Feature Tensor Pipeline.

Validates the aligned extraction, unit conversions, missing-data mask handling,
and tensor shapes across IMD ground-truth, ERA5 predictors, and GFS forecasts.
"""

from pathlib import Path
import pytest
import numpy as np

from src.features.preprocessing import (
    FEATURE_CHANNELS,
    CHANNEL_UNITS,
    AlignedDaySample,
    build_aligned_day_sample,
    load_imd_target_day,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
)
from src.ingestion.spatial import CANONICAL_NUM_LATS, CANONICAL_NUM_LONS


@pytest.fixture
def imd_file(project_root: Path) -> Path:
    p = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    assert p.exists(), f"IMD file missing: {p}"
    return p


@pytest.fixture
def era5_file(project_root: Path) -> Path:
    p = project_root / "data" / "raw" / "era5.nc"
    assert p.exists(), f"ERA5 file missing: {p}"
    return p


@pytest.fixture
def gfs_dir(project_root: Path) -> Path:
    p = project_root / "data" / "raw" / "gfs_exact_test"
    assert p.exists(), f"GFS exact dir missing: {p}"
    return p


class TestPreprocessingPipeline:
    """Validate aligned feature and target tensor generation."""

    def test_feature_channels_and_units_specification(self):
        """Verify defined feature channels and corresponding physical units."""
        assert len(FEATURE_CHANNELS) == 15
        assert len(CHANNEL_UNITS) == 15

        for ch in FEATURE_CHANNELS:
            assert ch in CHANNEL_UNITS, f"Missing unit mapping for {ch}"

        # Confirm critical unit standards
        assert CHANNEL_UNITS["gfs_tp_24h"] == "mm"
        assert CHANNEL_UNITS["gfs_prmsl_mean"] == "hPa"
        assert CHANNEL_UNITS["era5_msl_mean"] == "hPa"
        assert CHANNEL_UNITS["gfs_2t_mean"] == "°C"
        assert CHANNEL_UNITS["era5_t2m_mean"] == "°C"

    def test_load_imd_target_day(self, imd_file: Path):
        """Verify loading target day rainfall and mask from IMD."""
        rainfall, valid_mask = load_imd_target_day(imd_file, target_date="2024-06-21")

        assert rainfall.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert valid_mask.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert valid_mask.dtype == bool

        # IMD has valid land cells and masked ocean cells
        valid_count = int(valid_mask.sum())
        assert valid_count > 0, "Expected non-zero valid land cells"
        assert valid_count < valid_mask.size, "Expected ocean points to be masked"

        # Valid cells must have non-negative finite rainfall
        valid_vals = rainfall[valid_mask]
        assert np.all(np.isfinite(valid_vals))
        assert np.all(valid_vals >= 0.0)
        assert np.all(np.isnan(rainfall[~valid_mask]))

    def test_load_era5_in_window_predictors(self, era5_file: Path):
        """Verify ERA5 predictors extraction and unit conversions."""
        era5_feats = load_era5_in_window_predictors(era5_file, target_date="2024-06-21")

        expected_keys = [
            "era5_msl_mean",
            "era5_t2m_mean",
            "era5_d2m_mean",
            "era5_u10_mean",
            "era5_v10_mean",
            "era5_tcwv_mean",
            "era5_wind_speed",
        ]
        for k in expected_keys:
            assert k in era5_feats
            arr = era5_feats[k]
            assert arr.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
            assert np.all(np.isfinite(arr)), f"Non-finite values found in {k}"

        # Physical sanity checks
        msl = era5_feats["era5_msl_mean"]
        assert 950.0 <= np.min(msl) and np.max(msl) <= 1050.0, f"MSL out of range: [{np.min(msl)}, {np.max(msl)}]"

        t2m = era5_feats["era5_t2m_mean"]
        # Includes high-altitude Himalayan terrain in Ladakh/Kashmir at 38°N
        assert -30.0 <= np.min(t2m) and np.max(t2m) <= 55.0, f"T2M out of range: [{np.min(t2m)}, {np.max(t2m)}]"

        wspd = era5_feats["era5_wind_speed"]
        assert np.all(wspd >= 0.0)

    def test_load_gfs_forecast_predictors(self, gfs_dir: Path):
        """Verify GFS forecast predictors extraction, 24h precipitation, and unit conversions."""
        gfs_feats = load_gfs_forecast_predictors(gfs_dir, date_str="20240621", cycle_str="00")

        expected_keys = [
            "gfs_tp_24h",
            "gfs_prmsl_mean",
            "gfs_2t_mean",
            "gfs_2d_mean",
            "gfs_10u_mean",
            "gfs_10v_mean",
            "gfs_pwat_mean",
            "gfs_wind_speed",
        ]
        for k in expected_keys:
            assert k in gfs_feats
            arr = gfs_feats[k]
            assert arr.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
            assert np.all(np.isfinite(arr)), f"Non-finite values found in {k}"

        # 24h precipitation non-negativity
        tp_24h = gfs_feats["gfs_tp_24h"]
        assert np.all(tp_24h >= 0.0), "GFS 24h rainfall must be strictly non-negative"
        assert np.max(tp_24h) > 0.0, "Expected non-zero rainfall over monsoon domain"

        # Pressure and temperature sanity
        prmsl = gfs_feats["gfs_prmsl_mean"]
        assert 950.0 <= np.min(prmsl) and np.max(prmsl) <= 1050.0

        t2m = gfs_feats["gfs_2t_mean"]
        assert -30.0 <= np.min(t2m) and np.max(t2m) <= 55.0

    def test_build_aligned_day_sample_structure(
        self, imd_file: Path, era5_file: Path, gfs_dir: Path
    ):
        """Verify end-to-end AlignedDaySample tensor creation for validation target day."""
        sample = build_aligned_day_sample(
            imd_file=imd_file,
            era5_file=era5_file,
            gfs_dir=gfs_dir,
            target_date="2024-06-21",
        )

        assert isinstance(sample, AlignedDaySample)
        assert sample.target_date == "2024-06-21"

        # Tensor shapes
        # X: (15, 127, 121)
        assert sample.features.shape == (15, 127, 121)
        assert sample.features.dtype == np.float32
        assert np.all(np.isfinite(sample.features)), "Feature tensor contains NaNs or infs"

        # y: (127, 121)
        assert sample.target.shape == (127, 121)
        assert sample.target.dtype == np.float32

        # valid_mask: (127, 121)
        assert sample.valid_mask.shape == (127, 121)
        assert sample.valid_mask.dtype == bool

        # Coordinate arrays
        assert sample.lats.shape == (127,)
        assert sample.lons.shape == (121,)
        assert sample.lats[0] == 6.50 and sample.lats[-1] == 38.00
        assert sample.lons[0] == 68.00 and sample.lons[-1] == 98.00

        # Metadata validation
        assert sample.metadata["target_date"] == "2024-06-21"
        assert sample.metadata["num_channels"] == 15
        assert sample.metadata["spatial_shape"] == (127, 121)
        assert sample.metadata["valid_land_cells"] == int(sample.valid_mask.sum())
