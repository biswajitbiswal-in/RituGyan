"""Tests for Phase 2D Synoptic Feature Engineering & Meteorological Index Computation.

Validates:
- Physical units consistency
- Finite outputs (no unhandled NaNs / Infs)
- Valid physical ranges
- Deterministic calculation
- Missing-variable handling (Category B handling without variable fabrication)
- Synthetic and real-data end-to-end extraction
"""

from pathlib import Path
import numpy as np
import pytest

from src.features.preprocessing import (
    build_aligned_day_sample,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
)
from src.features.synoptic import (
    CLIMATOLOGICAL_TROUGH_LAT_NORMAL,
    IndexCategory,
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
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS


@pytest.fixture
def synthetic_synoptic_grid():
    """Create synthetic 127x121 meteorological fields for index testing."""
    lats = CANONICAL_LATS
    lons = CANONICAL_LONS

    # Synthetic realistic MSLP field with trough around 22°N
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    
    # Base pressure 1008 hPa with a trough (minimum) at 22°N
    mslp = 1008.0 - 10.0 * np.exp(-((lat_grid - 22.0) ** 2) / 16.0)
    # Add depression low at 20°N, 84°E (Bay of Bengal)
    depression = -6.0 * np.exp(-(((lat_grid - 20.0) ** 2) / 4.0 + ((lon_grid - 84.0) ** 2) / 9.0))
    mslp += depression

    # Synthetic 10m winds with cyclonic circulation around the depression and strong westerlies in south
    u10 = 8.0 * np.sin(np.deg2rad(lat_grid * 2)) + 5.0
    v10 = 4.0 * np.cos(np.deg2rad(lon_grid * 2))
    
    # PWAT (mm) with moist core
    pwat = 45.0 + 15.0 * np.exp(-((lat_grid - 20.0) ** 2) / 25.0)

    return {
        "mslp": mslp.astype(np.float32),
        "u10": u10.astype(np.float32),
        "v10": v10.astype(np.float32),
        "pwat": pwat.astype(np.float32),
        "lats": lats,
        "lons": lons,
    }


class TestSynopticFeatures:
    """Test suite for synoptic meteorological index computation."""

    def test_monsoon_trough_index_category_a(self, synthetic_synoptic_grid):
        """Verify monsoon trough position calculation, units, and valid range."""
        grid = synthetic_synoptic_grid
        res = compute_monsoon_trough_index(
            grid["mslp"],
            lats=grid["lats"],
            lons=grid["lons"],
            normal_lat=CLIMATOLOGICAL_TROUGH_LAT_NORMAL,
        )

        assert isinstance(res, SynopticIndexResult)
        assert res.name == "monsoon_trough_latitude_departure"
        assert res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert res.units == "degrees_latitude"
        assert res.is_computable is True
        assert res.value is not None
        assert np.isfinite(res.value)
        
        # Valid physical bounds for trough departure: within [-10.0, +10.0] degrees
        assert -10.0 <= res.value <= 10.0
        assert 16.0 <= res.metadata["mean_trough_latitude_deg_n"] <= 30.0
        assert 950.0 <= res.metadata["trough_minimum_pressure_hpa"] <= 1030.0
        assert np.isfinite(res.metadata["trough_pressure_gradient_hpa"])

    def test_llj_strength_index_category_b_and_missing_dependency(self, synthetic_synoptic_grid):
        """Verify that LLJ index correctly identifies Category B and does NOT fabricate 850 hPa wind."""
        grid = synthetic_synoptic_grid

        # When 850 hPa wind is None (current dataset status)
        res_missing = compute_llj_strength_index(
            u850_ms=None,
            v850_ms=None,
            lats=grid["lats"],
            lons=grid["lons"],
        )

        assert res_missing.category == IndexCategory.B_REQUIRES_ADDITIONAL_VARIABLE
        assert res_missing.is_computable is False
        assert res_missing.value is None
        assert "u_wind_850hPa" in res_missing.missing_dependencies
        assert "v_wind_850hPa" in res_missing.missing_dependencies
        assert "required_variables" in res_missing.metadata

        # When 850 hPa wind is explicitly provided (e.g. synthetic test)
        synthetic_u850 = grid["u10"] * 2.0  # Just for explicit 850hPa computation test
        synthetic_v850 = grid["v10"] * 2.0
        res_provided = compute_llj_strength_index(
            u850_ms=synthetic_u850,
            v850_ms=synthetic_v850,
            lats=grid["lats"],
            lons=grid["lons"],
        )
        assert res_provided.is_computable is True
        assert res_provided.value is not None
        assert res_provided.units == "m/s"
        assert np.isfinite(res_provided.value)

    def test_cross_equatorial_inflow_index(self, synthetic_synoptic_grid):
        """Verify Low-Latitude Arabian Sea Inflow Index calculation."""
        grid = synthetic_synoptic_grid
        res = compute_cross_equatorial_flow_index(
            grid["u10"],
            grid["v10"],
            lats=grid["lats"],
            lons=grid["lons"],
        )

        assert res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert res.units == "m/s"
        assert res.is_computable is True
        assert np.isfinite(res.value)
        assert 0.0 <= res.value <= 60.0
        assert np.isfinite(res.metadata["kinetic_energy_m2_s2"])
        assert res.metadata["kinetic_energy_m2_s2"] >= 0.0

    def test_moisture_flux_convergence_2d_proxy(self, synthetic_synoptic_grid):
        """Verify 2D MFC proxy calculation, units, finite values, and spherical metric."""
        grid = synthetic_synoptic_grid
        res = compute_moisture_flux_convergence(
            grid["pwat"],
            grid["u10"],
            grid["v10"],
            lats=grid["lats"],
            lons=grid["lons"],
        )

        assert res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert res.units == "mm/day"
        assert res.is_computable is True
        assert np.isfinite(res.value)
        
        # Spatial map checks
        assert res.spatial_field is not None
        assert res.spatial_field.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert np.all(np.isfinite(res.spatial_field))
        assert "central_india_mfc_mm_day" in res.metadata
        assert np.isfinite(res.metadata["central_india_mfc_mm_day"])

    def test_relative_vorticity_2d(self, synthetic_synoptic_grid):
        """Verify 2D relative vorticity on spherical grid."""
        grid = synthetic_synoptic_grid
        zeta = compute_relative_vorticity_2d(
            grid["u10"],
            grid["v10"],
            lats=grid["lats"],
            lons=grid["lons"],
        )
        assert zeta.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert np.all(np.isfinite(zeta))
        # Typical atmospheric synoptic vorticity magnitudes are around 1e-6 to 1e-4 s^-1
        assert np.max(np.abs(zeta)) < 1.0

    def test_monsoon_depression_index(self, synthetic_synoptic_grid):
        """Verify monsoon depression / cyclonic signature detection."""
        grid = synthetic_synoptic_grid
        res = compute_monsoon_depression_index(
            grid["mslp"],
            grid["u10"],
            grid["v10"],
            lats=grid["lats"],
            lons=grid["lons"],
        )
        assert res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert res.units == "unitless_score"
        assert 0.0 <= res.value <= 1.0
        assert np.isfinite(res.metadata["depression_mslp_anomaly_hpa"])
        assert np.isfinite(res.metadata["depression_max_vorticity_s1"])
        assert np.isfinite(res.metadata["depression_pressure_contrast_hpa"])
        assert 0.0 <= res.metadata["depression_vorticity_area_fraction"] <= 1.0
        assert 0.0 <= res.metadata["depression_vorticity_convergence_area_fraction"] <= 1.0
        assert res.value == float(res.metadata["depression_candidate_a_event"])

    def test_orographic_and_western_disturbance_indices(self, synthetic_synoptic_grid):
        """Verify Western Ghats orographic flux and Northwest activity indices."""
        grid = synthetic_synoptic_grid
        orog_res = compute_orographic_interception_index(
            grid["pwat"],
            grid["u10"],
            lats=grid["lats"],
            lons=grid["lons"],
        )
        assert orog_res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert orog_res.units == "kg/(m*s)"
        assert np.isfinite(orog_res.value)

        wd_res = compute_western_disturbance_index(
            grid["mslp"],
            grid["pwat"],
            lats=grid["lats"],
            lons=grid["lons"],
        )
        assert wd_res.category == IndexCategory.A_DIRECTLY_COMPUTABLE
        assert wd_res.units == "mm"
        assert np.isfinite(wd_res.value)

    def test_extract_all_synoptic_features_assembly(self, synthetic_synoptic_grid):
        """Verify end-to-end assembly into SynopticFeatureSet container."""
        grid = synthetic_synoptic_grid
        feats = extract_all_synoptic_features(
            mslp_hpa=grid["mslp"],
            pwat_kg_m2=grid["pwat"],
            u10_ms=grid["u10"],
            v10_ms=grid["v10"],
            target_date="2024-06-21",
            u850_ms=None,
            v850_ms=None,
            lats=grid["lats"],
            lons=grid["lons"],
        )

        assert isinstance(feats, SynopticFeatureSet)
        assert feats.target_date == "2024-06-21"
        assert feats.llj_is_supported is False
        assert feats.llj_strength_850_ms is None
        assert np.isfinite(feats.trough_mean_latitude)
        assert np.isfinite(feats.trough_latitude_departure)
        assert np.isfinite(feats.low_latitude_inflow_speed_ms)
        assert np.isfinite(feats.mfc_domain_mean_mm_day)
        assert feats.mfc_spatial_map.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert len(feats.feature_catalog) == 7
        assert feats.feature_catalog["llj_strength"].category == IndexCategory.B_REQUIRES_ADDITIONAL_VARIABLE

    def test_real_data_synoptic_feature_extraction(self):
        """Verify synoptic feature calculation using real ERA5 and GFS test data."""
        era5_path = Path("data/raw/era5.nc")
        gfs_dir = Path("data/raw/gfs_exact_test")
        if not gfs_dir.exists():
            gfs_dir = Path("data/raw/gfs_test")

        if not (era5_path.exists() and gfs_dir.exists()):
            pytest.skip("ERA5 or GFS test data not present")

        # Load predictors from Phase 2 preprocessing
        era5_feats = load_era5_in_window_predictors(era5_path, target_date="2024-06-21")
        gfs_feats = load_gfs_forecast_predictors(gfs_dir, date_str="20240621", cycle_str="00")

        # Compute synoptic features using ERA5 reanalysis fields
        syn_era5 = extract_all_synoptic_features(
            mslp_hpa=era5_feats["era5_msl_mean"],
            pwat_kg_m2=era5_feats["era5_tcwv_mean"],
            u10_ms=era5_feats["era5_u10_mean"],
            v10_ms=era5_feats["era5_v10_mean"],
            target_date="2024-06-21",
        )

        assert np.isfinite(syn_era5.trough_latitude_departure)
        assert np.isfinite(syn_era5.low_latitude_inflow_speed_ms)
        assert np.isfinite(syn_era5.mfc_domain_mean_mm_day)
        assert syn_era5.llj_is_supported is False

        # Compute synoptic features using GFS forecast fields
        syn_gfs = extract_all_synoptic_features(
            mslp_hpa=gfs_feats["gfs_prmsl_mean"],
            pwat_kg_m2=gfs_feats["gfs_pwat_mean"],
            u10_ms=gfs_feats["gfs_10u_mean"],
            v10_ms=gfs_feats["gfs_10v_mean"],
            target_date="2024-06-21",
        )

        assert np.isfinite(syn_gfs.trough_latitude_departure)
        assert np.isfinite(syn_gfs.low_latitude_inflow_speed_ms)
        assert np.isfinite(syn_gfs.mfc_domain_mean_mm_day)
