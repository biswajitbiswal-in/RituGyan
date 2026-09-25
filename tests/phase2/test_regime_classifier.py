"""Tests for Phase 2D Rule-Based Synoptic Regime Classifier.

Validates:
- Deterministic mapping across all 6 predefined regimes:
  0: ACTIVE_MONSOON
  1: BREAK_MONSOON
  2: MONSOON_DEPRESSION
  3: OROGRAPHIC_MONSOON
  4: COASTAL_REGIME
  5: WESTERN_DISTURBANCE
- Strict rule thresholds and evidence generation
- Fallback behavior on edge cases and ambiguous data
- Output ranges, confidence bounds in [0.0, 1.0], finite values
- Real-data classification integration
"""

from pathlib import Path
import numpy as np
import pytest

from src.features.preprocessing import (
    build_aligned_day_sample,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
    load_imd_target_day,
)
from src.features.regimes import (
    REGIME_CATALOG,
    RegimeClassificationResult,
    SynopticRegimeClassifier,
)
from src.features.synoptic import extract_all_synoptic_features
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS


@pytest.fixture
def classifier():
    """Create SynopticRegimeClassifier instance."""
    return SynopticRegimeClassifier(min_confidence_threshold=0.50)


def create_mock_synoptic_scenario(scenario_type: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Helper to synthesize meteorological grids corresponding to specific regimes."""
    lats = CANONICAL_LATS
    lons = CANONICAL_LONS
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Default baseline fields
    mslp = np.full((127, 121), 1008.0, dtype=np.float32)
    u10 = np.full((127, 121), 5.0, dtype=np.float32)
    v10 = np.full((127, 121), 2.0, dtype=np.float32)
    pwat = np.full((127, 121), 40.0, dtype=np.float32)
    rain = np.zeros((127, 121), dtype=np.float32)

    if scenario_type == "ACTIVE_MONSOON":
        # Trough south of normal (20°N), strong smooth inflow (8.5 m/s), high PWAT (52 mm), widespread rain
        mslp = 1008.0 - 8.0 * np.exp(-((lat_grid - 20.0) ** 2) / 20.0)
        u10 = np.full((127, 121), 8.5, dtype=np.float32)
        v10 = np.full((127, 121), 3.5, dtype=np.float32)
        pwat = np.full((127, 121), 52.0, dtype=np.float32)
        rain = np.where((lat_grid >= 18.0) & (lat_grid <= 25.0), 30.0, 8.0).astype(np.float32)

    elif scenario_type == "BREAK_MONSOON":
        # Trough shifted to foothills (27°N -> departure > +2.5°), dry central India, rain at foothills
        mslp = 1008.0 - 10.0 * np.exp(-((lat_grid - 27.5) ** 2) / 10.0)
        pwat = np.where(lat_grid > 26.0, 48.0, 30.0).astype(np.float32)
        rain = np.where(lat_grid > 26.0, 35.0, 0.5).astype(np.float32)

    elif scenario_type == "MONSOON_DEPRESSION":
        # Strong low in Bay of Bengal / Central India (20°N, 84°E), -5 hPa anomaly, strong cyclonic vorticity
        dist_sq = ((lat_grid - 20.0) ** 2) / 4.0 + ((lon_grid - 84.0) ** 2) / 8.0
        mslp = 1006.0 - 7.0 * np.exp(-dist_sq)
        # Add cyclonic wind circulation
        u10 += -12.0 * (lat_grid - 20.0) * np.exp(-dist_sq)
        v10 += 12.0 * (lon_grid - 84.0) * np.exp(-dist_sq)
        rain = np.where(dist_sq < 3.0, 85.0, 2.0).astype(np.float32)

    elif scenario_type == "OROGRAPHIC_MONSOON":
        # Heavy onshore wind impinging Western Ghats (14 m/s), PWAT 55 mm, heavy rain on Ghats
        u10 = np.where((lat_grid >= 8.0) & (lat_grid <= 18.0), 14.0, 3.0).astype(np.float32)
        pwat = np.full((127, 121), 55.0, dtype=np.float32)
        ghats_box = (lat_grid >= 10.0) & (lat_grid <= 18.0) & (lon_grid >= 73.0) & (lon_grid <= 75.5)
        rain = np.where(ghats_box, 90.0, 2.0).astype(np.float32)

    elif scenario_type == "WESTERN_DISTURBANCE":
        # Activity in NW (32°N, 75°E), peninsular inflow weak (< 4 m/s)
        u10 = np.full((127, 121), 2.5, dtype=np.float32)
        v10 = np.full((127, 121), 1.0, dtype=np.float32)
        pwat = np.where(lat_grid > 28.0, 38.0, 20.0).astype(np.float32)
        nw_box = (lat_grid >= 28.0) & (lat_grid <= 36.0) & (lon_grid >= 70.0) & (lon_grid <= 78.0)
        rain = np.where(nw_box, 45.0, 0.0).astype(np.float32)

    elif scenario_type == "COASTAL_REGIME":
        # Localized rain on coasts, no large synoptic system
        u10 = np.full((127, 121), 5.5, dtype=np.float32)
        v10 = np.full((127, 121), 3.0, dtype=np.float32)
        coastal_strip = (lat_grid >= 10.0) & (lat_grid <= 20.0) & (lon_grid >= 72.0) & (lon_grid <= 73.5)
        rain = np.where(coastal_strip, 25.0, 1.0).astype(np.float32)

    feats = extract_all_synoptic_features(
        mslp_hpa=mslp,
        pwat_kg_m2=pwat,
        u10_ms=u10,
        v10_ms=v10,
        target_date="2024-06-21",
    )

    return feats, rain, np.ones((127, 121), dtype=bool)


class TestSynopticRegimeClassifier:
    """Test suite for the rule-based synoptic regime classifier."""

    @pytest.mark.parametrize(
        "scenario,expected_regime_id,expected_code",
        [
            ("ACTIVE_MONSOON", 0, "ACTIVE_MONSOON"),
            ("BREAK_MONSOON", 1, "BREAK_MONSOON"),
            ("MONSOON_DEPRESSION", 2, "MONSOON_DEPRESSION"),
            ("OROGRAPHIC_MONSOON", 3, "OROGRAPHIC_MONSOON"),
            ("WESTERN_DISTURBANCE", 5, "WESTERN_DISTURBANCE"),
            ("COASTAL_REGIME", 4, "COASTAL_REGIME"),
        ],
    )
    def test_regime_classification_scenarios(
        self,
        classifier: SynopticRegimeClassifier,
        scenario: str,
        expected_regime_id: int,
        expected_code: str,
    ):
        """Verify that representative meteorological patterns trigger each respective regime."""
        feats, rain, mask = create_mock_synoptic_scenario(scenario)
        result = classifier.classify(feats, rainfall_field=rain, valid_mask=mask)

        assert isinstance(result, RegimeClassificationResult)
        assert result.regime_id == expected_regime_id
        assert result.regime_code == expected_code
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence >= 0.50
        assert len(result.triggered_rules) > 0
        assert isinstance(result.evidence, dict)

    def test_classification_determinism(self, classifier: SynopticRegimeClassifier):
        """Verify that identical meteorological inputs always produce identical classification results."""
        feats, rain, mask = create_mock_synoptic_scenario("ACTIVE_MONSOON")

        res1 = classifier.classify(feats, rainfall_field=rain, valid_mask=mask)
        res2 = classifier.classify(feats, rainfall_field=rain, valid_mask=mask)

        assert res1.regime_id == res2.regime_id
        assert res1.regime_code == res2.regime_code
        assert res1.confidence == res2.confidence
        assert res1.triggered_rules == res2.triggered_rules

    def test_fallback_behavior_on_ambiguous_data(self, classifier: SynopticRegimeClassifier):
        """Verify graceful fallback when inputs do not match distinct regime signatures."""
        # Create flat neutral field with low scores across all regimes
        mslp = np.full((127, 121), 1013.25, dtype=np.float32)
        u10 = np.full((127, 121), 0.5, dtype=np.float32)
        v10 = np.full((127, 121), 0.5, dtype=np.float32)
        pwat = np.full((127, 121), 25.0, dtype=np.float32)
        rain = np.zeros((127, 121), dtype=np.float32)

        feats = extract_all_synoptic_features(
            mslp_hpa=mslp,
            pwat_kg_m2=pwat,
            u10_ms=u10,
            v10_ms=v10,
            target_date="2024-06-21",
        )

        result = classifier.classify(feats, rainfall_field=rain)
        assert isinstance(result, RegimeClassificationResult)
        assert result.fallback_applied is True
        assert result.confidence == 0.50
        assert result.regime_id in [0, 4]  # Valid fallback regimes
        assert any("Fallback" in r for r in result.triggered_rules)

    def test_real_dataset_classification(self, classifier: SynopticRegimeClassifier):
        """Verify regime classification on real ingested sample date 2024-06-21."""
        era5_path = Path("data/raw/era5.nc")
        gfs_dir = Path("data/raw/gfs_exact_test")
        imd_path = Path("data/raw/RF25_ind2024_rfp25.nc")

        if not (era5_path.exists() and gfs_dir.exists() and imd_path.exists()):
            pytest.skip("Test datasets not available")

        # Load predictors & IMD target
        era5_feats = load_era5_in_window_predictors(era5_path, target_date="2024-06-21")
        imd_rain, valid_mask = load_imd_target_day(imd_path, target_date="2024-06-21")

        syn_feats = extract_all_synoptic_features(
            mslp_hpa=era5_feats["era5_msl_mean"],
            pwat_kg_m2=era5_feats["era5_tcwv_mean"],
            u10_ms=era5_feats["era5_u10_mean"],
            v10_ms=era5_feats["era5_v10_mean"],
            target_date="2024-06-21",
        )

        result = classifier.classify(syn_feats, rainfall_field=imd_rain, valid_mask=valid_mask)

        assert 0 <= result.regime_id <= 5
        assert result.regime_code in [r.code for r in REGIME_CATALOG.values()]
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.triggered_rules) > 0
        assert "trough_departure_deg" in result.evidence
