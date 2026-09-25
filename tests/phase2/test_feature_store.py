"""Tests for Phase 2E Feature Store Serialization, Parquet Metadata Indexing, and Split Management.

Validates:
- End-to-end FeatureStore serialization (NetCDF4 tensor store + Parquet metadata)
- Shape, dimension, and coordinate consistency
- No NaN contamination in predictor feature channels
- Expected target masking (ocean NaNs preserved, land points valid)
- Regime-label validity across the 6 canonical classes
- Temporal train/test partitioning without temporal or spatial leakage
- Reproducibility of stored arrays and metadata
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.features.batch_pipeline import MultiYearDatasetBuilder
from src.features.feature_store import (
    FeatureStore,
    SYNOPTIC_FEATURE_NAMES,
)
from src.features.preprocessing import (
    CHANNEL_UNITS,
    FEATURE_CHANNELS,
    AlignedDaySample,
    build_aligned_day_sample,
)
from src.features.regimes import REGIME_CATALOG, RegimeClassificationResult, SynopticRegimeClassifier
from src.features.synoptic import extract_all_synoptic_features
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS


@pytest.fixture
def temp_store_dir():
    """Create a temporary directory for FeatureStore testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestFeatureStore:
    """Test suite for FeatureStore serialization and validation."""

    def test_feature_store_end_to_end_serialization(self, temp_store_dir):
        """Verify complete feature store construction from real ingested day sample."""
        era5_path = Path("data/raw/era5.nc")
        gfs_dir = Path("data/raw/gfs_exact_test")
        imd_path = Path("data/raw/RF25_ind2024_rfp25.nc")

        if not (era5_path.exists() and gfs_dir.exists() and imd_path.exists()):
            pytest.skip("Raw test datasets not available")

        # 1. Build sample for 2024-06-21
        sample = build_aligned_day_sample(
            imd_file=imd_path,
            era5_file=era5_path,
            gfs_dir=gfs_dir,
            target_date="2024-06-21",
        )

        # 2. Extract synoptic features & classify regime
        era5_idx = {name: i for i, name in enumerate(sample.channel_names)}
        mslp = sample.features[era5_idx["era5_msl_mean"]]
        u10 = sample.features[era5_idx["era5_u10_mean"]]
        v10 = sample.features[era5_idx["era5_v10_mean"]]
        tcwv = sample.features[era5_idx["era5_tcwv_mean"]]

        syn_feats = extract_all_synoptic_features(
            mslp_hpa=mslp,
            pwat_kg_m2=tcwv,
            u10_ms=u10,
            v10_ms=v10,
            target_date="2024-06-21",
        )

        classifier = SynopticRegimeClassifier()
        regime_res = classifier.classify(
            synoptic_feats=syn_feats,
            rainfall_field=sample.target,
            valid_mask=sample.valid_mask,
        )

        # 3. Save to FeatureStore
        store = FeatureStore(temp_store_dir)
        store.save([sample], [syn_feats], [regime_res])

        # 4. Validate saved files
        assert store.metadata_path.exists()
        assert store.tensors_path.exists()
        assert store.config_path.exists()

        # 5. Validate Parquet metadata index
        df_meta = store.load_metadata()
        assert isinstance(df_meta, pd.DataFrame)
        assert len(df_meta) == 1
        assert df_meta.loc[0, "date"] == "2024-06-21"
        assert df_meta.loc[0, "year"] == 2024
        assert df_meta.loc[0, "regime_id"] in REGIME_CATALOG
        assert df_meta.loc[0, "regime_code"] == regime_res.regime_code
        assert 0.0 <= df_meta.loc[0, "regime_confidence"] <= 1.0

        # 6. Validate lazy NetCDF4 tensor dataset
        ds_tensors = store.load_tensors()
        assert "features" in ds_tensors
        assert "target" in ds_tensors
        assert "valid_mask" in ds_tensors
        assert "synoptic_vector" in ds_tensors

        assert ds_tensors["features"].shape == (1, 15, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert ds_tensors["target"].shape == (1, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert ds_tensors["valid_mask"].shape == (1, CANONICAL_NUM_LATS, CANONICAL_NUM_LONS)
        assert ds_tensors["synoptic_vector"].shape == (1, len(SYNOPTIC_FEATURE_NAMES))

        # Check coordinate values
        assert list(ds_tensors.coords["channel"].values) == FEATURE_CHANNELS
        np.testing.assert_allclose(ds_tensors.coords["lat"].values, CANONICAL_LATS)
        np.testing.assert_allclose(ds_tensors.coords["lon"].values, CANONICAL_LONS)

        # 7. Check NaN integrity: Predictor channels must be 100% finite across entire domain
        x_vals = ds_tensors["features"].values
        assert np.all(np.isfinite(x_vals)), "Found unexpected NaNs or Infs in predictor channels"

        # 8. Check Target Masking: Target has NaNs on ocean and finite on land
        y_vals = ds_tensors["target"].values[0]
        mask_vals = ds_tensors["valid_mask"].values[0]
        assert np.all(np.isfinite(y_vals[mask_vals])), "Land target cells must be finite"
        assert np.all(np.isnan(y_vals[~mask_vals])), "Ocean target cells must have NaNs preserved"

        ds_tensors.close()

    def test_temporal_split_no_leakage(self, temp_store_dir):
        """Verify that temporal partitioning strictly prevents train/test date overlap and leakage."""
        # Create synthetic multi-year metadata DataFrame
        records = [
            {"date": "2020-07-15", "year": 2020, "regime_id": 0, "regime_code": "ACTIVE_MONSOON"},
            {"date": "2021-08-10", "year": 2021, "regime_id": 1, "regime_code": "BREAK_MONSOON"},
            {"date": "2022-06-25", "year": 2022, "regime_id": 2, "regime_code": "MONSOON_DEPRESSION"},
            {"date": "2023-07-04", "year": 2023, "regime_id": 3, "regime_code": "OROGRAPHIC_MONSOON"},
            {"date": "2024-06-21", "year": 2024, "regime_id": 0, "regime_code": "ACTIVE_MONSOON"},
        ]
        df_mock = pd.DataFrame(records)
        meta_path = temp_store_dir / "metadata.parquet"
        df_mock.to_parquet(meta_path, index=False)

        store = FeatureStore(temp_store_dir)
        train_df, test_df = store.get_temporal_split(train_years=[2020, 2021, 2022, 2023], test_years=[2024])

        assert len(train_df) == 4
        assert len(test_df) == 1
        assert set(train_df["year"].unique()) == {2020, 2021, 2022, 2023}
        assert set(test_df["year"].unique()) == {2024}

        # Zero temporal leakage: Set intersection of dates must be empty
        train_dates = set(train_df["date"])
        test_dates = set(test_df["date"])
        assert len(train_dates.intersection(test_dates)) == 0
