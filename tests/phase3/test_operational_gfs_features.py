"""Focused tests for the versioned GFS-only Stage 1 feature representation."""

from pathlib import Path

import numpy as np
import pytest

from src.features.feature_store import SYNOPTIC_FEATURE_NAMES
from src.features.operational_gfs import (
    CHANNEL_SOURCE_MAP,
    OPERATIONAL_CHANNELS,
    build_operational_channel_tensor,
    build_operational_day_features,
    build_operational_synoptic_features,
    operational_feature_names,
    synoptic_vector,
)
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS


def _gfs_fixture() -> dict[str, np.ndarray]:
    shape = (len(CANONICAL_LATS), len(CANONICAL_LONS))
    return {
        "gfs_tp_24h": np.full(shape, 5.0, dtype=np.float32),
        "gfs_prmsl_mean": np.full(shape, 1000.0, dtype=np.float32),
        "gfs_2t_mean": np.full(shape, 28.0, dtype=np.float32),
        "gfs_2d_mean": np.full(shape, 22.0, dtype=np.float32),
        "gfs_10u_mean": np.full(shape, 4.0, dtype=np.float32),
        "gfs_10v_mean": np.full(shape, 3.0, dtype=np.float32),
        "gfs_pwat_mean": np.full(shape, 45.0, dtype=np.float32),
        "gfs_wind_speed": np.full(shape, 5.0, dtype=np.float32),
    }


def test_operational_layout_is_122_features_and_stable():
    names = operational_feature_names()
    assert len(OPERATIONAL_CHANNELS) == 15
    assert len(SYNOPTIC_FEATURE_NAMES) == 17
    assert len(names) == 122
    assert names[:15] == tuple(
        f"spatial__{channel}__domain__area_weighted_mean" for channel in OPERATIONAL_CHANNELS
    )
    assert names[-17] == "synoptic__trough_mean_latitude"


def test_former_era5_slots_map_to_gfs_without_era5_values():
    gfs = _gfs_fixture()
    tensor = build_operational_channel_tensor(gfs)
    assert tensor.shape == (15, 127, 121)
    for index, channel in enumerate(OPERATIONAL_CHANNELS):
        np.testing.assert_array_equal(tensor[index], gfs[CHANNEL_SOURCE_MAP[channel]])
    assert np.all(np.isfinite(tensor))


def test_gfs_synoptic_vector_is_finite_and_deterministic():
    gfs = _gfs_fixture()
    first = synoptic_vector(build_operational_synoptic_features(gfs, "2024-06-21"))
    second = synoptic_vector(build_operational_synoptic_features(gfs, "2024-06-21"))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (17,)
    assert np.all(np.isfinite(first))


def test_operational_day_builder_does_not_call_era5(monkeypatch):
    expected_tensor = np.zeros((15, 127, 121), dtype=np.float32)
    expected_synoptic = np.ones(17, dtype=np.float32)

    def fake_loader(gfs_dir, date_str, cycle_str):
        return _gfs_fixture()

    monkeypatch.setattr("src.features.operational_gfs.load_gfs_forecast_predictors", fake_loader)
    tensor, vector = build_operational_day_features("unused", "2024-06-21")
    assert tensor.shape == expected_tensor.shape
    assert vector.shape == expected_synoptic.shape
    assert np.all(np.isfinite(tensor))
    assert np.all(np.isfinite(vector))


@pytest.mark.integration
def test_real_single_cycle_gfs_build_is_era5_free(project_root: Path):
    gfs_dir = project_root / "data/raw/gfs_single_cycle_test"
    if not gfs_dir.exists():
        pytest.skip("GFS single-cycle fixture unavailable")
    tensor, vector = build_operational_day_features(gfs_dir, "2024-06-21")
    assert tensor.shape == (15, 127, 121)
    assert vector.shape == (17,)
    assert np.all(np.isfinite(tensor))
    assert np.all(np.isfinite(vector))
