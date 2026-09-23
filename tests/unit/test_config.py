"""Unit tests for configuration loading, validation, and domain rules."""

import pytest
from pydantic import ValidationError

from src.utils.config import (
    DomainConfig,
    RainfallThresholdsConfig,
    RituGyanConfig,
    load_config,
)


@pytest.mark.unit
def test_default_config_loading(default_config: RituGyanConfig):
    """Test that default_config.yaml loads properly with all required sections."""
    assert default_config.system.project_name == "RituGyan"
    assert default_config.domain.region_name == "India"
    assert default_config.domain.lat_min == 6.0
    assert default_config.domain.lat_max == 38.0
    assert default_config.domain.lon_min == 68.0
    assert default_config.domain.lon_max == 98.0
    assert default_config.domain.grid_resolution == 0.25


@pytest.mark.unit
def test_domain_grid_calculations(default_config: RituGyanConfig):
    """Verify calculated bounding box and grid dimensions."""
    bbox = default_config.domain.bbox
    assert bbox == (6.0, 38.0, 68.0, 98.0)
    # (38.0 - 6.0) / 0.25 + 1 = 32 / 0.25 + 1 = 129
    assert default_config.domain.num_lats == 129
    # (98.0 - 68.0) / 0.25 + 1 = 30 / 0.25 + 1 = 121
    assert default_config.domain.num_lons == 121


@pytest.mark.unit
def test_domain_invalid_lat_lon():
    """Verify validation triggers when min is greater than max."""
    with pytest.raises(ValidationError):
        DomainConfig(lat_min=40.0, lat_max=20.0)


@pytest.mark.unit
def test_rainfall_categorization():
    """Verify IMD rainfall category classification rules."""
    thresholds = RainfallThresholdsConfig()
    assert thresholds.categorize_rainfall(0.0) == "No Rain"
    assert thresholds.categorize_rainfall(1.2) == "Very Light"
    assert thresholds.categorize_rainfall(10.0) == "Light"
    assert thresholds.categorize_rainfall(45.0) == "Moderate"
    assert thresholds.categorize_rainfall(80.0) == "Heavy"
    assert thresholds.categorize_rainfall(150.0) == "Very Heavy"
    assert thresholds.categorize_rainfall(250.0) == "Extremely Heavy"


@pytest.mark.unit
def test_regime_lookup(default_config: RituGyanConfig):
    """Verify lookup methods for synoptic regimes by id and code."""
    assert len(default_config.regimes) == 6
    
    active = default_config.get_regime_by_code("ACTIVE_MONSOON")
    assert active is not None
    assert active.id == 0
    assert active.name == "Active Monsoon"

    depression = default_config.get_regime_by_id(2)
    assert depression is not None
    assert depression.code == "MONSOON_DEPRESSION"

    missing = default_config.get_regime_by_code("NON_EXISTENT")
    assert missing is None


@pytest.mark.unit
def test_alert_level_determination(default_config: RituGyanConfig):
    """Verify IMD district alert logic (Green, Yellow, Orange, Red)."""
    assert default_config.determine_alert_level(prob_heavy=0.1, expected_rain_mm=5.0) == "GREEN"
    assert default_config.determine_alert_level(prob_heavy=0.35) == "YELLOW"
    assert default_config.determine_alert_level(prob_heavy=0.70) == "ORANGE"
    assert default_config.determine_alert_level(prob_very_heavy=0.35) == "ORANGE"
    assert default_config.determine_alert_level(prob_very_heavy=0.65) == "RED"
    assert default_config.determine_alert_level(prob_extremely_heavy=0.35) == "RED"
    assert default_config.determine_alert_level(expected_rain_mm=210.0) == "RED"


@pytest.mark.unit
def test_path_resolution(default_config: RituGyanConfig, project_root):
    """Verify paths are properly resolved against the project root."""
    resolved = default_config.paths.resolve_paths(project_root)
    assert resolved["data_raw"].is_absolute()
    assert resolved["data_raw"] == project_root / "data" / "raw"
    assert resolved["models"] == project_root / "models" / "saved"
