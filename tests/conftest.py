"""Pytest configuration and shared fixtures for RituGyan."""

from pathlib import Path
import pytest

from src.utils.config import RituGyanConfig, load_config


@pytest.fixture
def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).parent.parent.resolve()


@pytest.fixture
def default_config(project_root: Path) -> RituGyanConfig:
    """Load and provide the standard RituGyanConfig instance."""
    config_file = project_root / "configs" / "default_config.yaml"
    return load_config(config_file, force_reload=True)


@pytest.fixture
def sample_india_grid():
    """Generate synthetic spatial coordinates matching the Indian bounding box."""
    import numpy as np
    lats = np.arange(6.0, 38.0 + 0.25, 0.25)
    lons = np.arange(68.0, 98.0 + 0.25, 0.25)
    return {
        "lats": lats,
        "lons": lons,
        "shape": (len(lats), len(lons)),
    }
