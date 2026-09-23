"""Unit tests for repository layout, logging, and package imports."""

from pathlib import Path
import pytest

from src.utils.logger import get_logger


@pytest.mark.unit
def test_directory_structure_exists(project_root: Path):
    """Verify that all standard directories mandated by the implementation plan exist."""
    required_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "shapefiles",
        project_root / "data" / "static",
        project_root / "models" / "saved",
        project_root / "src" / "ingestion",
        project_root / "src" / "features",
        project_root / "src" / "models" / "stage1_regime",
        project_root / "src" / "models" / "stage2_correction",
        project_root / "src" / "evaluation",
        project_root / "src" / "api" / "routes",
        project_root / "src" / "ui",
        project_root / "src" / "utils",
        project_root / "configs",
        project_root / "notebooks",
        project_root / "tests" / "unit",
        project_root / "tests" / "integration",
    ]
    for directory in required_dirs:
        assert directory.exists(), f"Missing required directory: {directory}"
        assert directory.is_dir(), f"Path is not a directory: {directory}"


@pytest.mark.unit
def test_module_imports():
    """Verify that all top-level subpackages can be cleanly imported."""
    import src
    import src.ingestion
    import src.features
    import src.models
    import src.models.stage1_regime
    import src.models.stage2_correction
    import src.evaluation
    import src.api
    import src.ui
    import src.utils

    assert src.__version__ == "0.1.0"


@pytest.mark.unit
def test_logger_functionality(tmp_path: Path):
    """Verify structured logger creation and file writing."""
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger", log_level="DEBUG", log_file=log_file)
    logger.info("Test info message")
    logger.debug("Test debug message")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test info message" in content
    assert "Test debug message" in content
