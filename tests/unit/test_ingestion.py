"""Unit tests for dataset readers, format detection, and inspection utilities."""

from pathlib import Path
import pytest

from src.ingestion.readers import (
    detect_file_format,
    discover_datasets,
    open_imd_dataset,
    open_gfs_dataset,
    setup_eccodes_environment,
)
from src.ingestion.inspector import (
    inspect_imd_dataset,
    inspect_gfs_dataset,
    check_imd_gfs_compatibility,
)


def test_setup_eccodes_environment():
    """Verify that ecCodes setup executes and returns a boolean status."""
    status = setup_eccodes_environment()
    assert isinstance(status, bool)


def test_detect_file_format_nonexistent():
    """Ensure FileNotFoundError is raised when inspecting non-existent file."""
    with pytest.raises(FileNotFoundError):
        detect_file_format("data/raw/non_existent_file.nc")


def test_open_dataset_nonexistent():
    """Ensure FileNotFoundError is raised when opening non-existent datasets."""
    with pytest.raises(FileNotFoundError):
        open_imd_dataset("data/raw/non_existent_imd.nc")

    with pytest.raises(FileNotFoundError):
        open_gfs_dataset("data/raw/non_existent_gfs.grib2")


def test_discover_datasets(project_root: Path):
    """Test raw dataset discovery in the workspace."""
    discovery = discover_datasets(project_root / "data" / "raw")
    assert "imd" in discovery
    assert "gfs" in discovery
    assert "era5" in discovery
    assert discovery["era5"]["status"].startswith("CURRENTLY PROCESSING")


def test_imd_inspection_real_data(project_root: Path):
    """Inspect actual IMD NetCDF dataset if present in data/raw."""
    imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    if not imd_path.exists():
        pytest.skip("IMD dataset RF25_ind2024_rfp25.nc not found in data/raw")

    report = inspect_imd_dataset(imd_path)
    assert report["dataset_type"] == "IMD_OBSERVED_RAINFALL"
    assert "RAINFALL" in report["data_variables"]
    assert report["latitude"]["min"] == 6.5
    assert report["latitude"]["max"] == 38.5
    assert report["latitude"]["spacing"] == 0.25
    assert report["longitude"]["min"] == 66.5
    assert report["longitude"]["max"] == 100.0
    assert report["longitude"]["spacing"] == 0.25
    assert report["time"]["count"] == 366


def test_gfs_inspection_real_data(project_root: Path):
    """Inspect actual GFS GRIB2 dataset if present in data/raw."""
    gfs_path = project_root / "data" / "raw" / "gfs.0p25.2024062100.f000.grib2"
    if not gfs_path.exists():
        pytest.skip("GFS dataset gfs.0p25.2024062100.f000.grib2 not found in data/raw")

    report = inspect_gfs_dataset(gfs_path)
    assert report["dataset_type"] == "NOAA_GFS_FORECAST"
    assert report["file_format"] == "grib2"
    assert report["dimensions"]["latitude"] == 721
    assert report["dimensions"]["longitude"] == 1440
    assert report["latitude"]["step"] == 0.25
    assert report["longitude"]["step"] == 0.25
    assert "prmsl" in report["variables_actually_present"]


def test_compatibility_check(project_root: Path):
    """Test compatibility analyzer between IMD and GFS reports."""
    imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    gfs_path = project_root / "data" / "raw" / "gfs.0p25.2024062100.f000.grib2"

    if not imd_path.exists() or not gfs_path.exists():
        pytest.skip("IMD or GFS dataset not available for compatibility check")

    imd_report = inspect_imd_dataset(imd_path)
    gfs_report = inspect_gfs_dataset(gfs_path)
    compat = check_imd_gfs_compatibility(imd_report, gfs_report)

    assert compat["spatial_overlap"]["lat_overlap"] is True
    assert compat["spatial_overlap"]["lon_overlap"] is True
    assert compat["grid_resolution"]["match"] is True
    assert len(compat["required_alignment_operations"]) > 0


def test_gfs_forecast_slices_inspection(project_root: Path):
    """Test inspection and variable inventory of acquired GFS forecast slices."""
    gfs_test_dir = project_root / "data" / "raw" / "gfs_test"
    if not gfs_test_dir.exists():
        pytest.skip("gfs_test directory does not exist")

    leads = ["f006", "f012", "f018", "f024"]
    for lead in leads:
        gfs_file = gfs_test_dir / f"gfs.0p25.2024062100.{lead}.grib2"
        assert gfs_file.exists(), f"Missing GFS forecast file for {lead}"
        
        report = inspect_gfs_dataset(gfs_file)
        assert report["file_format"] == "grib2"
        assert report["dimensions"]["latitude"] == 721
        assert report["dimensions"]["longitude"] == 1440
        
        # Verify required variables
        vars_present = set(report.get("variables_actually_present", []))
        # Short names in cfgrib/eccodes for these variables
        assert "prmsl" in vars_present or "PRMSL" in vars_present
        assert "2t" in vars_present or "TMP" in vars_present or "t2m" in vars_present
        assert "10u" in vars_present or "UGRD" in vars_present or "u10" in vars_present
        assert "10v" in vars_present or "VGRD" in vars_present or "v10" in vars_present
        assert "tp" in vars_present or "APCP" in vars_present
        assert "pwat" in vars_present or "PWAT" in vars_present or "tcwv" in vars_present

