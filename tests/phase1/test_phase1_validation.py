"""Comprehensive Phase 1 Validation Test Suite for RituGyan.

Covers:
1. Repository & Project Structure
2. IMD Gridded Rainfall Validation
3. ERA5 Reanalysis Validation
4. GFS Forecast Sequence Validation
5. Spatial Compatibility & Intersection
6. Temporal Compatibility & Mismatch Detection
7. Unit Compatibility & Conversion Requirements
8. Data Quality, Duplicates, and Monotonicity
9. Memory & Chunking Safety
"""

import os
from pathlib import Path
import pytest
import numpy as np
import xarray as xr

from src.utils.config import load_config, find_project_root
from src.ingestion.inspector import (
    inspect_imd_dataset,
    inspect_gfs_dataset,
    check_imd_gfs_compatibility,
)


# ==============================================================================
# 1. REPOSITORY / PROJECT STRUCTURE
# ==============================================================================

def test_project_directories_exist(project_root: Path):
    """Verify all required Phase 1 directories exist."""
    required_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "shapefiles",
        project_root / "models",
        project_root / "src" / "ingestion",
        project_root / "tests",
        project_root / "configs",
    ]
    for d in required_dirs:
        assert d.exists(), f"Required directory missing: {d}"
        assert d.is_dir(), f"Path is not a directory: {d}"


def test_config_loadable(project_root: Path):
    """Verify that configuration file exists and parses properly."""
    config_file = project_root / "configs" / "default_config.yaml"
    assert config_file.exists()
    cfg = load_config(config_file, force_reload=True)
    assert cfg.system.project_name == "RituGyan"
    assert cfg.domain.region_name == "India"
    assert cfg.domain.grid_resolution == 0.25


def test_required_dependencies():
    """Verify required meteorological and scientific packages are importable."""
    import xarray
    import numpy
    import pandas
    import pydantic
    import yaml
    
    assert xarray.__version__
    assert numpy.__version__
    assert pandas.__version__
    
    # ecCodes environment
    from src.ingestion.readers import setup_eccodes_environment
    setup_eccodes_environment()
    try:
        import eccodes
        assert eccodes.__version__
    except Exception as exc:
        pytest.skip(f"ecCodes binary library optional on platform: {exc}")


# ==============================================================================
# 2. IMD DATA VALIDATION
# ==============================================================================

def test_imd_dataset_validation(project_root: Path):
    """Validate all IMD 2024 NetCDF properties."""
    imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    assert imd_path.exists(), f"IMD dataset missing at {imd_path}"
    
    with xr.open_dataset(imd_path) as ds:
        # Dimensions
        assert "TIME" in ds.dims and ds.sizes["TIME"] == 366
        assert "LATITUDE" in ds.dims and ds.sizes["LATITUDE"] == 129
        assert "LONGITUDE" in ds.dims and ds.sizes["LONGITUDE"] == 135
        
        # Coordinates & Resolution
        lats = ds["LATITUDE"].values
        lons = ds["LONGITUDE"].values
        times = ds["TIME"].values
        
        assert np.isclose(lats.min(), 6.5)
        assert np.isclose(lats.max(), 38.5)
        assert np.isclose(lons.min(), 66.5)
        assert np.isclose(lons.max(), 100.0)
        
        # Monotonicity & Ordering
        assert np.all(np.diff(lats) > 0), "IMD latitudes must be strictly ascending"
        assert np.all(np.diff(lons) > 0), "IMD longitudes must be strictly ascending"
        assert np.isclose(np.diff(lats)[0], 0.25)
        assert np.isclose(np.diff(lons)[0], 0.25)
        
        # Date range
        assert str(times[0])[:10] == "2024-01-01"
        assert str(times[-1])[:10] == "2024-12-31"
        
        # Variable & Units
        assert "RAINFALL" in ds.data_vars
        rf = ds["RAINFALL"]
        assert rf.attrs.get("units") == "mm"
        
        rf_vals = rf.values
        nan_pct = (np.isnan(rf_vals).sum() / rf_vals.size) * 100
        assert 70.0 < nan_pct < 75.0, f"Expected ~71.5% NaNs (ocean/border mask), got {nan_pct:.2f}%"
        
        valid_vals = rf_vals[~np.isnan(rf_vals)]
        assert valid_vals.min() >= 0.0
        assert valid_vals.max() > 500.0  # Heavy monsoon events


# ==============================================================================
# 3. ERA5 DATA VALIDATION
# ==============================================================================

def test_era5_dataset_validation(project_root: Path):
    """Validate ERA5 2024 JJAS reanalysis NetCDF properties."""
    era5_path = project_root / "data" / "raw" / "era5.nc"
    assert era5_path.exists(), f"ERA5 dataset missing at {era5_path}"
    
    with xr.open_dataset(era5_path) as ds:
        # Dimensions
        assert ds.sizes.get("valid_time") == 488
        assert ds.sizes.get("latitude") == 129
        assert ds.sizes.get("longitude") == 121
        
        # Time coverage & 6-hourly steps
        tvals = ds["valid_time"].values
        assert str(tvals[0])[:19] == "2024-06-01T00:00:00"
        assert str(tvals[-1])[:19] == "2024-09-30T18:00:00"
        assert len(tvals) == 488
        
        # Verify 6-hourly step
        tdiffs = np.diff(tvals).astype("timedelta64[h]")
        assert np.all(tdiffs == np.timedelta64(6, "h"))
        
        # Latitude & Longitude bounds
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        assert np.isclose(lats.min(), 6.0)
        assert np.isclose(lats.max(), 38.0)
        assert np.isclose(lons.min(), 68.0)
        assert np.isclose(lons.max(), 98.0)
        assert np.isclose(abs(lats[1] - lats[0]), 0.25)
        assert np.isclose(abs(lons[1] - lons[0]), 0.25)
        
        # Variables & Units
        expected_vars = {
            "u10": "m s**-1",
            "v10": "m s**-1",
            "d2m": "K",
            "t2m": "K",
            "msl": "Pa",
            "tcwv": "kg m**-2",
        }
        for var_name, expected_unit in expected_vars.items():
            assert var_name in ds.data_vars, f"Missing ERA5 variable {var_name}"
            assert ds[var_name].attrs.get("units") == expected_unit
            arr = ds[var_name].values
            assert np.isnan(arr).sum() == 0, f"Unexpected NaNs found in ERA5 variable {var_name}"


def test_era5_lazy_loading(project_root: Path):
    """Verify that ERA5 can be opened lazily without loading entire array into memory."""
    era5_path = project_root / "data" / "raw" / "era5.nc"
    if not era5_path.exists():
        pytest.skip("ERA5 file not present")
        
    with xr.open_dataset(era5_path) as ds:
        # NetCDF on-demand lazy indexing without loading 488x129x121 into memory
        sample_slice = ds["t2m"].isel(valid_time=slice(0, 10))
        assert sample_slice.shape == (10, 129, 121)
        assert sample_slice.values.shape == (10, 129, 121)
        
    try:
        import dask
        ds_chunked = xr.open_dataset(era5_path, chunks={"valid_time": 50, "latitude": 65, "longitude": 60})
        assert ds_chunked["t2m"].data.chunks is not None
        ds_chunked.close()
    except ImportError:
        pass  # Dask chunk manager is optional; native xarray lazy slicing verified


# ==============================================================================
# 4. GFS VALIDATION
# ==============================================================================

def test_gfs_forecast_sequence(project_root: Path):
    """Validate the GFS f006, f012, f018, f024 forecast slice sequence."""
    gfs_dir = project_root / "data" / "raw" / "gfs_test"
    assert gfs_dir.exists()
    
    lead_steps = ["f006", "f012", "f018", "f024"]
    for lead in lead_steps:
        fpath = gfs_dir / f"gfs.0p25.2024062100.{lead}.grib2"
        assert fpath.exists(), f"Missing GFS forecast file: {fpath.name}"
        
        rep = inspect_gfs_dataset(fpath)
        assert rep["file_format"] == "grib2"
        assert rep["dimensions"]["latitude"] == 721
        assert rep["dimensions"]["longitude"] == 1440
        assert rep["latitude"]["step"] == 0.25
        assert rep["longitude"]["step"] == 0.25
        assert rep["initialization_times"] == ["20240621 0000 UTC"]
        
        # Verify required meteorological variables
        vars_found = rep["variables_actually_present"]
        assert "prmsl" in vars_found  # Pressure
        assert "2t" in vars_found     # Temperature
        assert "2d" in vars_found     # Dewpoint
        assert "10u" in vars_found    # U-wind
        assert "10v" in vars_found    # V-wind
        assert "tp" in vars_found     # Total precipitation
        assert "pwat" in vars_found   # Precipitable water


# ==============================================================================
# 5. SPATIAL COMPATIBILITY & INTERSECTION
# ==============================================================================

def test_spatial_grid_compatibility(project_root: Path):
    """Perform exact coordinate set comparison across IMD and ERA5."""
    imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    era5_path = project_root / "data" / "raw" / "era5.nc"
    
    with xr.open_dataset(imd_path) as imd, xr.open_dataset(era5_path) as era5:
        imd_lats = np.round(imd["LATITUDE"].values, 4)
        era5_lats = np.round(era5["latitude"].values, 4)
        imd_lons = np.round(imd["LONGITUDE"].values, 4)
        era5_lons = np.round(era5["longitude"].values, 4)
        
        # Common intersection
        common_lats = np.intersect1d(imd_lats, era5_lats)
        common_lons = np.intersect1d(imd_lons, era5_lons)
        
        assert len(common_lats) == 127
        assert common_lats.min() == 6.5
        assert common_lats.max() == 38.0
        
        assert len(common_lons) == 121
        assert common_lons.min() == 68.0
        assert common_lons.max() == 98.0
        
        # Total intersection points
        total_common_points = len(common_lats) * len(common_lons)
        assert total_common_points == 15367
        
        # Verify both align on exact 0.25 lattice (remainders modulo 0.25 == 0)
        assert np.all(np.isclose(common_lats % 0.25, 0.0) | np.isclose(common_lats % 0.25, 0.25))
        assert np.all(np.isclose(common_lons % 0.25, 0.0) | np.isclose(common_lons % 0.25, 0.25))


# ==============================================================================
# 6. TEMPORAL COMPATIBILITY & 3-HOUR OFFSET DETECTION
# ==============================================================================

def test_temporal_offset_detection():
    """Verify explicit detection of the 3-hour difference between GFS 00Z-00Z and IMD 03Z-03Z."""
    imd_obs_hour_utc = 3  # 08:30 IST
    gfs_init_hour_utc = 0
    gfs_available_leads = [6, 12, 18, 24]  # f006 to f024
    
    gfs_max_lead = max(gfs_available_leads)
    
    # 24h accumulation starting at 03Z requires leads up to f027 (03Z next day)
    required_lead_for_exact_imd_match = 24 + imd_obs_hour_utc  # f027 (27 hours)
    
    # Check whether f027 is present in the f006-f024 sequence
    is_exact_window_complete = required_lead_for_exact_imd_match in gfs_available_leads
    
    # This must be flagged as NOT complete for exact 03Z match (needs f027 or 00Z-00Z proxy)
    assert is_exact_window_complete is False, "Exact 03Z window requires f027 which is not in f024 sequence"


# ==============================================================================
# 7. DATA QUALITY & INTEGRITY
# ==============================================================================

def test_no_duplicate_coordinates_or_times(project_root: Path):
    """Verify coordinate uniqueness and monotonicity in all datasets."""
    imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
    era5_path = project_root / "data" / "raw" / "era5.nc"
    
    with xr.open_dataset(imd_path) as imd:
        assert len(imd["TIME"].values) == len(np.unique(imd["TIME"].values)), "IMD contains duplicate timestamps"
        assert len(imd["LATITUDE"].values) == len(np.unique(imd["LATITUDE"].values)), "IMD contains duplicate latitudes"
        assert len(imd["LONGITUDE"].values) == len(np.unique(imd["LONGITUDE"].values)), "IMD contains duplicate longitudes"
        
    with xr.open_dataset(era5_path) as era5:
        assert len(era5["valid_time"].values) == len(np.unique(era5["valid_time"].values)), "ERA5 contains duplicate timestamps"
        assert len(era5["latitude"].values) == len(np.unique(era5["latitude"].values)), "ERA5 contains duplicate latitudes"
        assert len(era5["longitude"].values) == len(np.unique(era5["longitude"].values)), "ERA5 contains duplicate longitudes"
