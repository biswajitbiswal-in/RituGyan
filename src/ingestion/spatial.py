"""Spatial indexing and coordinate alignment utilities for RituGyan.

Performs pure index-based coordinate slicing for IMD, ERA5, and GFS grids
to extract the common Indian subgrid without interpolation or resampling.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import numpy as np
import xarray as xr

# Canonical common subgrid constants for India domain
CANONICAL_LAT_MIN: float = 6.50
CANONICAL_LAT_MAX: float = 38.00
CANONICAL_LON_MIN: float = 68.00
CANONICAL_LON_MAX: float = 98.00
CANONICAL_GRID_STEP: float = 0.25
CANONICAL_NUM_LATS: int = 127
CANONICAL_NUM_LONS: int = 121
CANONICAL_TOTAL_POINTS: int = CANONICAL_NUM_LATS * CANONICAL_NUM_LONS  # 15,367

# Expected canonical coordinate vectors (strictly ascending)
CANONICAL_LATS: np.ndarray = np.linspace(
    CANONICAL_LAT_MIN, CANONICAL_LAT_MAX, CANONICAL_NUM_LATS, dtype=np.float64
)
CANONICAL_LONS: np.ndarray = np.linspace(
    CANONICAL_LON_MIN, CANONICAL_LON_MAX, CANONICAL_NUM_LONS, dtype=np.float64
)

# GFS 0.25° Global grid slicing constants (lat descending 90 to -90, lon ascending 0 to 359.75)
GFS_GLOBAL_NUM_LATS: int = 721
GFS_GLOBAL_NUM_LONS: int = 1440
GFS_LAT_START_IDX: int = int(round((90.0 - CANONICAL_LAT_MAX) / CANONICAL_GRID_STEP))  # 208 (for 38.0°N)
GFS_LAT_END_IDX: int = int(round((90.0 - CANONICAL_LAT_MIN) / CANONICAL_GRID_STEP))    # 334 (for 6.5°N)
GFS_LON_START_IDX: int = int(round(CANONICAL_LON_MIN / CANONICAL_GRID_STEP))           # 272 (for 68.0°E)
GFS_LON_END_IDX: int = int(round(CANONICAL_LON_MAX / CANONICAL_GRID_STEP))             # 392 (for 98.0°E)


def get_canonical_grid() -> Dict[str, Any]:
    """Return dictionary with canonical spatial grid coordinates and metadata.
    
    Returns:
        Dict with canonical latitudes, longitudes, shapes, and boundaries.
    """
    return {
        "lats": CANONICAL_LATS.copy(),
        "lons": CANONICAL_LONS.copy(),
        "shape": (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS),
        "num_lats": CANONICAL_NUM_LATS,
        "num_lons": CANONICAL_NUM_LONS,
        "total_points": CANONICAL_TOTAL_POINTS,
        "lat_range": (CANONICAL_LAT_MIN, CANONICAL_LAT_MAX),
        "lon_range": (CANONICAL_LON_MIN, CANONICAL_LON_MAX),
        "grid_step": CANONICAL_GRID_STEP,
        "lat_ordering": "ascending",
        "lon_ordering": "ascending",
    }


def extract_imd_subgrid(
    ds: xr.Dataset,
    lat_name: str = "LATITUDE",
    lon_name: str = "LONGITUDE",
) -> xr.Dataset:
    """Extract common 127x121 subgrid from an IMD NetCDF Dataset via index slicing.
    
    Args:
        ds: IMD xarray Dataset.
        lat_name: Name of the latitude coordinate (default 'LATITUDE').
        lon_name: Name of the longitude coordinate (default 'LONGITUDE').
        
    Returns:
        xr.Dataset sliced to [6.50°N..38.00°N, 68.00°E..98.00°E].
    """
    sub_ds = ds.sel(
        {
            lat_name: slice(CANONICAL_LAT_MIN, CANONICAL_LAT_MAX),
            lon_name: slice(CANONICAL_LON_MIN, CANONICAL_LON_MAX),
        }
    )
    # Verify exact coordinate equality
    assert sub_ds[lat_name].shape[0] == CANONICAL_NUM_LATS, (
        f"IMD extracted lat shape {sub_ds[lat_name].shape} != {CANONICAL_NUM_LATS}"
    )
    assert sub_ds[lon_name].shape[0] == CANONICAL_NUM_LONS, (
        f"IMD extracted lon shape {sub_ds[lon_name].shape} != {CANONICAL_NUM_LONS}"
    )
    assert np.allclose(sub_ds[lat_name].values, CANONICAL_LATS), "IMD latitudes mismatch canonical grid"
    assert np.allclose(sub_ds[lon_name].values, CANONICAL_LONS), "IMD longitudes mismatch canonical grid"
    return sub_ds


def extract_era5_subgrid(
    ds: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    ascending_lat: bool = True,
) -> xr.Dataset:
    """Extract common 127x121 subgrid from an ERA5 NetCDF Dataset via index slicing.
    
    ERA5 natively stores latitudes in descending order (38.0 down to 6.0).
    When ascending_lat is True (default), the latitude axis is re-ordered to ascending.
    
    Args:
        ds: ERA5 xarray Dataset.
        lat_name: Name of latitude coordinate (default 'latitude').
        lon_name: Name of longitude coordinate (default 'longitude').
        ascending_lat: If True, re-orders latitude to ascending.
        
    Returns:
        xr.Dataset sliced to [6.50°N..38.00°N, 68.00°E..98.00°E].
    """
    # ERA5 native latitude is descending (38.0 to 6.0)
    sub_ds = ds.sel(
        {
            lat_name: slice(CANONICAL_LAT_MAX, CANONICAL_LAT_MIN),
            lon_name: slice(CANONICAL_LON_MIN, CANONICAL_LON_MAX),
        }
    )
    if ascending_lat:
        sub_ds = sub_ds.sortby(lat_name, ascending=True)

    assert sub_ds[lat_name].shape[0] == CANONICAL_NUM_LATS, (
        f"ERA5 extracted lat shape {sub_ds[lat_name].shape} != {CANONICAL_NUM_LATS}"
    )
    assert sub_ds[lon_name].shape[0] == CANONICAL_NUM_LONS, (
        f"ERA5 extracted lon shape {sub_ds[lon_name].shape} != {CANONICAL_NUM_LONS}"
    )
    if ascending_lat:
        assert np.allclose(sub_ds[lat_name].values, CANONICAL_LATS), "ERA5 latitudes mismatch canonical grid"
    assert np.allclose(sub_ds[lon_name].values, CANONICAL_LONS), "ERA5 longitudes mismatch canonical grid"
    return sub_ds


def extract_gfs_subgrid_array(
    global_array: np.ndarray,
    ascending_lat: bool = True,
) -> np.ndarray:
    """Extract common 127x121 subgrid from a 2D global GFS array (721x1440) via pure index slicing.
    
    GFS global arrays are row-major (721 lats descending 90 to -90, 1440 lons ascending 0 to 359.75).
    
    Args:
        global_array: 2D numpy array of shape (721, 1440).
        ascending_lat: If True, reverses the latitude axis to ascending (6.5 to 38.0).
        
    Returns:
        2D numpy array of shape (127, 121) covering [6.50°N..38.00°N, 68.00°E..98.00°E].
    """
    if global_array.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (721, 1440), got shape {global_array.shape}")
    if global_array.shape != (GFS_GLOBAL_NUM_LATS, GFS_GLOBAL_NUM_LONS):
        raise ValueError(
            f"Expected shape ({GFS_GLOBAL_NUM_LATS}, {GFS_GLOBAL_NUM_LONS}), got {global_array.shape}"
        )

    # Slice rows (latitudes: 208..334 inclusive) and cols (longitudes: 272..392 inclusive)
    sliced = global_array[
        GFS_LAT_START_IDX : GFS_LAT_END_IDX + 1,
        GFS_LON_START_IDX : GFS_LON_END_IDX + 1,
    ]

    if ascending_lat:
        # Flip rows to ascending (6.50°N at index 0, 38.00°N at index 126)
        sliced = sliced[::-1, :]

    assert sliced.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS), (
        f"GFS sliced subgrid shape {sliced.shape} != ({CANONICAL_NUM_LATS}, {CANONICAL_NUM_LONS})"
    )
    return np.ascontiguousarray(sliced)
