"""Automated validation tests for Phase 2B: Spatial Alignment Specification & Common Subgrid Indexing.

Validates coordinate arrays, resolution, monotonicity, orientation handling,
longitude indexing, no-interpolation guarantee, and exact numerical equality
across:
1. IMD gridded daily rainfall
2. ERA5 6-hourly reanalysis
3. GFS 0.25° forecast fields
"""

from pathlib import Path
import pytest
import xarray as xr
import numpy as np

from src.ingestion.readers import setup_eccodes_environment
from src.ingestion.spatial import (
    CANONICAL_LAT_MIN,
    CANONICAL_LAT_MAX,
    CANONICAL_LON_MIN,
    CANONICAL_LON_MAX,
    CANONICAL_GRID_STEP,
    CANONICAL_NUM_LATS,
    CANONICAL_NUM_LONS,
    CANONICAL_TOTAL_POINTS,
    CANONICAL_LATS,
    CANONICAL_LONS,
    get_canonical_grid,
    extract_imd_subgrid,
    extract_era5_subgrid,
    extract_gfs_subgrid_array,
)


@pytest.fixture(scope="module")
def eccodes_ready():
    """Ensure ecCodes environment is configured."""
    ready = setup_eccodes_environment()
    assert ready, "ecCodes C-library must be reachable for GRIB2 inspection"
    import eccodes
    return eccodes


class TestSpatialAlignment:
    """Validate spatial alignment specification and index-based extraction."""

    def test_canonical_grid_specification(self):
        """Validate canonical Indian subgrid constants and coordinate arrays."""
        grid = get_canonical_grid()
        assert grid["num_lats"] == 127
        assert grid["num_lons"] == 121
        assert grid["total_points"] == 15367
        assert grid["shape"] == (127, 121)
        assert grid["grid_step"] == 0.25

        lats = grid["lats"]
        lons = grid["lons"]

        # Strictly ascending monotonicity
        assert np.all(np.diff(lats) > 0), "Canonical latitudes must be strictly ascending"
        assert np.all(np.diff(lons) > 0), "Canonical longitudes must be strictly ascending"

        # Step size equality
        assert np.allclose(np.diff(lats), 0.25), "Latitude spacing must be exactly 0.25°"
        assert np.allclose(np.diff(lons), 0.25), "Longitude spacing must be exactly 0.25°"

        # Bounds
        assert lats[0] == 6.50
        assert lats[-1] == 38.00
        assert lons[0] == 68.00
        assert lons[-1] == 98.00

        # No half-grid offsets: every coordinate must be an exact multiple of 0.25
        assert np.all(np.isclose(np.remainder(lats, 0.25), 0.0) | np.isclose(np.remainder(lats, 0.25), 0.25))
        assert np.all(np.isclose(np.remainder(lons, 0.25), 0.0) | np.isclose(np.remainder(lons, 0.25), 0.25))

    def test_imd_spatial_indexing_and_extraction(self, project_root: Path):
        """Validate IMD dataset coordinate extraction and bounds."""
        imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
        assert imd_path.exists(), f"IMD dataset missing: {imd_path}"

        ds = xr.open_dataset(imd_path)
        raw_lats = ds["LATITUDE"].values
        raw_lons = ds["LONGITUDE"].values

        # Check raw coordinates
        assert raw_lats.shape == (129,)
        assert raw_lons.shape == (135,)
        assert np.all(np.diff(raw_lats) > 0), "Raw IMD latitude must be ascending"
        assert np.all(np.diff(raw_lons) > 0), "Raw IMD longitude must be ascending"

        # Extract subgrid via pure indexing
        sub_ds = extract_imd_subgrid(ds)
        sub_lats = sub_ds["LATITUDE"].values
        sub_lons = sub_ds["LONGITUDE"].values

        assert sub_lats.shape == (127,)
        assert sub_lons.shape == (121,)
        assert np.array_equal(sub_lats, CANONICAL_LATS)
        assert np.array_equal(sub_lons, CANONICAL_LONS)

        # Confirm exact sample values match raw dataset without interpolation
        raw_slice = ds["RAINFALL"].sel(
            LATITUDE=slice(CANONICAL_LAT_MIN, CANONICAL_LAT_MAX),
            LONGITUDE=slice(CANONICAL_LON_MIN, CANONICAL_LON_MAX),
        )
        assert np.array_equal(sub_ds["RAINFALL"].values, raw_slice.values, equal_nan=True)
        ds.close()

    def test_era5_spatial_indexing_and_orientation(self, project_root: Path):
        """Validate ERA5 coordinate slicing and latitude reordering."""
        era5_path = project_root / "data" / "raw" / "era5.nc"
        assert era5_path.exists(), f"ERA5 dataset missing: {era5_path}"

        ds = xr.open_dataset(era5_path)
        raw_lats = ds["latitude"].values
        raw_lons = ds["longitude"].values

        # Check raw ERA5 coordinates: latitude is descending
        assert raw_lats.shape == (129,)
        assert raw_lons.shape == (121,)
        assert np.all(np.diff(raw_lats) < 0), "Raw ERA5 latitude must be descending"
        assert np.all(np.diff(raw_lons) > 0), "Raw ERA5 longitude must be ascending"

        # Extract subgrid with ascending_lat=True
        sub_ds = extract_era5_subgrid(ds, ascending_lat=True)
        sub_lats = sub_ds["latitude"].values
        sub_lons = sub_ds["longitude"].values

        assert sub_lats.shape == (127,)
        assert sub_lons.shape == (121,)
        assert np.all(np.diff(sub_lats) > 0), "Extracted ERA5 latitude must be ascending"
        assert np.allclose(sub_lats, CANONICAL_LATS)
        assert np.allclose(sub_lons, CANONICAL_LONS)

        # Confirm exact value preservation (no interpolation)
        # Sliced at index level: raw latitude from index 0 (38.0) to 126 (6.5)
        raw_data = ds["t2m"].isel(valid_time=0, latitude=slice(0, 127), longitude=slice(0, 121)).values
        extracted_data = sub_ds["t2m"].isel(valid_time=0).values
        # Reversing raw_data latitude should match extracted_data exactly
        assert np.allclose(extracted_data, raw_data[::-1, :])
        ds.close()

    def test_gfs_spatial_indexing_and_longitude_handling(
        self, eccodes_ready, project_root: Path
    ):
        """Validate GFS global grid slicing, longitude indexing, and orientation."""
        eccodes = eccodes_ready
        gfs_path = project_root / "data" / "raw" / "gfs_exact_test" / "gfs.0p25.2024062100.f003.grib2"
        assert gfs_path.exists(), f"GFS lead file missing: {gfs_path}"

        with open(gfs_path, "rb") as fp:
            gid = eccodes.codes_grib_new_from_file(fp)
            assert gid is not None

            # Verify global grid definitions
            ni = eccodes.codes_get(gid, "Ni")
            nj = eccodes.codes_get(gid, "Nj")
            assert ni == 1440
            assert nj == 721

            lat_first = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            lat_last = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
            lon_first = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            lon_last = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")

            assert lat_first == 90.0 and lat_last == -90.0
            assert lon_first == 0.0 and lon_last == 359.75

            # Extract 2m temperature field
            raw_vals = eccodes.codes_get_values(gid)
            global_grid = raw_vals.reshape(721, 1440)

            # Subgrid extraction
            subgrid = extract_gfs_subgrid_array(global_grid, ascending_lat=True)
            assert subgrid.shape == (127, 121)

            # Verify GFS coordinate reconstruction
            gfs_lats_global = np.linspace(90.0, -90.0, 721)
            gfs_lons_global = np.linspace(0.0, 359.75, 1440)

            # Slicing indices: 208..334 for lat (38.0..6.5), 272..392 for lon (68.0..98.0)
            gfs_extracted_lats = gfs_lats_global[208:335][::-1]
            gfs_extracted_lons = gfs_lons_global[272:393]

            assert np.allclose(gfs_extracted_lats, CANONICAL_LATS)
            assert np.allclose(gfs_extracted_lons, CANONICAL_LONS)

            # Confirm exact value extraction without interpolation
            expected_sample = global_grid[334, 272]  # lat index 334 (6.5°N), lon index 272 (68.0°E)
            assert subgrid[0, 0] == expected_sample

            eccodes.codes_release(gid)

    def test_cross_dataset_coordinate_identity(self, project_root: Path):
        """Verify strict 1-to-1 numerical identity of coordinates across IMD, ERA5, and GFS."""
        # 1. IMD
        imd_ds = xr.open_dataset(project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc")
        imd_sub = extract_imd_subgrid(imd_ds)
        imd_lats = imd_sub["LATITUDE"].values
        imd_lons = imd_sub["LONGITUDE"].values

        # 2. ERA5
        era5_ds = xr.open_dataset(project_root / "data" / "raw" / "era5.nc")
        era5_sub = extract_era5_subgrid(era5_ds, ascending_lat=True)
        era5_lats = era5_sub["latitude"].values
        era5_lons = era5_sub["longitude"].values

        # 3. GFS
        gfs_lats = np.linspace(90.0, -90.0, 721)[208:335][::-1]
        gfs_lons = np.linspace(0.0, 359.75, 1440)[272:393]

        # Numerical identity checks: max absolute difference must be 0.0
        assert np.max(np.abs(imd_lats - era5_lats)) == 0.0, "IMD vs ERA5 latitude discrepancy"
        assert np.max(np.abs(imd_lons - era5_lons)) == 0.0, "IMD vs ERA5 longitude discrepancy"
        assert np.max(np.abs(imd_lats - gfs_lats)) == 0.0, "IMD vs GFS latitude discrepancy"
        assert np.max(np.abs(imd_lons - gfs_lons)) == 0.0, "IMD vs GFS longitude discrepancy"
        assert np.max(np.abs(era5_lats - gfs_lats)) == 0.0, "ERA5 vs GFS latitude discrepancy"
        assert np.max(np.abs(era5_lons - gfs_lons)) == 0.0, "ERA5 vs GFS longitude discrepancy"

        imd_ds.close()
        era5_ds.close()
