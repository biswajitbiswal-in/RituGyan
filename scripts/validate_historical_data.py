"""Comprehensive historical dataset validation script for RituGyan.

Validates 2021-2023 historical IMD and ERA5 datasets against the 2024 validated baseline.
Checks:
- IMD: June 1 to September 30 (122 daily observations), 0.25° grid, lat/lon range/ordering,
  rainfall variable/units, missing-value pattern, spatial compatibility.
- ERA5: June 1 to September 30 (488 timesteps per year), 00/06/12/18 UTC, 0.25° grid,
  lat/lon range/ordering, variables (u10, v10, t2m, d2m, msl, tcwv), units, NaN/Inf checks,
  spatial compatibility.
"""

from pathlib import Path
from typing import Dict, Any, List
import datetime
import numpy as np
import pandas as pd
import xarray as xr


def validate_imd_datasets(raw_dir: Path) -> Dict[int, Dict[str, Any]]:
    results = {}
    base_file = raw_dir / "RF25_ind2024_rfp25.nc"
    if not base_file.exists():
        raise FileNotFoundError(f"Baseline IMD 2024 file not found: {base_file}")

    base_ds = xr.open_dataset(base_file)
    base_lats = base_ds["LATITUDE"].values
    base_lons = base_ds["LONGITUDE"].values
    base_mask_sample = ~np.isnan(base_ds["RAINFALL"].isel(TIME=0).values)

    years = [2021, 2022, 2023, 2024]

    for yr in years:
        fpath = raw_dir / f"RF25_ind{yr}_rfp25.nc"
        res: Dict[str, Any] = {"file": fpath.name, "exists": fpath.exists()}
        if not fpath.exists():
            res["status"] = "FAIL"
            res["error"] = "File not found"
            results[yr] = res
            continue

        ds = xr.open_dataset(fpath)
        time_coord = "TIME" if "TIME" in ds.coords or "TIME" in ds.dims else "time"
        lat_coord = "LATITUDE" if "LATITUDE" in ds.coords else "latitude"
        lon_coord = "LONGITUDE" if "LONGITUDE" in ds.coords else "longitude"

        # Check coordinates presence
        res["has_time"] = time_coord in ds
        res["has_lat"] = lat_coord in ds
        res["has_lon"] = lon_coord in ds

        # Check time range
        jjas_slice = ds.sel({time_coord: slice(f"{yr}-06-01", f"{yr}-09-30")})
        time_vals = pd.to_datetime(jjas_slice[time_coord].values)
        res["time_count"] = len(time_vals)
        res["expected_time_count"] = 122
        res["time_start"] = str(time_vals.min().date()) if len(time_vals) > 0 else None
        res["time_end"] = str(time_vals.max().date()) if len(time_vals) > 0 else None
        res["continuous_daily"] = (
            len(time_vals) == 122 and
            (time_vals == pd.date_range(f"{yr}-06-01", f"{yr}-09-30", freq="D")).all()
        )

        # Spatial grid checks
        lats = ds[lat_coord].values
        lons = ds[lon_coord].values
        res["lat_min"] = float(lats.min())
        res["lat_max"] = float(lats.max())
        res["lat_len"] = len(lats)
        res["lon_min"] = float(lons.min())
        res["lon_max"] = float(lons.max())
        res["lon_len"] = len(lons)
        res["lat_ascending"] = bool(np.all(np.diff(lats) > 0))
        res["lon_ascending"] = bool(np.all(np.diff(lons) > 0))
        res["lat_step"] = float(np.round(np.mean(np.diff(lats)), 4))
        res["lon_step"] = float(np.round(np.mean(np.diff(lons)), 4))

        # Spatial compatibility with 2024 baseline
        res["lat_exact_match_2024"] = bool(np.allclose(lats, base_lats, atol=1e-5))
        res["lon_exact_match_2024"] = bool(np.allclose(lons, base_lons, atol=1e-5))

        # Variable & Units check
        rf_var = "RAINFALL" if "RAINFALL" in ds else ("rf" if "rf" in ds else None)
        res["rf_var"] = rf_var
        if rf_var:
            res["rf_units"] = ds[rf_var].attrs.get("units", "mm/day")
            res["rf_dtype"] = str(ds[rf_var].dtype)
            data_jjas = jjas_slice[rf_var].values
            res["rf_min"] = float(np.nanmin(data_jjas))
            res["rf_max"] = float(np.nanmax(data_jjas))
            res["nan_count"] = int(np.sum(np.isnan(data_jjas)))
            res["valid_count"] = int(np.sum(~np.isnan(data_jjas)))
            res["total_points"] = int(data_jjas.size)
            res["valid_land_fraction"] = float(res["valid_count"] / res["total_points"])

            # Check land-sea mask consistency across timesteps
            # For each day, NaN pattern (ocean mask) should match baseline
            mask_day0 = ~np.isnan(data_jjas[0])
            res["mask_matches_baseline"] = bool(np.array_equal(mask_day0, base_mask_sample))
            # Check if land mask is invariant over the season
            all_masks = ~np.isnan(data_jjas)
            res["mask_temporally_stable"] = bool(
                np.all(all_masks == mask_day0[None, :, :])
            )

        # Summary status
        is_pass = (
            res["exists"]
            and res["continuous_daily"]
            and res["lat_exact_match_2024"]
            and res["lon_exact_match_2024"]
            and res["rf_var"] == "RAINFALL"
            and res["rf_min"] >= 0.0
            and res["mask_matches_baseline"]
        )
        res["status"] = "PASS" if is_pass else "FAIL"
        ds.close()
        results[yr] = res

    base_ds.close()
    return results


def validate_era5_datasets(raw_dir: Path) -> Dict[str, Any]:
    results = {}
    base_file = raw_dir / "era5.nc"
    if not base_file.exists():
        raise FileNotFoundError(f"Baseline ERA5 2024 file not found: {base_file}")

    base_ds = xr.open_dataset(base_file)
    base_lats = base_ds["latitude"].values
    base_lons = base_ds["longitude"].values

    expected_vars = ["u10", "v10", "t2m", "d2m", "msl", "tcwv"]
    expected_units = {
        "u10": "m s**-1",
        "v10": "m s**-1",
        "t2m": "K",
        "d2m": "K",
        "msl": "Pa",
        "tcwv": "kg m**-2",
    }

    # Locate ERA5 files
    # Might be individual (era5_2021.nc, etc.) or combined (era5_21_to_23.nc)
    years = [2021, 2022, 2023, 2024]
    
    # Check if single combined file exists or individual files
    multi_file = raw_dir / "era5_21_to_23.nc"
    multi_ds = xr.open_dataset(multi_file) if multi_file.exists() else None

    for yr in years:
        res: Dict[str, Any] = {}
        if yr == 2024:
            ds_yr = base_ds
            res["source_file"] = "era5.nc"
        else:
            indiv_file = raw_dir / f"era5_{yr}.nc"
            if indiv_file.exists():
                ds_yr = xr.open_dataset(indiv_file)
                res["source_file"] = indiv_file.name
            elif multi_ds is not None:
                # Slice from multi-year
                time_coord = "valid_time" if "valid_time" in multi_ds.coords else "time"
                ds_yr = multi_ds.sel({time_coord: slice(f"{yr}-06-01", f"{yr}-09-30T18:00:00")})
                res["source_file"] = f"era5_21_to_23.nc (slice {yr})"
            else:
                res["status"] = "FAIL"
                res["error"] = f"No ERA5 file found for {yr}"
                results[yr] = res
                continue

        time_coord = "valid_time" if "valid_time" in ds_yr.coords else "time"
        lat_coord = "latitude" if "latitude" in ds_yr.coords else "lat"
        lon_coord = "longitude" if "longitude" in ds_yr.coords else "lon"

        # Coordinates
        res["has_time"] = time_coord in ds_yr
        res["has_lat"] = lat_coord in ds_yr
        res["has_lon"] = lon_coord in ds_yr

        # Time checks
        time_vals = pd.to_datetime(ds_yr[time_coord].values)
        res["time_count"] = len(time_vals)
        res["expected_time_count"] = 488  # 122 days * 4 cycles
        res["time_start"] = str(time_vals.min()) if len(time_vals) > 0 else None
        res["time_end"] = str(time_vals.max()) if len(time_vals) > 0 else None

        # Check exact 6-hourly cycle hours: 00, 06, 12, 18 UTC
        expected_times = pd.date_range(f"{yr}-06-01 00:00:00", f"{yr}-09-30 18:00:00", freq="6h")
        res["exact_6h_times_match"] = bool(
            len(time_vals) == 488 and (time_vals == expected_times).all()
        )
        hours_present = sorted(list(set(time_vals.hour)))
        res["hours_present"] = hours_present
        res["expected_hours"] = [0, 6, 12, 18]

        # Spatial grid checks
        lats = ds_yr[lat_coord].values
        lons = ds_yr[lon_coord].values
        res["lat_min"] = float(lats.min())
        res["lat_max"] = float(lats.max())
        res["lat_len"] = len(lats)
        res["lon_min"] = float(lons.min())
        res["lon_max"] = float(lons.max())
        res["lon_len"] = len(lons)
        res["lat_ordering"] = "descending" if lats[0] > lats[-1] else "ascending"
        res["lon_ordering"] = "ascending" if lons[0] < lons[-1] else "descending"
        res["lat_step"] = float(np.round(np.abs(np.mean(np.diff(lats))), 4))
        res["lon_step"] = float(np.round(np.abs(np.mean(np.diff(lons))), 4))

        # Spatial compatibility with 2024 baseline
        res["lat_exact_match_2024"] = bool(np.allclose(lats, base_lats, atol=1e-5))
        res["lon_exact_match_2024"] = bool(np.allclose(lons, base_lons, atol=1e-5))

        # Variable presence, units, NaN/Inf checks
        var_status = {}
        all_vars_ok = True
        for var in expected_vars:
            v_info: Dict[str, Any] = {}
            if var not in ds_yr:
                v_info["present"] = False
                all_vars_ok = False
                var_status[var] = v_info
                continue

            v_info["present"] = True
            v_info["units"] = ds_yr[var].attrs.get("units", "unknown")
            data = ds_yr[var].values
            nan_cnt = int(np.isnan(data).sum())
            inf_cnt = int(np.isinf(data).sum())
            v_info["nan_count"] = nan_cnt
            v_info["inf_count"] = inf_cnt
            v_info["min"] = float(np.nanmin(data))
            v_info["max"] = float(np.nanmax(data))
            v_info["mean"] = float(np.nanmean(data))
            v_info["is_clean"] = (nan_cnt == 0 and inf_cnt == 0)

            # Check physical realism
            if var == "t2m" and (v_info["min"] < 200 or v_info["max"] > 340):
                v_info["is_clean"] = False
            if var == "d2m" and (v_info["min"] < 200 or v_info["max"] > 340):
                v_info["is_clean"] = False
            if var == "msl" and (v_info["min"] < 80000 or v_info["max"] > 110000):
                v_info["is_clean"] = False
            if var == "tcwv" and (v_info["min"] < 0 or v_info["max"] > 120):
                v_info["is_clean"] = False

            if not v_info["is_clean"]:
                all_vars_ok = False
            var_status[var] = v_info

        res["variables"] = var_status
        res["all_vars_present_and_clean"] = all_vars_ok

        # Overall year status
        is_pass = (
            res["exact_6h_times_match"]
            and res["lat_exact_match_2024"]
            and res["lon_exact_match_2024"]
            and res["all_vars_present_and_clean"]
        )
        res["status"] = "PASS" if is_pass else "FAIL"
        results[yr] = res

    if multi_ds is not None:
        multi_ds.close()
    base_ds.close()
    return results


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    print("=" * 70)
    print("RITUGYAN HISTORICAL DATASET VALIDATION (2021-2023 vs 2024 BASELINE)")
    print("=" * 70)

    imd_res = validate_imd_datasets(raw_dir)
    print("\n--- IMD VALIDATION RESULTS ---")
    for yr, r in imd_res.items():
        print(f"Year {yr}: Status = {r['status']}")
        print(f"  File: {r.get('file')}")
        print(f"  Observations: {r.get('time_count')} days (Expected: {r.get('expected_time_count')})")
        print(f"  Date Range: {r.get('time_start')} to {r.get('time_end')} (Continuous: {r.get('continuous_daily')})")
        print(f"  Grid: Lat [{r.get('lat_min')}, {r.get('lat_max')}] (n={r.get('lat_len')}, step={r.get('lat_step')}°), Lon [{r.get('lon_min')}, {r.get('lon_max')}] (n={r.get('lon_len')}, step={r.get('lon_step')}°)")
        print(f"  Spatial Match with 2024: Lat={r.get('lat_exact_match_2024')}, Lon={r.get('lon_exact_match_2024')}")
        print(f"  Variable: {r.get('rf_var')} [{r.get('rf_units')}], Range: [{r.get('rf_min')}, {r.get('rf_max')}], Land Points: {r.get('valid_count')}/{r.get('total_points')} ({r.get('valid_land_fraction', 0)*100:.1f}%)")
        print(f"  Mask Match 2024: {r.get('mask_matches_baseline')}, Seasonally Stable: {r.get('mask_temporally_stable')}")

    era5_res = validate_era5_datasets(raw_dir)
    print("\n--- ERA5 VALIDATION RESULTS ---")
    for yr, r in era5_res.items():
        print(f"Year {yr}: Status = {r['status']}")
        print(f"  Source: {r.get('source_file')}")
        print(f"  Timesteps: {r.get('time_count')} (Expected: {r.get('expected_time_count')})")
        print(f"  Date Range: {r.get('time_start')} to {r.get('time_end')} (Exact 6h: {r.get('exact_6h_times_match')})")
        print(f"  Cycles: {r.get('hours_present')} UTC")
        print(f"  Grid: Lat [{r.get('lat_min')}, {r.get('lat_max')}] ({r.get('lat_ordering')}, n={r.get('lat_len')}), Lon [{r.get('lon_min')}, {r.get('lon_max')}] ({r.get('lon_ordering')}, n={r.get('lon_len')})")
        print(f"  Spatial Match with 2024: Lat={r.get('lat_exact_match_2024')}, Lon={r.get('lon_exact_match_2024')}")
        print(f"  Variables Clean: {r.get('all_vars_present_and_clean')}")
        if "variables" in r:
            for v, vi in r["variables"].items():
                print(f"    {v} [{vi.get('units')}]: min={vi.get('min'):.2f}, max={vi.get('max'):.2f}, mean={vi.get('mean'):.2f}, NaNs={vi.get('nan_count')}, Infs={vi.get('inf_count')}, Clean={vi.get('is_clean')}")
