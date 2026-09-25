"""Acquisition and validation of a complete GFS 00Z exact forecast cycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import xarray as xr

from src.ingestion.gfs_downloader import download_gfs_lead_slice
from src.ingestion.readers import detect_file_format, open_gfs_dataset, setup_eccodes_environment
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_LEADS = ["f003", "f006", "f009", "f012", "f015", "f018", "f021", "f024", "f027"]
TARGET_DATE = "20240621"
TARGET_CYCLE = "00"
OUTPUT_DIR = Path("data/raw/gfs_exact_test")

REQUIRED_VARS_CANONICAL = [
    "total precipitation (APCP)",
    "mean sea level pressure (PRMSL)",
    "2m temperature (TMP_2m)",
    "2m dewpoint (DPT_2m)",
    "10m U wind (UGRD_10m)",
    "10m V wind (VGRD_10m)",
    "precipitable water (PWAT)",
]


def acquire_exact_gfs_cycle() -> Dict[str, Any]:
    """Download the 9 required forecast lead slices."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_eccodes_environment()

    manifest: Dict[str, Any] = {
        "dataset_name": "NOAA GFS 0.25 Degree - Exact IMD-Aligned Cycle",
        "initialization_time": f"{TARGET_DATE} {TARGET_CYCLE}:00 UTC",
        "date": TARGET_DATE,
        "cycle": TARGET_CYCLE,
        "leads_requested": REQUIRED_LEADS,
        "output_directory": str(OUTPUT_DIR),
        "files": {},
    }

    for lead in REQUIRED_LEADS:
        out_file = OUTPUT_DIR / f"gfs.0p25.{TARGET_DATE}{TARGET_CYCLE}.{lead}.grib2"
        logger.info(f"Downloading GFS slice: {lead} -> {out_file.name}")
        info = download_gfs_lead_slice(
            date_str=TARGET_DATE,
            cycle_str=TARGET_CYCLE,
            lead_str=lead,
            output_path=out_file,
        )
        manifest["files"][lead] = info
        logger.info(f"Successfully downloaded {lead} ({info['total_mb']} MB, {info['records_count']} records)")

    return manifest


def validate_file(file_path: Path, expected_lead: str) -> Dict[str, Any]:
    """Perform rigorous validation of a downloaded GRIB2 forecast lead file."""
    report: Dict[str, Any] = {
        "file_name": file_path.name,
        "path": str(file_path),
        "exists": file_path.exists(),
        "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2) if file_path.exists() else 0,
        "format": None,
        "is_valid_grib2": False,
        "readable_eccodes": False,
        "initialization_time": None,
        "forecast_lead": None,
        "lead_match": False,
        "grid_resolution_deg": None,
        "spatial_shape": None,
        "variables_found": [],
        "required_variables_status": {},
        "status": "FAIL",
        "errors": [],
    }

    if not file_path.exists() or report["size_bytes"] == 0:
        report["errors"].append("File does not exist or is empty.")
        return report

    # 1. Format check
    fmt = detect_file_format(file_path)
    report["format"] = fmt
    report["is_valid_grib2"] = (fmt == "grib2")

    if not report["is_valid_grib2"]:
        report["errors"].append(f"Expected GRIB2 format, detected: {fmt}")
        return report

    # 2. Read with cfgrib / ecCodes via open_gfs_dataset
    try:
        import cfgrib
        datasets = cfgrib.open_datasets(str(file_path))
        report["readable_eccodes"] = True
    except Exception as exc:
        report["errors"].append(f"cfgrib open_datasets failed: {exc}")
        return report

    all_vars: Dict[str, Any] = {}
    init_times = set()
    step_ranges = set()
    lats_shape = None
    lons_shape = None
    lat_res = None
    lon_res = None

    for ds in datasets:
        for vname, da in ds.data_vars.items():
            all_vars[vname] = {
                "long_name": getattr(da, "long_name", vname),
                "units": getattr(da, "units", ""),
                "shape": list(da.shape),
                "dims": list(da.dims),
            }
        if "time" in ds.coords:
            t_val = ds.coords["time"].values
            init_times.add(str(t_val))
        if "step" in ds.coords:
            s_val = ds.coords["step"].values
            step_ranges.add(str(s_val))
        if "latitude" in ds.coords and "longitude" in ds.coords:
            lat = ds.coords["latitude"].values
            lon = ds.coords["longitude"].values
            lats_shape = len(lat)
            lons_shape = len(lon)
            lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else None
            lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else None

    report["variables_found"] = list(all_vars.keys())
    report["spatial_shape"] = (lats_shape, lons_shape)
    report["grid_resolution_deg"] = (round(lat_res, 3), round(lon_res, 3)) if lat_res and lon_res else None
    report["initialization_time"] = list(init_times)
    report["forecast_lead"] = list(step_ranges)

    # Validate lead matching
    lead_hours = int(expected_lead.replace("f", ""))
    lead_match = False
    for s in step_ranges:
        if f"{lead_hours}" in s or f"{lead_hours:02d}" in s:
            lead_match = True
    report["lead_match"] = lead_match

    # Verify resolution (0.25 deg)
    res_ok = (report["grid_resolution_deg"] == (0.25, 0.25))

    # Check required variables
    var_map = {
        "total precipitation (APCP)": any(k in all_vars for k in ["tp", "apcp", "acpcp"]),
        "mean sea level pressure (PRMSL)": any(k in all_vars for k in ["prmsl", "msl"]),
        "2m temperature (TMP_2m)": any(k in all_vars for k in ["t2m", "2t", "tmp"]),
        "2m dewpoint (DPT_2m)": any(k in all_vars for k in ["d2m", "2d", "dpt"]),
        "10m U wind (UGRD_10m)": any(k in all_vars for k in ["u10", "10u", "u"]),
        "10m V wind (VGRD_10m)": any(k in all_vars for k in ["v10", "10v", "v"]),
        "precipitable water (PWAT)": any(k in all_vars for k in ["pwat"]),
    }
    report["required_variables_status"] = var_map
    all_vars_present = all(var_map.values())

    if report["is_valid_grib2"] and report["readable_eccodes"] and res_ok and all_vars_present:
        report["status"] = "PASS"
    else:
        if not res_ok:
            report["errors"].append(f"Grid resolution {report['grid_resolution_deg']} != (0.25, 0.25)")
        if not all_vars_present:
            missing = [k for k, v in var_map.items() if not v]
            report["errors"].append(f"Missing required variables: {missing}")
        report["status"] = "FAIL"

    return report


def main() -> None:
    logger.info("=== Starting GFS Exact Cycle Acquisition ===")
    manifest = acquire_exact_gfs_cycle()

    logger.info("=== Starting Validation of Downloaded Files ===")
    validation_results = {}
    total_size_bytes = 0

    all_passed = True
    for lead in REQUIRED_LEADS:
        fpath = OUTPUT_DIR / f"gfs.0p25.{TARGET_DATE}{TARGET_CYCLE}.{lead}.grib2"
        v_res = validate_file(fpath, lead)
        validation_results[lead] = v_res
        total_size_bytes += v_res["size_bytes"]
        logger.info(
            f"Validation for {lead}: Status={v_res['status']} | "
            f"Size={v_res['size_mb']} MB | Res={v_res['grid_resolution_deg']} | "
            f"Vars={v_res['variables_found']}"
        )
        if v_res["status"] != "PASS":
            all_passed = False

    summary = {
        "dataset": "GFS 0.25 Exact Cycle",
        "target_date": TARGET_DATE,
        "cycle": f"{TARGET_CYCLE}Z",
        "output_directory": str(OUTPUT_DIR),
        "total_files": len(REQUIRED_LEADS),
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "all_validations_passed": all_passed,
        "files_validated": validation_results,
    }

    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Validation summary saved to {manifest_file}")
    print("\n" + "=" * 60)
    print(f"TOTAL FILES DOWNLOADED: {len(REQUIRED_LEADS)}")
    print(f"TOTAL DOWNLOAD SIZE: {round(total_size_bytes / (1024 * 1024), 2)} MB")
    print(f"INITIALIZATION TIME: {TARGET_DATE} {TARGET_CYCLE}:00 UTC")
    print(f"AVAILABLE FORECAST LEADS: {REQUIRED_LEADS}")
    print(f"OVERALL VALIDATION STATUS: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
