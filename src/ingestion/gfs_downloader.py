"""Targeted GFS forecast acquisition utility using HTTP byte-range extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

from src.ingestion.readers import setup_eccodes_environment
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Base URL for NOAA GFS Open Data Dissemination (AWS S3)
NOAA_GFS_S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Target variables and their matching patterns in GFS .idx files
TARGET_VARIABLE_PATTERNS = {
    "PRMSL": [":PRMSL:mean sea level:"],
    "TMP_2m": [":TMP:2 m above ground:"],
    "DPT_2m": [":DPT:2 m above ground:"],
    "UGRD_10m": [":UGRD:10 m above ground:"],
    "VGRD_10m": [":VGRD:10 m above ground:"],
    "PWAT": [":PWAT:entire atmosphere (considered as a single layer):"],
    "APCP": [":APCP:surface:"],
}


def fetch_gfs_idx(date_str: str, cycle_str: str, lead_str: str) -> List[Dict[str, Any]]:
    """Fetch and parse GFS .idx index file from NOAA AWS S3 bucket.
    
    Args:
        date_str: Date in YYYYMMDD format (e.g. '20240621').
        cycle_str: Cycle hour (e.g. '00').
        lead_str: Forecast lead (e.g. 'f006').
        
    Returns:
        List of parsed index records with line numbers, byte offsets, variables, and levels.
    """
    url = f"{NOAA_GFS_S3_BASE}/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.{lead_str}.idx"
    logger.info(f"Fetching GFS index from: {url}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "RituGyan-Meteorology/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch GFS index from {url}: {exc}") from exc

    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    parsed = []
    for line in lines:
        parts = line.split(":")
        if len(parts) >= 6:
            parsed.append({
                "line_num": int(parts[0]),
                "offset": int(parts[1]),
                "date": parts[2],
                "var": parts[3],
                "level": parts[4],
                "step": parts[5],
                "raw": line,
            })
    return parsed


def find_byte_ranges(
    parsed_idx: List[Dict[str, Any]],
    target_vars: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Determine start and end byte offsets for each requested variable record.
    
    Args:
        parsed_idx: List of parsed GRIB index entries.
        target_vars: Optional list of target variable keys. Defaults to all in TARGET_VARIABLE_PATTERNS.
        
    Returns:
        List of matching records with calculated start_byte and end_byte.
    """
    if target_vars is None:
        target_vars = list(TARGET_VARIABLE_PATTERNS.keys())

    patterns_to_check = []
    for tvar in target_vars:
        if tvar in TARGET_VARIABLE_PATTERNS:
            patterns_to_check.extend(TARGET_VARIABLE_PATTERNS[tvar])

    selected_records = []
    total_entries = len(parsed_idx)

    for i, entry in enumerate(parsed_idx):
        raw_line = entry["raw"]
        matched = False
        matched_var_key = None
        for tvar in target_vars:
            for pat in TARGET_VARIABLE_PATTERNS.get(tvar, []):
                if pat in raw_line:
                    matched = True
                    matched_var_key = tvar
                    break
            if matched:
                break

        if matched:
            start_byte = entry["offset"]
            end_byte = parsed_idx[i + 1]["offset"] - 1 if (i + 1 < total_entries) else None
            
            selected_records.append({
                "var_key": matched_var_key,
                "short_name": entry["var"],
                "level": entry["level"],
                "step": entry["step"],
                "raw": entry["raw"],
                "start_byte": start_byte,
                "end_byte": end_byte,
            })

    return selected_records


def download_gfs_lead_slice(
    date_str: str,
    cycle_str: str,
    lead_str: str,
    output_path: Path,
    target_vars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Download and assemble selected GRIB2 variable slices for a single forecast lead.
    
    Args:
        date_str: Forecast date (YYYYMMDD).
        cycle_str: Forecast cycle (HH).
        lead_str: Forecast lead (fFFF).
        output_path: Target .grib2 file path.
        target_vars: Optional list of variable keys.
        
    Returns:
        Dict with download statistics and variable details.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grib_url = f"{NOAA_GFS_S3_BASE}/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.{lead_str}"
    
    parsed_idx = fetch_gfs_idx(date_str, cycle_str, lead_str)
    records = find_byte_ranges(parsed_idx, target_vars)

    if not records:
        raise ValueError(f"No matching variable records found for {lead_str} in GFS index.")

    logger.info(f"Downloading {len(records)} variable slices for {lead_str} from {grib_url}")
    
    extracted_records_info = []
    total_bytes = 0

    with open(output_path, "wb") as out_f:
        for rec in records:
            start = rec["start_byte"]
            end = rec["end_byte"]
            range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
            
            req = urllib.request.Request(
                grib_url,
                headers={
                    "Range": range_header,
                    "User-Agent": "RituGyan-Meteorology/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Failed downloading byte range {range_header} from {grib_url}: {exc}"
                ) from exc

            if not data.startswith(b"GRIB"):
                raise ValueError(
                    f"Downloaded slice for {rec['raw']} did not begin with GRIB header bytes. "
                    f"Got: {data[:16]}"
                )

            out_f.write(data)
            slice_size = len(data)
            total_bytes += slice_size

            extracted_records_info.append({
                "var_key": rec["var_key"],
                "short_name": rec["short_name"],
                "level": rec["level"],
                "step": rec["step"],
                "size_bytes": slice_size,
            })

    return {
        "file_name": output_path.name,
        "file_path": str(output_path),
        "source_url": grib_url,
        "lead_time": lead_str,
        "cycle": f"{date_str} {cycle_str}:00 UTC",
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
        "records_count": len(extracted_records_info),
        "records": extracted_records_info,
    }
