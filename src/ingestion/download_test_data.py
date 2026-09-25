"""Script to download small GFS forecast test dataset for single cycle."""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.gfs_downloader import download_gfs_lead_slice
from src.utils.logger import get_logger

logger = get_logger(__name__)


def acquire_test_dataset() -> Dict[str, Any]:
    """Download GFS forecast test slices for 2024-06-21 00 UTC."""
    date_str = "20240621"
    cycle_str = "00"
    leads = ["f006", "f012", "f018", "f024"]
    out_dir = Path("data/raw/gfs_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "initialization_time": f"{date_str} {cycle_str}:00 UTC",
        "date": date_str,
        "cycle": cycle_str,
        "download_method": "HTTP byte-range filtered extraction via NOAA GFS AWS S3 Open Data",
        "source_bucket": "https://noaa-gfs-bdp-pds.s3.amazonaws.com",
        "required_variables": [
            "APCP",
            "PRMSL",
            "TMP_2m",
            "DPT_2m",
            "UGRD_10m",
            "VGRD_10m",
            "PWAT",
        ],
        "files": {},
    }

    for lead in leads:
        out_file = out_dir / f"gfs.0p25.{date_str}{cycle_str}.{lead}.grib2"
        logger.info(f"Starting acquisition for {lead} -> {out_file.name}")
        info = download_gfs_lead_slice(
            date_str=date_str,
            cycle_str=cycle_str,
            lead_str=lead,
            output_path=out_file,
        )
        manifest["files"][lead] = info
        logger.info(f"Completed {lead}: {info['total_mb']} MB ({info['records_count']} records)")

    metadata_file = out_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest successfully written to {metadata_file}")
    return manifest


if __name__ == "__main__":
    acquire_test_dataset()
