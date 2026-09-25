"""Multi-source preprocessing and feature tensor construction for RituGyan.

Constructs aligned (channels, 127, 121) feature tensors and (127, 121) target tensors
by combining:
1. IMD 24-hour daily rainfall ground truth [mm]
2. ERA5 6-hourly atmospheric state predictors [hPa, °C, m/s, kg/m^2]
3. GFS forecast sequence (f003-f027) with cumulative precipitation differencing [mm]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import glob
import numpy as np
import xarray as xr

from src.ingestion.readers import setup_eccodes_environment
from src.ingestion.spatial import (
    CANONICAL_LATS,
    CANONICAL_LONS,
    CANONICAL_NUM_LATS,
    CANONICAL_NUM_LONS,
    extract_era5_subgrid,
    extract_gfs_subgrid_array,
    extract_imd_subgrid,
    get_canonical_grid,
)
from src.utils.config import find_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard channel definitions and physical units
FEATURE_CHANNELS: List[str] = [
    "gfs_tp_24h",       # GFS 24h accumulated rainfall A(f027) - A(f003) [mm]
    "gfs_prmsl_mean",   # GFS mean sea-level pressure [hPa]
    "gfs_2t_mean",      # GFS mean 2m temperature [°C]
    "gfs_2d_mean",      # GFS mean 2m dewpoint temperature [°C]
    "gfs_10u_mean",     # GFS mean 10m zonal wind [m/s]
    "gfs_10v_mean",     # GFS mean 10m meridional wind [m/s]
    "gfs_pwat_mean",    # GFS mean precipitable water [kg/m^2]
    "gfs_wind_speed",   # GFS mean wind speed sqrt(u^2 + v^2) [m/s]
    "era5_msl_mean",    # ERA5 mean sea-level pressure [hPa]
    "era5_t2m_mean",    # ERA5 mean 2m temperature [°C]
    "era5_d2m_mean",    # ERA5 mean 2m dewpoint temperature [°C]
    "era5_u10_mean",    # ERA5 mean 10m zonal wind [m/s]
    "era5_v10_mean",    # ERA5 mean 10m meridional wind [m/s]
    "era5_tcwv_mean",   # ERA5 mean total column water vapour [kg/m^2]
    "era5_wind_speed",  # ERA5 mean wind speed sqrt(u^2 + v^2) [m/s]
]

CHANNEL_UNITS: Dict[str, str] = {
    "gfs_tp_24h": "mm",
    "gfs_prmsl_mean": "hPa",
    "gfs_2t_mean": "°C",
    "gfs_2d_mean": "°C",
    "gfs_10u_mean": "m/s",
    "gfs_10v_mean": "m/s",
    "gfs_pwat_mean": "kg/m^2",
    "gfs_wind_speed": "m/s",
    "era5_msl_mean": "hPa",
    "era5_t2m_mean": "°C",
    "era5_d2m_mean": "°C",
    "era5_u10_mean": "m/s",
    "era5_v10_mean": "m/s",
    "era5_tcwv_mean": "kg/m^2",
    "era5_wind_speed": "m/s",
}


@dataclass
class AlignedDaySample:
    """Container for one temporally and spatially aligned day sample."""
    target_date: str
    features: np.ndarray        # Shape: (num_channels, 127, 121), dtype=float32
    target: np.ndarray          # Shape: (127, 121), dtype=float32 (NaNs preserved for ocean)
    valid_mask: np.ndarray      # Shape: (127, 121), dtype=bool (True where IMD observation exists)
    channel_names: List[str]
    channel_units: Dict[str, str]
    lats: np.ndarray            # Shape: (127,), ascending
    lons: np.ndarray            # Shape: (121,), ascending
    metadata: Dict[str, Any]


def load_imd_target_day(
    file_path: Union[str, Path],
    target_date: str = "2024-06-21",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load IMD daily rainfall for target day D, extracted to the common 127x121 subgrid.
    
    Args:
        file_path: Path to IMD NetCDF file.
        target_date: Target date string 'YYYY-MM-DD'.
        
    Returns:
        Tuple of:
          - target array (127, 121) float32 with NaNs preserved
          - valid_mask array (127, 121) bool (True for valid land points)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"IMD dataset not found: {path}")

    ds = xr.open_dataset(path)
    if "TIME" not in ds.coords and "TIME" not in ds.dims:
        raise KeyError("Expected coordinate 'TIME' in IMD dataset")

    # Select target date
    day_ds = ds.sel(TIME=target_date)
    sub_ds = extract_imd_subgrid(day_ds)
    rainfall = sub_ds["RAINFALL"].values.astype(np.float32)
    valid_mask = ~np.isnan(rainfall)

    ds.close()
    return rainfall, valid_mask


def load_era5_in_window_predictors(
    file_path: Union[str, Path],
    target_date: str = "2024-06-21",
) -> Dict[str, np.ndarray]:
    """Load ERA5 6-hourly snapshots within the IMD accumulation window and compute aggregates.
    
    Timestamps for target day D:
      - D 06:00 UTC
      - D 12:00 UTC
      - D 18:00 UTC
      - D+1 00:00 UTC
      
    Args:
        file_path: Path to ERA5 NetCDF file.
        target_date: Target date string 'YYYY-MM-DD'.
        
    Returns:
        Dict with mean ERA5 feature arrays (each shape 127x121 float32) with converted units.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 dataset not found: {path}")

    ds = xr.open_dataset(path)
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"

    dt_d = datetime.date.fromisoformat(target_date)
    dt_next = dt_d + datetime.timedelta(days=1)

    era5_timestamps = [
        f"{dt_d.isoformat()}T06:00:00",
        f"{dt_d.isoformat()}T12:00:00",
        f"{dt_d.isoformat()}T18:00:00",
        f"{dt_next.isoformat()}T00:00:00",
    ]

    # Slice time window and canonical subgrid
    window_ds = ds.sel({time_coord: era5_timestamps})
    sub_ds = extract_era5_subgrid(window_ds, ascending_lat=True)

    # Compute temporal mean across the 4 in-window snapshots
    u10 = sub_ds["u10"].mean(dim=time_coord).values.astype(np.float32)            # m/s
    v10 = sub_ds["v10"].mean(dim=time_coord).values.astype(np.float32)            # m/s
    msl = (sub_ds["msl"].mean(dim=time_coord).values / 100.0).astype(np.float32)  # Pa -> hPa
    t2m = (sub_ds["t2m"].mean(dim=time_coord).values - 273.15).astype(np.float32) # K -> °C
    d2m = (sub_ds["d2m"].mean(dim=time_coord).values - 273.15).astype(np.float32) # K -> °C
    tcwv = sub_ds["tcwv"].mean(dim=time_coord).values.astype(np.float32)          # kg/m^2
    wspd = np.sqrt(u10**2 + v10**2).astype(np.float32)                            # m/s

    ds.close()

    return {
        "era5_msl_mean": msl,
        "era5_t2m_mean": t2m,
        "era5_d2m_mean": d2m,
        "era5_u10_mean": u10,
        "era5_v10_mean": v10,
        "era5_tcwv_mean": tcwv,
        "era5_wind_speed": wspd,
    }


def load_gfs_forecast_predictors(
    gfs_dir: Union[str, Path],
    date_str: str = "20240621",
    cycle_str: str = "00",
) -> Dict[str, np.ndarray]:
    """Load GFS sequence (f003-f027), compute cumulative 24h precipitation, and aggregate predictors.
    
    Args:
        gfs_dir: Path to directory containing GFS lead GRIB2 files.
        date_str: Forecast init date 'YYYYMMDD'.
        cycle_str: Forecast cycle '00'.
        
    Returns:
        Dict with GFS feature arrays (each shape 127x121 float32) with converted units.
    """
    setup_eccodes_environment()
    import eccodes

    gfs_path = Path(gfs_dir)
    lead_files = sorted(gfs_path.glob(f"gfs.0p25.{date_str}{cycle_str}.f*.grib2"))
    if not lead_files:
        raise FileNotFoundError(f"No GFS lead files matching {date_str}{cycle_str} in {gfs_path}")

    accum_0: Dict[int, np.ndarray] = {}
    gfs_vars: Dict[str, List[np.ndarray]] = {
        "prmsl": [],
        "2t": [],
        "2d": [],
        "10u": [],
        "10v": [],
        "pwat": [],
    }

    for fpath in lead_files:
        lead = int(fpath.stem.split(".")[-1].replace("f", ""))
        with open(fpath, "rb") as fp:
            while True:
                gid = eccodes.codes_grib_new_from_file(fp)
                if gid is None:
                    break
                sn = eccodes.codes_get(gid, "shortName")
                if sn == "tp":
                    start_step = eccodes.codes_get(gid, "startStep")
                    if start_step == 0:
                        raw_tp = eccodes.codes_get_values(gid).reshape(721, 1440)
                        accum_0[lead] = extract_gfs_subgrid_array(raw_tp, ascending_lat=True)
                elif sn in gfs_vars:
                    raw_v = eccodes.codes_get_values(gid).reshape(721, 1440)
                    gfs_vars[sn].append(extract_gfs_subgrid_array(raw_v, ascending_lat=True))
                eccodes.codes_release(gid)

    if 3 not in accum_0 or 27 not in accum_0:
        raise ValueError(
            f"Missing required leads for 24h differencing: f003 ({3 in accum_0}), f027 ({27 in accum_0})"
        )

    # 1. 24-hour precipitation accumulation over [03Z(D) -> 03Z(D+1)]
    gfs_tp_24h = (accum_0[27] - accum_0[3]).astype(np.float32)
    # Clip small numerical artifacts to zero
    gfs_tp_24h = np.maximum(gfs_tp_24h, 0.0)

    # 2. Atmospheric predictor means across the forecast sequence
    prmsl = (np.mean(gfs_vars["prmsl"], axis=0) / 100.0).astype(np.float32)     # Pa -> hPa
    t2m = (np.mean(gfs_vars["2t"], axis=0) - 273.15).astype(np.float32)        # K -> °C
    d2m = (np.mean(gfs_vars["2d"], axis=0) - 273.15).astype(np.float32)        # K -> °C
    u10 = np.mean(gfs_vars["10u"], axis=0).astype(np.float32)                  # m/s
    v10 = np.mean(gfs_vars["10v"], axis=0).astype(np.float32)                  # m/s
    pwat = np.mean(gfs_vars["pwat"], axis=0).astype(np.float32)                # kg/m^2
    wspd = np.sqrt(u10**2 + v10**2).astype(np.float32)                         # m/s

    return {
        "gfs_tp_24h": gfs_tp_24h,
        "gfs_prmsl_mean": prmsl,
        "gfs_2t_mean": t2m,
        "gfs_2d_mean": d2m,
        "gfs_10u_mean": u10,
        "gfs_10v_mean": v10,
        "gfs_pwat_mean": pwat,
        "gfs_wind_speed": wspd,
    }


def build_aligned_day_sample(
    imd_file: Union[str, Path],
    era5_file: Union[str, Path],
    gfs_dir: Union[str, Path],
    target_date: str = "2024-06-21",
) -> AlignedDaySample:
    """Build a complete, aligned (C, 127, 121) feature tensor and (127, 121) target tensor for Day D.
    
    Args:
        imd_file: Path to IMD rainfall NetCDF.
        era5_file: Path to ERA5 NetCDF.
        gfs_dir: Path to directory containing GFS lead files (f003-f027).
        target_date: Target date string 'YYYY-MM-DD'.
        
    Returns:
        AlignedDaySample containing features, target, mask, and coordinate metadata.
    """
    date_compact = target_date.replace("-", "")

    # 1. Load IMD target & mask
    target_rain, valid_mask = load_imd_target_day(imd_file, target_date=target_date)

    # 2. Load ERA5 predictors
    era5_feats = load_era5_in_window_predictors(era5_file, target_date=target_date)

    # 3. Load GFS predictors
    gfs_feats = load_gfs_forecast_predictors(gfs_dir, date_str=date_compact, cycle_str="00")

    # 4. Assemble feature tensor in fixed channel order
    all_feats = {**gfs_feats, **era5_feats}
    feature_planes: List[np.ndarray] = []
    for ch in FEATURE_CHANNELS:
        if ch not in all_feats:
            raise KeyError(f"Missing required channel in assembled features: {ch}")
        arr = all_feats[ch]
        assert arr.shape == (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS), (
            f"Channel {ch} has unexpected shape {arr.shape} != ({CANONICAL_NUM_LATS}, {CANONICAL_NUM_LONS})"
        )
        feature_planes.append(arr)

    feature_tensor = np.stack(feature_planes, axis=0).astype(np.float32)

    metadata = {
        "target_date": target_date,
        "gfs_cycle": f"{date_compact}00",
        "spatial_shape": (CANONICAL_NUM_LATS, CANONICAL_NUM_LONS),
        "num_channels": len(FEATURE_CHANNELS),
        "valid_land_cells": int(valid_mask.sum()),
        "total_cells": int(valid_mask.size),
        "valid_land_fraction": float(valid_mask.mean()),
        "rainfall_max_mm": float(np.nanmax(target_rain)),
        "rainfall_mean_mm": float(np.nanmean(target_rain)),
        "gfs_tp_max_mm": float(np.max(gfs_feats["gfs_tp_24h"])),
        "gfs_tp_mean_mm": float(np.mean(gfs_feats["gfs_tp_24h"])),
    }

    return AlignedDaySample(
        target_date=target_date,
        features=feature_tensor,
        target=target_rain,
        valid_mask=valid_mask,
        channel_names=FEATURE_CHANNELS,
        channel_units=CHANNEL_UNITS,
        lats=CANONICAL_LATS,
        lons=CANONICAL_LONS,
        metadata=metadata,
    )
