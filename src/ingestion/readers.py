"""Dataset readers and discovery utilities for RituGyan."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import xarray as xr

from src.utils.config import find_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_eccodes_environment() -> bool:
    """Ensure ecCodes C-library and definitions are reachable on Windows and Linux.
    
    Returns:
        bool: True if ecCodes library is available, False otherwise.
    """
    try:
        import ecmwflibs  # type: ignore

        dll_path = ecmwflibs.find("eccodes")
        if dll_path:
            dll_dir = os.path.dirname(dll_path)
            if dll_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            os.environ["ECCODES_DIR"] = dll_dir
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception:
                    pass
    except ImportError:
        pass

    try:
        import eccodes  # noqa: F401
        return True
    except Exception as exc:
        logger.debug(f"ecCodes library check returned: {exc}")
        return False


def detect_file_format(file_path: Union[str, Path]) -> str:
    """Inspect magic bytes to accurately detect data file format.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        str: Format description ('netcdf3', 'netcdf4/hdf5', 'grib2', 'grib1', 'directory', or 'unknown').
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        return "directory"

    with open(path, "rb") as f:
        magic = f.read(8)

    if magic.startswith(b"CDF\x01") or magic.startswith(b"CDF\x02"):
        return "netcdf3"
    elif magic.startswith(b"\x89HDF\r\n\x1a\n"):
        return "netcdf4/hdf5"
    elif magic.startswith(b"GRIB"):
        edition = magic[7] if len(magic) >= 8 else 0
        if edition == 2:
            return "grib2"
        elif edition == 1:
            return "grib1"
        return "grib"
    else:
        # Fallback to extension check
        ext = path.suffix.lower()
        if ext in [".nc", ".nc4", ".netcdf"]:
            return "netcdf"
        elif ext in [".grib", ".grib2", ".grb", ".grb2"]:
            return "grib2"
        return "unknown"


def discover_datasets(raw_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Discover available IMD, GFS, and ERA5 raw datasets in the raw data directory.
    
    Args:
        raw_dir: Optional path to data/raw directory. If None, resolved from config.
        
    Returns:
        Dict with discovered file paths, formats, and statuses.
    """
    if raw_dir is None:
        cfg = load_config()
        root = find_project_root()
        raw_path = (root / cfg.paths.data_raw_dir).resolve()
    else:
        raw_path = Path(raw_dir).resolve()

    discovery: Dict[str, Any] = {
        "raw_dir": str(raw_path),
        "raw_dir_exists": raw_path.exists(),
        "imd": {"status": "UNAVAILABLE", "files": []},
        "gfs": {"status": "UNAVAILABLE", "files": []},
        "era5": {"status": "CURRENTLY PROCESSING/DOWNLOADING — not used in this phase", "files": []},
    }

    if not raw_path.exists():
        logger.warning(f"Raw data directory does not exist: {raw_path}")
        return discovery

    # Iterate over all files, including within subdirectories (e.g. gfs_test)
    for file_path in raw_path.rglob("*"):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        fname_lower = file_path.name.lower()
        if fname_lower.endswith(".json") or fname_lower.endswith(".txt") or fname_lower.endswith(".idx"):
            continue

        fmt = detect_file_format(file_path)

        # IMD identification (NetCDF gridded rainfall)
        if ("rf" in fname_lower or "imd" in fname_lower or "rain" in fname_lower) and "nc" in fmt:
            discovery["imd"]["files"].append({
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "format": fmt,
            })
            discovery["imd"]["status"] = "AVAILABLE"

        # GFS identification (GRIB2 or NetCDF)
        elif "gfs" in fname_lower or "grib" in fmt:
            discovery["gfs"]["files"].append({
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "format": fmt,
            })
            discovery["gfs"]["status"] = "AVAILABLE"

        # ERA5 identification
        elif "era5" in fname_lower or "ecmwf" in fname_lower:
            discovery["era5"]["files"].append({
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "format": fmt,
            })

    return discovery


def open_imd_dataset(
    file_path: Union[str, Path],
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """Safely open an IMD NetCDF gridded rainfall dataset in read-only mode.
    
    Args:
        file_path: Path to the IMD NetCDF file.
        chunks: Optional chunking dictionary for Dask/lazy loading.
        
    Returns:
        xr.Dataset: The opened IMD dataset.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"IMD dataset file does not exist: {path}")

    fmt = detect_file_format(path)
    if "netcdf" not in fmt and fmt != "unknown":
        logger.warning(f"Expected NetCDF format for IMD file, but detected: {fmt}")

    # Open with xarray (read-only)
    ds = xr.open_dataset(path, chunks=chunks, decode_times=True)
    return ds


def open_gfs_dataset(
    file_path: Union[str, Path],
    filter_by_keys: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[xr.Dataset, List[xr.Dataset]]:
    """Safely open a GFS forecast dataset (GRIB2 or NetCDF) in read-only mode.
    
    Args:
        file_path: Path to GFS file.
        filter_by_keys: Optional dictionary of GRIB filter keys (e.g. {'typeOfLevel': 'meanSea'}).
        kwargs: Additional arguments passed to cfgrib or xr.open_dataset.
        
    Returns:
        Union[xr.Dataset, List[xr.Dataset]]: The opened GFS dataset or list of datasets.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GFS dataset file does not exist: {path}")

    fmt = detect_file_format(path)

    if fmt == "grib2" or fmt == "grib" or fmt == "grib1":
        setup_eccodes_environment()
        try:
            import cfgrib  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cfgrib / ecCodes for GRIB2 file {path}: {exc}. "
                "Ensure eccodes and ecmwflibs or conda-forge eccodes are installed."
            ) from exc

        if filter_by_keys:
            backend_kwargs = kwargs.pop("backend_kwargs", {})
            backend_kwargs["filter_by_keys"] = filter_by_keys
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs, **kwargs)
        else:
            # Return list of dataset groups if multiple parameter tables exist
            return cfgrib.open_datasets(str(path), **kwargs)

    elif "netcdf" in fmt:
        return xr.open_dataset(path, **kwargs)

    else:
        # Attempt standard xarray open
        logger.warning(f"Format for GFS file {path} is '{fmt}'. Attempting xr.open_dataset.")
        return xr.open_dataset(path, **kwargs)
