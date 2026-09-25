"""Data ingestion connectors, readers, and metadata inspection modules for RituGyan."""

from src.ingestion.inspector import (
    check_imd_gfs_compatibility,
    inspect_gfs_dataset,
    inspect_imd_dataset,
)
from src.ingestion.readers import (
    detect_file_format,
    discover_datasets,
    open_gfs_dataset,
    open_imd_dataset,
    setup_eccodes_environment,
)
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

__all__ = [
    "detect_file_format",
    "discover_datasets",
    "open_imd_dataset",
    "open_gfs_dataset",
    "setup_eccodes_environment",
    "inspect_imd_dataset",
    "inspect_gfs_dataset",
    "check_imd_gfs_compatibility",
    "CANONICAL_LATS",
    "CANONICAL_LONS",
    "CANONICAL_NUM_LATS",
    "CANONICAL_NUM_LONS",
    "get_canonical_grid",
    "extract_imd_subgrid",
    "extract_era5_subgrid",
    "extract_gfs_subgrid_array",
]
