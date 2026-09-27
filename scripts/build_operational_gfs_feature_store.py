"""Build the isolated GFS-only Stage 1 operational feature store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.features.operational_gfs import (  # noqa: E402
    OPERATIONAL_STORE_DEFAULT,
    build_operational_feature_store,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-store", type=Path, default=repo_root / "data/processed/feature_store")
    parser.add_argument("--gfs-root", type=Path, default=repo_root / "data/raw/gfs_historical")
    parser.add_argument("--output", type=Path, default=repo_root / OPERATIONAL_STORE_DEFAULT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-canonical-gfs",
        action="store_true",
        help="Build historical validation store from the canonical store's GFS planes only; never reads ERA5 planes.",
    )
    args = parser.parse_args()
    output = build_operational_feature_store(
        canonical_store_dir=args.canonical_store,
        gfs_root_dir=args.gfs_root,
        output_dir=args.output,
        workers=args.workers,
        reuse_canonical_gfs=args.reuse_canonical_gfs,
    )
    print(f"Operational GFS-only feature store: {output}")


if __name__ == "__main__":
    main()
