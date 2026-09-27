"""
Build the full multi-year RituGyan feature store (2021-2024 JJAS).

Streams day-by-day through all 488 aligned samples (IMD + ERA5 + GFS historical),
classifies synoptic regime without IMD rainfall (leak-free), and serializes to:
  data/processed/feature_store/tensors.nc
  data/processed/feature_store/metadata.parquet
  data/processed/feature_store/store_config.json

Run from the repo root:
  python scripts/build_feature_store.py
"""

import datetime
import sys
import time
from pathlib import Path

# ── Pre-load pyarrow DLL before large NumPy arrays occupy address space ───────
# On Windows, pyarrow._parquet.pyd cannot be loaded after ~500 MB of NumPy
# arrays have fragmented the process address space.  Importing it here — at
# process startup, while memory is clean — keeps the DLL resident for the
# serialisation step that runs after all 488 samples are accumulated.
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    pass  # pandas will fall back to fastparquet if available

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.features.feature_store import FeatureStore, SYNOPTIC_FEATURE_NAMES
from src.features.preprocessing import (
    AlignedDaySample,
    FEATURE_CHANNELS,
    build_aligned_day_sample,
    load_era5_in_window_predictors,
    load_gfs_forecast_predictors,
    load_imd_target_day,
)
from src.features.regimes import RegimeClassificationResult, SynopticRegimeClassifier
from src.features.synoptic import SynopticFeatureSet, extract_all_synoptic_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
YEARS = [2021, 2022, 2023, 2024]
RAW_DIR = repo_root / "data" / "raw"
PROCESSED_DIR = repo_root / "data" / "processed" / "feature_store"

ERA5_MULTI = RAW_DIR / "era5_21_to_23.nc"   # covers 2021-2023
ERA5_2024  = RAW_DIR / "era5.nc"             # covers 2024


def jjas_dates(year: int):
    """Yield ISO date strings for Jun 1 – Sep 30 of the given year."""
    start = datetime.date(year, 6, 1)
    end   = datetime.date(year, 9, 30)
    cur   = start
    while cur <= end:
        yield cur.isoformat()
        cur += datetime.timedelta(days=1)


def build():
    classifier = SynopticRegimeClassifier()
    store = FeatureStore(PROCESSED_DIR)

    samples: list[AlignedDaySample] = []
    synoptic_list: list[SynopticFeatureSet] = []
    regime_list: list[RegimeClassificationResult] = []

    total_expected = len(YEARS) * 122
    processed = 0
    skipped = 0
    t0 = time.time()

    for year in YEARS:
        imd_file  = RAW_DIR / f"RF25_ind{year}_rfp25.nc"
        era5_file = ERA5_2024 if year == 2024 else ERA5_MULTI
        gfs_dir   = RAW_DIR / "gfs_historical" / str(year)

        if not imd_file.exists():
            logger.error(f"IMD file missing for {year}: {imd_file}")
            sys.exit(1)
        if not era5_file.exists():
            logger.error(f"ERA5 file missing for {year}: {era5_file}")
            sys.exit(1)
        if not gfs_dir.exists():
            logger.error(f"GFS dir missing for {year}: {gfs_dir}")
            sys.exit(1)

        year_dates = list(jjas_dates(year))
        logger.info(f"=== Processing {year}: {len(year_dates)} JJAS days ===")

        for dt_str in year_dates:
            try:
                # ── 1. Build aligned day sample ───────────────────────────
                sample = build_aligned_day_sample(
                    imd_file=imd_file,
                    era5_file=era5_file,
                    gfs_dir=gfs_dir,
                    target_date=dt_str,
                )

                # ── 2. Extract synoptic features from ERA5 ────────────────
                era5_idx = {name: i for i, name in enumerate(sample.channel_names)}
                syn_feats = extract_all_synoptic_features(
                    mslp_hpa  = sample.features[era5_idx["era5_msl_mean"]],
                    pwat_kg_m2= sample.features[era5_idx["era5_tcwv_mean"]],
                    u10_ms    = sample.features[era5_idx["era5_u10_mean"]],
                    v10_ms    = sample.features[era5_idx["era5_v10_mean"]],
                    target_date=dt_str,
                    lats=sample.lats,
                    lons=sample.lons,
                )

                # ── 3. Leak-free regime classification ────────────────────
                # rainfall_field=None: IMD observed rainfall must NOT be
                # passed here.  It is the Stage 2 supervised target and
                # would constitute target leakage for Stage 1 training.
                regime_res = classifier.classify(
                    synoptic_feats=syn_feats,
                    rainfall_field=None,
                    valid_mask=sample.valid_mask,
                    lats=sample.lats,
                    lons=sample.lons,
                )

                samples.append(sample)
                synoptic_list.append(syn_feats)
                regime_list.append(regime_res)
                processed += 1

                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    eta  = (total_expected - processed) / rate
                    logger.info(
                        f"  Progress: {processed}/{total_expected} days "
                        f"({100*processed/total_expected:.1f}%) | "
                        f"Speed: {rate:.1f} days/s | ETA: {eta/60:.1f} min"
                    )

            except Exception as exc:
                logger.error(f"  FAILED {dt_str}: {exc}")
                skipped += 1

    # ── 4. Serialize ─────────────────────────────────────────────────────────
    logger.info(f"Serializing {len(samples)} samples to feature store...")
    store.save(samples, synoptic_list, regime_list)

    elapsed = time.time() - t0
    logger.info(
        f"\n{'='*72}\n"
        f"FEATURE STORE BUILD COMPLETE\n"
        f"  Processed : {processed} days\n"
        f"  Skipped   : {skipped} days\n"
        f"  Duration  : {elapsed/60:.1f} minutes\n"
        f"  Output    : {PROCESSED_DIR}\n"
        f"{'='*72}"
    )
    return processed, skipped


if __name__ == "__main__":
    processed, skipped = build()
    sys.exit(0 if skipped == 0 else 1)
