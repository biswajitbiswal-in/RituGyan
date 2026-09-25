# Phase 2E: Feature Store Architecture & Multi-Year Serialization Report

**Date:** September 25, 2026  
**Status:** **PASS**  
**Serialized Store Location:** `data/processed/feature_store/`  
**Storage Architecture:** NetCDF4 (Multidimensional Tensors) + Apache Parquet (Tabular Metadata)  
**Author:** RituGyan Multi-Year Ingestion & Dataset Audit Engine  

---

## Executive Summary

Phase 2E establishes the permanent, memory-efficient feature store for RituGyan. The architecture couples **lazy, chunked multidimensional tensor storage** with a **high-speed columnar Parquet metadata index**, enforcing strict temporal segregation to prevent data leakage during machine learning training.

All serialized arrays, channel orderings, physical units, coordinate grids, and regime labels have been verified by end-to-end automated tests (`tests/phase2/test_multiyear_dataset.py`, `tests/phase2/test_feature_store.py`).

---

## 1. Feature Store Storage Architecture

The feature store is organized into three complementary files inside `data/processed/feature_store/`:

```
data/processed/feature_store/
├── metadata.parquet       # Apache Parquet tabular index (date, year, month, regime, metrics)
├── tensors.nc             # NetCDF4 multidimensional tensor store (X, y, mask, synoptic_vector)
└── store_config.json      # Coordinate metadata, channel list, and physical unit catalog
```

### 1.1 Multidimensional Tensor Layout (`tensors.nc`)
* **Feature Tensor ($X$)**: Shape `(N, 15, 127, 121)`, `dtype=float32`
  - 15 multi-source channels (GFS 24h differenced precipitation, GFS atmospheric state means, ERA5 in-window reanalysis means, wind speed magnitudes).
  - **Zero NaN contamination**: 100% finite across all grid cells.
* **Ground Truth Target ($y$)**: Shape `(N, 127, 121)`, `dtype=float32`
  - IMD 24-hour daily rainfall in $\text{mm}$.
  - NaNs preserved over surrounding ocean bodies; land points contain valid continuous rainfall measurements.
* **Valid Land Mask**: Shape `(N, 127, 121)`, `dtype=bool`
  - `True` for valid Indian land observation stations (4,964 cells); `False` for ocean / unobserved borders.
* **Synoptic Feature Matrix**: Shape `(N, 17)`, `dtype=float32`
  - 17 physical synoptic diagnostic indices (trough departure, low-latitude inflow, moisture convergence, vorticity, etc.).
* **Regime Label Vectors**: Shape `(N,)`, `dtype=int32` (`regime_id`) and `(N,)`, `dtype=float32` (`regime_confidence`).

### 1.2 Tabular Metadata Index (`metadata.parquet`)
Enables zero-overhead filtering, subsetting, and split extraction via `pyarrow` / `pandas`:
- `date`, `year`, `month`, `day`
- `regime_id` (0..5), `regime_code`, `regime_name`
- `regime_confidence` ($[0.0, 1.0]$), `fallback_applied` (`bool`)
- `valid_land_cells`, `rainfall_max_mm`, `rainfall_mean_mm`, `gfs_tp_max_mm`, `gfs_tp_mean_mm`
- `trough_latitude_departure`, `inflow_speed_ms`, `domain_pwat_mm`, `mfc_central_india_mm_day`, `depression_mslp_anomaly`

---

## 2. Temporal Partitioning Strategy (Zero-Leakage Guard)

```
========================================================================================
                          TEMPORAL DATASET PARTITIONING
========================================================================================

TRAIN PARTITION (JJAS 2020 – 2023):
  * Period: 2020-06-01 to 2023-09-30 (4 Seasons = 488 expected JJAS days)
  * Target: Model training & hyperparameter cross-validation across diverse regimes
  * Constraint: Completely segregated in time from test partition

TEST / BENCHMARK PARTITION (JJAS 2024):
  * Period: 2024-06-01 to 2024-09-30 (1 Season = 122 expected JJAS days)
  * Target: Out-of-sample operational evaluation and extreme event verification
  * Constraint: Zero grid cell or temporal leakage from training years
========================================================================================
```

### Why Random Grid-Cell Splitting is Strictly Prohibited
In spatial meteorological fields, adjacent grid cells on the same day exhibit strong spatial auto-correlation ($r > 0.90$). Randomly splitting cells from the same date across train and test sets leads to massive data leakage and artificially inflated model performance. RituGyan enforces **strict calendar-year temporal partitioning**.

---

## 3. Storage Footprint & Memory Efficiency

| Asset | Precision / Format | Size per Day (1 Sample) | Projected Full 5-Year Size (610 Days) |
|---|---|---|---|
| **Predictor Tensors ($X$)** | `float32` | $15 \times 127 \times 121 \times 4\text{ B} \approx 922\text{ KB}$ | $\approx 562\text{ MB}$ |
| **Target ($y$) & Mask** | `float32` + `bool` | $(127 \times 121 \times 4) + (127 \times 121 \times 1) \approx 76\text{ KB}$ | $\approx 46\text{ MB}$ |
| **Synoptic Vector** | `float32` | $17 \times 4\text{ B} = 68\text{ B}$ | $\approx 41\text{ KB}$ |
| **Parquet Index** | Snappy compressed | $\approx 12\text{ KB}$ | $\approx 150\text{ KB}$ |
| **TOTAL FEATURE STORE** | **NetCDF4 + Parquet** | **$\approx 1.01\text{ MB}$** | **$\approx 608\text{ MB}$** |

### Memory Management Protocols
- **Day-by-Day Streaming Ingestion**: Each day is processed independently and written to disk without holding multiple seasons in RAM.
- **Lazy Xarray Access**: Models load slices on-demand using memory-mapped array views during training.

---

## 4. Current Feature Store State & Ingested Sample Metrics

* **Stored Aligned Days**: `1 day` (`2024-06-21`)
* **Storage Location**: `data/processed/feature_store/`
* **Feature Tensor Size**: `(1, 15, 127, 121)` float32 ($1,015,750\text{ bytes}$)
* **Class / Regime Identified**: `MONSOON_DEPRESSION` (ID: 2, confidence: 1.00)
* **Valid Land Observations**: 4,964 grid cells
* **Target Rainfall Max / Mean**: $244.78\text{ mm}$ / $5.26\text{ mm}$
* **GFS 24h Differenced Rain Max / Mean**: $86.25\text{ mm}$ / $7.68\text{ mm}$

---

## 5. Automated Test Verification

Automated test execution across the new Phase 2E suites:
- [`tests/phase2/test_multiyear_dataset.py`](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/tests/phase2/test_multiyear_dataset.py): **2 passed** (validates availability audit, date accounting, download requirements).
- [`tests/phase2/test_feature_store.py`](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/tests/phase2/test_feature_store.py): **2 passed** (validates serialization, Parquet metadata querying, zero-NaN predictor integrity, target masking, and temporal split non-leakage).

Full repository test suite:
```
============================== test session starts ==============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\biswa\OneDrive\Documents\GitHub\RituGyan
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.13.0
collected 68 items

tests\phase1\test_phase1_validation.py ..........                        [ 14%]
tests\phase2\test_feature_store.py ..                                    [ 17%]
tests\phase2\test_multiyear_dataset.py ..                                [ 20%]
tests\phase2\test_preprocessing.py .....                                 [ 27%]
tests\phase2\test_regime_classifier.py .........                         [ 41%]
tests\phase2\test_spatial_alignment.py .....                             [ 48%]
tests\phase2\test_synoptic_features.py .........                         [ 61%]
tests\phase2\test_temporal_alignment.py ........                         [ 73%]
tests\unit\test_config.py .......                                        [ 83%]
tests\unit\test_environment.py ...                                       [ 88%]
tests\unit\test_ingestion.py ........                                    [100%]

====================== 68 passed, 47 warnings in 31.03s =======================
```
