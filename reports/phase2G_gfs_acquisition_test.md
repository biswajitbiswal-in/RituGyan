# Phase 2G: Single-Cycle GFS Acquisition and Pipeline Test Report

**Author:** RituGyan Core Engineering Team  
**Date:** September 2026  
**Status:** COMPLETED & VALIDATED (PASS)  
**Scope:** Strict single-cycle test acquisition of NOAA GFS 0.25° numerical weather prediction forecast sequence for initialization **2024-06-21 00:00 UTC** across all 9 required forecast leads (`f003`–`f027`).

---

## 1. Executive Summary

| Parameter | Specification | Validation Result | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Source** | NOAA Big Data Program (AWS S3: `noaa-gfs-bdp-pds`) | HTTP targeted byte-range extraction | **PASS** |
| **Target Date & Cycle** | `2024-06-21 00Z` (`2024-06-21 00:00:00 UTC`) | Exact match across all 9 lead files | **PASS** |
| **Forecast Leads** | `f003, f006, f009, f012, f015, f018, f021, f024, f027` (9 leads) | All 9 leads acquired without gaps | **PASS** |
| **Total Download Size** | Targeted variable slices only (~51.45 MB total) | 51.45 MB (vs ~4.5 GB for full global GRIBs) | **PASS** |
| **Target Variables** | `APCP, PRMSL, TMP_2m, DPT_2m, UGRD_10m, VGRD_10m, PWAT` (7 vars) | All 7 variables verified in all 9 files | **PASS** |
| **Global Grid Shape** | $721 \times 1440$ ($0.25^\circ \times 0.25^\circ$) | Exact $721 \times 1440$ row-major grid | **PASS** |
| **Extracted Subgrid** | $127 \times 121$ ($[6.5^\circ\text{N}, 38.0^\circ\text{N}] \times [68.0^\circ\text{E}, 98.0^\circ\text{E}]$) | $\Delta_{\text{coord}} = 0.0$ match with canonical grid | **PASS** |
| **Predictor Cleanliness** | Zero NaNs / Zero Infs across all predictor fields | 100% clean numerical fields | **PASS** |
| **Precipitation Differencing**| $A(f027) - A(f003) \ge 0.0$ (24h accumulation semantics) | Valid ($A(f027) \ge A(f003)$ monotonic) | **PASS** |
| **Phase 2 Ingestion** | Consumed by `load_gfs_forecast_predictors` & `build_aligned_day_sample` | Shape $(15, 127, 121)$ generated seamlessly | **PASS** |
| **Repository Test Suite** | Full pytest regression suite | 97 / 97 tests passing ($100\%$) | **PASS** |

---

## 2. File Inventory & Storage Isolation

All acquired single-cycle test files are stored in an isolated directory (`data/raw/gfs_single_cycle_test/`) to guarantee zero interference with existing raw files:

| File Name | Forecast Lead | Step (Hours) | Size (Bytes) | Size (MB) | GRIB2 Status | ecCodes Readable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gfs.0p25.2024062100.f003.grib2` | `f003` | $+3\text{ h}$ | $5,818,681$ | $5.55\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f006.grib2` | `f006` | $+6\text{ h}$ | $5,936,090$ | $5.66\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f009.grib2` | `f009` | $+9\text{ h}$ | $5,941,019$ | $5.67\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f012.grib2` | `f012` | $+12\text{ h}$ | $6,034,611$ | $5.76\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f015.grib2` | `f015` | $+15\text{ h}$ | $6,003,604$ | $5.73\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f018.grib2` | `f018` | $+18\text{ h}$ | $6,080,018$ | $5.80\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f021.grib2` | `f021` | $+21\text{ h}$ | $6,024,756$ | $5.75\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f024.grib2` | `f024` | $+24\text{ h}$ | $6,076,501$ | $5.80\text{ MB}$ | **VALID** | **YES** |
| `gfs.0p25.2024062100.f027.grib2` | `f027` | $+27\text{ h}$ | $6,034,123$ | $5.75\text{ MB}$ | **VALID** | **YES** |
| **Total / Summary** | **9 Leads** | **3h–27h** | **53,949,403** | **51.45 MB** | **100% PASS** | **100% PASS** |

---

## 3. Variable Diagnostics, Dimensions & Units

Each forecast lead file encapsulates exactly the 7 target variables required by RituGyan's synoptic and localized precipitation models:

| Variable Name | GRIB2 ShortName | Level / Layer | GRIB2 Native Units | Converted Pipeline Units | Subgrid Value Range (Sample f003) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Precipitation** | `tp` (APCP) | `surface` | $\text{kg m}^{-2}$ (mm) | $\text{mm}$ | $[0.00, 114.28]\text{ mm}$ |
| **Pressure Reduced to MSL** | `prmsl` | `mean sea level` | $\text{Pa}$ | $\text{hPa}$ ($/ 100$) | $[989.42, 1009.61]\text{ hPa}$ |
| **2-Metre Temperature** | `2t` (TMP) | `2 m above ground`| $\text{K}$ | $^\circ\text{C}$ ($- 273.15$) | $[1.45, 43.12]\ ^\circ\text{C}$ |
| **2-Metre Dewpoint Temp**| `2d` (DPT) | `2 m above ground`| $\text{K}$ | $^\circ\text{C}$ ($- 273.15$) | $[-15.20, 29.35]\ ^\circ\text{C}$ |
| **10-Metre U-Wind** | `10u` (UGRD) | `10 m above ground`| $\text{m s}^{-1}$ | $\text{m/s}$ | $[-13.24, 18.52]\text{ m/s}$ |
| **10-Metre V-Wind** | `10v` (VGRD) | `10 m above ground`| $\text{m s}^{-1}$ | $\text{m/s}$ | $[-11.85, 16.74]\text{ m/s}$ |
| **Precipitable Water** | `pwat` (PWAT) | `entire atmosphere`| $\text{kg m}^{-2}$ | $\text{kg/m}^2$ | $[12.80, 84.60]\text{ kg/m}^2$ |

### 3.1 Spatial Dimensions & Coordinates
- **Global Native Grid:** $721 \text{ latitudes} \times 1440 \text{ longitudes}$ ($0.25^\circ$ uniform resolution).
- **Subgrid Extracted:** $127 \text{ latitudes} \times 121 \text{ longitudes}$ (15,367 grid points).
- **Latitude Domain:** $6.50^\circ\text{N}$ to $38.00^\circ\text{N}$ ($0.25^\circ$ step, reordered to ascending).
- **Longitude Domain:** $68.00^\circ\text{E}$ to $98.00^\circ\text{E}$ ($0.25^\circ$ step, strictly ascending).
- **Numerical Cleanliness:** **0 NaNs, 0 Infs** across all 9 lead files.

---

## 4. Precipitation Differencing Semantics

In NOAA GFS 0.25° forecasts initialized at 00 UTC:
- `f003`: Cumulative surface precipitation from 00Z to 03Z ($0 \to 3\text{h}$).
- `f027`: Cumulative surface precipitation from 00Z to 27Z ($0 \to 27\text{h}$, i.e., next day 03Z).
- **IMD 24-Hour Aligned Window:** $[03\text{Z}(D) \to 03\text{Z}(D+1)] = A(f027) - A(f003)$.

### Verification Results:
- `f003` Mean Subgrid Accumulation: $2.14\text{ mm}$
- `f027` Mean Subgrid Accumulation: $18.62\text{ mm}$
- Difference $A(f027) - A(f003) \ge 0.0$ is strictly satisfied at all 15,367 grid points.
- 24-Hour GFS Forecast Rainfall Mean over India domain: $16.48\text{ mm}$ (Max: $286.40\text{ mm}$).

---

## 5. Pipeline Integration & Regression Test Suite

1. **Preprocessing Pipeline Compatibility:**
   - `load_gfs_forecast_predictors()` was executed on `data/raw/gfs_single_cycle_test/` and yielded all 8 required GFS feature channels (`gfs_tp_24h`, `gfs_prmsl_mean`, `gfs_2t_mean`, `gfs_2d_mean`, `gfs_10u_mean`, `gfs_10v_mean`, `gfs_pwat_mean`, `gfs_wind_speed`).
   - `build_aligned_day_sample()` successfully merged IMD ground truth, ERA5 reanalysis predictors, and newly acquired GFS forecast tensors into a canonical $(15, 127, 121)$ sample for `2024-06-21`.

2. **Full Repository Pytest Results:**
   - Total Tests Executed: **97**
   - Passed: **97**
   - Failed: **0**
   - Pass Rate: **100%**

---

## 6. Conclusion & Recommendation

The single-cycle GFS acquisition test is **100% SUCCESSFUL**. The pipeline is:
- **Fast & Bandwidth-Efficient:** Byte-range extraction saves >98% network bandwidth per lead.
- **Resumable & Atomic:** Downloads write to temporary `.tmp` files and atomically finalize.
- **Robust:** Includes exponential backoff retries and strict GRIB2 structure verification.
- **Fully Integrated:** Confirmed drop-in compatibility with RituGyan's multi-year feature pipeline.
