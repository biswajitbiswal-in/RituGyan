# Phase 2F: Historical IMD and ERA5 Data Validation Report

**Author:** RituGyan Core Engineering Team  
**Date:** September 2026  
**Status:** COMPLETED & PASSED  
**Scope:** Strict physical, temporal, spatial, and numerical validation of historical IMD (2021–2023) and ERA5 (2021–2023) datasets against the validated 2024 baseline.

---

## 1. Executive Summary

| Dataset Source | Target Period | Grid Resolution | Temporal Coverage | Physical Cleanliness | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Gridded Rainfall (2021)** | JJAS 2021 | 0.25° ($129 \times 135$) | 122 / 122 days (Continuous) | Land mask & units match 2024 | **PASS** |
| **IMD Gridded Rainfall (2022)** | JJAS 2022 | 0.25° ($129 \times 135$) | 122 / 122 days (Continuous) | Land mask & units match 2024 | **PASS** |
| **IMD Gridded Rainfall (2023)** | JJAS 2023 | 0.25° ($129 \times 135$) | 122 / 122 days (Continuous) | Land mask & units match 2024 | **PASS** |
| **IMD Gridded Rainfall (2024)** | JJAS 2024 | 0.25° ($129 \times 135$) | 122 / 122 days (Continuous) | Baseline Reference | **PASS** |
| **ERA5 Reanalysis (2021)** | JJAS 2021 | 0.25° ($129 \times 121$) | 488 / 488 steps (00,06,12,18Z) | 0 NaNs / 0 Infs across all 6 vars | **PASS** |
| **ERA5 Reanalysis (2022)** | JJAS 2022 | 0.25° ($129 \times 121$) | 488 / 488 steps (00,06,12,18Z) | 0 NaNs / 0 Infs across all 6 vars | **PASS** |
| **ERA5 Reanalysis (2023)** | JJAS 2023 | 0.25° ($129 \times 121$) | 488 / 488 steps (00,06,12,18Z) | 0 NaNs / 0 Infs across all 6 vars | **PASS** |
| **ERA5 Reanalysis (2024)** | JJAS 2024 | 0.25° ($129 \times 121$) | 488 / 488 steps (00,06,12,18Z) | Baseline Reference | **PASS** |

**Conclusion:** All downloaded historical IMD (2021–2023) and ERA5 (2021–2023) datasets are **100% physically, spatially, and temporally consistent** with the validated 2024 reference datasets. The repository is **FULLY READY** to proceed to the historical GFS acquisition phase.

---

## 2. IMD Gridded Rainfall Validation (2021–2024)

### 2.1 File Inventory & Metadata
- `data/raw/RF25_ind2021_rfp25.nc` (25.4 MB)
- `data/raw/RF25_ind2022_rfp25.nc` (25.4 MB)
- `data/raw/RF25_ind2023_rfp25.nc` (25.4 MB)
- `data/raw/RF25_ind2024_rfp25.nc` (25.5 MB)

### 2.2 Detailed IMD Audit Matrix

| Metric / Check | 2021 | 2022 | 2023 | 2024 (Baseline) | Validation Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target Period** | June 1 – Sept 30 | June 1 – Sept 30 | June 1 – Sept 30 | June 1 – Sept 30 | **MATCH** |
| **Daily Observations** | 122 days | 122 days | 122 days | 122 days | **PASS (122/122)** |
| **Daily Continuity** | Continuous (0 gaps)| Continuous (0 gaps)| Continuous (0 gaps)| Continuous (0 gaps)| **PASS** |
| **Latitude Range** | $[6.5^\circ\text{N}, 38.5^\circ\text{N}]$ | $[6.5^\circ\text{N}, 38.5^\circ\text{N}]$ | $[6.5^\circ\text{N}, 38.5^\circ\text{N}]$ | $[6.5^\circ\text{N}, 38.5^\circ\text{N}]$ | **EXACT MATCH** |
| **Longitude Range** | $[66.5^\circ\text{E}, 100.0^\circ\text{E}]$ | $[66.5^\circ\text{E}, 100.0^\circ\text{E}]$ | $[66.5^\circ\text{E}, 100.0^\circ\text{E}]$ | $[66.5^\circ\text{E}, 100.0^\circ\text{E}]$ | **EXACT MATCH** |
| **Grid Resolution** | $0.25^\circ \times 0.25^\circ$ ($129 \times 135$) | $0.25^\circ \times 0.25^\circ$ ($129 \times 135$) | $0.25^\circ \times 0.25^\circ$ ($129 \times 135$) | $0.25^\circ \times 0.25^\circ$ ($129 \times 135$) | **EXACT MATCH** |
| **Coordinate Ordering** | Lat: Ascending<br>Lon: Ascending | Lat: Ascending<br>Lon: Ascending | Lat: Ascending<br>Lon: Ascending | Lat: Ascending<br>Lon: Ascending | **EXACT MATCH** |
| **Variable Name & Units**| `RAINFALL` [mm] | `RAINFALL` [mm] | `RAINFALL` [mm] | `RAINFALL` [mm] | **EXACT MATCH** |
| **Value Range [min, max]**| $[0.0, 526.71]$ mm | $[0.0, 979.14]$ mm | $[0.0, 449.08]$ mm | $[0.0, 660.55]$ mm | **REALISTIC** |
| **Land Mask Points / Day**| 4,964 land points | 4,964 land points | 4,964 land points | 4,964 land points | **EXACT MATCH** |
| **Total Land Points (JJAS)**| 605,608 | 605,608 | 605,608 | 605,608 | **EXACT MATCH** |
| **Land Fraction** | 28.50% | 28.50% | 28.50% | 28.50% | **EXACT MATCH** |
| **Mask Stability** | 100% Invariant | 100% Invariant | 100% Invariant | 100% Invariant | **PASS** |
| **Spatial Compatibility**| $\Delta_{\text{coord}} = 0.0$ | $\Delta_{\text{coord}} = 0.0$ | $\Delta_{\text{coord}} = 0.0$ | Baseline | **PASS** |

---

## 3. ERA5 Reanalysis Validation (2021–2024)

### 3.1 File Inventory & Metadata
- `data/raw/era5_21_to_23.nc` (273.1 MB) — Consolidated 2021–2023 JJAS 6-hourly reanalysis
- `data/raw/era5.nc` (85.7 MB) — Validated 2024 JJAS 6-hourly reanalysis baseline

### 3.2 Detailed ERA5 Audit Matrix

| Metric / Check | 2021 | 2022 | 2023 | 2024 (Baseline) | Validation Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target Period** | June 1 – Sept 30 | June 1 – Sept 30 | June 1 – Sept 30 | June 1 – Sept 30 | **MATCH** |
| **Total Timesteps** | 488 ($122 \times 4$) | 488 ($122 \times 4$) | 488 ($122 \times 4$) | 488 ($122 \times 4$) | **PASS (488/488)** |
| **Synoptic Cycles** | 00, 06, 12, 18 UTC | 00, 06, 12, 18 UTC | 00, 06, 12, 18 UTC | 00, 06, 12, 18 UTC | **EXACT MATCH** |
| **Latitude Range** | $[38.0^\circ\text{N}, 6.0^\circ\text{N}]$ | $[38.0^\circ\text{N}, 6.0^\circ\text{N}]$ | $[38.0^\circ\text{N}, 6.0^\circ\text{N}]$ | $[38.0^\circ\text{N}, 6.0^\circ\text{N}]$ | **EXACT MATCH** |
| **Longitude Range** | $[68.0^\circ\text{E}, 98.0^\circ\text{E}]$ | $[68.0^\circ\text{E}, 98.0^\circ\text{E}]$ | $[68.0^\circ\text{E}, 98.0^\circ\text{E}]$ | $[68.0^\circ\text{E}, 98.0^\circ\text{E}]$ | **EXACT MATCH** |
| **Grid Dimensions** | $129 \times 121$ ($0.25^\circ$) | $129 \times 121$ ($0.25^\circ$) | $129 \times 121$ ($0.25^\circ$) | $129 \times 121$ ($0.25^\circ$) | **EXACT MATCH** |
| **Coordinate Ordering** | Lat: Descending<br>Lon: Ascending | Lat: Descending<br>Lon: Ascending | Lat: Descending<br>Lon: Ascending | Lat: Descending<br>Lon: Ascending | **EXACT MATCH** |
| **NaN / Inf Check** | 0 NaNs, 0 Infs | 0 NaNs, 0 Infs | 0 NaNs, 0 Infs | 0 NaNs, 0 Infs | **CLEAN (100%)** |
| **Spatial Compatibility**| $\Delta_{\text{coord}} = 0.0$ | $\Delta_{\text{coord}} = 0.0$ | $\Delta_{\text{coord}} = 0.0$ | Baseline | **PASS** |

### 3.3 ERA5 Variables & Physical Diagnostics

#### Year 2021
- `u10` [$\text{m/s}$]: Min = $-15.16$, Max = $+16.44$, Mean = $2.32$ (NaNs: 0, Infs: 0) — **CLEAN**
- `v10` [$\text{m/s}$]: Min = $-13.88$, Max = $+16.54$, Mean = $1.38$ (NaNs: 0, Infs: 0) — **CLEAN**
- `t2m` [$\text{K}$]: Min = $245.82$, Max = $321.25$, Mean = $296.27$ (NaNs: 0, Infs: 0) — **CLEAN**
- `d2m` [$\text{K}$]: Min = $232.85$, Max = $303.75$, Mean = $291.28$ (NaNs: 0, Infs: 0) — **CLEAN**
- `msl` [$\text{Pa}$]: Min = $98,628.06$, Max = $104,020.44$, Mean = $100,764.90$ (NaNs: 0, Infs: 0) — **CLEAN**
- `tcwv` [$\text{kg/m}^2$]: Min = $0.30$, Max = $89.67$, Mean = $43.31$ (NaNs: 0, Infs: 0) — **CLEAN**

#### Year 2022
- `u10` [$\text{m/s}$]: Min = $-14.60$, Max = $+17.72$, Mean = $2.07$ (NaNs: 0, Infs: 0) — **CLEAN**
- `v10` [$\text{m/s}$]: Min = $-14.79$, Max = $+31.63$, Mean = $1.23$ (NaNs: 0, Infs: 0) — **CLEAN**
- `t2m` [$\text{K}$]: Min = $235.25$, Max = $320.92$, Mean = $296.36$ (NaNs: 0, Infs: 0) — **CLEAN**
- `d2m` [$\text{K}$]: Min = $232.08$, Max = $304.20$, Mean = $291.34$ (NaNs: 0, Infs: 0) — **CLEAN**
- `msl` [$\text{Pa}$]: Min = $98,443.75$, Max = $104,666.44$, Mean = $100,731.96$ (NaNs: 0, Infs: 0) — **CLEAN**
- `tcwv` [$\text{kg/m}^2$]: Min = $0.21$, Max = $84.40$, Mean = $43.26$ (NaNs: 0, Infs: 0) — **CLEAN**

#### Year 2023
- `u10` [$\text{m/s}$]: Min = $-22.05$, Max = $+24.51$, Mean = $2.44$ (NaNs: 0, Infs: 0) — **CLEAN**
- `v10` [$\text{m/s}$]: Min = $-19.18$, Max = $+29.15$, Mean = $1.50$ (NaNs: 0, Infs: 0) — **CLEAN**
- `t2m` [$\text{K}$]: Min = $241.32$, Max = $319.36$, Mean = $296.51$ (NaNs: 0, Infs: 0) — **CLEAN**
- `d2m` [$\text{K}$]: Min = $233.86$, Max = $304.36$, Mean = $291.48$ (NaNs: 0, Infs: 0) — **CLEAN**
- `msl` [$\text{Pa}$]: Min = $96,835.06$, Max = $104,574.25$, Mean = $100,788.61$ (NaNs: 0, Infs: 0) — **CLEAN**
- `tcwv` [$\text{kg/m}^2$]: Min = $0.29$, Max = $91.27$, Mean = $43.30$ (NaNs: 0, Infs: 0) — **CLEAN**

#### Year 2024 (Baseline)
- `u10` [$\text{m/s}$]: Min = $-12.06$, Max = $+18.50$, Mean = $2.35$ (NaNs: 0, Infs: 0) — **CLEAN**
- `v10` [$\text{m/s}$]: Min = $-15.17$, Max = $+17.20$, Mean = $1.55$ (NaNs: 0, Infs: 0) — **CLEAN**
- `t2m` [$\text{K}$]: Min = $245.98$, Max = $320.76$, Mean = $296.82$ (NaNs: 0, Infs: 0) — **CLEAN**
- `d2m` [$\text{K}$]: Min = $235.28$, Max = $304.66$, Mean = $291.91$ (NaNs: 0, Infs: 0) — **CLEAN**
- `msl` [$\text{Pa}$]: Min = $98,444.12$, Max = $103,788.00$, Mean = $100,714.36$ (NaNs: 0, Infs: 0) — **CLEAN**
- `tcwv` [$\text{kg/m}^2$]: Min = $0.41$, Max = $94.94$, Mean = $45.62$ (NaNs: 0, Infs: 0) — **CLEAN**

---

## 4. Test Suite Execution & Verification

The complete project test suite was executed, including:
- Unit and ingestion tests (`tests/unit/`, `tests/phase1/`)
- Spatial and temporal alignment suites (`tests/phase2/test_spatial_alignment.py`, `tests/phase2/test_temporal_alignment.py`)
- Synoptic feature extraction and regime classification (`tests/phase2/test_synoptic_features.py`, `tests/phase2/test_regime_classifier.py`)
- Multi-source batch ingestion and feature store verification (`tests/phase2/test_multiyear_dataset.py`, `tests/phase2/test_feature_store.py`)
- Historical dataset validation suite (`tests/phase2/test_historical_validation.py`)

### Test Results Summary:
- **Total Tests Executed:** 85
- **Passed:** 85
- **Failed:** 0
- **Pass Rate:** **100%**

---

## 5. Decision & Next Step Recommendation

| Year | IMD Status | ERA5 Status | Overall Historical Status |
| :--- | :--- | :--- | :--- |
| **2021** | **PASS** (122 / 122 days) | **PASS** (488 / 488 steps) | **VALIDATED** |
| **2022** | **PASS** (122 / 122 days) | **PASS** (488 / 488 steps) | **VALIDATED** |
| **2023** | **PASS** (122 / 122 days) | **PASS** (488 / 488 steps) | **VALIDATED** |
| **2024** | **PASS** (122 / 122 days) | **PASS** (488 / 488 steps) | **VALIDATED** |

### Historical GFS Acquisition Readiness: **READY**
1. Ground truth IMD rainfall targets for 2021–2024 are fully available and validated ($4 \times 122 = 488$ daily rainfall fields).
2. Boundary and synoptic state ERA5 reanalysis fields for 2021–2024 are fully available and validated ($4 \times 488 = 1,952$ 6-hourly snapshots).
3. The spatial subgrid extracting pipeline ($127 \times 121$) is verified to align IMD, ERA5, and GFS grids identically.
4. GFS forecast downloading can safely proceed for JJAS 2021–2024 knowing that both IMD targets and ERA5 predictors are sound and aligned.
