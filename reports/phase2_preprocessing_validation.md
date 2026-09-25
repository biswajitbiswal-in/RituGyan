# Phase 2C: Multi-Source Preprocessing & Aligned Feature Tensor Pipeline Report — RituGyan

**Date:** September 25, 2026  
**Status:** **PASS**  
**Author:** RituGyan Pipeline Diagnostic & Feature Engineering Engine  

---

## Executive Summary

Phase 2C establishes and validates the end-to-end multi-source preprocessing pipeline that fuses:
1. **IMD 24-Hour Gridded Rainfall Observation** ($y \in \mathbb{R}^{127 \times 121}$ with explicit land/sea mask $\mathcal{M} \in \{0, 1\}^{127 \times 121}$).
2. **ERA5 6-Hourly Reanalysis Predictors** ($6\text{ physical fields} + 1\text{ derived wind speed}$ averaged across in-window timestamps `06Z`, `12Z`, `18Z`, `00Z(D+1)`).
3. **NOAA GFS Forecast Predictors & Cumulative Rainfall** ($7\text{ fields}$ including the validated 24h precipitation differencing $\mathcal{A}_{\text{f027}} - \mathcal{A}_{\text{f003}} + 1\text{ derived wind speed}$).

Using the validation fixture date **$D = \text{2024-06-21}$**, the pipeline extracts, converts, and stacks all features into a structured tensor $X \in \mathbb{R}^{15 \times 127 \times 121}$ and target $y \in \mathbb{R}^{127 \times 121}$ with zero interpolation, zero NaN-filling on observations, and rigorous unit standardizations.

---

## 1. Feature Channel Catalog & Physical Units

The assembled feature tensor $X$ contains **15 distinct meteorological channels**:

| Index | Channel Name | Source Dataset | Physical Meaning | Native Raw Unit | Converted Target Unit | Transformation Operation |
|---|---|---|---|---|---|---|
| **0** | `gfs_tp_24h` | NOAA GFS | 24h accumulated forecast precipitation | `kg m**-2` | **`mm`** | $\mathcal{A}_{\text{f027}} - \mathcal{A}_{\text{f003}}$ |
| **1** | `gfs_prmsl_mean` | NOAA GFS | Mean sea-level pressure | `Pa` | **`hPa`** | $\text{value} / 100.0$ |
| **2** | `gfs_2t_mean` | NOAA GFS | Mean 2m air temperature | `K` | **`°C`** | $\text{value} - 273.15$ |
| **3** | `gfs_2d_mean` | NOAA GFS | Mean 2m dewpoint temperature | `K` | **`°C`** | $\text{value} - 273.15$ |
| **4** | `gfs_10u_mean` | NOAA GFS | Mean 10m zonal wind component | `m s**-1` | **`m/s`** | Identity |
| **5** | `gfs_10v_mean` | NOAA GFS | Mean 10m meridional wind component | `m s**-1` | **`m/s`** | Identity |
| **6** | `gfs_pwat_mean` | NOAA GFS | Mean total precipitable water | `kg m**-2` | **`kg/m^2`** | Identity |
| **7** | `gfs_wind_speed` | NOAA GFS (Derived) | Mean 10m total wind speed | `m s**-1` | **`m/s`** | $\sqrt{u^2 + v^2}$ |
| **8** | `era5_msl_mean` | ECMWF ERA5 | In-window mean sea-level pressure | `Pa` | **`hPa`** | $\text{value} / 100.0$ |
| **9** | `era5_t2m_mean` | ECMWF ERA5 | In-window mean 2m air temperature | `K` | **`°C`** | $\text{value} - 273.15$ |
| **10** | `era5_d2m_mean` | ECMWF ERA5 | In-window mean 2m dewpoint | `K` | **`°C`** | $\text{value} - 273.15$ |
| **11** | `era5_u10_mean` | ECMWF ERA5 | In-window mean 10m zonal wind | `m s**-1` | **`m/s`** | Identity |
| **12** | `era5_v10_mean` | ECMWF ERA5 | In-window mean 10m meridional wind | `m s**-1` | **`m/s`** | Identity |
| **13** | `era5_tcwv_mean` | ECMWF ERA5 | In-window mean total column water vapour | `kg m**-2` | **`kg/m^2`** | Identity |
| **14** | `era5_wind_speed` | ECMWF ERA5 (Derived)| In-window mean 10m total wind speed | `m s**-1` | **`m/s`** | $\sqrt{u^2 + v^2}$ |

---

## 2. Target Variable & Missing Data Handling

* **Target Variable ($y$)**: IMD 24-hour daily accumulated rainfall [$\text{mm}$] covering $[D\text{ 03:00 UTC} \to D+1\text{ 03:00 UTC}]$.
* **Spatial Shape**: `(127, 121)` ($15,367$ grid cells).
* **Missing Data / Ocean Mask Handling**:
  - IMD gridded observation data contains valid records exclusively over the Indian landmass; maritime and extra-territorial cells within the $[6.50^\circ\text{N} \dots 38.00^\circ\text{N}, 68.00^\circ\text{E} \dots 98.00^\circ\text{E}]$ bounding box are marked as `NaN`.
  - **Zero-Filling Prohibition**: Missing observations are **NOT** silently filled with zeros. Doing so would corrupt the training distribution by treating unmonitored ocean points as "zero rain".
  - **Explicit Boolean Mask**: The pipeline outputs an explicit `valid_mask` array ($\text{bool}$, shape `(127, 121)`):
    $$\text{valid\_mask}[i, j] = \begin{cases} \text{True} & \text{if } y[i, j] \text{ is finite (monitored Indian land cell)} \\ \text{False} & \text{if } y[i, j] \text{ is NaN (ocean / masked cell)} \end{cases}$$
  - For target day `2024-06-21`:
    - Valid land cells: **$4,964$** ($32.30\%$)
    - Masked ocean/non-monitored cells: **$10,403$** ($67.70\%$)

---

## 3. One-Day Fixture Validation Statistics (2024-06-21)

| Tensor Component | Shape | Dtype | Min Value | Max Value | Mean Value | NaN / Inf Count |
|---|---|---|---|---|---|---|
| **Feature Tensor ($X$)** | `(15, 127, 121)` | `float32` | Vary by channel | Vary by channel | Vary by channel | **0 / 0** (100% complete) |
| `gfs_tp_24h` | `(127, 121)` | `float32` | `0.00 mm` | `86.25 mm` | `7.68 mm` | 0 |
| `gfs_prmsl_mean` | `(127, 121)` | `float32` | `992.83 hPa` | `1008.23 hPa` | `1001.45 hPa` | 0 |
| `gfs_2t_mean` | `(127, 121)` | `float32` | `-18.84 °C` | `41.65 °C` | `28.61 °C` | 0 |
| `gfs_2d_mean` | `(127, 121)` | `float32` | `-24.81 °C` | `27.75 °C` | `18.59 °C` | 0 |
| `gfs_pwat_mean` | `(127, 121)` | `float32` | `3.15 kg/m^2` | `74.05 kg/m^2` | `48.27 kg/m^2` | 0 |
| `era5_msl_mean` | `(127, 121)` | `float32` | `994.12 hPa` | `1008.92 hPa` | `1002.34 hPa` | 0 |
| `era5_t2m_mean` | `(127, 121)` | `float32` | `-11.62 °C` | `35.77 °C` | `26.85 °C` | 0 |
| `era5_tcwv_mean` | `(127, 121)` | `float32` | `4.62 kg/m^2` | `69.31 kg/m^2` | `46.12 kg/m^2` | 0 |
| **Target Rainfall ($y$)** | `(127, 121)` | `float32` | `0.00 mm` | `244.78 mm` | `5.26 mm` (land) | 10,403 NaNs (Ocean) |
| **Valid Mask ($\mathcal{M}$)** | `(127, 121)` | `bool` | `False` (0) | `True` (1) | `0.3230` | 0 |

---

## 4. Automated Test Suite Results

Test execution across the preprocessing test suite (`tests/phase2/test_preprocessing.py`):

```
tests/phase2/test_preprocessing.py::TestPreprocessingPipeline::test_feature_channels_and_units_specification PASSED
tests/phase2/test_preprocessing.py::TestPreprocessingPipeline::test_load_imd_target_day PASSED
tests/phase2/test_preprocessing.py::TestPreprocessingPipeline::test_load_era5_in_window_predictors PASSED
tests/phase2/test_preprocessing.py::TestPreprocessingPipeline::test_load_gfs_forecast_predictors PASSED
tests/phase2/test_preprocessing.py::TestPreprocessingPipeline::test_build_aligned_day_sample_structure PASSED

======================== 5 passed in 3.72s ========================
```

---

## 5. Final Assessment & Next Steps

### FINAL STATUS: **PASS**

### Summary of Accomplishments:
1. **Standardized 15-Channel Pipeline**: Designed and validated `build_aligned_day_sample()` in `src/features/preprocessing.py`.
2. **Strict Physical Unit Standardization**: Surface pressure ($\text{Pa} \to \text{hPa}$), temperature ($K \to ^\circ\text{C}$), rainfall ($\text{mm}$), wind speed ($\text{m/s}$), moisture ($\text{kg/m}^2$).
3. **Safe Missing-Data Architecture**: Explicit `valid_mask` separates monitored Indian land cells from unmonitored ocean points without zero-filling artifacts.
4. **Zero-Interpolation Guarantee**: Preserves raw discrete quarter-degree grid values across all sources.

### Next Exact Step (Phase 2D):
Proceed to **Phase 2D: Synoptic Feature Engineering & Meteorological Index Computation**:
- Implement synoptic-scale monsoon index calculators (e.g. Monsoon Trough Position Index, Low-Level Jet (LLJ) 850 hPa Wind Shear, Moisture Flux Convergence, Cross-Equatorial Flow Index) and regime categorization functions.
