# Stage 1 Operational GFS Feature Implementation

**Feature version:** `stage1_operational_gfs_v1`  
**Status:** Predictor implementation complete; operational Stage 1 model has not been retrained.  
**Scope:** GFS-only feature representation and historical validation store.

## 1. Objective

This version replaces the retrospective same-day ERA5 predictor representation with forecast-time GFS equivalents while preserving the established 15-channel order, seven spatial regions, 17 synoptic concepts, and 122-feature dimensionality. It is implemented separately from `data/processed/feature_store/`.

No LightGBM model was retrained, no existing model artifact was changed, Candidate A was not changed, and Stage 2/API/dashboard work was not started.

## 2. Architecture

The operational path is:

1. Read the existing GFS 00Z `f003`-`f027` GRIB2 sequence using `load_gfs_forecast_predictors()`.
2. Map the eight GFS field products into the stable 15-channel Stage 1 layout.
3. Calculate the 17 synoptic scalars from GFS `prmsl`, `pwat`, `10u`, and `10v` only.
4. Apply the same seven region definitions and cosine-latitude area-weighted aggregation at Stage 1 transformation time.
5. Copy canonical historical targets, masks, and Candidate A labels read-only for retrospective evaluation metadata. They are never inputs to steps 2-4.

Implementation: `src/features/operational_gfs.py`  
Build entry point: `scripts/build_operational_gfs_feature_store.py`

## 3. GFS Variable Mapping

| Operational channel slot | GFS source | Units |
|---|---|---|
| `gfs_tp_24h` | `tp`: `A(0->27) - A(0->3)` | mm |
| `gfs_prmsl_mean` | `prmsl` forecast mean | hPa |
| `gfs_2t_mean` | `2t` forecast mean | deg C |
| `gfs_2d_mean` | `2d` forecast mean | deg C |
| `gfs_10u_mean` | `10u` forecast mean | m/s |
| `gfs_10v_mean` | `10v` forecast mean | m/s |
| `gfs_pwat_mean` | `pwat` forecast mean | kg/m^2 |
| `gfs_wind_speed` | `sqrt(gfs_10u_mean^2 + gfs_10v_mean^2)` | m/s |
| Former `era5_msl_mean` slot | `gfs_prmsl_mean` | hPa |
| Former `era5_t2m_mean` slot | `gfs_2t_mean` | deg C |
| Former `era5_d2m_mean` slot | `gfs_2d_mean` | deg C |
| Former `era5_u10_mean` slot | `gfs_10u_mean` | m/s |
| Former `era5_v10_mean` slot | `gfs_10v_mean` | m/s |
| Former `era5_tcwv_mean` slot | `gfs_pwat_mean` | kg/m^2 |
| Former `era5_wind_speed` slot | `gfs_wind_speed` | m/s |

The former ERA5 slot names are retained in the stable channel order for representation comparison. The store metadata records their GFS source mapping explicitly.

## 4. Spatial Features

The seven unchanged regions are `domain`, `depression_track`, `central_india`, `arabian_sea_gateway`, `western_ghats`, `northwest_india`, and `himalayan_foothills`. Each of the 15 channels is aggregated over each region using the existing cosine(latitude) weights. This produces 105 spatial features in the same region-major/channel-minor order.

## 5. Synoptic Mapping

All 17 names and their conceptual order are retained. The source arrays are now GFS forecast means:

* Trough position, departure, minimum, and gradient use GFS MSLP.
* Inflow speed, zonal/meridional flow, and kinetic energy use GFS 10-m winds.
* Domain and Central India MFC use the existing spherical 2-D `-div(PWAT * V10)` surface proxy.
* Depression anomaly uses GFS MSLP; surface vorticity uses GFS 10-m winds.
* Western Ghats flux uses `GFS PWAT * GFS 10u`.
* Northwest and domain moisture/pressure metrics use GFS PWAT and MSLP.

MFC is explicitly **not** vertically integrated moisture flux convergence. Depression and MFC thresholds are inherited only for representation continuity and require GFS-specific recalibration/revalidation before operational model training.

## 6. Candidate A Limitation

Candidate A labels are copied unchanged from the canonical store. They remain ERA5-conditioned, research-derived proxy labels rather than independent meteorological ground truth. This operational predictor version does not regenerate labels and does not alter regime IDs.

## 7. GFS Precipitation Limitation

`gfs_tp_24h` is retained because it is available from the forecast at issuance after the 00Z cycle and exactly represents the IMD-aligned 03Z-03Z forecast window through `A(0->27)-A(0->3)`. It is also a forecast precipitation outcome proxy, so its inclusion requires a future ablation and validation before final Stage 1 selection.

## 8. Store Structure and Provenance

The versioned output is `data/processed/feature_store_operational_gfs/` and contains:

* `tensors.nc`: GFS-only predictor tensor, historical target/mask copies, synoptic vector, and unchanged label IDs.
* `metadata.parquet`: unchanged historical label metadata plus existing evaluation metadata.
* `feature_names.json`: deterministic 122-name operational order.
* `store_config.json`: version, source mapping, regions, split definitions, limitations, and provenance.

The forecast-time source path is `data/raw/gfs_historical/{year}`. For the completed historical store, `--reuse-canonical-gfs` selected only the eight GFS channel planes from the canonical tensor, avoiding a second decode of 4,392 GRIB files. The operational output contains no ERA5 channel values; the source mode is recorded in `store_config.json`. The raw-GFS path remains available for independent rebuilds.

## 9. Validation

Validation covers channel mapping, deterministic feature ordering, finite GFS-only synoptic values, real single-cycle GFS construction, store shape, date coverage, chronological splits, and leakage/provenance metadata.

Expected design invariants are:

* **488 samples:** 122 JJAS days for each of 2021-2024.
* **Chronology:** 2021-2022 train, 2023 validation, 2024 test.
* **Predictor tensor:** `(488, 15, 127, 121)`.
* **Synoptic tensor:** `(488, 17)`.
* **Feature names:** 122, with exact stable ordering.
* **Data quality:** zero predictor or synoptic NaN/Inf values.
* **GFS mapping:** exact equality between operational channels and the selected canonical GFS planes.
* **ERA5 predictor dependency:** absent from the operational tensor; `era5_predictors_used` is recorded as `false`.

## 10. Tests

Focused tests are in `tests/phase3/test_operational_gfs_features.py`: **5 passed**. The existing repository suite also passed: **115 passed**. No LightGBM training test was run and no operational model was trained.

## 11. Known Limitations

The labels are research proxies, the depression/MFC thresholds have not been recalibrated for GFS forecast error, and `gfs_tp_24h` needs ablation. Numerical distributions must be compared with the original ERA5 representation before retraining. A true 850-hPa LLJ feature is not included and would require additional pressure-level GFS data. The raw-GFS rebuild path was not completed in this run because decoding the full historical GRIB corpus exceeded the available execution window; the completed historical artifact used the explicitly recorded canonical-GFS fast path.

## 12. Next Step

Review the completed store and distribution comparison, then decide whether to recalibrate synoptic thresholds and run the planned `gfs_tp_24h` ablation. Only after those checks should a separately versioned Stage 1 LightGBM model be trained.
