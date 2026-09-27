# Stage 1 Operational Feature Availability Audit

**Date:** 2026-09-27  
**Scope:** Existing Stage 1 LightGBM artifact `stage1_lightgbm_5class_candidate_a_v1`  
**Mode:** Audit only. No model, feature store, labels, datasets, or pipeline code were modified.

## Executive Finding

The saved Stage 1 model has exactly **122 features**: 105 spatial summaries (`7 regions x 15 channels`) and 17 stored synoptic scalars. The trained model does not read `target`, `valid_mask`, rainfall metadata, or Candidate A labels as predictors.

However, the current feature store is a **research/retrospective representation, not an operational predictor set**:

* 56 spatial features use GFS forecast fields and are operationally usable as `OPERATIONAL_WITH_GFS`.
* 49 spatial features use same-day, in-window ERA5 reanalysis and are `ERA5_REANALYSIS`.
* All 17 synoptic scalars are calculated from those ERA5 fields and are therefore also `ERA5_REANALYSIS`.
* No current Stage 1 feature is `OBSERVATION_DEPENDENT` or `TARGET_OR_LABEL_DERIVED`.
* The 66 ERA5-dependent features should not be silently used at live forecast issuance. Most are conceptually replaceable with GFS equivalents; the depression/vorticity and moisture-convergence proxies need meteorological validation before adoption.

The principal operational risk is therefore **availability mismatch, not remaining target leakage**. The reported feature-importance concentration in ERA5 and ERA5-derived synoptic variables is materially affected by this mismatch.

## 1. Implementation Trace

| Item | Actual implementation evidence |
|---|---|
| Training entry point | `scripts/train_stage1_regime.py` calls `train_stage1_from_store()`; the model fitting implementation is `src/models/stage1_regime/classifier.py::train_stage1_model`. |
| Predictor construction | `classifier.py::_sample_predictors()` computes cosine-latitude area-weighted means over `SPATIAL_REGIONS`, then appends the stored synoptic vector. |
| Feature order | Region-major, channel-minor spatial features, followed by `synoptic__...` names. The canonical saved order is `models/regime_classifier/stage1_lightgbm_5class_candidate_a_v1/feature_names.json`. |
| Count | `7 x len(FEATURE_CHANNELS) + len(SYNOPTIC_FEATURE_NAMES) = 7 x 15 + 17 = 122`. The artifact metadata independently records `feature_count: 122`. |
| Model loader inputs | `load_stage1_datasets()` reads tensor `features`, tensor `synoptic_vector`, and metadata `date/regime_id/regime_code`. It does not read tensor `target` or `valid_mask`. |
| Spatial calculation | For each region and channel: `sum(value * cos(latitude)) / sum(cos(latitude))`; no target or land-mask weighting is applied. |
| Source tensor construction | `build_aligned_day_sample()` loads IMD target/mask, ERA5 window predictors, and GFS forecast predictors. The 15 feature planes contain only the named GFS and ERA5 channels. |
| Synoptic construction | `scripts/build_feature_store.py` and `batch_pipeline.py` pass `era5_msl_mean`, `era5_tcwv_mean`, `era5_u10_mean`, and `era5_v10_mean` to `extract_all_synoptic_features()`. |

## 2. Operational Status Categories Used

`OPERATIONAL` means calculable from issuance-time information without a special forecast dependency. `OPERATIONAL_WITH_GFS` means it is operationally available from the GFS forecast sequence. `ERA5_REANALYSIS` means the current implementation uses retrospective same-day ERA5. `OBSERVATION_DEPENDENT` means it needs an observation unavailable for the forecast horizon. `TARGET_OR_LABEL_DERIVED` means direct or indirect target/label leakage. `UNCERTAIN` means provenance or timing cannot be established from the implementation.

## 3. Exhaustive 122-Feature Inventory

The following table is exhaustive. Each spatial row expands to **seven concrete saved feature names**, one for each region in `domain`, `depression_track`, `central_india`, `arabian_sea_gateway`, `western_ghats`, `northwest_india`, and `himalayan_foothills`. Thus the table enumerates `15 x 7 = 105` exact names by explicit template, followed by all 17 exact synoptic names. The templates match `feature_names.json` character-for-character.

### 3.1 Spatial features: 105 names

| Exact feature-name template (`{region}` takes each of the seven regions) | Source | Calculation | Status | Reason / operational disposition |
|---|---|---|---|---|
| `spatial__gfs_tp_24h__{region}__area_weighted_mean` | GFS `tp` | Cosine-latitude mean of `A(0->27)-A(0->3)` | OPERATIONAL_WITH_GFS | Forecast precipitation from 00Z f003/f027 is available after the cycle; no IMD rain is used. |
| `spatial__gfs_prmsl_mean__{region}__area_weighted_mean` | GFS `prmsl` | Mean of GFS forecast MSLP fields, Pa to hPa, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_2t_mean__{region}__area_weighted_mean` | GFS `2t` | Mean GFS 2-m temperature, K to C, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_2d_mean__{region}__area_weighted_mean` | GFS `2d` | Mean GFS 2-m dewpoint, K to C, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_10u_mean__{region}__area_weighted_mean` | GFS `10u` | Mean forecast 10-m zonal wind, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_10v_mean__{region}__area_weighted_mean` | GFS `10v` | Mean forecast 10-m meridional wind, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_pwat_mean__{region}__area_weighted_mean` | GFS `pwat` | Mean forecast precipitable water, then area-weighted | OPERATIONAL_WITH_GFS | Future GFS forecast field can be produced operationally. |
| `spatial__gfs_wind_speed__{region}__area_weighted_mean` | GFS `10u`, `10v` | `sqrt(u10^2+v10^2)` on each field, then mean and area-weight | OPERATIONAL_WITH_GFS | Derived only from operational GFS fields. |
| `spatial__era5_msl_mean__{region}__area_weighted_mean` | ERA5 `msl` | Mean of D06/D12/D18/D+1 00Z same-day ERA5 MSLP, Pa to hPa, then area-weighted | ERA5_REANALYSIS | Those in-window analysis values are retrospective and unavailable at issuance; replace with GFS `prmsl`. |
| `spatial__era5_t2m_mean__{region}__area_weighted_mean` | ERA5 `t2m` | Mean of four same-day ERA5 snapshots, K to C, then area-weighted | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS `2t`. |
| `spatial__era5_d2m_mean__{region}__area_weighted_mean` | ERA5 `d2m` | Mean of four same-day ERA5 snapshots, K to C, then area-weighted | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS `2d`. |
| `spatial__era5_u10_mean__{region}__area_weighted_mean` | ERA5 `u10` | Mean of four same-day ERA5 snapshots, then area-weighted | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS `10u`. |
| `spatial__era5_v10_mean__{region}__area_weighted_mean` | ERA5 `v10` | Mean of four same-day ERA5 snapshots, then area-weighted | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS `10v`. |
| `spatial__era5_tcwv_mean__{region}__area_weighted_mean` | ERA5 `tcwv` | Mean of four same-day ERA5 snapshots, then area-weighted | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS `pwat`. |
| `spatial__era5_wind_speed__{region}__area_weighted_mean` | ERA5 `u10`, `v10` | `sqrt(u10^2+v10^2)` per snapshot, temporal mean, then area-weight | ERA5_REANALYSIS | Same-day reanalysis is not an issuance-time input; replace with GFS wind speed. |

**Expansion note:** `{region}` is exactly each of the seven region strings listed above, so every one of the 105 saved spatial names is covered. The artifact file is the authoritative concrete expansion and was not changed.

### 3.2 Synoptic features: 17 exact names

| Exact feature | Source | Calculation | Status | Reason / operational disposition |
|---|---|---|---|---|
| `synoptic__trough_mean_latitude` | ERA5 `msl` | Mean latitude of column-wise minimum MSLP in 16-30N, 72-86E | ERA5_REANALYSIS | Same-day ERA5 trough analysis unavailable at issuance; reconstruct from GFS MSLP. |
| `synoptic__trough_latitude_departure` | ERA5 `msl` | Trough mean latitude minus fixed 22.5N climatology | ERA5_REANALYSIS | Same dependency; GFS MSLP equivalent is plausible. |
| `synoptic__trough_min_pressure_hpa` | ERA5 `msl` | Minimum MSLP in trough corridor | ERA5_REANALYSIS | Same dependency; use GFS MSLP forecast. |
| `synoptic__trough_pressure_gradient_hpa` | ERA5 `msl` | MSLP at 10N/78E minus trough minimum | ERA5_REANALYSIS | Same dependency; use GFS MSLP forecast. |
| `synoptic__low_latitude_inflow_speed_ms` | ERA5 `u10`, `v10` | Mean `sqrt(u10^2+v10^2)` in 6.5-12N, 68-78E | ERA5_REANALYSIS | Same-day ERA5 surface wind unavailable; GFS 10-m wind equivalent is practical. |
| `synoptic__low_latitude_inflow_zonal_ms` | ERA5 `u10` | Mean zonal wind in gateway | ERA5_REANALYSIS | Replace with the GFS 10u gateway mean. |
| `synoptic__low_latitude_inflow_meridional_ms` | ERA5 `v10` | Mean meridional wind in gateway | ERA5_REANALYSIS | Replace with the GFS 10v gateway mean. |
| `synoptic__low_latitude_kinetic_energy` | ERA5 `u10`, `v10` | `0.5 * mean(u10^2+v10^2)` in gateway | ERA5_REANALYSIS | Replace with the same proxy calculated from GFS 10-m winds. |
| `synoptic__mfc_domain_mean_mm_day` | ERA5 `tcwv`, `u10`, `v10` | Spherical 2-D `-div(PWAT * surface wind)`, converted to mm/day, domain mean | ERA5_REANALYSIS | GFS can supply an analogous proxy, but its scientific validity and forecast error need validation. |
| `synoptic__mfc_central_india_mean_mm_day` | ERA5 `tcwv`, `u10`, `v10` | Same 2-D MFC proxy, central-India regional mean | ERA5_REANALYSIS | GFS equivalent is possible; validate proxy skill before replacement. |
| `synoptic__depression_mslp_anomaly_hpa` | ERA5 `msl` | Track-box minimum MSLP minus domain mean | ERA5_REANALYSIS | GFS MSLP equivalent is practical, but thresholds require recalibration/validation. |
| `synoptic__depression_max_vorticity_s1` | ERA5 `u10`, `v10` | Maximum spherical relative vorticity in depression track box | ERA5_REANALYSIS | GFS 10-m wind can provide the same surface proxy; meteorological validity is uncertain. |
| `synoptic__orographic_ghats_zonal_flux` | ERA5 `tcwv`, `u10` | Mean `PWAT * u10` over Western Ghats corridor | ERA5_REANALYSIS | Replace with GFS `pwat * 10u`; thresholds must be checked after source substitution. |
| `synoptic__nw_india_min_mslp_hpa` | ERA5 `msl` | Northwest-box minimum MSLP | ERA5_REANALYSIS | Replace with GFS MSLP forecast. |
| `synoptic__nw_india_mean_pwat_mm` | ERA5 `tcwv` | Northwest-box mean PWAT | ERA5_REANALYSIS | Replace with GFS PWAT forecast. |
| `synoptic__domain_mean_pwat_mm` | ERA5 `tcwv` | Whole-domain mean TCWV | ERA5_REANALYSIS | Replace with GFS PWAT forecast. |
| `synoptic__domain_mean_mslp_hpa` | ERA5 `msl` | Whole-domain mean MSLP | ERA5_REANALYSIS | Replace with GFS MSLP forecast. |

## 4. ERA5 Dependency Trace

ERA5 enters the trained model twice: directly through 49 spatial ERA5 summaries and indirectly through all 17 synoptic scalars. It is not merely a diagnostic side channel. The batch builder explicitly selects ERA5 channel planes before calling `extract_all_synoptic_features()`, and the resulting 17 values are serialized into `synoptic_vector` and loaded by Stage 1.

The implementation uses ERA5 timestamps D 06Z, D 12Z, D 18Z, and D+1 00Z. These are retrospective in-window atmospheric states, not information available when the 00Z GFS forecast is issued. The ERA5 features are therefore unavailable for live inference even though they are physically meaningful retrospective predictors. They may be retained for a research benchmark, or replaced by GFS forecast fields for an operational model. No replacement should be implemented without retraining and revalidation; this audit does not do that.

The current repository already contains the required GFS surface variables for direct replacements: `prmsl`, `2t`, `2d`, `10u`, `10v`, `pwat`, and `tp`. It does **not** contain pressure-level GFS `u850/v850`; those would require additional pressure-level data and are not needed to reproduce the current 17-feature vector.

## 5. Leakage and Label Audit

* **Observed IMD target `y`:** Loaded while building each sample and serialized in `target`, but never read by `load_stage1_datasets()` or `_sample_predictors()`. The target-change regression test confirms predictor matrices are unchanged when `target` is replaced.
* **`valid_mask`:** Passed to Candidate A classifier, but when `rainfall_field=None`, `_compute_rainfall_distribution()` returns zero/default rainfall evidence and does not inspect the mask. Stage 1 itself never reads it.
* **Observed rainfall:** `batch_pipeline.py` and `scripts/build_feature_store.py` explicitly call `classify(..., rainfall_field=None)`. The earlier `rainfall_field=None` issue is therefore fixed in the current path.
* **Candidate A labels:** `regime_id` and `regime_code` are read as labels only. They are not included in the feature matrix. `predict_regime_for_date()` displays the label for comparison after prediction; it does not feed it to the model.
* **Indirect label leakage:** The labels are deterministic rules over the same ERA5-derived synoptic values that are also model predictors. This is not target leakage, but it creates a **label-proxy dependence/circularity**: the model learns to reproduce an ERA5-conditioned research rule. It is a major interpretation limitation for operational deployment.
* **Post-event information:** The ERA5 values are in-window and may include atmospheric states after the nominal issuance time. They are not observed rainfall, but they are still unavailable at issuance and must be removed/replaced for an operational claim.

## 6. Candidate A Label Audit

Candidate A receives the ERA5-derived synoptic feature set. Its rules use trough position, low-latitude inflow, domain PWAT, MFC, depression pressure/vorticity proxies, and Western Ghats flux. In the actual Stage 1 build, `rainfall_field=None`, so rainfall concentration rules are disabled and `valid_mask` does not reveal the target.

Candidate A labels should be treated as **research-derived proxy labels**, not independent meteorological ground truth. ERA5 use in the labels does not create a train/test date leak because the split is chronological (2021-2022 train, 2023 validation, 2024 test), but it means every split is evaluated against labels produced by the same ERA5-conditioned rule system. A future operational classifier can still be trained against these labels, but its result should be described as emulating a research proxy, not as learning independently observed regime truth. The methodology must explicitly document the ERA5-conditioned label provenance and the need for operational relabeling or independent verification.

## 7. Proposed Operational Feature Set

### KEEP

Keep the 56 GFS spatial features in the design, subject to confirming the issuance cycle and lead availability: all seven-region summaries of `gfs_tp_24h`, `gfs_prmsl_mean`, `gfs_2t_mean`, `gfs_2d_mean`, `gfs_10u_mean`, `gfs_10v_mean`, `gfs_pwat_mean`, and `gfs_wind_speed`.

### REPLACE

Replace the 49 ERA5 spatial features one-for-one with the corresponding GFS channel and region. Replace the following synoptic features with GFS-derived equivalents: trough position/departure/pressure metrics, low-latitude inflow metrics, Western Ghats moisture flux, Northwest MSLP/PWAT, and domain MSLP/PWAT. The closest source variables are the existing GFS `prmsl`, `10u`, `10v`, and `pwat` fields.

### INVESTIGATE

Before replacing, validate GFS versions of both MFC features, `depression_max_vorticity_s1`, and `depression_mslp_anomaly_hpa` against independent meteorological diagnostics. Recheck all Candidate A thresholds after source substitution. Investigate whether the 24-hour GFS precipitation predictor is appropriate for Stage 1 regime classification, since it is a forecast outcome proxy rather than an atmospheric state field.

### REMOVE

Remove current same-day ERA5 features from any claimed live operational inference if no GFS reconstruction is available. Do not substitute them with observed rainfall, IMD target summaries, valid-land statistics derived from future observations, or Candidate A labels.

## 8. Feature Migration Plan

| Current feature family | Current source | Proposed replacement | Proposed source | Already in repository? | Additional pressure-level GFS data? |
|---|---|---|---|---|---|
| All 7-region `era5_msl_mean` summaries | ERA5 `msl` | Matching region mean `gfs_prmsl_mean` | GFS `prmsl` | Yes | No |
| All 7-region `era5_t2m_mean` summaries | ERA5 `t2m` | Matching region mean `gfs_2t_mean` | GFS `2t` | Yes | No |
| All 7-region `era5_d2m_mean` summaries | ERA5 `d2m` | Matching region mean `gfs_2d_mean` | GFS `2d` | Yes | No |
| All 7-region `era5_u10_mean` summaries | ERA5 `u10` | Matching region mean `gfs_10u_mean` | GFS `10u` | Yes | No |
| All 7-region `era5_v10_mean` summaries | ERA5 `v10` | Matching region mean `gfs_10v_mean` | GFS `10v` | Yes | No |
| All 7-region `era5_tcwv_mean` summaries | ERA5 `tcwv` | Matching region mean `gfs_pwat_mean` | GFS `pwat` | Yes | No |
| All 7-region `era5_wind_speed` summaries | ERA5 `u10/v10` | Matching region mean `gfs_wind_speed` | GFS `10u/10v` | Yes | No |
| Trough latitude, departure, minimum and gradient | ERA5 `msl` | Same index on forecast MSLP | GFS `prmsl` | Yes | No |
| Low-latitude inflow speed, u, v, kinetic energy | ERA5 `u10/v10` | Same gateway proxies | GFS `10u/10v` | Yes | No |
| Domain and central-India MFC proxies | ERA5 `tcwv/u10/v10` | Same 2-D proxy, with validation | GFS `pwat/10u/10v` | Yes | No |
| Depression MSLP anomaly | ERA5 `msl` | Same pressure anomaly | GFS `prmsl` | Yes | No |
| Depression maximum vorticity | ERA5 `u10/v10` | Same surface-vorticity proxy, with validation | GFS `10u/10v` | Yes | No |
| Western Ghats zonal moisture flux | ERA5 `tcwv/u10` | Same flux proxy | GFS `pwat/10u` | Yes | No |
| Northwest MSLP/PWAT and domain MSLP/PWAT | ERA5 `msl/tcwv` | Same regional/domain summaries | GFS `prmsl/pwat` | Yes | No |

The repository’s optional 850-hPa LLJ implementation is separate from the current 122 features. If a future model adds a true LLJ feature, it will require GFS pressure-level `u850` and `v850`; current surface GFS data is not a valid substitute.

## 9. Final Decision

The baseline remains unchanged and is valid as a **retrospective ERA5-conditioned Stage 1 research baseline**. It is not yet a fully operational feature set. Operationalization requires a separate feature-store/model version using GFS-only predictors, re-derived synoptic features, Candidate A label interpretation documented as proxy supervision, and full retraining/revalidation. Those actions are intentionally outside this audit.
