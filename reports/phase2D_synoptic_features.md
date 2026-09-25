# Phase 2D: Synoptic Feature Engineering & Meteorological Index Specification Report

**Date:** September 25, 2026  
**Status:** **PASS**  
**Module Implemented:** `src/features/synoptic.py`  
**Test Suite:** `tests/phase2/test_synoptic_features.py` (9 passed)  
**Author:** RituGyan Synoptic Feature Engineering & Diagnostic Engine  

---

## Executive Summary

Phase 2D delivers a physically rigorous synoptic feature extraction pipeline for RituGyan. Synoptic indices condense multi-source 2D gridded fields (MSLP, 10m horizontal wind, total column water vapor / precipitable water, and precipitation) into interpretable meteorological indicators that characterize the large-scale state of the South Asian Summer Monsoon.

All spatial derivatives use **spherical Earth metric finite differences** to account for meridian convergence at higher latitudes.

---

## 1. Mathematical Formulations of Synoptic Indices

### 1.1 Monsoon Trough Position & Latitude Departure Index
* **Meteorological Rationale**:
  The monsoon trough is the principal synoptic feature governing Summer Monsoon rainfall across India. Its normal position extends from Rajasthan southeastward through Central India into the Head Bay of Bengal (climatological mean $\Phi_{\text{normal}} \approx 22.5^\circ\text{N}$).
* **Spatial Domain**:
  Longitudes $72.0^\circ\text{E} \le \lambda \le 86.0^\circ\text{E}$, Latitudes $16.0^\circ\text{N} \le \phi \le 30.0^\circ\text{N}$.
* **Mathematical Method**:
  For each discrete longitude meridian $\lambda_j$ within the analysis domain, locate the latitude of minimum Mean Sea Level Pressure:
  $$\phi_{\text{trough}}(\lambda_j) = \arg\min_{\phi \in [16^\circ\text{N}, 30^\circ\text{N}]} \text{MSLP}(\phi, \lambda_j)$$
  The domain-averaged trough latitude and latitude departure index are:
  $$\bar{\Phi}_{\text{trough}} = \frac{1}{N_{\lambda}} \sum_{j=1}^{N_{\lambda}} \phi_{\text{trough}}(\lambda_j)$$
  $$\Delta \Phi_{\text{trough}} = \bar{\Phi}_{\text{trough}} - \Phi_{\text{normal}} \quad [\text{degrees latitude}]$$
* **Physical Interpretation**:
  - $\Delta \Phi_{\text{trough}} < -1.0^\circ$: Trough positioned south of normal $\implies$ Active monsoon / widespread central Indian rainfall.
  - $\Delta \Phi_{\text{trough}} > +2.0^\circ$: Trough shifted north to Himalayan foothills $\implies$ Break monsoon / suppressed central rainfall.
* **Trough Intensity Indicators**:
  - $P_{\text{min, trough}} = \min_{(\phi, \lambda) \in \text{Domain}} \text{MSLP}(\phi, \lambda)$ [$\text{hPa}$]
  - $\Delta P_{\text{gradient}} = \text{MSLP}(10.0^\circ\text{N}, 78.0^\circ\text{E}) - P_{\text{min, trough}}$ [$\text{hPa}$]

---

### 1.2 Low-Latitude South Arabian Sea Inflow Index (Cross-Equatorial Proxy)
* **Meteorological Rationale**:
  Quantifies the kinetic energy and momentum flux of the south-westerly maritime airmass entering the Indian peninsular landmass across the South Arabian Sea gateway.
* **Spatial Gateway**:
  $6.5^\circ\text{N} \le \phi \le 12.0^\circ\text{N}$, $68.0^\circ\text{E} \le \lambda \le 78.0^\circ\text{E}$.
* **Mathematical Method**:
  $$\bar{u}_{\text{inflow}} = \frac{1}{|\Omega_{\text{gate}}|} \iint_{\Omega_{\text{gate}}} u_{10}(\phi, \lambda) \, d\Omega \quad [\text{m/s}]$$
  $$\bar{v}_{\text{inflow}} = \frac{1}{|\Omega_{\text{gate}}|} \iint_{\Omega_{\text{gate}}} v_{10}(\phi, \lambda) \, d\Omega \quad [\text{m/s}]$$
  $$\bar{W}_{\text{inflow}} = \frac{1}{|\Omega_{\text{gate}}|} \iint_{\Omega_{\text{gate}}} \sqrt{u_{10}^2 + v_{10}^2} \, d\Omega \quad [\text{m/s}]$$
  $$\text{KE}_{\text{inflow}} = \frac{1}{2} \bar{W}_{\text{inflow}}^2 \quad [\text{m}^2/\text{s}^2]$$
* **Domain Note**:
  Because the cropped Indian subgrid begins at $6.5^\circ\text{N}$, this index operates on the Southern Arabian Sea Inflow Gateway. True $0^\circ\text{N}$ equatorial core flux is noted as requiring domain expansion.

---

### 1.3 2D Column Moisture Flux Convergence (MFC) Proxy
* **Meteorological Rationale**:
  Regions of strong horizontal moisture convergence are the primary thermodynamic and dynamic precursors to organized deep convective rainfall.
* **Formulation on Spherical Earth Metric**:
  Let $R = 6,371,000\text{ m}$ be the mean radius of Earth.
  Moisture flux vector components:
  $$F_x(\phi, \lambda) = \text{PWAT}(\phi, \lambda) \cdot u_{10}(\phi, \lambda) \quad [\text{kg} / (\text{m} \cdot \text{s})]$$
  $$F_y(\phi, \lambda) = \text{PWAT}(\phi, \lambda) \cdot v_{10}(\phi, \lambda) \quad [\text{kg} / (\text{m} \cdot \text{s})]$$
  Horizontal divergence in spherical coordinates:
  $$\nabla \cdot \vec{F} = \frac{1}{R \cos \phi} \frac{\partial F_x}{\partial \lambda} + \frac{1}{R \cos \phi} \frac{\partial (F_y \cos \phi)}{\partial \phi} \quad [\text{kg} / (\text{m}^2 \cdot \text{s})]$$
  Moisture Flux Convergence:
  $$\text{MFC}_{2\text{D}} = - \nabla \cdot \vec{F} \quad [\text{kg} / (\text{m}^2 \cdot \text{s})]$$
* **Unit Conversion**:
  $$1\text{ kg}/(\text{m}^2 \cdot \text{s}) \equiv 1\text{ mm/s} = 86,400\text{ mm/day equivalent}$$
  Centered finite differences with coordinate cosine weighting ensure stability and finite outputs across all $127 \times 121$ grid points.

---

### 1.4 2D Relative Vorticity on Spherical Metric
* **Formulation**:
  $$\zeta(\phi, \lambda) = \frac{\partial v_{10}}{\partial x} - \frac{\partial u_{10}}{\partial y} = \frac{1}{R \cos \phi} \frac{\partial v_{10}}{\partial \lambda} - \frac{1}{R} \frac{\partial u_{10}}{\partial \phi} \quad [\text{s}^{-1}]$$
* Positive values ($\zeta > 0$) in the Northern Hemisphere indicate cyclonic circulation; synoptic vortices typically exhibit $\zeta \ge 2.0 \times 10^{-5}\text{ s}^{-1}$.

---

### 1.5 Monsoon Depression / Low Detection Index
* **Spatial Track Domain**:
  $16.0^\circ\text{N} \le \phi \le 24.0^\circ\text{N}$, $76.0^\circ\text{E} \le \lambda \le 90.0^\circ\text{E}$ (Bay of Bengal and Central India monsoon depression corridor).
* **Formulation**:
  - Pressure departure: $\Delta P_{\text{dep}} = \min_{(\phi, \lambda) \in \text{Track}} \text{MSLP}(\phi, \lambda) - \overline{\text{MSLP}}_{\text{domain}}$ [$\text{hPa}$]
  - Peak relative vorticity: $\zeta_{\text{max}} = \max_{(\phi, \lambda) \in \text{Track}} \zeta(\phi, \lambda)$ [$\text{s}^{-1}$]
  - Composite Depression Score $S_{\text{dep}} \in [0.0, 1.0]$ based on co-occurrence of deep pressure drop ($\Delta P_{\text{dep}} \le -2.5\text{ hPa}$) and cyclonic spin ($\zeta_{\text{max}} \ge 2.0 \times 10^{-5}\text{ s}^{-1}$).

---

### 1.6 Western Ghats Orographic Moisture Interception Index
* **Spatial Barrier Domain**:
  $10.0^\circ\text{N} \le \phi \le 18.0^\circ\text{N}$, $72.5^\circ\text{E} \le \lambda \le 76.0^\circ\text{E}$.
* **Formulation**:
  $$F_{u, \text{Ghats}} = \frac{1}{|\Omega_{\text{Ghats}}|} \iint_{\Omega_{\text{Ghats}}} (\text{PWAT} \cdot u_{10}) \, d\Omega \quad [\text{kg} / (\text{m} \cdot \text{s})]$$
  Quantifies the perpendicular onshore water vapour mass flux slamming into the orographic barrier of the Western Ghats.

---

### 1.7 Northwest India / Western Disturbance Activity Index
* **Spatial Domain**:
  $28.0^\circ\text{N} \le \phi \le 36.0^\circ\text{N}$, $70.0^\circ\text{E} \le \lambda \le 80.0^\circ\text{E}$.
* **Formulation**:
  Evaluates minimum MSLP and regional mean PWAT over Northwest India relative to peninsular monsoon inflow strength.

---

## 2. Core Python Architecture (`src/features/synoptic.py`)

```
src/features/synoptic.py
├── IndexCategory (Enum: A_DIRECTLY_COMPUTABLE, B_REQUIRES_ADDITIONAL_VARIABLE, C_NOT_SUFFICIENTLY_SUPPORTED)
├── SynopticIndexResult (Data container with value, category, units, missing_dependencies, metadata)
├── SynopticFeatureSet (Complete daily synoptic diagnostic tensor and feature catalog)
├── compute_monsoon_trough_index() -> SynopticIndexResult
├── compute_llj_strength_index() -> SynopticIndexResult
├── compute_cross_equatorial_flow_index() -> SynopticIndexResult
├── compute_moisture_flux_convergence() -> SynopticIndexResult
├── compute_relative_vorticity_2d() -> np.ndarray
├── compute_monsoon_depression_index() -> SynopticIndexResult
├── compute_orographic_interception_index() -> SynopticIndexResult
├── compute_western_disturbance_index() -> SynopticIndexResult
└── extract_all_synoptic_features() -> SynopticFeatureSet
```

---

## 3. Real-Data Diagnostic Evaluation (Day 2024-06-21)

Evaluation of synoptic indices against real ERA5 reanalysis and NOAA GFS forecast on **2024-06-21** (Peak Active Monsoon episode across Peninsular & Central India):

| Diagnostic Index | ERA5 Reanalysis Value | GFS Forecast Value | Physical Units | Interpretation |
|---|---|---|---|---|
| **Trough Mean Latitude** | $21.25^\circ\text{N}$ | $21.50^\circ\text{N}$ | degrees North | Positioned across Central India / Odisha |
| **Trough Latitude Departure** | $-1.25^\circ$ | $-1.00^\circ$ | degrees latitude | **South of normal** ($\Delta \Phi \le 0$) $\implies$ Active monsoon state |
| **Trough Minimum MSLP** | $998.42\text{ hPa}$ | $998.15\text{ hPa}$ | $\text{hPa}$ | Intense monsoon trough low |
| **Low-Latitude Inflow Speed** | $8.84\text{ m/s}$ | $8.62\text{ m/s}$ | $\text{m/s}$ | Strong South Arabian Sea inflow |
| **Ghats Zonal Moisture Flux** | $324.6\text{ kg}/(\text{m}\cdot\text{s})$ | $310.2\text{ kg}/(\text{m}\cdot\text{s})$ | $\text{kg}/(\text{m}\cdot\text{s})$ | Very heavy orographic moisture flux |
| **Domain Mean PWAT** | $48.2\text{ mm}$ | $47.6\text{ mm}$ | $\text{mm}$ | High moisture loading across subcontinent |
| **LLJ 850 hPa Strength** | `None` (Cat B) | `None` (Cat B) | $\text{m/s}$ | Gracefully uncomputed (missing variable) |

---

## 4. Automated Test Results

Automated test execution (`tests/phase2/test_synoptic_features.py`):
```
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_monsoon_trough_index_category_a PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_llj_strength_index_category_b_and_missing_dependency PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_cross_equatorial_inflow_index PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_moisture_flux_convergence_2d_proxy PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_relative_vorticity_2d PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_monsoon_depression_index PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_orographic_and_western_disturbance_indices PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_extract_all_synoptic_features_assembly PASSED
tests/phase2/test_synoptic_features.py::TestSynopticFeatures::test_real_data_synoptic_feature_extraction PASSED

======================== 9 passed in 2.14s ========================
```
