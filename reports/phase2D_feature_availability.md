# Phase 2D: Synoptic Feature & Meteorological Variable Availability Report

**Date:** September 25, 2026  
**Status:** **AUDIT COMPLETE — CATEGORY CLASSIFICATION VERIFIED**  
**Cycle Validated:** GFS 00Z Cycle (`2024-06-21 00:00 UTC`), IMD Target Day (`2024-06-21`), ERA5 JJAS 2024  
**Author:** RituGyan Synoptic Feature Engineering & Diagnostic Engine  

---

## Executive Summary

Phase 2D implements and validates synoptic meteorological feature engineering and a transparent rule-based weather-regime classifier for RituGyan. In accordance with strict meteorological integrity protocols:
1. **Pressure-Level Variable Audit**: Direct inspection of all acquired raw datasets confirmed that current files contain **surface and single-level atmospheric variables only** (`msl/prmsl`, `10u/u10`, `10v/v10`, `2t/t2m`, `2d/d2m`, `pwat/tcwv`, `tp/RAINFALL`).
2. **Zero-Fabrication Policy Enforced**: The pipeline strictly refuses to fabricate or proxy 850 hPa isobaric winds from surface 10m winds. Indices requiring pressure-level data are formally classified as **Category B** with automated missing-dependency handling.
3. **Availability Classification**: Every proposed synoptic feature has been systematically audited, documented, and assigned a data availability status (Category A, B, or C).

---

## 1. Raw Dataset Variable & Pressure-Level Inventory

| Dataset | File Path | Ingested Variables | Pressure Levels Present | Spatial Domain |
|---|---|---|---|---|
| **ERA5 Reanalysis** | `data/raw/era5.nc` | `u10`, `v10`, `t2m`, `d2m`, `msl`, `tcwv` | **None** (Single-level surface / column) | $6.0^\circ\text{N} - 38.0^\circ\text{N}$, $68.0^\circ\text{E} - 98.0^\circ\text{E}$ |
| **GFS Forecast Sequence** | `data/raw/gfs_exact_test/gfs.0p25.*.grib2` (f003–f027) | `10u`, `10v`, `2t`, `2d`, `prmsl`, `pwat`, `tp` | **None** in operational leads $f003-f027$ | Global $0.25^\circ \rightarrow$ cropped $6.5^\circ\text{N}-38.0^\circ\text{N}, 68.0^\circ\text{E}-98.0^\circ\text{E}$ |
| **IMD Observations** | `data/raw/RF25_ind2024_rfp25.nc` | `RAINFALL` | **None** (Surface 24h accumulation) | $6.5^\circ\text{N} - 38.5^\circ\text{N}$, $66.5^\circ\text{E} - 100.0^\circ\text{E}$ |

---

## 2. Proposed Synoptic Index Classification Matrix

| # | Proposed Synoptic Feature | Classification | Availability Status | Operational Support in Current Pipeline |
|---|---|---|---|---|
| **1** | **Monsoon Trough Position & Latitude Departure** | **Category A** | Directly Computable | Fully computed using 2D MSLP fields across $16^\circ-30^\circ\text{N}, 72^\circ-86^\circ\text{E}$. |
| **2** | **Low-Level Jet (LLJ) Strength Index** | **Category B** | Computable Only After Acquiring Required Variable | Requires $u_{850}, v_{850}$ isobaric wind. Pipeline detects missing variables without fabricating data. |
| **3** | **Cross-Equatorial Flow Index** | **Category A / B** | Gateway Proxy Supported (Cat A) / Equatorial Core (Cat B) | Low-Latitude Arabian Sea Inflow Gateway ($6.5^\circ-12.0^\circ\text{N}, 68^\circ-78^\circ\text{E}$) computed at 10m. True $0^\circ\text{N}$ equator requires domain extension. |
| **4** | **Integrated Moisture Flux Convergence (MFC)** | **Category A / B** | 2D Column Proxy Supported (Cat A) / Multi-Level VIMFC (Cat B) | 2D Column-Moisture Divergence $-\nabla \cdot (\text{PWAT} \cdot \vec{v}_{10})$ computed on spherical metric. Multi-level VIMFC requires 3D isobaric profiles. |
| **5** | **Monsoon Depression / Cyclonic System Index** | **Category A** | Directly Computable | Computed from MSLP anomaly and relative vorticity $\zeta = \frac{\partial v_{10}}{\partial x} - \frac{\partial u_{10}}{\partial y}$ over $16^\circ-24^\circ\text{N}, 76^\circ-90^\circ\text{E}$. |
| **6** | **Western Ghats Orographic Interception Index** | **Category A** | Directly Computable | Computed from perpendicular zonal moisture flux $\text{PWAT} \cdot u_{10}$ over $10^\circ-18^\circ\text{N}, 72.5^\circ-76.0^\circ\text{E}$. |
| **7** | **Western Disturbance Activity Index** | **Category A** | Directly Computable | Computed from Northwest MSLP anomaly and PWAT activity over $28^\circ-36^\circ\text{N}, 70^\circ-80^\circ\text{E}$. |
| **8** | **Synoptic Regime Classifier (6 Predefined Regimes)** | **Category A** | Directly Computable | Fully implemented rule-based / heuristic classifier mapping multi-source state to 6 canonical regimes. |

---

## 3. Detailed Review of Missing Variables (Category B)

### 3.1 Low-Level Jet (Findlater Jet) at 850 hPa
- **Meteorological Requirement**: Somali Low-Level Jet core reaches peak intensity at the 850 hPa pressure level (~1.5 km altitude), where frictional decoupling from the surface boundary layer allows winds to accelerate to 30–50+ kt (15–25 m/s).
- **Current Data Status**: Acquired GFS forecast files ($f003-f027$) and ERA5 single-level NetCDF only contain 10m surface winds (`10u`/`u10`, `10v`/`v10`). Surface winds are heavily retarded by surface drag ($C_d$) and do not represent the 850 hPa jet core speed.
- **Project Rule Adherence**: The system **refuses to fabricate or derive 850 hPa wind from 10m wind**.
- **Exact Required Variables for Future Ingestion**:
  1. `u` (Zonal wind, `typeOfLevel: isobaricInhPa`, `level: 850`, units: `m/s`)
  2. `v` (Meridional wind, `typeOfLevel: isobaricInhPa`, `level: 850`, units: `m/s`)
  from ERA5 pressure-level dataset and GFS forecast lead sequence.

### 3.2 Vertically Integrated Moisture Flux Convergence (VIMFC)
- **Meteorological Requirement**:
  $$\text{VIMFC} = -\frac{1}{g} \int_{300}^{p_s} \nabla \cdot (q \, \vec{v}) \, dp$$
  Requires 3D atmospheric profiles of specific humidity $q(p)$ and horizontal wind vectors $\vec{v}(p)$ across multiple pressure levels (1000, 925, 850, 700, 500, 300 hPa).
- **Current Data Status**: Acquired datasets contain 2D column-integrated precipitable water (`pwat`/`tcwv`) and 2D surface 10m winds.
- **Operational Implementation**:
  - The pipeline provides a rigorous **2D Column-Moisture Divergence Proxy**: $-\nabla \cdot (\text{PWAT} \cdot \vec{v}_{10})$ in $\text{mm/day}$ equivalent.
  - The true multi-level VIMFC remains classified as **Category B** until multi-level profile variables are acquired.

---

## 4. Verification and Automated Handling

All modules in `src/features/synoptic.py` implement explicit availability guards:
```python
# Low-Level Jet index output when 850 hPa wind is unacquired
SynopticIndexResult(
    name="low_level_jet_850hpa_strength",
    value=None,
    category=IndexCategory.B_REQUIRES_ADDITIONAL_VARIABLE,
    units="m/s",
    is_computable=False,
    missing_dependencies=["u_wind_850hPa", "v_wind_850hPa"],
    metadata={"reason": "Low-Level Jet strength requires 850 hPa isobaric wind (u, v)..."}
)
```

Automated test verification in `tests/phase2/test_synoptic_features.py` confirms that:
1. Category B indices return `is_computable=False` and `value=None` gracefully without crashing.
2. No fabricated values are generated.
3. Category A features compute deterministically with valid units and physical ranges.
