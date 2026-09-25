# RituGyan - Phase 2 Technical Decision Document
## Temporal Alignment Strategy for Multi-Source Meteorological Fusion

**Document ID:** TD-PHASE2-001  
**Status:** APPROVED / SPECIFICATION  
**Author:** RituGyan Architecture Team  
**Date:** 2026-09-25  

---

### 1. Executive Summary & Objective

In Phase 2 (Preprocessing & Feature Engineering), RituGyan fuses three distinct meteorological data streams:
1. **IMD Gridded Rainfall** (Ground Truth Target, $y$)
2. **NOAA GFS Forecasts** (Numerical Weather Prediction Predictor, $X_{\text{GFS}}$)
3. **ECMWF ERA5 Reanalysis** (Atmospheric & Thermodynamic Baseline Context, $X_{\text{ERA5}}$)

A rigorous temporal alignment strategy is required because these three data sources operate on different diurnal reporting cycles, initialization steps, and accumulation intervals. This document establishes the formal definitions, explains the 3-hour diurnal mismatch, assesses dataset sufficiency, and defines the alignment convention for both research/operational benchmarks and the prototype implementation.

---

### 2. Definition of One Training Sample

A single training instance in the RituGyan dataset is defined as:

$$\mathcal{S}_i = \left( \mathbf{X}_{\text{GFS}}^{(i)}, \mathbf{X}_{\text{ERA5}}^{(i)}, \mathbf{X}_{\text{Static}}^{(i)}, y_{\text{IMD}}^{(i)} \right)_{\text{grid}=(lat, lon), \text{day}=D}$$

Where:
* **Unit of Analysis:** A single spatial grid cell $(lat, lon)$ at the target resolution (or a spatial sub-grid / patch) for target date $D$.
* **Target ($y_{\text{IMD}}$):** 24-hour total rainfall accumulation recorded in millimeters ($\text{mm}$) for Day $D$.
* **Dynamic Predictors ($\mathbf{X}_{\text{GFS}}$):** Atmospheric state variables and predicted precipitation accumulation corresponding to the target accumulation window.
* **Environmental/Synoptic Context ($\mathbf{X}_{\text{ERA5}}$):** Historical synoptic background state (moisture, dynamical forcing, thermodynamic stability) over the 24-hour window.
* **Static Descriptors ($\mathbf{X}_{\text{Static}}$):** Topographic elevation, slope, aspect, distance to coast, and land surface characteristics.

---

### 3. IMD Rainfall Target Window Definition

* **Standard Observation Cycle:** India Meteorological Department (IMD) daily rainfall observations are recorded universally at **08:30 IST (Indian Standard Time)**.
* **UTC Equivalent:** Indian Standard Time is $\text{UTC} + 05:30$.  
  $$08:30\text{ IST} = 03:00\text{ UTC (03Z)}$$
* **Target Accumulation Interval:**
  $$\text{IMD Target Window for Day } D = \left[ D\text{ 03:00 UTC} \longrightarrow D+1\text{ 03:00 UTC} \right] \quad (24\text{ hours})$$
* Any daily value indexed as date $D$ in IMD gridded products represents the rainfall accumulated during the 24-hour period ending at 08:30 IST on Day $D+1$ (or Day $D$ depending on dataset index standard; formally spanning $03\text{Z}(D) \to 03\text{Z}(D+1)$).

---

### 4. GFS Forecast Lead Times & Predictor Windows

GFS runs are initialized at synoptic cycles: $00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$.  
For a Day-1 operational forecast initialized at $00\text{Z}$ on Day $D$:

* $f000 = D\text{ 00:00 UTC}$ (Analysis / Init)
* $f003 = D\text{ 03:00 UTC}$ (Start of IMD target window)
* $f006 = D\text{ 06:00 UTC}$
* $f012 = D\text{ 12:00 UTC}$
* $f018 = D\text{ 18:00 UTC}$
* $f024 = D+1\text{ 00:00 UTC}$
* $f027 = D+1\text{ 03:00 UTC}$ (End of IMD target window)

To construct an **exact 24-hour GFS forecast accumulation matching IMD 03Z–03Z**, the model requires:
$$\text{APCP}_{\text{exact}} = \text{APCP}_{f003 \to f027} = \sum_{t \in \{f006, f009, f012, f015, f018, f021, f024, f027\}} \Delta \text{APCP}_t$$

---

### 5. ERA5 Reanalysis Alignment

ERA5 reanalysis slices are available at 6-hourly intervals: $00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$.

* **Instantaneous Thermodynamic & Dynamic Slices:**
  * Slices within the target window: $D\text{ 06:00 UTC}$, $D\text{ 12:00 UTC}$, $D\text{ 18:00 UTC}$, and $D+1\text{ 00:00 UTC}$.
  * These 4 timestamps span the exact interior of the $03\text{Z}(D) \to 03\text{Z}(D+1)$ window, centered around diurnal convective peaks.
* **Accumulated Variables (Total Precipitation, Fluxes):**
  * Sum of ERA5 hourly or 6-hourly step accumulations spanning $[03\text{Z}, 03\text{Z}]$ (or $[06\text{Z}(D) \to 06\text{Z}(D+1)]$ as nearest 6-hourly proxy).

---

### 6. The 3-Hour Mismatch Analysis

#### Why the Mismatch Occurs
* **Meteorological convention:** IMD operates on local standard time ($08:30\text{ IST} = 03:00\text{ UTC}$), while Global NWP (GFS) and ECMWF Reanalysis are aligned with UTC standard synoptic hours ($00\text{Z}, 06\text{Z}, 12\text{Z}, 18\text{Z}$).
* **Impact of substituting 00Z–00Z for 03Z–03Z:**
  1. A $00\text{Z} \to 00\text{Z}$ window misses the $00\text{Z} \to 03\text{Z}$ rainfall on Day $D+1$ and incorrectly includes the $00\text{Z} \to 03\text{Z}$ rainfall from Day $D$.
  2. While monsoon convective systems over the Indian subcontinent typically peak in the late afternoon/early evening ($09\text{Z} \to 14\text{Z}$ / $14:30 \to 19:30\text{ IST}$), coastal and orographic heavy rainfall systems (e.g. Western Ghats and Northeast) frequently experience strong nocturnal and early morning convection ($00\text{Z} \to 03\text{Z}$).
  3. Treating $00\text{Z} \to 00\text{Z}$ as an exact substitute introduces phase error and degrades extreme precipitation skill scores (CSI / ETS / Brier Score).

---

### 7. Evaluation of GFS Lead Times & Test Data Sufficiency

* **Requirement for Exact Alignment:** GFS forecast lead steps **$f003$ and $f027$** from the $00\text{Z}$ cycle are strictly mandatory.
* **Status of Current Test Data:**
  * The Phase 1 test download (`data/raw/gfs/gfs_20240601_00z.grib2`) contains forecast steps:  
    `f000, f006, f012, f018, f024`.
  * Step **$f027$ is missing** from the current test dataset.
  * Step **$f003$ is missing** from the current test dataset.
* **Explicit Finding:**  
  **The current GFS test data is INSUFFICIENT for exact observational temporal alignment.**

---

### 8. Separation of Temporal Alignment Paradigms

To ensure scientific transparency, RituGyan formally distinguishes three alignment paradigms:

```
+---------------------------------------------------------------------------------------------------------+
|                                    1. EXACT OBSERVATIONAL ALIGNMENT                                      |
|  Target (IMD):       [ Day D 03:00 UTC -------------------------------> Day D+1 03:00 UTC ]             |
|  GFS Predictor:      00Z Init -> [ f003 ------------------------------> f027 ]                           |
|  ERA5 Context:       [ 06Z(D), 12Z(D), 18Z(D), 00Z(D+1) ]                                               |
|  Use Case:           Scientific publication, official benchmark, final production model.                |
+---------------------------------------------------------------------------------------------------------+
|                                    2. OPERATIONAL FORECAST ALIGNMENT                                    |
|  Inference Time:     Day D ~04:00 UTC (as soon as GFS 00Z run arrives)                                  |
|  Predictor Input:    GFS 00Z forecast cycles (f003 to f027 for 24h lead)                                |
|  Context Input:      GFS Analysis (00Z) or Real-time NWP state (ERA5 unavailable due to ~5-day latency) |
|  Target Predicted:   IMD 24h Rainfall ending Day D+1 08:30 IST                                          |
|  Use Case:           Live operational deployment for disaster management and reservoir operations.      |
+---------------------------------------------------------------------------------------------------------+
|                                   3. PROTOTYPE / DEMO APPROXIMATION                                     |
|  Target (IMD):       Day D Index (03Z -> 03Z)                                                           |
|  GFS Predictor:      00Z Init -> [ f000 ------------------------------> f024 ] (00Z -> 00Z)             |
|  ERA5 Context:       Day D Daily Aggregates (00Z, 06Z, 12Z, 18Z)                                        |
|  Offset:             3-hour temporal offset acknowledged and documented as a proxy approximation.       |
|  Use Case:           Rapid pipeline validation, CI/CD sanity runs, and baseline code testing.          |
+---------------------------------------------------------------------------------------------------------+
```

---

### 9. Documented Convention & Recommendation

1. **For Production Pipeline & Full Dataset Ingestion:**
   * Update the GFS ingestion specification (`configs/default_config.yaml` and download scripts) to request forecast steps:  
     `f003, f006, f009, f012, f015, f018, f021, f024, f027`.
   * Implement exact $03\text{Z} \to 03\text{Z}$ accumulation matching.

2. **For Phase 1/Phase 2 Initial Pipeline Testing:**
   * Acknowledge the current 5-step test file as a **Proxy Approximation ($00\text{Z} \to 00\text{Z}$ vs $03\text{Z} \to 03\text{Z}$)**.
   * Under no circumstance will $00\text{Z} \to 00\text{Z}$ be silently labeled as exact. All pipeline metadata will explicitly log `temporal_alignment: proxy_00z_03z_offset`.

---

### 10. Summary Decision Block

```text
RECOMMENDED ALIGNMENT: EXACT (with documented PROXY mode for existing test fixtures)
REASON: Exact observational alignment requires matching IMD's 08:30 IST (03Z) accumulation window using GFS 00Z forecast steps f003 through f027 to avoid phase errors in morning convective precipitation.
DATA REQUIRED: GFS 00Z forecast steps [f003, f006, f009, f012, f015, f018, f021, f024, f027]; IMD daily gridded rainfall (03Z-03Z); ERA5 6-hourly synoptic timesteps [06Z, 12Z, 18Z, 00Z].
CURRENT GFS DATA SUFFICIENT: NO
```
