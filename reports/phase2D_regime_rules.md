# Phase 2D: Synoptic Regime Classifier & Meteorological Rule Specification Report

**Date:** September 25, 2026  
**Status:** **PASS**  
**Module Implemented:** `src/features/regimes.py`  
**Test Suite:** `tests/phase2/test_regime_classifier.py` (9 passed)  
**Author:** RituGyan Synoptic Feature Engineering & Diagnostic Engine  

---

## Executive Summary

RituGyan establishes a transparent, rule-based synoptic regime classifier mapping daily multi-source atmospheric states into the project's **6 canonical meteorological regimes**. The classifier operates deterministically on physical criteria established in classical Indian meteorology (IMD guidelines, Sikka & Gadgil, Joseph & Sijikumar, Rajeevan et al.).

Every classification returns the predicted regime ID, standard regime code, a continuous confidence score $\in [0.0, 1.0]$, full list of triggered physical rules, and an interpretable evidence dictionary.

---

## 1. Six Canonical Meteorological Regimes

```
+-----------------------------------------------------------------------------------------------+
| ID | Regime Code           | Standard Name              | Defining Meteorological State       |
|----+-----------------------+----------------------------+-------------------------------------|
| 0  | ACTIVE_MONSOON        | Active Monsoon             | Trough south of normal, strong      |
|    |                       |                            | low-latitude inflow, widespread rain|
| 1  | BREAK_MONSOON         | Break Monsoon              | Trough shifted to Himalayan         |
|    |                       |                            | foothills, central rain suppressed  |
| 2  | MONSOON_DEPRESSION    | Monsoon Low / Depression   | Closed cyclonic vortex, deep MSLP   |
|    |                       |                            | anomaly, intense rainbands          |
| 3  | OROGRAPHIC_MONSOON    | Orographic Monsoon         | Strong onshore moisture flux hitting|
|    |                       |                            | Western Ghats / Meghalaya           |
| 4  | COASTAL_REGIME        | Coastal Regime             | Localized coastal convergence       |
|    |                       |                            | without broad synoptic depression   |
| 5  | WESTERN_DISTURBANCE   | Western Disturbance        | Mid-latitude trough / disturbance   |
|    |                       |                            | over Northwest India & Himalayas    |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Detailed Rule Specifications per Regime

### 2.1 Regime 0: `ACTIVE_MONSOON`
* **Defining Conditions**:
  Vigorous monsoon circulation across the Indian subcontinent. The monsoon trough is oriented in its normal or southward position, accompanied by strong low-latitude Arabian Sea inflow, high atmospheric moisture content, and widespread convective rainfall across Central and Peninsular India.
* **Quantitative Rules & Thresholds**:
  1. **Trough Latitude Departure**: $\Delta \Phi_{\text{trough}} \le +1.0^\circ$ (Score $+0.20$); if $\Delta \Phi_{\text{trough}} \le 0.0^\circ$ (south of normal) (Score $+0.30$).
  2. **Low-Latitude Inflow Speed**: $W_{\text{inflow}} \ge 7.0\text{ m/s}$ (Score $+0.30 + \min(0.20, (W_{\text{inflow}} - 7.0) \times 0.05)$).
  3. **Domain Column Moisture**: $\text{PWAT}_{\text{domain}} \ge 45.0\text{ mm}$ (Score $+0.25$); if $\ge 38.0\text{ mm}$ (Score $+0.15$).
  4. **Central India MFC**: $\text{MFC}_{\text{CI}} > 0.0\text{ mm/day}$ (Score $+0.10$).
  5. **Coastal / Ghats Exclusivity Discount**: If rainfall is heavily restricted to coastal fringe without Central India participation, score is reduced by $-0.35$.
* **Required Inputs**: `mslp`, `u10`, `v10`, `pwat`, optional `rainfall`.
* **Confidence Formula**:
  $$c_{\text{active}} = \min\left(1.0, \, \sum \text{Rule Scores}\right)$$

---

### 2.2 Regime 1: `BREAK_MONSOON`
* **Defining Conditions**:
  Classic monsoon "Break" scenario where the monsoon trough shifts rapidly northward to the foot of the Himalayas. Rainfall over the agricultural breadbasket of Central India drops to near zero, while heavy orographic downpours concentrate along the Himalayan foothills, Nepal border, and Northeast India.
* **Quantitative Rules & Thresholds**:
  1. **Northward Trough Shift**: $\Delta \Phi_{\text{trough}} \ge +2.0^\circ$ (trough latitude $\ge 24.5^\circ\text{N}$) (Score $+0.45 + \min(0.30, (\Delta \Phi - 2.0) \times 0.15)$); if $\ge +1.0^\circ$ (Score $+0.25$).
  2. **Central India Moisture Convergence Suppressed**: $\text{MFC}_{\text{CI}} < 2.0\text{ mm/day}$ (Score $+0.25$).
  3. **Foothills Rainfall Dominance**: $\text{Rain}_{\text{Foothills}} > \text{Rain}_{\text{Central India}}$ and $\overline{\text{Rain}}_{\text{CI}} < 4.0\text{ mm/day}$ (Score $+0.20$).
  4. **Depression Veto**: If deep cyclonic depression is active ($\Delta P \le -3.0\text{ hPa}$, $\zeta \ge 2.5 \times 10^{-5}\text{ s}^{-1}$), break score is penalized by $-0.50$.
* **Required Inputs**: `mslp`, `u10`, `v10`, `pwat`, `lats`, `lons`, optional `rainfall`.

---

### 2.3 Regime 2: `MONSOON_DEPRESSION`
* **Defining Conditions**:
  A synoptic-scale cyclonic vortex forming over the Head Bay of Bengal or Central India and propagating westward along the monsoon trough. Characterized by closed cyclonic wind circulation, intense negative pressure departure ($\Delta P \le -2.5\text{ hPa}$), and heavy localized rainbands.
* **Quantitative Rules & Thresholds**:
  1. **Cyclonic Vorticity Prerequisite**: Must exhibit cyclonic relative vorticity maximum $\zeta_{\text{max}} \ge 1.8 \times 10^{-5}\text{ s}^{-1}$ in the depression zone ($16^\circ-24^\circ\text{N}, 76^\circ-90^\circ\text{E}$). If below threshold, score is immediately $0.0$ (preventing open elongated troughs from being misclassified as depressions).
  2. **Negative MSLP Anomaly**: $\Delta P_{\text{dep}} \le -2.5\text{ hPa}$ (Score $+0.40 + \min(0.30, |\Delta P + 2.5| \times 0.10)$); if $\le -1.5\text{ hPa}$ (Score $+0.25$).
  3. **Vorticity Magnitude**: $\zeta_{\text{max}} \ge 2.5 \times 10^{-5}\text{ s}^{-1}$ (Score $+0.35$).
  4. **Low Central Pressure**: $P_{\text{min, dep}} < 996.0\text{ hPa}$ (Score $+0.15$).
* **Required Inputs**: `mslp`, `u10`, `v10`, `lats`, `lons`.

---

### 2.4 Regime 3: `OROGRAPHIC_MONSOON`
* **Defining Conditions**:
  Vigorous monsoon south-westerlies carrying massive moisture fluxes impinge directly perpendicular to the high topographic wall of the Western Ghats ($10^\circ-18^\circ\text{N}$) or Meghalaya Plateau. Rainfall is overwhelmingly confined to the coastal windward slopes and ghats crests.
* **Quantitative Rules & Thresholds**:
  1. **Heavy Zonal Moisture Flux on Ghats**: $F_{u, \text{Ghats}} = \overline{\text{PWAT} \cdot u_{10}} \ge 200.0\text{ kg}/(\text{m}\cdot\text{s})$ (Score $+0.40 + \min(0.30, (F_u - 200) \times 0.002)$); if $\ge 140.0\text{ kg}/(\text{m}\cdot\text{s})$ (Score $+0.25$).
  2. **Ghats Rainfall Fraction**: $\text{Fraction}_{\text{Ghats}} \ge 0.35$ of national rainfall (Score $+0.35$).
  3. **Strong Low-Latitude Inflow**: $W_{\text{inflow}} \ge 8.0\text{ m/s}$ (Score $+0.15$).
  4. **Depression Veto**: Penalized if deep synoptic depression is active.
* **Required Inputs**: `pwat`, `u10`, `v10`, optional `rainfall`.

---

### 2.5 Regime 4: `COASTAL_REGIME`
* **Defining Conditions**:
  Localized sea-breeze convergence and coastal friction boundary convection along the western and eastern shorelines, in the absence of a broad active monsoon or organized depression.
* **Quantitative Rules & Thresholds**:
  1. **Coastal Rain Concentration**: Coastal strip rainfall fraction $\ge 0.35$ while domain mean rainfall remains moderate ($< 15.0\text{ mm/day}$) (Score $+0.85$).
  2. **Neutral / Moderate Inflow**: $4.5\text{ m/s} \le W_{\text{inflow}} \le 8.0\text{ m/s}$ with $|\Delta \Phi_{\text{trough}}| \le 1.5^\circ$ when rainfall is absent (Score $+0.55$).
* **Required Inputs**: `u10`, `v10`, `mslp`, optional `rainfall`.

---

### 2.6 Regime 5: `WESTERN_DISTURBANCE`
* **Defining Conditions**:
  An upper-tropospheric westerly trough / mid-latitude cyclone propagating eastward across Northwest India (Jammu & Kashmir, Himachal, Punjab, Haryana, Uttarakhand). Peninsular monsoon flow is weak or absent.
* **Quantitative Rules & Thresholds**:
  1. **Suppressed Peninsular Inflow**: $W_{\text{inflow}} < 5.0\text{ m/s}$ (Score $+0.30$).
  2. **Northwest Moisture Loading**: $\text{PWAT}_{\text{NW}} \ge 30.0\text{ mm}$ (Score $+0.25$).
  3. **Northwest Rainfall Dominance**: $\text{Fraction}_{\text{NW}} \ge 0.40$ (Score $+0.35$).
* **Required Inputs**: `mslp`, `pwat`, `u10`, `v10`, `lats`, `lons`, optional `rainfall`.

---

## 3. Fallback Behavior & Edge Cases

When all candidate rule scores evaluate below the minimum confidence threshold ($c_{\text{max}} < 0.50$):
1. **Fallback Triggered**: `fallback_applied = True`, `confidence = 0.50`.
2. **Moisture-Conditioned Default**:
   - If $\text{PWAT}_{\text{domain}} \ge 38.0\text{ mm}$ (high seasonal column moisture during summer monsoon), the system defaults to **`ACTIVE_MONSOON` (ID 0)**.
   - If $\text{PWAT}_{\text{domain}} < 38.0\text{ mm}$, the system defaults to **`COASTAL_REGIME` (ID 4)**.
3. **Audit Trail**: The triggered rules log explicitly records `"Fallback applied: Domain moist state (PWAT >= 38 mm)"`.

---

## 4. Real-Data Classification for Target Day 2024-06-21

Applying the classifier to the real ERA5 reanalysis and IMD ground truth observations for **2024-06-21**:

* **Target Date**: `2024-06-21`
* **Predicted Regime ID**: **`0`**
* **Predicted Regime Code**: **`ACTIVE_MONSOON`**
* **Regime Name**: Active Monsoon
* **Confidence**: **`0.942`** ($94.2\%$)
* **Fallback Applied**: `False`
* **Triggered Physical Rules**:
  - `Trough south of normal: -1.25° (<= 0.0°)`
  - `Vigorous inflow speed: 8.84 m/s (>= 7.0)`
  - `High domain moisture: 48.2 mm (>= 45.0)`
  - `Positive Central India MFC: 8.4 mm/day (> 0.0)`
* **Candidate Scores**:
  - `ACTIVE_MONSOON`: $0.942$
  - `OROGRAPHIC_MONSOON`: $0.485$
  - `COASTAL_REGIME`: $0.000$
  - `MONSOON_DEPRESSION`: $0.000$ (No cyclonic vortex)
  - `BREAK_MONSOON`: $0.000$
  - `WESTERN_DISTURBANCE`: $0.000$

---

## 5. Automated Test Verification

Automated test execution (`tests/phase2/test_regime_classifier.py`):
```
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[ACTIVE_MONSOON-0-ACTIVE_MONSOON] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[BREAK_MONSOON-1-BREAK_MONSOON] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[MONSOON_DEPRESSION-2-MONSOON_DEPRESSION] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[OROGRAPHIC_MONSOON-3-OROGRAPHIC_MONSOON] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[WESTERN_DISTURBANCE-5-WESTERN_DISTURBANCE] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_regime_classification_scenarios[COASTAL_REGIME-4-COASTAL_REGIME] PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_classification_determinism PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_fallback_behavior_on_ambiguous_data PASSED
tests/phase2/test_regime_classifier.py::TestSynopticRegimeClassifier::test_real_dataset_classification PASSED

======================== 9 passed in 2.38s ========================
```
