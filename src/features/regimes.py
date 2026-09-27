"""Transparent rule-based synoptic regime classifier for RituGyan.

Classifies daily atmospheric states into 6 predefined meteorological regimes:
- ID 0: ACTIVE_MONSOON
- ID 1: BREAK_MONSOON
- ID 2: MONSOON_DEPRESSION
- ID 3: OROGRAPHIC_MONSOON
- ID 4: COASTAL_REGIME
- ID 5: WESTERN_DISTURBANCE

Strict design principles:
1. Deterministic and fully interpretable (no black-box ML).
2. Explicit rule thresholds based on IMD meteorological criteria.
3. Detailed evidence dictionaries and quantitative confidence calculation.
4. Robust, documented fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.features.synoptic import SynopticFeatureSet
from src.ingestion.spatial import CANONICAL_LATS, CANONICAL_LONS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeDefinition:
    """Metadata definition for a synoptic regime."""
    regime_id: int
    code: str
    name: str
    description: str
    primary_indicators: List[str]


# The 6 Canonical Project Regimes
REGIME_CATALOG: Dict[int, RegimeDefinition] = {
    0: RegimeDefinition(
        regime_id=0,
        code="ACTIVE_MONSOON",
        name="Active Monsoon",
        description="Strong cross-equatorial inflow, trough south of or at normal, widespread heavy rainfall across central India.",
        primary_indicators=[
            "Trough latitude departure <= +1.0°",
            "Low-latitude Arabian Sea inflow speed >= 6.5 m/s",
            "Domain mean PWAT >= 40 mm",
            "Positive Central India moisture convergence",
        ],
    ),
    1: RegimeDefinition(
        regime_id=1,
        code="BREAK_MONSOON",
        name="Break Monsoon",
        description="Monsoon trough shifted to Himalayan foothills, rainfall suppressed over Central India, rain confined to foothills/NE.",
        primary_indicators=[
            "Trough latitude departure >= +2.0° (shifted north to foothills)",
            "Central India moisture convergence suppressed (< 5 mm/day)",
            "Absence of synoptic depression in Central India",
        ],
    ),
    2: RegimeDefinition(
        regime_id=2,
        code="MONSOON_DEPRESSION",
        name="Monsoon Low / Depression",
        description="Closed cyclonic circulation with marked negative MSLP anomaly and high relative vorticity over BoB/Central India.",
        primary_indicators=[
            "Depression MSLP anomaly <= -2.5 hPa (or <= -1.5 hPa with vorticity)",
            "Relative vorticity maximum >= 2.0e-5 s^-1 in 16°N-24°N, 76°E-90°E",
            "Intense organized convective rainbands",
        ],
    ),
    3: RegimeDefinition(
        regime_id=3,
        code="OROGRAPHIC_MONSOON",
        name="Orographic Monsoon",
        description="Strong perpendicular onshore westerly moisture flux impinging on Western Ghats / Meghalaya terrain.",
        primary_indicators=[
            "Western Ghats zonal moisture flux >= 180 kg/(m*s)",
            "Precipitation heavily concentrated along coastal/ghats topography",
            "Absence of deep synoptic depression",
        ],
    ),
    4: RegimeDefinition(
        regime_id=4,
        code="COASTAL_REGIME",
        name="Coastal Regime",
        description="Localized convective convergence driven by thermal and frictional land-sea boundaries.",
        primary_indicators=[
            "Precipitation concentrated along coastline grid cells",
            "Moderate domain-wide precipitation without deep synoptic system",
        ],
    ),
    5: RegimeDefinition(
        regime_id=5,
        code="WESTERN_DISTURBANCE",
        name="Western Disturbance",
        description="Mid-latitude synoptic trough / disturbance propagating across Northwest India and Western Himalayas.",
        primary_indicators=[
            "Precipitation and moisture active primarily over Northwest India (>= 28°N)",
            "Peninsular / low-latitude monsoon inflow suppressed (< 5.0 m/s)",
            "Negative MSLP anomaly over Northwest sector",
        ],
    ),
}


@dataclass
class RegimeClassificationResult:
    """Output of the rule-based synoptic regime classifier."""
    target_date: str
    regime_id: int
    regime_code: str
    regime_name: str
    confidence: float
    fallback_applied: bool
    triggered_rules: List[str]
    rule_scores: Dict[str, float]
    evidence: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SynopticRegimeClassifier:
    """Transparent, deterministic rule-based meteorological regime classifier."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.50,
        normal_trough_lat: float = 22.5,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.normal_trough_lat = normal_trough_lat

    def classify(
        self,
        synoptic_feats: SynopticFeatureSet,
        rainfall_field: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        lats: np.ndarray = CANONICAL_LATS,
        lons: np.ndarray = CANONICAL_LONS,
    ) -> RegimeClassificationResult:
        """Classify a day sample into one of the 6 synoptic regimes.
        
        Args:
            synoptic_feats: Precomputed synoptic indices from `extract_all_synoptic_features`.
            rainfall_field: Optional 2D array (127, 121) of daily rainfall (IMD or GFS forecast) [mm].
            valid_mask: Optional 2D bool array for valid land cells.
            lats: Canonical latitude coordinate array.
            lons: Canonical longitude coordinate array.
            
        Returns:
            RegimeClassificationResult with regime ID, code, confidence, and complete evidence.
        """
        evidence: Dict[str, Any] = {
            "target_date": synoptic_feats.target_date,
            "trough_departure_deg": synoptic_feats.trough_latitude_departure,
            "trough_mean_lat_deg_n": synoptic_feats.trough_mean_latitude,
            "trough_pmin_hpa": synoptic_feats.trough_min_pressure_hpa,
            "low_latitude_inflow_speed_ms": synoptic_feats.low_latitude_inflow_speed_ms,
            "domain_mean_pwat_mm": synoptic_feats.domain_mean_pwat_mm,
            "mfc_central_india_mm_day": synoptic_feats.mfc_central_india_mean_mm_day,
            "depression_mslp_anomaly_hpa": synoptic_feats.depression_mslp_anomaly_hpa,
            "depression_max_vorticity_s1": synoptic_feats.depression_max_vorticity_s1,
            "depression_pressure_contrast_hpa": synoptic_feats.depression_pressure_contrast_hpa,
            "depression_vorticity_area_fraction": synoptic_feats.depression_vorticity_area_fraction,
            "depression_vorticity_convergence_area_fraction": (
                synoptic_feats.depression_vorticity_convergence_area_fraction
            ),
            "depression_candidate_a_event": synoptic_feats.depression_candidate_a_event,
            "orographic_ghats_zonal_flux": synoptic_feats.orographic_ghats_zonal_flux,
            "nw_india_min_mslp_hpa": synoptic_feats.nw_india_min_mslp_hpa,
            "nw_india_mean_pwat_mm": synoptic_feats.nw_india_mean_pwat_mm,
        }

        # Spatial rainfall partitions (if rainfall grid is provided)
        rain_evidence = self._compute_rainfall_distribution(rainfall_field, valid_mask, lats, lons)
        evidence.update(rain_evidence)

        # Evaluate candidate scores and rules for all 6 regimes
        candidate_scores: Dict[int, float] = {}
        candidate_rules: Dict[int, List[str]] = {}

        # 1. MONSOON_DEPRESSION (ID 2)
        dep_score, dep_rules = self._evaluate_monsoon_depression(synoptic_feats, evidence)
        candidate_scores[2] = dep_score
        candidate_rules[2] = dep_rules

        # 2. BREAK_MONSOON (ID 1)
        break_score, break_rules = self._evaluate_break_monsoon(synoptic_feats, evidence)
        candidate_scores[1] = break_score
        candidate_rules[1] = break_rules

        # 3. OROGRAPHIC_MONSOON (ID 3)
        orog_score, orog_rules = self._evaluate_orographic_monsoon(synoptic_feats, evidence)
        candidate_scores[3] = orog_score
        candidate_rules[3] = orog_rules

        # 4. WESTERN_DISTURBANCE (ID 5)
        wd_score, wd_rules = self._evaluate_western_disturbance(synoptic_feats, evidence)
        candidate_scores[5] = wd_score
        candidate_rules[5] = wd_rules

        # 5. ACTIVE_MONSOON (ID 0)
        active_score, active_rules = self._evaluate_active_monsoon(synoptic_feats, evidence)
        candidate_scores[0] = active_score
        candidate_rules[0] = active_rules

        # 6. COASTAL_REGIME (ID 4)
        coastal_score, coastal_rules = self._evaluate_coastal_regime(synoptic_feats, evidence)
        candidate_scores[4] = coastal_score
        candidate_rules[4] = coastal_rules

        rule_score_summary = {
            REGIME_CATALOG[rid].code: round(score, 3) for rid, score in candidate_scores.items()
        }

        # Pick best regime with highest confidence score
        best_regime_id = max(candidate_scores, key=lambda k: candidate_scores[k])
        best_score = candidate_scores[best_regime_id]
        fallback_applied = False

        # Fallback if no regime cleared threshold
        if best_score < self.min_confidence_threshold:
            fallback_applied = True
            if synoptic_feats.domain_mean_pwat_mm >= 38.0:
                best_regime_id = 0  # Default to ACTIVE_MONSOON during moist monsoon
                best_score = 0.50
                candidate_rules[0].append("Fallback applied: Domain moist state (PWAT >= 38 mm)")
            else:
                best_regime_id = 4  # Default to COASTAL_REGIME
                best_score = 0.50
                candidate_rules[4].append("Fallback applied: Quiescent / moderate coastal state")

        reg_def = REGIME_CATALOG[best_regime_id]

        return RegimeClassificationResult(
            target_date=synoptic_feats.target_date,
            regime_id=best_regime_id,
            regime_code=reg_def.code,
            regime_name=reg_def.name,
            confidence=float(np.clip(best_score, 0.0, 1.0)),
            fallback_applied=fallback_applied,
            triggered_rules=candidate_rules[best_regime_id],
            rule_scores=rule_score_summary,
            evidence=evidence,
            metadata={"candidate_scores_raw": {str(k): v for k, v in candidate_scores.items()}},
        )

    def _compute_rainfall_distribution(
        self,
        rainfall_field: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute regional rainfall fractions for Ghats, Central India, Foothills, and Northwest."""
        if rainfall_field is None:
            return {
                "rainfall_data_available": False,
                "ghats_rain_fraction": 0.0,
                "central_india_rain_fraction": 0.0,
                "foothills_rain_fraction": 0.0,
                "nw_rain_fraction": 0.0,
                "coastal_rain_fraction": 0.0,
                "domain_mean_rain_mm": 0.0,
            }

        rain = np.nan_to_num(rainfall_field, nan=0.0)
        if valid_mask is not None:
            rain = np.where(valid_mask, rain, 0.0)

        total_rain_sum = float(np.sum(rain))
        safe_total = max(total_rain_sum, 1e-4)

        # 1. Western Ghats: 8°N-18°N, 73°E-76°E
        ghats_mask = (lats[:, None] >= 8.0) & (lats[:, None] <= 18.0) & (lons[None, :] >= 73.0) & (lons[None, :] <= 76.0)
        ghats_sum = float(np.sum(rain[ghats_mask]))

        # 2. Central India: 18°N-25°N, 75°E-85°E
        ci_mask = (lats[:, None] >= 18.0) & (lats[:, None] <= 25.0) & (lons[None, :] >= 75.0) & (lons[None, :] <= 85.0)
        ci_sum = float(np.sum(rain[ci_mask]))

        # 3. Himalayan Foothills: 26°N-30°N, 78°E-90°E
        fh_mask = (lats[:, None] >= 26.0) & (lats[:, None] <= 30.0) & (lons[None, :] >= 78.0) & (lons[None, :] <= 90.0)
        fh_sum = float(np.sum(rain[fh_mask]))

        # 4. Northwest India: 28°N-36°N, 70°E-78°E
        nw_mask = (lats[:, None] >= 28.0) & (lats[:, None] <= 36.0) & (lons[None, :] >= 70.0) & (lons[None, :] <= 78.0)
        nw_sum = float(np.sum(rain[nw_mask]))

        # 5. Coastal Belt (West & East coasts)
        coastal_mask = (
            ((lats[:, None] >= 8.0) & (lats[:, None] <= 22.0) & (lons[None, :] >= 72.0) & (lons[None, :] <= 74.0)) |
            ((lats[:, None] >= 8.0) & (lats[:, None] <= 22.0) & (lons[None, :] >= 79.5) & (lons[None, :] <= 87.0))
        )
        coastal_sum = float(np.sum(rain[coastal_mask]))

        return {
            "rainfall_data_available": True,
            "ghats_rain_fraction": ghats_sum / safe_total,
            "central_india_rain_fraction": ci_sum / safe_total,
            "foothills_rain_fraction": fh_sum / safe_total,
            "nw_rain_fraction": nw_sum / safe_total,
            "coastal_rain_fraction": coastal_sum / safe_total,
            "domain_mean_rain_mm": float(np.mean(rain)),
            "ghats_mean_rain_mm": float(np.mean(rain[ghats_mask])) if np.any(ghats_mask) else 0.0,
            "ci_mean_rain_mm": float(np.mean(rain[ci_mask])) if np.any(ci_mask) else 0.0,
        }

    def _evaluate_monsoon_depression(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate the Candidate A Category-A surface proxy for regime 2."""
        rules = []
        if syn.depression_candidate_a_event:
            rules.extend([
                f"Local pressure contrast: {syn.depression_pressure_contrast_hpa:.2f} hPa (<= -2.0 hPa)",
                (
                    "Cyclonic vorticity area fraction: "
                    f"{syn.depression_vorticity_area_fraction:.3f} (>= 0.10)"
                ),
                (
                    "Vorticity with positive 10 m convergence area fraction: "
                    f"{syn.depression_vorticity_convergence_area_fraction:.3f} (>= 0.05)"
                ),
                "Candidate A Category-A 10 m surface proxy; not validated depression detection",
            ])
            return 1.0, rules

        return 0.0, rules

    def _evaluate_break_monsoon(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate BREAK_MONSOON (ID 1) rules."""
        rules = []
        score = 0.0

        # Criterion A: Marked northward shift of trough towards foothills
        if syn.trough_latitude_departure >= 2.0:
            rules.append(f"Trough shifted north by +{syn.trough_latitude_departure:.2f}° (>= +2.0°)")
            score += 0.45 + min(0.30, (syn.trough_latitude_departure - 2.0) * 0.15)
        elif syn.trough_latitude_departure >= 1.0:
            rules.append(f"Trough shifted north by +{syn.trough_latitude_departure:.2f}° (>= +1.0°)")
            score += 0.25

        # Criterion B: Moisture flux convergence suppressed in Central India
        if syn.mfc_central_india_mean_mm_day < 2.0:
            rules.append(f"Central India MFC suppressed: {syn.mfc_central_india_mean_mm_day:.1f} mm/day (< 2.0)")
            score += 0.25

        # Criterion C: Foothills rain dominance over Central India (if rain available)
        if ev.get("rainfall_data_available", False):
            if ev["foothills_rain_fraction"] > ev["central_india_rain_fraction"] and ev["ci_mean_rain_mm"] < 4.0:
                rules.append(f"Foothills rain ({ev['foothills_rain_fraction']:.2f}) exceeds Central India ({ev['central_india_rain_fraction']:.2f})")
                score += 0.20

        # Penalize only when the Candidate A surface proxy event is present.
        if syn.depression_candidate_a_event:
            score = max(0.0, score - 0.50)

        return min(1.0, score), rules

    def _evaluate_orographic_monsoon(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate OROGRAPHIC_MONSOON (ID 3) rules."""
        rules = []
        score = 0.0

        # Criterion A: Strong onshore zonal moisture flux along Western Ghats
        if syn.orographic_ghats_zonal_flux >= 200.0:
            rules.append(f"Heavy Ghats onshore moisture flux: {syn.orographic_ghats_zonal_flux:.1f} kg/(m*s) (>= 200)")
            score += 0.40 + min(0.30, (syn.orographic_ghats_zonal_flux - 200.0) * 0.002)
        elif syn.orographic_ghats_zonal_flux >= 140.0:
            rules.append(f"Moderate Ghats onshore moisture flux: {syn.orographic_ghats_zonal_flux:.1f} kg/(m*s) (>= 140)")
            score += 0.25

        # Criterion B: Rain concentrated on Ghats / West Coast
        if ev.get("rainfall_data_available", False):
            if ev["ghats_rain_fraction"] >= 0.35:
                rules.append(f"High Ghats rainfall fraction: {ev['ghats_rain_fraction']:.2f} (>= 0.35)")
                score += 0.35
            elif ev["ghats_rain_fraction"] >= 0.20:
                score += 0.15

        # Strong peninsular inflow
        if syn.low_latitude_inflow_speed_ms >= 8.0:
            rules.append(f"Strong low-latitude inflow: {syn.low_latitude_inflow_speed_ms:.1f} m/s (>= 8.0)")
            score += 0.15

        # Penalize only when the Candidate A surface proxy event is present.
        if syn.depression_candidate_a_event:
            score = max(0.0, score - 0.40)

        return min(1.0, score), rules

    def _evaluate_western_disturbance(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate WESTERN_DISTURBANCE (ID 5) rules."""
        rules = []
        score = 0.0

        # Criterion A: Northwest activity and weak peninsular monsoon flow
        if syn.low_latitude_inflow_speed_ms < 5.0:
            rules.append(f"Suppressed monsoon inflow: {syn.low_latitude_inflow_speed_ms:.1f} m/s (< 5.0)")
            score += 0.30
        
        if syn.nw_india_mean_pwat_mm >= 30.0:
            rules.append(f"Elevated NW moisture: {syn.nw_india_mean_pwat_mm:.1f} mm (>= 30.0)")
            score += 0.25

        if ev.get("rainfall_data_available", False):
            if ev["nw_rain_fraction"] >= 0.40:
                rules.append(f"Rainfall dominated by NW sector: {ev['nw_rain_fraction']:.2f} (>= 0.40)")
                score += 0.35

        return min(1.0, score), rules

    def _evaluate_active_monsoon(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate ACTIVE_MONSOON (ID 0) rules."""
        rules = []
        score = 0.0

        # Criterion A: Trough south of normal or normal position
        if syn.trough_latitude_departure <= 0.0:
            rules.append(f"Trough south of normal: {syn.trough_latitude_departure:.2f}° (<= 0.0°)")
            score += 0.30
        elif syn.trough_latitude_departure <= 1.0:
            rules.append(f"Trough near normal: {syn.trough_latitude_departure:.2f}° (<= +1.0°)")
            score += 0.20

        # Criterion B: Strong low-latitude Arabian Sea inflow
        if syn.low_latitude_inflow_speed_ms >= 7.0:
            rules.append(f"Vigorous inflow speed: {syn.low_latitude_inflow_speed_ms:.1f} m/s (>= 7.0)")
            score += 0.30 + min(0.20, (syn.low_latitude_inflow_speed_ms - 7.0) * 0.05)
        elif syn.low_latitude_inflow_speed_ms >= 5.5:
            rules.append(f"Moderate inflow speed: {syn.low_latitude_inflow_speed_ms:.1f} m/s (>= 5.5)")
            score += 0.15

        # Criterion C: Elevated domain moisture
        if syn.domain_mean_pwat_mm >= 45.0:
            rules.append(f"High domain moisture: {syn.domain_mean_pwat_mm:.1f} mm (>= 45.0)")
            score += 0.25
        elif syn.domain_mean_pwat_mm >= 38.0:
            rules.append(f"Moderate domain moisture: {syn.domain_mean_pwat_mm:.1f} mm (>= 38.0)")
            score += 0.15

        # Criterion D: Positive Central India MFC
        if syn.mfc_central_india_mean_mm_day > 0.0:
            rules.append(f"Positive Central India MFC: {syn.mfc_central_india_mean_mm_day:.1f} mm/day (> 0.0)")
            score += 0.10

        # If rainfall is available and dominated by coastal fringe or ghats without CI rain, discount active score
        if ev.get("rainfall_data_available", False):
            if ev.get("coastal_rain_fraction", 0.0) >= 0.35 and ev.get("central_india_rain_fraction", 0.0) < 0.20:
                score = max(0.0, score - 0.35)

        return min(1.0, score), rules

    def _evaluate_coastal_regime(
        self,
        syn: SynopticFeatureSet,
        ev: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Evaluate COASTAL_REGIME (ID 4) rules."""
        rules = []
        score = 0.0

        if ev.get("rainfall_data_available", False):
            if ev["coastal_rain_fraction"] >= 0.35 and ev["domain_mean_rain_mm"] < 15.0:
                rules.append(f"Coastal rain dominance ({ev['coastal_rain_fraction']:.2f}) under moderate domain rain")
                score += 0.85
            elif ev["coastal_rain_fraction"] >= 0.20:
                score += 0.40
        else:
            # When rain is absent, moderate inflow with neutral trough
            if 4.5 <= syn.low_latitude_inflow_speed_ms <= 8.0 and abs(syn.trough_latitude_departure) <= 1.5:
                rules.append("Moderate inflow and neutral trough (coastal boundary convergence proxy)")
                score += 0.55

        return min(1.0, score), rules
