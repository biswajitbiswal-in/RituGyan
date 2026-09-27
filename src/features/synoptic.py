"""Synoptic feature engineering and meteorological index calculators for RituGyan.

Computes synoptic meteorological features and indices from multi-source 2D grids:
1. Monsoon Trough Position & Latitude Departure Index (Category A)
2. Low-Level Jet (LLJ) Strength Index (Category B — requires 850 hPa isobaric wind)
3. Cross-Equatorial / Low-Latitude Inflow Index (Category A for proxy / B for 0°N core)
4. Moisture Flux Convergence (MFC) Proxy (Category A for 2D proxy / B for 3D VIMFC)
5. Monsoon Depression / Cyclonic Circulation Index (Category A)
6. Western Ghats Orographic Interception Index (Category A)
7. Western Disturbance Index (Category A)

Strict meteorological rules:
- No fabrication of 850 hPa winds from 10m surface winds.
- Explicit index availability classification (Category A, B, C).
- Deterministic finite outputs with valid physical units and spherical metric divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.ingestion.spatial import (
    CANONICAL_LATS,
    CANONICAL_LONS,
    CANONICAL_NUM_LATS,
    CANONICAL_NUM_LONS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mean Earth radius for spherical finite differences [meters]
EARTH_RADIUS_M = 6371000.0

# Climatological normal monsoon trough latitude (mean across 72°E - 86°E)
CLIMATOLOGICAL_TROUGH_LAT_NORMAL = 22.5  # degrees North

DEPRESSION_PRESSURE_CONTRAST_THRESHOLD_HPA = -2.0
DEPRESSION_VORTICITY_THRESHOLD_S1 = 2.5e-5
DEPRESSION_VORTICITY_AREA_FRACTION_MIN = 0.10
DEPRESSION_VORTICITY_CONVERGENCE_AREA_FRACTION_MIN = 0.05


class IndexCategory(str, Enum):
    """Classification of meteorological index data availability."""
    A_DIRECTLY_COMPUTABLE = "A_DIRECTLY_COMPUTABLE"
    B_REQUIRES_ADDITIONAL_VARIABLE = "B_REQUIRES_ADDITIONAL_VARIABLE"
    C_NOT_SUFFICIENTLY_SUPPORTED = "C_NOT_SUFFICIENTLY_SUPPORTED"


class MissingDependencyError(RuntimeError):
    """Raised when an index is requested that requires unacquired variables."""
    pass


@dataclass
class SynopticIndexResult:
    """Container for the output of a synoptic index calculation."""
    name: str
    value: Optional[float]
    category: IndexCategory
    units: str
    is_computable: bool
    missing_dependencies: List[str] = field(default_factory=list)
    spatial_field: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynopticFeatureSet:
    """Assembled synoptic features and diagnostic indices for a given day."""
    target_date: str
    # 1. Monsoon Trough
    trough_mean_latitude: float                  # [degrees N]
    trough_latitude_departure: float             # [degrees latitude departure from 22.5°N]
    trough_min_pressure_hpa: float               # [hPa]
    trough_pressure_gradient_hpa: float          # [hPa between 10°N and trough minimum]
    
    # 2. Low-Level Jet (LLJ)
    llj_strength_850_ms: Optional[float]         # [m/s] (None if unacquired)
    llj_is_supported: bool                       # False with current dataset
    
    # 3. Cross-Equatorial / Low-Latitude Arabian Sea Inflow
    low_latitude_inflow_speed_ms: float          # [m/s] mean wind speed in 6.5°N-12.0°N, 68°E-78°E
    low_latitude_inflow_zonal_ms: float          # [m/s]
    low_latitude_inflow_meridional_ms: float     # [m/s]
    low_latitude_kinetic_energy: float           # [m^2/s^2]
    
    # 4. Moisture Flux Convergence (MFC) Proxy
    mfc_domain_mean_mm_day: float                # [mm/day equivalent]
    mfc_central_india_mean_mm_day: float         # [mm/day equivalent]
    mfc_spatial_map: np.ndarray                  # (127, 121) array in mm/day equivalent
    
    # 5. Monsoon Depression / Cyclonic System
    depression_min_mslp_hpa: float               # [hPa] minimum MSLP in 16°N-24°N, 76°E-90°E
    depression_mslp_anomaly_hpa: float           # [hPa] departure from domain mean
    depression_max_vorticity_s1: float           # [s^-1] relative vorticity maximum
    
    # 6. Orographic Interception
    orographic_ghats_zonal_flux: float           # [kg/(m*s)] zonal moisture flux over Western Ghats
    
    # 7. Northwest / Western Disturbance Activity
    nw_india_min_mslp_hpa: float                 # [hPa] MSLP over 28°N-36°N, 70°E-80°E
    nw_india_mean_pwat_mm: float                 # [mm] PWAT over Northwest sector
    
    # Broad Domain Moisture State
    domain_mean_pwat_mm: float                   # [mm / kg/m^2]
    domain_mean_mslp_hpa: float                  # [hPa]

    # Candidate A Category-A surface depression proxy diagnostics
    depression_pressure_contrast_hpa: Optional[float] = None
    depression_vorticity_area_fraction: Optional[float] = None
    depression_vorticity_convergence_area_fraction: Optional[float] = None
    depression_candidate_a_event: bool = False
    
    # Index Classification Catalog
    feature_catalog: Dict[str, SynopticIndexResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_monsoon_trough_index(
    mslp_hpa: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    normal_lat: float = CLIMATOLOGICAL_TROUGH_LAT_NORMAL,
    lat_bounds: Tuple[float, float] = (16.0, 30.0),
    lon_bounds: Tuple[float, float] = (72.0, 86.0),
) -> SynopticIndexResult:
    """Calculate Monsoon Trough Axis Position, Latitude Departure, and Intensity from MSLP.
    
    Method:
      1. Crop MSLP to Central/North India trough corridor (16°N-30°N, 72°E-86°E).
      2. Along each meridional column within lon_bounds, locate the latitude of minimum MSLP.
      3. Compute the domain-averaged trough axis latitude Phi_trough.
      4. Compute departure: Delta Phi = Phi_trough - Phi_normal [degrees latitude].
         - Negative values (Delta Phi < -1.0°): Trough south of normal (Active monsoon).
         - Positive values (Delta Phi > +1.5°): Trough shifted to Himalayan foothills (Break monsoon).
      5. Identify minimum central pressure and meridional pressure gradient against peninsular India (10°N).
      
    Classification: Category A (Directly computable from available MSLP grids).
    """
    assert mslp_hpa.shape == (len(lats), len(lons)), (
        f"MSLP shape {mslp_hpa.shape} mismatch with coordinates ({len(lats)}, {len(lons)})"
    )

    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_lats = lats[lat_mask]
    sub_lons = lons[lon_mask]
    sub_mslp = mslp_hpa[np.ix_(lat_mask, lon_mask)]

    # For each longitude slice, find latitude of minimum MSLP
    min_lat_indices = np.argmin(sub_mslp, axis=0)
    trough_lats_per_lon = sub_lats[min_lat_indices]

    mean_trough_lat = float(np.mean(trough_lats_per_lon))
    latitude_departure = float(mean_trough_lat - normal_lat)
    trough_pmin = float(np.min(sub_mslp))

    # Pressure at 10°N, 78°E (South Peninsular baseline)
    lat_10_idx = int(np.argmin(np.abs(lats - 10.0)))
    lon_78_idx = int(np.argmin(np.abs(lons - 78.0)))
    peninsular_mslp = float(mslp_hpa[lat_10_idx, lon_78_idx])
    pressure_gradient = float(peninsular_mslp - trough_pmin)

    metadata = {
        "mean_trough_latitude_deg_n": mean_trough_lat,
        "latitude_departure_deg": latitude_departure,
        "normal_climatological_latitude_deg_n": normal_lat,
        "trough_minimum_pressure_hpa": trough_pmin,
        "peninsular_reference_mslp_hpa": peninsular_mslp,
        "trough_pressure_gradient_hpa": pressure_gradient,
        "lon_slices_analyzed": len(sub_lons),
        "lat_range_analyzed": list(lat_bounds),
        "lon_range_analyzed": list(lon_bounds),
    }

    return SynopticIndexResult(
        name="monsoon_trough_latitude_departure",
        value=latitude_departure,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="degrees_latitude",
        is_computable=True,
        missing_dependencies=[],
        metadata=metadata,
    )


def compute_llj_strength_index(
    u850_ms: Optional[np.ndarray] = None,
    v850_ms: Optional[np.ndarray] = None,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    lat_bounds: Tuple[float, float] = (8.0, 16.0),
    lon_bounds: Tuple[float, float] = (68.0, 76.0),
) -> SynopticIndexResult:
    """Evaluate Low-Level Jet (Findlater Jet) Strength at 850 hPa.
    
    Theoretical Definition:
      Core speed of the lower tropospheric Somali Jet at 850 hPa (~1.5 km ASL) over the
      Arabian Sea and Peninsular India gateway.
      
    Classification: Category B (Computable only after acquiring 850 hPa isobaric wind fields).
    
    IMPORTANT:
      Surface 10m winds are NOT 850 hPa winds. We strictly refuse to fabricate an 850 hPa
      wind field from 10m surface winds. If 850 hPa winds are absent, this function returns
      an uncomputed result with category B and missing dependencies noted.
    """
    missing_deps = []
    if u850_ms is None:
        missing_deps.append("u_wind_850hPa")
    if v850_ms is None:
        missing_deps.append("v_wind_850hPa")

    if missing_deps:
        return SynopticIndexResult(
            name="low_level_jet_850hpa_strength",
            value=None,
            category=IndexCategory.B_REQUIRES_ADDITIONAL_VARIABLE,
            units="m/s",
            is_computable=False,
            missing_dependencies=missing_deps,
            metadata={
                "reason": (
                    "Low-Level Jet strength requires 850 hPa isobaric wind (u, v). "
                    "Current pipeline datasets only contain surface 10m winds. "
                    "Per project policy, 850 hPa winds are not fabricated."
                ),
                "required_variables": ["u (isobaricInhPa: 850)", "v (isobaricInhPa: 850)"],
                "target_domain": {"lat_bounds": lat_bounds, "lon_bounds": lon_bounds},
            },
        )

    # If provided (e.g. in future ingestion or unit test fixture)
    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_u = u850_ms[np.ix_(lat_mask, lon_mask)]
    sub_v = v850_ms[np.ix_(lat_mask, lon_mask)]
    speed = np.sqrt(sub_u**2 + sub_v**2)
    mean_speed = float(np.mean(speed))
    max_speed = float(np.max(speed))

    return SynopticIndexResult(
        name="low_level_jet_850hpa_strength",
        value=mean_speed,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="m/s",
        is_computable=True,
        missing_dependencies=[],
        metadata={"mean_speed_ms": mean_speed, "max_speed_ms": max_speed},
    )


def compute_cross_equatorial_flow_index(
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    lat_bounds: Tuple[float, float] = (6.5, 12.0),
    lon_bounds: Tuple[float, float] = (68.0, 78.0),
) -> SynopticIndexResult:
    """Calculate Low-Latitude South Arabian Sea Inflow Index (Cross-Equatorial Proxy).
    
    Spatial Domain:
      Southern Inflow Gateway (6.5°N - 12.0°N, 68.0°E - 78.0°E).
      
    Theoretical Note:
      True cross-equatorial flux at 0°N requires a domain spanning the equator (0°N-5°N, 40°E-55°E),
      which is outside the cropped Indian subgrid (6.5°N-38.0°N).
      This index computes the low-latitude Arabian Sea surface inflow into peninsular India
      at 10m altitude, serving as the operational surface synoptic proxy.
      
    Classification:
      - Category A for Low-Latitude Arabian Sea Inflow Gateway (6.5°N-12.0°N, 68°E-78°E).
      - Note attached explaining boundary proxy vs true equatorial core.
    """
    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_u = u10_ms[np.ix_(lat_mask, lon_mask)]
    sub_v = v10_ms[np.ix_(lat_mask, lon_mask)]
    speed = np.sqrt(sub_u**2 + sub_v**2)

    mean_speed = float(np.mean(speed))
    mean_u = float(np.mean(sub_u))
    mean_v = float(np.mean(sub_v))
    kinetic_energy = float(0.5 * np.mean(speed**2))

    metadata = {
        "mean_inflow_speed_ms": mean_speed,
        "mean_zonal_inflow_ms": mean_u,
        "mean_meridional_inflow_ms": mean_v,
        "kinetic_energy_m2_s2": kinetic_energy,
        "domain_gateway_lats": list(lat_bounds),
        "domain_gateway_lons": list(lon_bounds),
        "proxy_nature": "Low-Latitude South Arabian Sea Inflow Gateway (6.5°N-12.0°N)",
        "true_equator_0n_supported": False,
    }

    return SynopticIndexResult(
        name="low_latitude_inflow_speed",
        value=mean_speed,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="m/s",
        is_computable=True,
        missing_dependencies=[],
        metadata=metadata,
    )


def compute_moisture_flux_convergence(
    pwat_kg_m2: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
) -> SynopticIndexResult:
    """Calculate 2D Column-Integrated Moisture Flux Convergence (MFC Proxy).
    
    Formulation:
      Moisture Flux Vector:
        F_x = PWAT * u_10   [kg / (m * s)]
        F_y = PWAT * v_10   [kg / (m * s)]
        
      Spherical Earth Divergence:
        div(F) = 1 / (R * cos(phi)) * d(F_x)/d(lambda) + 1 / (R * cos(phi)) * d(F_y * cos(phi))/d(phi)
        
      Moisture Flux Convergence:
        MFC = - div(F)      [kg / (m^2 * s)]
        
      Conversion to mm/day equivalent:
        1 kg/(m^2 * s) = 1 mm/s = 86,400 mm/day
        
    Classification:
      - Category A for 2D Column-Moisture Advective Convergence Proxy (from PWAT and 10m wind).
      - Category B for True Multi-Level Vertically Integrated Moisture Flux Convergence (VIMFC),
        which requires 3D specific humidity and wind profiles across pressure levels.
    """
    assert pwat_kg_m2.shape == (len(lats), len(lons))
    assert u10_ms.shape == (len(lats), len(lons))
    assert v10_ms.shape == (len(lats), len(lons))

    # Moisture flux components [kg / (m * s)]
    fx = pwat_kg_m2 * u10_ms
    fy = pwat_kg_m2 * v10_ms

    n_lats = len(lats)
    n_lons = len(lons)

    # Coordinate grids in radians
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    cos_lat = np.cos(lat_rad)[:, np.newaxis]  # shape (127, 1)
    cos_lat_safe = np.maximum(cos_lat, 1e-4)

    # Grid spacings in radians (assuming uniform spacing, 0.25 deg = 0.00436332 rad)
    dlat_rad = np.gradient(lat_rad)  # 1D array of shape (127,)
    dlon_rad = np.gradient(lon_rad)  # 1D array of shape (121,)

    # Finite differences along longitude (axis=1) and latitude (axis=0)
    # d(Fx)/d(lambda):
    dfx_dlon = np.gradient(fx, axis=1) / dlon_rad[np.newaxis, :]
    
    # d(Fy * cos(phi))/d(phi):
    fy_cos = fy * cos_lat
    dfy_cos_dlat = np.gradient(fy_cos, axis=0) / dlat_rad[:, np.newaxis]

    # Spherical divergence [kg / (m^2 * s)]
    div_f = (dfx_dlon + dfy_cos_dlat) / (EARTH_RADIUS_M * cos_lat_safe)

    # Moisture Flux Convergence = - div(F) [kg / (m^2 * s)]
    mfc_kg_m2_s = -div_f

    # Convert to mm/day equivalent: 1 kg/m^2/s = 86400 mm/day
    mfc_mm_day = mfc_kg_m2_s * 86400.0

    # Ensure finite outputs
    mfc_mm_day = np.nan_to_num(mfc_mm_day, nan=0.0, posinf=500.0, neginf=-500.0)

    # Regional averages:
    domain_mean = float(np.mean(mfc_mm_day))

    # Central India core zone: 18°N-25°N, 75°E-85°E
    c_lat_mask = (lats >= 18.0) & (lats <= 25.0)
    c_lon_mask = (lons >= 75.0) & (lons <= 85.0)
    central_india_mean = float(np.mean(mfc_mm_day[np.ix_(c_lat_mask, c_lon_mask)]))

    metadata = {
        "domain_mean_mfc_mm_day": domain_mean,
        "central_india_mfc_mm_day": central_india_mean,
        "mfc_min_mm_day": float(np.min(mfc_mm_day)),
        "mfc_max_mm_day": float(np.max(mfc_mm_day)),
        "proxy_type": "2D Column-Moisture Convergence Proxy (-div(PWAT * V_10))",
        "true_3d_vimfc_supported": False,
    }

    return SynopticIndexResult(
        name="moisture_flux_convergence_2d_proxy",
        value=domain_mean,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="mm/day",
        is_computable=True,
        missing_dependencies=[],
        spatial_field=mfc_mm_day.astype(np.float32),
        metadata=metadata,
    )


def compute_relative_vorticity_2d(
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
) -> np.ndarray:
    """Compute 2D relative vorticity zeta = dv/dx - du/dy on spherical grid [s^-1]."""
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    cos_lat = np.cos(lat_rad)[:, np.newaxis]
    cos_lat_safe = np.maximum(cos_lat, 1e-4)

    dlat_rad = np.gradient(lat_rad)
    dlon_rad = np.gradient(lon_rad)

    # dv/dx = 1 / (R * cos(phi)) * dv/dlon
    dv_dlon = np.gradient(v10_ms, axis=1) / dlon_rad[np.newaxis, :]
    dv_dx = dv_dlon / (EARTH_RADIUS_M * cos_lat_safe)

    # du/dy = 1 / R * du/dlat
    du_dlat = np.gradient(u10_ms, axis=0) / dlat_rad[:, np.newaxis]
    du_dy = du_dlat / EARTH_RADIUS_M

    # Vorticity zeta = dv/dx - du/dy [s^-1]
    zeta = dv_dx - du_dy
    return np.nan_to_num(zeta, nan=0.0).astype(np.float32)


def compute_horizontal_convergence_2d(
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
) -> np.ndarray:
    """Compute positive horizontal convergence from 10 m winds on a sphere [s^-1]."""
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    cos_lat = np.cos(lat_rad)[:, np.newaxis]
    cos_lat_safe = np.maximum(cos_lat, 1e-4)

    dlat_rad = np.gradient(lat_rad)
    dlon_rad = np.gradient(lon_rad)
    du_dlon = np.gradient(u10_ms, axis=1) / dlon_rad[np.newaxis, :]
    dvc_dlat = np.gradient(v10_ms * cos_lat, axis=0) / dlat_rad[:, np.newaxis]
    divergence = (du_dlon + dvc_dlat) / (EARTH_RADIUS_M * cos_lat_safe)
    return np.nan_to_num(-divergence, nan=0.0).astype(np.float32)


def compute_monsoon_depression_index(
    mslp_hpa: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    lat_bounds: Tuple[float, float] = (16.0, 24.0),
    lon_bounds: Tuple[float, float] = (76.0, 90.0),
) -> SynopticIndexResult:
    """Evaluate a Category-A 10 m surface proxy for a monsoon depression.

    Candidate A requires a local pressure contrast plus spatially coherent 10 m
    cyclonic vorticity and convergence in the core track. It is not a validated
    meteorological depression detector and does not infer pressure-level flow.
    """
    domain_mean_mslp = float(np.mean(mslp_hpa))

    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_mslp = mslp_hpa[np.ix_(lat_mask, lon_mask)]
    min_mslp = float(np.min(sub_mslp))
    mslp_anomaly = float(min_mslp - domain_mean_mslp)
    min_row, min_col = np.unravel_index(np.argmin(sub_mslp), sub_mslp.shape)
    center_lat = float(lats[lat_mask][min_row])
    center_lon = float(lons[lon_mask][min_col])

    lat_radians = np.deg2rad(lats[:, np.newaxis])
    center_lat_radians = np.deg2rad(center_lat)
    delta_lat = lat_radians - center_lat_radians
    delta_lon = np.deg2rad(lons[np.newaxis, :] - center_lon)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_radians) * np.cos(center_lat_radians) * np.sin(delta_lon / 2.0) ** 2
    )
    distance_degrees = np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))
    inner_mask = distance_degrees <= 1.0
    annulus_mask = (distance_degrees >= 2.0) & (distance_degrees <= 4.0)
    area_weights = np.broadcast_to(np.cos(lat_radians), mslp_hpa.shape)

    def weighted_mean(field: np.ndarray, mask: np.ndarray) -> float:
        valid = mask & np.isfinite(field)
        if not np.any(valid):
            raise ValueError("Depression pressure region contains no finite grid cells")
        return float(np.sum(field[valid] * area_weights[valid]) / np.sum(area_weights[valid]))

    inner_mean_mslp = weighted_mean(mslp_hpa, inner_mask)
    annulus_mean_mslp = weighted_mean(mslp_hpa, annulus_mask)
    pressure_contrast = inner_mean_mslp - annulus_mean_mslp

    # Spatially aggregated cyclonic vorticity and co-located wind convergence.
    zeta = compute_relative_vorticity_2d(u10_ms, v10_ms, lats=lats, lons=lons)
    sub_zeta = zeta[np.ix_(lat_mask, lon_mask)]
    max_vorticity = float(np.max(sub_zeta))
    convergence = compute_horizontal_convergence_2d(u10_ms, v10_ms, lats=lats, lons=lons)
    sub_convergence = convergence[np.ix_(lat_mask, lon_mask)]
    box_weights = area_weights[np.ix_(lat_mask, lon_mask)]
    total_box_area = float(np.sum(box_weights))
    cyclonic_mask = sub_zeta >= DEPRESSION_VORTICITY_THRESHOLD_S1
    vorticity_area_fraction = float(np.sum(box_weights * cyclonic_mask) / total_box_area)
    joint_area_fraction = float(
        np.sum(box_weights * (cyclonic_mask & (sub_convergence > 0.0))) / total_box_area
    )

    pressure_condition = pressure_contrast <= DEPRESSION_PRESSURE_CONTRAST_THRESHOLD_HPA
    circulation_condition = (
        vorticity_area_fraction >= DEPRESSION_VORTICITY_AREA_FRACTION_MIN
        and joint_area_fraction >= DEPRESSION_VORTICITY_CONVERGENCE_AREA_FRACTION_MIN
    )
    candidate_a_event = pressure_condition and circulation_condition
    score = 1.0 if candidate_a_event else 0.0

    metadata = {
        "depression_min_mslp_hpa": min_mslp,
        "depression_inner_mean_mslp_hpa": inner_mean_mslp,
        "depression_annulus_mean_mslp_hpa": annulus_mean_mslp,
        "depression_pressure_contrast_hpa": pressure_contrast,
        "depression_mslp_anomaly_hpa": mslp_anomaly,
        "depression_max_vorticity_s1": max_vorticity,
        "depression_vorticity_area_fraction": vorticity_area_fraction,
        "depression_vorticity_convergence_area_fraction": joint_area_fraction,
        "depression_candidate_a_event": candidate_a_event,
        "depression_score": score,
        "search_bounds_lat": list(lat_bounds),
        "search_bounds_lon": list(lon_bounds),
        "pressure_contrast_threshold_hpa": DEPRESSION_PRESSURE_CONTRAST_THRESHOLD_HPA,
        "vorticity_threshold_s1": DEPRESSION_VORTICITY_THRESHOLD_S1,
        "vorticity_area_fraction_min": DEPRESSION_VORTICITY_AREA_FRACTION_MIN,
        "vorticity_convergence_area_fraction_min": DEPRESSION_VORTICITY_CONVERGENCE_AREA_FRACTION_MIN,
        "proxy_classification": "Category-A 10 m surface proxy; not validated depression detection",
    }

    return SynopticIndexResult(
        name="monsoon_depression_index",
        value=score,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="unitless_score",
        is_computable=True,
        missing_dependencies=[],
        metadata=metadata,
    )


def compute_orographic_interception_index(
    pwat_kg_m2: np.ndarray,
    u10_ms: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    lat_bounds: Tuple[float, float] = (10.0, 18.0),
    lon_bounds: Tuple[float, float] = (72.5, 76.0),
) -> SynopticIndexResult:
    """Calculate Zonal Moisture Flux Interception on Western Ghats Terrain.
    
    Formula:
      F_u = mean(PWAT * u_10) over Western Ghats barrier corridor (10°N-18°N, 72.5°E-76.0°E).
      Units: kg / (m * s).
      
    Classification: Category A (Directly computable from PWAT and 10m zonal wind).
    """
    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_pwat = pwat_kg_m2[np.ix_(lat_mask, lon_mask)]
    sub_u = u10_ms[np.ix_(lat_mask, lon_mask)]
    flux_u = sub_pwat * sub_u

    mean_flux = float(np.mean(flux_u))
    max_flux = float(np.max(flux_u))

    metadata = {
        "mean_zonal_moisture_flux_kg_m_s": mean_flux,
        "max_zonal_moisture_flux_kg_m_s": max_flux,
        "corridor_lats": list(lat_bounds),
        "corridor_lons": list(lon_bounds),
    }

    return SynopticIndexResult(
        name="orographic_ghats_zonal_moisture_flux",
        value=mean_flux,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="kg/(m*s)",
        is_computable=True,
        missing_dependencies=[],
        metadata=metadata,
    )


def compute_western_disturbance_index(
    mslp_hpa: np.ndarray,
    pwat_kg_m2: np.ndarray,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
    lat_bounds: Tuple[float, float] = (28.0, 36.0),
    lon_bounds: Tuple[float, float] = (70.0, 80.0),
) -> SynopticIndexResult:
    """Calculate Northwest India Synoptic Activity (Western Disturbance Proxy).
    
    Domain:
      Northwest India / Western Himalayas (28.0°N-36.0°N, 70.0°E-80.0°E).
      
    Classification: Category A (Directly computable from MSLP and PWAT).
    """
    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])

    sub_mslp = mslp_hpa[np.ix_(lat_mask, lon_mask)]
    sub_pwat = pwat_kg_m2[np.ix_(lat_mask, lon_mask)]

    min_mslp = float(np.min(sub_mslp))
    mean_pwat = float(np.mean(sub_pwat))

    metadata = {
        "nw_india_min_mslp_hpa": min_mslp,
        "nw_india_mean_pwat_mm": mean_pwat,
        "nw_bounds_lat": list(lat_bounds),
        "nw_bounds_lon": list(lon_bounds),
    }

    return SynopticIndexResult(
        name="western_disturbance_nw_activity",
        value=mean_pwat,
        category=IndexCategory.A_DIRECTLY_COMPUTABLE,
        units="mm",
        is_computable=True,
        missing_dependencies=[],
        metadata=metadata,
    )


def extract_all_synoptic_features(
    mslp_hpa: np.ndarray,
    pwat_kg_m2: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
    target_date: str = "2024-06-21",
    u850_ms: Optional[np.ndarray] = None,
    v850_ms: Optional[np.ndarray] = None,
    lats: np.ndarray = CANONICAL_LATS,
    lons: np.ndarray = CANONICAL_LONS,
) -> SynopticFeatureSet:
    """Assemble all synoptic indices and diagnostic features into a unified container."""
    catalog: Dict[str, SynopticIndexResult] = {}

    # 1. Monsoon Trough Position & Latitude Departure
    trough_res = compute_monsoon_trough_index(mslp_hpa, lats=lats, lons=lons)
    catalog["monsoon_trough"] = trough_res

    # 2. Low-Level Jet (LLJ) Strength Index (Category B)
    llj_res = compute_llj_strength_index(u850_ms=u850_ms, v850_ms=v850_ms, lats=lats, lons=lons)
    catalog["llj_strength"] = llj_res

    # 3. Cross-Equatorial / Low-Latitude Inflow Index (Category A)
    inflow_res = compute_cross_equatorial_flow_index(u10_ms, v10_ms, lats=lats, lons=lons)
    catalog["low_latitude_inflow"] = inflow_res

    # 4. Moisture Flux Convergence 2D Proxy (Category A)
    mfc_res = compute_moisture_flux_convergence(pwat_kg_m2, u10_ms, v10_ms, lats=lats, lons=lons)
    catalog["moisture_flux_convergence"] = mfc_res

    # 5. Monsoon Depression Index (Category A)
    dep_res = compute_monsoon_depression_index(mslp_hpa, u10_ms, v10_ms, lats=lats, lons=lons)
    catalog["monsoon_depression"] = dep_res

    # 6. Orographic Interception Index (Category A)
    orog_res = compute_orographic_interception_index(pwat_kg_m2, u10_ms, lats=lats, lons=lons)
    catalog["orographic_interception"] = orog_res

    # 7. Western Disturbance Activity (Category A)
    wd_res = compute_western_disturbance_index(mslp_hpa, pwat_kg_m2, lats=lats, lons=lons)
    catalog["western_disturbance"] = wd_res

    domain_mean_pwat = float(np.mean(pwat_kg_m2))
    domain_mean_mslp = float(np.mean(mslp_hpa))

    return SynopticFeatureSet(
        target_date=target_date,
        trough_mean_latitude=trough_res.metadata["mean_trough_latitude_deg_n"],
        trough_latitude_departure=trough_res.value,
        trough_min_pressure_hpa=trough_res.metadata["trough_minimum_pressure_hpa"],
        trough_pressure_gradient_hpa=trough_res.metadata["trough_pressure_gradient_hpa"],
        llj_strength_850_ms=llj_res.value,
        llj_is_supported=llj_res.is_computable,
        low_latitude_inflow_speed_ms=inflow_res.value,
        low_latitude_inflow_zonal_ms=inflow_res.metadata["mean_zonal_inflow_ms"],
        low_latitude_inflow_meridional_ms=inflow_res.metadata["mean_meridional_inflow_ms"],
        low_latitude_kinetic_energy=inflow_res.metadata["kinetic_energy_m2_s2"],
        mfc_domain_mean_mm_day=mfc_res.value,
        mfc_central_india_mean_mm_day=mfc_res.metadata["central_india_mfc_mm_day"],
        mfc_spatial_map=mfc_res.spatial_field,
        depression_min_mslp_hpa=dep_res.metadata["depression_min_mslp_hpa"],
        depression_mslp_anomaly_hpa=dep_res.metadata["depression_mslp_anomaly_hpa"],
        depression_max_vorticity_s1=dep_res.metadata["depression_max_vorticity_s1"],
        orographic_ghats_zonal_flux=orog_res.value,
        nw_india_min_mslp_hpa=wd_res.metadata["nw_india_min_mslp_hpa"],
        nw_india_mean_pwat_mm=wd_res.value,
        domain_mean_pwat_mm=domain_mean_pwat,
        domain_mean_mslp_hpa=domain_mean_mslp,
        depression_pressure_contrast_hpa=dep_res.metadata["depression_pressure_contrast_hpa"],
        depression_vorticity_area_fraction=dep_res.metadata["depression_vorticity_area_fraction"],
        depression_vorticity_convergence_area_fraction=dep_res.metadata[
            "depression_vorticity_convergence_area_fraction"
        ],
        depression_candidate_a_event=dep_res.metadata["depression_candidate_a_event"],
        feature_catalog=catalog,
        metadata={"total_indices_evaluated": len(catalog)},
    )
