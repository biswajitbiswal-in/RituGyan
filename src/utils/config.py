"""Central configuration loader and schema validator for RituGyan."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator


class SystemConfig(BaseModel):
    """System-level runtime configuration."""
    project_name: str = "RituGyan"
    version: str = "0.1.0"
    log_level: str = "INFO"
    seed: int = 42


class DomainConfig(BaseModel):
    """Spatial and temporal domain boundaries for the Indian subcontinent."""
    region_name: str = "India"
    lat_min: float = 6.0
    lat_max: float = 38.0
    lon_min: float = 68.0
    lon_max: float = 98.0
    grid_resolution: float = 0.25
    temporal_accumulation_window_hours: int = 24
    observation_cycle_utc_hour: int = 3

    @field_validator("lat_max")
    @classmethod
    def validate_latitude(cls, v: float, info: Any) -> float:
        lat_min = info.data.get("lat_min", 6.0)
        if v <= lat_min:
            raise ValueError(f"lat_max ({v}) must be greater than lat_min ({lat_min})")
        return v

    @field_validator("lon_max")
    @classmethod
    def validate_longitude(cls, v: float, info: Any) -> float:
        lon_min = info.data.get("lon_min", 68.0)
        if v <= lon_min:
            raise ValueError(f"lon_max ({v}) must be greater than lon_min ({lon_min})")
        return v

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (lat_min, lat_max, lon_min, lon_max)."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    @property
    def num_lats(self) -> int:
        """Expected number of latitude grid points."""
        return int(round((self.lat_max - self.lat_min) / self.grid_resolution)) + 1

    @property
    def num_lons(self) -> int:
        """Expected number of longitude grid points."""
        return int(round((self.lon_max - self.lon_min) / self.grid_resolution)) + 1


class RainfallThresholdsConfig(BaseModel):
    """IMD rainfall intensity categories and thresholds in mm/day."""
    trace_cutoff: float = 0.1
    light_lower: float = 2.5
    moderate_lower: float = 15.6
    heavy: float = 64.5
    very_heavy: float = 115.6
    extremely_heavy: float = 204.5
    categories: Dict[str, List[float]] = Field(default_factory=dict)

    def categorize_rainfall(self, value: float) -> str:
        """Return the IMD rainfall category string for a given accumulation in mm/day."""
        if value < self.trace_cutoff:
            return "No Rain"
        elif value < self.light_lower:
            return "Very Light"
        elif value < self.moderate_lower:
            return "Light"
        elif value < self.heavy:
            return "Moderate"
        elif value < self.very_heavy:
            return "Heavy"
        elif value < self.extremely_heavy:
            return "Very Heavy"
        else:
            return "Extremely Heavy"


class AlertLevel(BaseModel):
    """Individual alert level properties."""
    name: str
    action: str
    description: str
    prob_heavy_threshold: Optional[float] = None
    prob_very_heavy_threshold: Optional[float] = None
    prob_extremely_heavy_threshold: Optional[float] = None


class RegimeDefinition(BaseModel):
    """Definition for an individual synoptic weather regime."""
    id: int
    code: str
    name: str
    description: str


class ModelParamsConfig(BaseModel):
    """Model hyperparameters, loss configurations, and fallback controls."""
    min_regime_confidence: float = 0.60
    max_physical_rain_limit_mm: float = 1000.0
    soft_blending_enabled: bool = True
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    quantiles: List[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    stage1_model_type: str = "lightgbm"
    stage2_model_type: str = "lightgbm"


class PathsConfig(BaseModel):
    """Filesystem storage paths."""
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_shapefiles_dir: str = "data/shapefiles"
    data_static_dir: str = "data/static"
    models_dir: str = "models/saved"
    notebooks_dir: str = "notebooks"
    logs_dir: str = "logs"

    def resolve_paths(self, base_dir: Path) -> Dict[str, Path]:
        """Resolve all relative paths against a project base directory."""
        return {
            "data_raw": (base_dir / self.data_raw_dir).resolve(),
            "data_processed": (base_dir / self.data_processed_dir).resolve(),
            "data_shapefiles": (base_dir / self.data_shapefiles_dir).resolve(),
            "data_static": (base_dir / self.data_static_dir).resolve(),
            "models": (base_dir / self.models_dir).resolve(),
            "notebooks": (base_dir / self.notebooks_dir).resolve(),
            "logs": (base_dir / self.logs_dir).resolve(),
        }


class ApiConfig(BaseModel):
    """FastAPI service configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "RituGyan AI Forecast API"
    version: str = "0.1.0"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class UiConfig(BaseModel):
    """Web frontend UI dashboard configuration (HTML/CSS/JS or React)."""
    port: int = 3000
    theme: str = "dark"
    default_lat: float = 20.5937
    default_lon: float = 78.9629
    default_zoom: int = 5
    api_base_url: str = "http://localhost:8000/api/v1"


class RituGyanConfig(BaseModel):
    """Top-level configuration container for the RituGyan system."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    rainfall_thresholds: RainfallThresholdsConfig = Field(default_factory=RainfallThresholdsConfig)
    alert_levels: Dict[str, AlertLevel] = Field(default_factory=dict)
    regimes: Dict[str, RegimeDefinition] = Field(default_factory=dict)
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    ui: UiConfig = Field(default_factory=UiConfig)

    def get_regime_by_id(self, regime_id: int) -> Optional[RegimeDefinition]:
        """Look up regime definition by its integer ID."""
        for regime in self.regimes.values():
            if regime.id == regime_id:
                return regime
        return None

    def get_regime_by_code(self, code: str) -> Optional[RegimeDefinition]:
        """Look up regime definition by code (e.g. 'ACTIVE_MONSOON')."""
        normalized_code = code.upper().strip()
        for regime in self.regimes.values():
            if regime.code == normalized_code:
                return regime
        return None

    def determine_alert_level(
        self,
        prob_heavy: float = 0.0,
        prob_very_heavy: float = 0.0,
        prob_extremely_heavy: float = 0.0,
        expected_rain_mm: float = 0.0,
    ) -> str:
        """Determine IMD color-coded alert level based on probabilities and rain amount."""
        if (
            prob_extremely_heavy >= 0.30
            or prob_very_heavy >= 0.60
            or expected_rain_mm >= self.rainfall_thresholds.extremely_heavy
        ):
            return "RED"
        elif (
            prob_very_heavy >= 0.30
            or prob_heavy >= 0.60
            or expected_rain_mm >= self.rainfall_thresholds.heavy
        ):
            return "ORANGE"
        elif (
            prob_heavy >= 0.30
            or expected_rain_mm >= self.rainfall_thresholds.moderate_lower
        ):
            return "YELLOW"
        else:
            return "GREEN"


_CACHED_CONFIG: Optional[RituGyanConfig] = None


def find_project_root() -> Path:
    """Detect project root directory by searching upwards for pyproject.toml or configs/."""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "default_config.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current


def load_config(config_path: Optional[str | Path] = None, force_reload: bool = False) -> RituGyanConfig:
    """Load, validate, and return the RituGyan configuration.
    
    Args:
        config_path: Path to YAML config file. If None, default_config.yaml is used.
        force_reload: Whether to force re-reading the file even if cached.
        
    Returns:
        Validated RituGyanConfig instance.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload and config_path is None:
        return _CACHED_CONFIG

    if config_path is None:
        env_path = os.getenv("RITUGYAN_CONFIG_PATH")
        if env_path:
            config_file = Path(env_path)
        else:
            root = find_project_root()
            config_file = root / "configs" / "default_config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        # Return sensible default config if file is missing
        config = RituGyanConfig()
    else:
        with open(config_file, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        config = RituGyanConfig.model_validate(raw_data)

    if config_path is None:
        _CACHED_CONFIG = config

    return config
