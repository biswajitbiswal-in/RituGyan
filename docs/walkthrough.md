# Walkthrough - Phase 0: Foundation, Tooling & Environment Setup

Phase 0 of **RituGyan** establishes the full architectural repository layout, dependency specifications, centralized Pydantic v2 + YAML configuration framework, structured logging utilities, and pytest automated validation suite.

---

## 1. Directory Structure

The following standardized repository structure was created with corresponding package initializers and `.gitkeep` markers:

```
RituGyan/
├── configs/
│   └── default_config.yaml           # Central domain, regime, alert, and threshold specs
├── data/
│   ├── raw/                          # External raw GRIB2/NetCDF files (gitignored)
│   ├── processed/                    # Harmonized Zarr stores and Parquet feature tables
│   ├── shapefiles/                   # India district & river basin polygon shapefiles
│   └── static/                       # SRTM DEM elevation rasters & distance-to-coast
├── models/
│   └── saved/                        # Trained LightGBM/XGBoost weights & calibrators
├── notebooks/                        # Research, exploratory analysis, and case studies
├── src/
│   ├── __init__.py                   # Package version definition (v0.1.0)
│   ├── ingestion/                    # NWP & observation downloaders & regridders
│   ├── features/                     # Synoptic indices, dynamic features, regime labeler
│   ├── models/
│   │   ├── stage1_regime/            # Stage 1 Multi-class Weather Regime Classifier
│   │   └── stage2_correction/        # Stage 2 Quantile Mapper, ML Corrector & Tail heads
│   ├── evaluation/                   # Verification engine (RMSE, ETS, CSI, POD, FAR, FSS)
│   ├── api/                          # FastAPI REST service & route handlers
│   │   └── routes/
│   ├── ui/                           # Streamlit interactive operational portal
│   └── utils/
│       ├── config.py                 # Pydantic v2 configuration parser & validator
│       └── logger.py                 # Structured logger with console & file handlers
├── tests/
│   ├── conftest.py                   # Pytest fixtures and mock spatial grid generator
│   └── unit/
│       ├── test_config.py            # Configuration validation tests
│       └── test_environment.py       # Layout, importability, and logging tests
├── .gitignore                        # Comprehensive ignores for large rasters/datasets/caches
├── pyproject.toml                    # Modern PEP 621 / setuptools build & packaging
└── requirements.txt                  # Pinned dependencies categorized by subsystem
```

---

## 2. Centralized Configuration Architecture

The configuration framework in [default_config.yaml](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/configs/default_config.yaml) and [config.py](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/src/utils/config.py) provides:

- **Geographic Domain:** Standardized bounding box (`6.0°N to 38.0°N`, `68.0°E to 98.0°E`) at $0.25^\circ$ resolution with computed $129 \times 121$ grid coordinates.
- **IMD Categorization Engine:** Real-time classification for 7 precipitation bins (No Rain, Very Light, Light, Moderate, Heavy, Very Heavy, Extremely Heavy).
- **Regime Catalog:** 6 synoptic regimes (`ACTIVE_MONSOON`, `BREAK_MONSOON`, `MONSOON_DEPRESSION`, `OROGRAPHIC_MONSOON`, `COASTAL_REGIME`, `WESTERN_DISTURBANCE`) accessible by integer ID or uppercase string code.
- **IMD Color Alerts:** Automated derivation of Green, Yellow, Orange, and Red district risk levels from tail exceedance probabilities and expected rain amounts.
- **Resilience Parameters:** Fallback minimum classification confidence ($0.60$) and physical rain limits.

---

## 3. Verification & Test Results

The unit test suite passed 10/10 tests cleanly:

```bash
$ python -m pytest tests/unit/ -v
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\biswa\OneDrive\Documents\GitHub\RituGyan
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 10 items

tests\unit\test_config.py::test_default_config_loading PASSED            [ 10%]
tests\unit\test_config.py::test_domain_grid_calculations PASSED          [ 20%]
tests\unit\test_config.py::test_domain_invalid_lat_lon PASSED            [ 30%]
tests\unit\test_config.py::test_rainfall_categorization PASSED           [ 40%]
tests\unit\test_config.py::test_regime_lookup PASSED                    [ 50%]
tests\unit\test_config.py::test_alert_level_determination PASSED        [ 60%]
tests\unit\test_config.py::test_path_resolution PASSED                  [ 70%]
tests\unit\test_environment.py::test_directory_structure_exists PASSED  [ 80%]
tests\unit\test_environment.py::test_module_imports PASSED              [ 90%]
tests\unit\test_environment.py::test_logger_functionality PASSED        [100%]

============================= 10 passed in 0.12s ==============================
```

Package installation in editable mode was also verified:
```bash
$ python -c "import src; import src.utils.config as cfg; print('Project:', cfg.load_config().system.project_name); print('Regimes:', len(cfg.load_config().regimes))"
Project: RituGyan
Regimes: 6
```
