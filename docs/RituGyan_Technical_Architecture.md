# Technical Architecture Document

## RituGyan — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

**Document Version:** 1.0  
**Status:** Approved Architecture Blueprint  
**Companions:** [RituGyan_PRD.md](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/docs/RituGyan_PRD.md), [RituGyan_Implementation_Plan.md](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/docs/RituGyan_Implementation_Plan.md), [RituGyan_Phase_Requirements_and_External_Dependencies.md](file:///c:/Users/biswa/OneDrive/Documents/GitHub/RituGyan/docs/RituGyan_Phase_Requirements_and_External_Dependencies.md)

---

## 1. Architecture Overview & Design Principles

### 1.1 Executive Summary
RituGyan is an operational-grade, AI/ML-driven meteorological post-processing platform designed to ingest raw Numerical Weather Prediction (NWP) forecasts, identify synoptic and local weather regimes across the Indian subcontinent, and apply regime-conditional bias correction and tail-risk probability estimators at high spatial resolution (grid and district level).

### 1.2 Core Architectural Principles
1. **Regime Conditioning over Static Global Correction:** Decouple correction algorithms based on meteorological dynamics (e.g., Depression vs. Break Monsoon) rather than applying a single blended statistical model.
2. **Tail-Risk Optimization:** Explicitly optimize for extreme precipitation events (>64.5 mm/day, >115.6 mm/day, >204.5 mm/day) rather than standard RMSE-only skill scores that wash out tail events.
3. **Decoupled, Modular Pipeline:** Ingestion, feature store, classification, correction, verification, and presentation layers operate as isolated, testable, and maintainable services.
4. **Resilience & Graceful Degradation:** The pipeline implements automatic fallback strategies (e.g., if regime classification confidence is low or auxiliary synoptic data fails, the system safely falls back to standard Empirical Quantile Mapping or raw NWP with clear diagnostic flags).
5. **Transparent, Regime-Stratified Verification:** Continuous scoring engine computes contingency tables, equitable threat scores, and spatial metrics (FSS) broken down by regime.

---

## 2. High-Level Architecture (C4 Model)

### 2.1 System Context (C4 Level 1)

```mermaid
flowchart TD
    subgraph External_Data_Sources [External Data Ingestion]
        NWP[NWP Providers<br/>NCMRWF, GFS, ECMWF]
        OBS[Observed Rainfall<br/>IMD Gridded 0.25°, IMERG]
        SYN[Synoptic & Reanalysis<br/>ERA5, IMD Bulletins]
        GEO[Geospatial Assets<br/>SRTM DEM, District Shapefiles]
    end

    subgraph RituGyan_Platform [RituGyan AI Platform]
        IngestEngine[Data Ingestion & Alignment Engine]
        FeatureStore[Feature Engineering & Store]
        RegimeEngine[Stage 1: Regime Classification Service]
        CorrectionEngine[Stage 2: Regime-Conditional Bias Correction]
        VerifEngine[Verification & Model Governance]
        CoreAPI[RituGyan Serving Core / FastAPI]
    end

    subgraph Downstream_Consumers [Downstream Stakeholders & Systems]
        IMD_Ops[IMD & Operational Forecasters]
        DM[Disaster Management Authorities NDMA/SDMAs]
        Agri[Agricultural Decision Support Systems]
        Dashboard[RituGyan Interactive Operational Portal]
    end

    NWP --> IngestEngine
    OBS --> IngestEngine
    SYN --> IngestEngine
    GEO --> IngestEngine

    IngestEngine --> FeatureStore
    FeatureStore --> RegimeEngine
    FeatureStore --> CorrectionEngine
    RegimeEngine --> CorrectionEngine
    CorrectionEngine --> VerifEngine
    CorrectionEngine --> CoreAPI
    VerifEngine --> CoreAPI

    CoreAPI --> Dashboard
    CoreAPI --> IMD_Ops
    CoreAPI --> DM
    CoreAPI --> Agri
```

### 2.2 System Container Architecture (C4 Level 2)

```mermaid
flowchart LR
    subgraph Storage_Tier [Storage & Persistence Tier]
        RawData[(Object Store: Raw NetCDF / GRIB2 / GeoTIFF)]
        ZarrStore[(Processed Zarr / Parquet Feature Store)]
        ModelRegistry[(Model Artifacts & MLflow Registry)]
        ResultsDB[(Forecast & Verification Store: PostgreSQL / PostGIS)]
    end

    subgraph Compute_Tier [Pipeline & Compute Tier]
        Scheduler[Task Scheduler / APScheduler / Celery]
        IngestJob[Data Harmonization & Regridding Worker]
        Stage1Worker[Stage 1: Regime Classifier Worker]
        Stage2Worker[Stage 2: Bias Correction & Prob. Worker]
        EvalWorker[Regime-Stratified Verification Worker]
    end

    subgraph Service_Tier [Application & Serving Tier]
        FastAPIApp[FastAPI REST / GeoJSON Service]
        WebPortalUI[Interactive Web Portal: React / HTML-CSS-JS Dashboard]
    end

    Scheduler --> IngestJob
    IngestJob --> RawData
    IngestJob --> ZarrStore
    
    Scheduler --> Stage1Worker
    ZarrStore --> Stage1Worker
    ModelRegistry --> Stage1Worker
    
    Scheduler --> Stage2Worker
    Stage1Worker --> Stage2Worker
    ZarrStore --> Stage2Worker
    ModelRegistry --> Stage2Worker
    Stage2Worker --> ResultsDB

    Scheduler --> EvalWorker
    ResultsDB --> EvalWorker
    EvalWorker --> ResultsDB

    ResultsDB --> FastAPIApp
    FastAPIApp --> WebPortalUI
```

---

## 3. Subsystem Detailed Specifications

### 3.1 Data Architecture & Ingestion Subsystem

```mermaid
flowchart TD
    A[Raw NWP GRIB2 / NetCDF4] --> D[Spatial Harmonization: xESMF / Bilinear & Conservative Regridding to 0.25° / 0.1°]
    B[IMD Observed / IMERG Gridded] --> D
    C[ERA5 Reanalysis / Static DEM] --> D
    
    D --> E[Temporal Alignment: Daily Accumulation 03:00 UTC / 08:30 IST]
    E --> F[Feature Construction: Zarr Cubes & Columnar Parquet]
    F --> G[Data Validation: Great Expectations Schema Assertions]
```

#### Ingestion Specifications:
- **Spatial Alignment:** Standardized target regular grid over the Indian domain (`6.0°N to 38.0°N`, `68.0°E to 98.0°E`) using conservative regridding (`xesmf.Regridder(method='conservative')`) for flux/precipitation and bilinear interpolation for continuous fields (temperature, geopotential height, MSLP).
- **Temporal Harmonization:** Accumulation window standardizing all NWP model steps to match IMD's 24-hour observation cycle (03:00 UTC of Day $T$ to 03:00 UTC of Day $T+1$, corresponding to 08:30 IST).
- **Static Covariates:**
  - SRTM Digital Elevation Model (DEM) aggregated to grid scale (mean elevation, slope, aspect).
  - Distance-to-coast raster (Euclidean distance transform).
  - Land-sea mask and regional district boundaries (Survey of India / IMD shapefiles).

---

### 3.2 Feature Store & Synoptic Representation

The Feature Store generates normalized spatial tensors and tabular vector embeddings per grid point and synoptic region.

| Feature Group | Parameters | Physical Meaning / Synoptic Value |
|---|---|---|
| **Pressure Fields** | MSLP, MSLP Anomaly, Geopotential Height (500 hPa, 200 hPa) | Identifies low-pressure systems, monsoon trough position, WD troughs |
| **Wind & Vorticity** | $u, v$ components at 850 hPa & 200 hPa, Wind Shear, Relative Vorticity | Low-level jet strength, monsoon trough shear zone, tropical easterly jet |
| **Moisture & Thermodynamics** | Specific Humidity (850 hPa, 700 hPa), Total Column Water Vapour (TCWV), Moisture Flux Divergence | Moisture availability and convergence zones |
| **Convective Activity** | Outgoing Longwave Radiation (OLR), Convective Available Potential Energy (CAPE) | Deep convection and active monsoon cloud bands |
| **Climatological Anomalies** | Antecedent 3-day / 7-day rainfall departure from normal | Soil moisture feedback and memory effects |
| **Topographic / Coastal** | Elevation, Terrain Gradient, Distance to Coastline | Orographic lift triggering (Western Ghats, Northeast hills) |

---

### 3.3 Stage 1: Weather Regime Classification Engine

```mermaid
flowchart LR
    subgraph Inputs
        SynopticFeatures[Synoptic Feature Vector<br/>Monsoon Trough Index, OLR, Vorticity]
        SpatialGrids[Spatial Wind & Pressure Fields]
        LocationMeta[District / Region Geo-Tag]
    end

    subgraph Classifier_Ensemble [Stage 1 Ensemble Classifier]
        XGB_Model[LightGBM / XGBoost Gradient Boosted Classifier]
        SpatialCNN[Spatial Feature Extractor / CNN optional]
        RuleEngine[IMD Onset/Break & Depression Heuristics]
    end

    subgraph Output
        RegimeID[Predicted Regime Label]
        Confidence[Soft Probability Distribution]
    end

    SynopticFeatures --> XGB_Model
    SpatialGrids --> SpatialCNN
    SynopticFeatures --> RuleEngine
    
    XGB_Model --> Output
    SpatialCNN --> Output
    RuleEngine --> Output
```

#### Meteorological Regimes Classification Catalog:
1. **Active Monsoon:** Strong cross-equatorial flow, low-level jet at 850 hPa > 30 knots, monsoon trough south of normal position, high OLR anomalies over central India.
2. **Break Monsoon:** Monsoon trough shifted north to Himalayan foothills, marked rainfall reduction over Central India, increased rainfall over NE India and foothills.
3. **Monsoon Low / Depression:** Closed cyclonic circulation up to mid-troposphere, pressure departure < -2 to -4 hPa, RSMC tracked low-pressure systems.
4. **Orographic Monsoon:** Heavy precipitation triggered along windward slopes (Western Ghats, Meghalaya plateau) driven by strong westerly/southwesterly moisture flux.
5. **Coastal Regime:** Strong land-sea thermal and friction contrasts along the western and eastern coastlines with localized convergence.
6. **Western Disturbance:** Mid-latitude upper-tropospheric westerly trough propagating across northwest India (prominent in transitional and pre/post-monsoon periods).

#### Classifier Data Contracts:
```python
class RegimeClassificationOutput(BaseModel):
    timestamp: datetime
    region_id: str
    predicted_regime: str  # e.g., "DEPRESSION_LOW"
    regime_confidence: float  # e.g., 0.89
    class_probabilities: Dict[str, float]
    synoptic_metrics: Dict[str, float]  # trough_index, shear_magnitude, etc.
```

---

### 3.4 Stage 2: Regime-Conditional Bias Correction & Tail-Risk Engine

```mermaid
flowchart TD
    RawNWP[Raw NWP Rainfall Forecast] --> Router{Regime Router}
    RegimeIn[Stage 1 Regime Label + Probabilities] --> Router
    
    Router -->|Active| M1[Active Regime Correction Model]
    Router -->|Break| M2[Break Regime Correction Model]
    Router -->|Depression| M3[Depression Regime Correction Model]
    Router -->|Orographic| M4[Orographic Regime Correction Model]
    Router -->|Coastal| M5[Coastal Regime Correction Model]
    Router -->|Western Disturbance| M6[WD Correction Model]
    
    M1 --> EnsembleCombiner[Unified Bias-Corrected Continuous Rainfall Grid]
    M2 --> EnsembleCombiner
    M3 --> EnsembleCombiner
    M4 --> EnsembleCombiner
    M5 --> EnsembleCombiner
    M6 --> EnsembleCombiner

    EnsembleCombiner --> TailModel[Extreme Tail Probabilistic Estimator]
    
    TailModel --> Out1[Corrected Rainfall mm/day]
    TailModel --> Out2[P Heavy > 64.5mm]
    TailModel --> Out3[P Very Heavy > 115.6mm]
    TailModel --> Out4[P Extremely Heavy > 204.5mm]
    
    Out1 --> ZonalAgg[Zonal Statistics & District Mapping Engine]
    Out2 --> ZonalAgg
    Out3 --> ZonalAgg
    Out4 --> ZonalAgg
    
    ZonalAgg --> DistrictProduct[District GeoJSON / Operational Alerts]
```

#### Dual-Correction Methodology:
1. **Tier 1 (Empirical Quantile Mapping - EQM Baseline):**
   $$x_{\text{corr}} = F_{\text{obs}, r}^{-1}\left(F_{\text{nwp}, r}(x_{\text{raw}})\right)$$
   Where $F_{\text{obs}, r}$ and $F_{\text{nwp}, r}$ are the cumulative distribution functions for regime $r$.

2. **Tier 2 (Unified LightGBM / XGBoost Non-Linear Residual Regressor):**
   $$y_{\text{corr}} = x_{\text{raw}} + f_{\theta}\left(x_{\text{raw}}, r, \mathbf{z}_{\text{synoptic}}, \mathbf{z}_{\text{topo}}\right)$$
   Where $r$ is the regime embedding vector, $\mathbf{z}_{\text{synoptic}}$ represents ambient moisture/wind features, and $\mathbf{z}_{\text{topo}}$ represents elevation and slope.

3. **Multi-Threshold Tail Classification Heads:**
   Binary probability classifiers trained with focal loss / balanced class weights to predict exceedance probabilities:
   - $P(\text{Rain} \ge 64.5\text{ mm/day})$ (IMD Heavy Rainfall)
   - $P(\text{Rain} \ge 115.6\text{ mm/day})$ (IMD Very Heavy Rainfall)
   - $P(\text{Rain} \ge 204.5\text{ mm/day})$ (IMD Extremely Heavy Rainfall)

---

### 3.5 Verification & Governance Subsystem

The verification engine operates asynchronously to ingest verified ground truth once IMD daily observations are finalized, producing regime-stratified skill scores.

```mermaid
flowchart LR
    ForecastData[(Forecast DB)] --> VerifService[Verification Engine]
    ObservedData[(Observation DB)] --> VerifService
    RegimeLabels[(Regime History)] --> VerifService

    VerifService --> ContinuousMetrics[Continuous: RMSE, MAE, Bias]
    VerifService --> CategoricalMetrics[Categorical: ETS, CSI, POD, FAR]
    VerifService --> SpatialMetrics[Spatial: Fractions Skill Score FSS]
    
    ContinuousMetrics --> ScorecardGen[Regime-Stratified Scorecard Generator]
    CategoricalMetrics --> ScorecardGen
    SpatialMetrics --> ScorecardGen
    
    ScorecardGen --> VerifReport[Verification Report & MLflow Dashboard]
```

#### Mathematical Formulations:
- **Equitable Threat Score (ETS):**
  $$\text{ETS} = \frac{H - H_{\text{random}}}{H + M + F - H_{\text{random}}}, \quad H_{\text{random}} = \frac{(H + M)(H + F)}{N}$$
- **Fraction Skill Score (FSS):**
  $$\text{FSS}_{(n)} = 1 - \frac{\text{MSE}_{(n)}}{\text{MSE}_{(n),\text{ref}}}$$
  Evaluated across spatial neighborhood sizes ($n = 1, 3, 5, 9$ grid squares) to evaluate spatial placement skill independent of point double-penalty errors.

---

## 4. API & Interface Architecture

### 4.1 FastAPI Service Endpoints

```
/api/v1
  ├── /forecast
  │     ├── GET  /latest?date=YYYY-MM-DD
  │     ├── GET  /district/{district_id}?date=YYYY-MM-DD
  │     └── GET  /grid/geojson?date=YYYY-MM-DD&threshold=heavy
  ├── /regime
  │     ├── GET  /current
  │     ├── GET  /history?start=...&end=...
  │     └── GET  /diagnostics/{region_id}
  ├── /verification
  │     ├── GET  /scorecard?regime=all&season=2024
  │     ├── GET  /metrics/timeseries
  │     └── GET  /report/export?format=pdf|json
  └── /health
        └── GET  /status
```

### 4.2 Forecast District Output Schema

```json
{
  "forecast_date": "2026-07-15",
  "issue_time": "2026-07-15T06:00:00Z",
  "district_id": "IN-MH-PUN",
  "district_name": "Pune",
  "state": "Maharashtra",
  "detected_regime": {
    "primary": "OROGRAPHIC_MONSOON",
    "confidence": 0.92,
    "secondary_influences": ["ACTIVE_MONSOON"]
  },
  "raw_nwp_forecast_mm": 48.5,
  "ritugyan_corrected_forecast_mm": 78.2,
  "uncertainty_interval_90": [62.0, 96.5],
  "exceedance_probabilities": {
    "heavy_gt_64_5mm": 0.76,
    "very_heavy_gt_115_6mm": 0.28,
    "extremely_heavy_gt_204_5mm": 0.04
  },
  "alert_level": "ORANGE",
  "verification_baseline_skill": {
    "historical_regime_rmse_reduction_pct": 28.4
  }
}
```

---

## 5. Technology Stack & Infrastructure Topology

### 5.1 Component Tech Stack Table

| Architectural Layer | Technologies / Frameworks | Justification |
|---|---|---|
| **Data Ingestion & Grids** | `xarray`, `netCDF4`, `cfgrib`, `xesmf`, `dask` | Native multi-dimensional meteorological raster handling and out-of-core computing |
| **Geospatial & Vector Ops** | `geopandas`, `shapely`, `rasterio`, `scipy` | Zonal district masking, spatial distance computations, CRS projections |
| **ML Modeling (Stage 1 & 2)** | `LightGBM`, `XGBoost`, `scikit-learn`, `PyTorch` | State-of-the-art tabular/spatial gradient boosting, fast inference, GPU training capability |
| **Verification & Metrics** | `xskillscore`, `pysteps`, `numpy` | Standardized WMO-compliant meteorological skill evaluation and spatial neighborhood FSS |
| **Model Tracking & Registry** | `MLflow`, `DVC` | Reproducible artifact tracking, model lineage, hyperparameter tuning |
| **API & Web Service** | `FastAPI`, `Uvicorn`, `Pydantic v2` | High-performance asynchronous REST API with automatic OpenAPI documentation |
| **Storage & Caching** | `Zarr` (grids), `Parquet` (tabular), `PostgreSQL/PostGIS` (metadata/districts) | Optimized tiered storage for large multidimensional time-series and GIS queries |
| **Interactive Visualization** | `React` / `HTML5+CSS+JS`, `Plotly.js`, `Leaflet` / `MapLibre` | Rich interactive operational web dashboard for forecast maps, regime diagnostics, and scorecards |

### 5.2 Deployment & Container Topology

```mermaid
flowchart TD
    subgraph Host_Or_Kubernetes_Cluster [Docker Compose / Kubernetes Cluster]
        subgraph Pipeline_Pods [Pipeline Services]
            WorkerIngest[Ingestion & Regridding Container]
            WorkerML[Stage 1 & 2 Inference Container]
            WorkerEval[Verification & Scorecard Container]
        end

        subgraph Core_Services [Serving & DB Services]
            FastAPIContainer[FastAPI Core Server]
            PostgresContainer[(PostgreSQL / PostGIS)]
            MLflowContainer[MLflow Tracking Server]
            WebPortalContainer[Interactive Web Portal UI Container]
        end

        subgraph Volume_Mounts [Shared Volumes / Object Storage]
            DataLake[(Zarr / NetCDF Storage: Local Disk / S3)]
        end
    end

    DataLake --- WorkerIngest
    DataLake --- WorkerML
    DataLake --- WorkerEval
    PostgresContainer --- FastAPIContainer
    FastAPIContainer --- WebPortalContainer
    MLflowContainer --- WorkerML
```

---

## 6. End-to-End Operational Workflow & Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Scheduler as Operational Cron / Scheduler
    participant Ingest as Ingestion & Preprocessing
    participant Stage1 as Stage 1: Regime Classifier
    participant Stage2 as Stage 2: Bias Correction & Prob.
    participant Store as Results Store (PostGIS/Zarr)
    participant API as FastAPI Serving
    actor User as Forecaster / Disaster Manager

    Scheduler->>Ingest: Trigger 00Z / 12Z Daily Ingestion
    Ingest->>Ingest: Download & Harmonize NWP + Synoptic Fields to 0.25° Grid
    Ingest->>Stage1: Provide Synoptic Feature Vector & Grids
    Stage1->>Stage1: Classify Regime (Active, Break, Depression, etc.) + Confidences
    Stage1->>Stage2: Pass Regime Class & Weighting Matrix
    Stage2->>Stage2: Run Regime-Conditional Regressor + Tail Classifiers
    Stage2->>Stage2: Compute District Zonal Averages & Threshold Exceedances
    Stage2->>Store: Persist Corrected Grid & District JSON Records
    Store-->>API: Data Available for Serving
    User->>API: Query District Forecast & Exceedance Probability
    API-->>User: Return GeoJSON Forecast, Regime Diagnostics & Risk Level
```

---

## 7. Resilience, Fallback & Quality Assurance

```mermaid
flowchart TD
    A[Start Daily Forecast Run] --> B{Synoptic Data Available?}
    B -->|Yes| C[Execute Stage 1 Full Classifier]
    B -->|No / Partial| D[Execute Spatial Location Fallback + Climatology Classifier]
    
    C --> E{Regime Confidence >= 0.60?}
    E -->|Yes| F[Apply Stage 2 Pure Regime-Specific Model]
    E -->|No| G[Apply Soft-Weighted Ensemble across Candidate Regimes]
    D --> G
    
    F --> H{ML Model Output Valid? Check NaN/Extremes}
    G --> H
    H -->|Pass| I[Publish RituGyan Corrected Forecast]
    H -->|Fail Assertion| J[Fallback to Empirical Quantile Mapping EQM Baseline]
    J --> I
```

1. **Missing Inflow Handling:** If external synoptic indices (e.g., OLR from satellite) are delayed, the system seamlessly falls back to NWP-derived proxies (e.g., NWP simulated OLR / vertical velocity).
2. **Confidence-Weighted Soft Correction:** If Stage 1 classification confidence falls below the strict threshold ($0.60$), Stage 2 blends correction predictions weighted proportionally by the regime class posterior probabilities $\sum p_k \cdot f_k(x)$.
3. **Physical Bound Assertions:** Precipitation outputs are strictly bounded $[0, \text{max\_physical\_limit}]$ with automatic clipping and logging of anomalous values.
4. **Data Drift & Concept Drift Monitoring:** MLflow monitors regime distribution shifts over rolling 30-day windows to detect changes in NWP model versions or climate anomalies.

---

## 8. Repository Codebase Organization Blueprint

```
RituGyan/
├── data/
│   ├── raw/                  # Downloaded raw GRIB2/NetCDF files (gitignored)
│   ├── processed/            # Regridded Zarr stores and Parquet tables
│   ├── shapefiles/           # India district and basin polygon shapefiles
│   └── static/               # Topography (DEM), distance-to-coast rasters
├── docs/
│   ├── RituGyan_PRD.md
│   ├── RituGyan_Technical_Approach.md
│   └── RituGyan_Technical_Architecture.md
├── src/
│   ├── ingestion/            # NWP, observation, and synoptic downloaders & regridders
│   │   ├── nwp_ingest.py
│   │   ├── obs_ingest.py
│   │   └── regridder.py
│   ├── features/             # Synoptic index calculators and spatial feature extraction
│   │   ├── synoptic_indices.py
│   │   └── feature_pipeline.py
│   ├── models/
│   │   ├── stage1_regime/    # Classifier trainers, inference, and regime definitions
│   │   │   ├── classifier.py
│   │   │   └── regime_definitions.py
│   │   └── stage2_correction/# Regime-conditional regressors & tail risk classifiers
│   │       ├── quantile_mapper.py
│   │       ├── ml_corrector.py
│   │       └── exceedance_classifier.py
│   ├── evaluation/           # Verification engine (RMSE, ETS, CSI, POD, FAR, FSS)
│   │   ├── metrics.py
│   │   ├── spatial_fss.py
│   │   └── scorecard.py
│   ├── api/                  # FastAPI web service endpoints and schemas
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── routes/
│   └── ui/                   # Web dashboard / React portal frontend components
│       └── app.py
├── tests/                    # Unit, integration, and meteorological validation tests
├── configs/                  # YAML configurations (grid specs, model hyperparameters)
├── notebooks/                # Exploratory data analysis, case studies & verification plots
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

*End of Architecture Document*
