# Phase 2B: Spatial Alignment Specification & Common Subgrid Indexing Report — RituGyan

**Date:** September 25, 2026  
**Status:** **PASS**  
**Author:** RituGyan Pipeline Diagnostic & Ingestion Engine  

---

## Executive Summary

Phase 2B defines, validates, and standardizes the spatial indexing architecture linking ground-truth observations (**IMD Daily Rainfall**), reanalysis fields (**ERA5 Reanalysis**), and numerical weather prediction forecasts (**NOAA GFS 0.25°**).

All three raw datasets natively share an exact quarter-degree ($0.25^\circ \approx 27.75\text{ km}$) lattice resolution. Through pure coordinate-index slicing (without any spatial interpolation, bilinear resampling, or regridding), an exact, coincident **$127 \times 121$ subgrid** ($15,367$ grid points) spanning the Indian domain was extracted and verified to have **$0.0000^\circ$ coordinate discrepancy** across all three datasets.

---

## 1. Original Dataset Spatial Geometries

| Dataset | Native Format | Native Grid Dimensions | Latitude Range & Ordering | Longitude Range & Ordering | Grid Spacing |
|---|---|---|---|---|---|
| **IMD Gridded Rainfall** | NetCDF-4 | $129 \times 135$ | $6.50^\circ\text{N} \to 38.50^\circ\text{N}$ (Ascending) | $66.50^\circ\text{E} \to 100.00^\circ\text{E}$ (Ascending) | $0.25^\circ$ |
| **ERA5 Reanalysis** | NetCDF-4 | $129 \times 121$ | $38.00^\circ\text{N} \to 6.00^\circ\text{N}$ (**Descending**) | $68.00^\circ\text{E} \to 98.00^\circ\text{E}$ (Ascending) | $0.25^\circ$ |
| **NOAA GFS Global** | WMO GRIB2 | $721 \times 1440$ | $90.00^\circ\text{N} \to -90.00^\circ\text{N}$ (**Descending**) | $0.00^\circ\text{E} \to 359.75^\circ\text{E}$ (Ascending) | $0.25^\circ$ |

---

## 2. Common Indian Subgrid Specification

The canonical bounding box is defined by the intersection of the spatial coverages:

* **Latitude Extent**: $[6.50^\circ\text{N}, 38.00^\circ\text{N}]$
  - Lower bound $6.50^\circ\text{N}$: Determined by IMD southern boundary ($6.50^\circ\text{N}$).
  - Upper bound $38.00^\circ\text{N}$: Determined by ERA5 northern boundary ($38.00^\circ\text{N}$).
  - Total Latitude Grid Points: **$127$** ($6.50, 6.75, 7.00, \dots, 37.75, 38.00$)
* **Longitude Extent**: $[68.00^\circ\text{E}, 98.00^\circ\text{E}]$
  - Lower bound $68.00^\circ\text{E}$: Determined by ERA5 western boundary ($68.00^\circ\text{E}$).
  - Upper bound $98.00^\circ\text{E}$: Determined by ERA5 eastern boundary ($98.00^\circ\text{E}$).
  - Total Longitude Grid Points: **$121$** ($68.00, 68.25, 68.50, \dots, 97.75, 98.00$)
* **Canonical Dimensions**: **$127 \times 121 = 15,367$ grid cells**
* **Canonical Coordinate Representation**:
  - Latitude: **Strictly Ascending** ($6.50^\circ\text{N} \to 38.00^\circ\text{N}$)
  - Longitude: **Strictly Ascending** ($68.00^\circ\text{E} \to 98.00^\circ\text{E}$)

---

## 3. Pure Index Slicing Method (Zero Interpolation)

Because all datasets lie on exact quarter-degree lattice points (`.00`, `.25`, `.50`, `.75`), extraction is performed strictly by index selection with zero resampling error:

### A. IMD Extraction
* **Latitude Index Slice**: `slice(6.50, 38.00)` $\to$ rows `0:127` (127 points, ascending).
* **Longitude Index Slice**: `slice(68.00, 98.00)` $\to$ cols `6:127` (121 points, ascending).
* **Orientation Action**: None required (native ordering is ascending $\times$ ascending).

### B. ERA5 Extraction
* **Latitude Index Slice**: `slice(38.00, 6.50)` $\to$ rows `0:127` (127 points, native descending).
* **Longitude Index Slice**: `slice(68.00, 98.00)` $\to$ cols `0:121` (121 points, native ascending).
* **Orientation Action**: Latitude axis is reversed/sorted ascending (`sub_ds.sortby('latitude', ascending=True)` or `array[::-1, :]`).

### C. GFS Global Extraction
* **Global Grid Dimensions**: $721 \text{ rows} \times 1440 \text{ cols}$.
* **Latitude Indexing Formula**:
  $$\text{Row Index} = \text{round}\left(\frac{90.0 - \text{lat}}{0.25}\right)$$
  - For $38.00^\circ\text{N}$: $\text{Row Index} = (90.0 - 38.0) / 0.25 = 208$
  - For $6.50^\circ\text{N}$: $\text{Row Index} = (90.0 - 6.5) / 0.25 = 334$
  - Sliced rows: `208 : 335` (127 points).
* **Longitude Indexing Formula**:
  $$\text{Col Index} = \text{round}\left(\frac{\text{lon}}{0.25}\right)$$
  - For $68.00^\circ\text{E}$: $\text{Col Index} = 68.0 / 0.25 = 272$
  - For $98.00^\circ\text{E}$: $\text{Col Index} = 98.0 / 0.25 = 392$
  - Sliced cols: `272 : 393` (121 points).
* **Orientation Action**: Rows are flipped along the latitude axis (`sliced[::-1, :]`) to produce ascending latitude ($6.50^\circ\text{N} \to 38.00^\circ\text{N}$).

---

## 4. Coordinate Equality & Half-Grid Verification

Direct numerical comparison of the sliced coordinate arrays:

```python
Target Shape: (127, 121)
Latitude range: 6.50 to 38.00 (step = 0.25)
Longitude range: 68.00 to 98.00 (step = 0.25)

Cross-Dataset Discrepancy Matrix:
- max(|IMD_lat - ERA5_lat|) = 0.00000000000000°
- max(|IMD_lon - ERA5_lon|) = 0.00000000000000°
- max(|IMD_lat - GFS_lat|)  = 0.00000000000000°
- max(|IMD_lon - GFS_lon|)  = 0.00000000000000°
- max(|ERA5_lat - GFS_lat|) = 0.00000000000000°
- max(|ERA5_lon - GFS_lon|) = 0.00000000000000°
```

* **Half-Grid Offset Check**: Every coordinate value modulo $0.25^\circ$ evaluates to $0.0000^\circ$. No Arakawa staggering, cell-center vs cell-edge discrepancies, or half-grid displacements exist.
* **Monotonicity**: All extracted 1D coordinate vectors are strictly monotonically increasing ($\Delta x = +0.25^\circ > 0$).

---

## 5. Memory & Processing Performance

* **Subgrid Memory Profile**:
  - Single 2D float32 subgrid ($127 \times 121$): **$61,468\text{ bytes} \approx 60.03\text{ KB}$**.
  - Full day feature snapshot (1 GFS 24h rainfall + 6 atmospheric predictors + 4 ERA5 timesteps): **$< 1.5\text{ MB}$**.
* **Index Slicing Efficiency**: Slicing occurs via direct memory strides in NumPy/C without creating intermediate full-resolution global buffers or computing interpolation weight matrices.

---

## 6. Automated Test Suite Results

Test execution (`tests/phase2/test_spatial_alignment.py`):

```
tests/phase2/test_spatial_alignment.py::TestSpatialAlignment::test_canonical_grid_specification PASSED
tests/phase2/test_spatial_alignment.py::TestSpatialAlignment::test_imd_spatial_indexing_and_extraction PASSED
tests/phase2/test_spatial_alignment.py::TestSpatialAlignment::test_era5_spatial_indexing_and_orientation PASSED
tests/phase2/test_spatial_alignment.py::TestSpatialAlignment::test_gfs_spatial_indexing_and_longitude_handling PASSED
tests/phase2/test_spatial_alignment.py::TestSpatialAlignment::test_cross_dataset_coordinate_identity PASSED

======================== 5 passed in 1.86s ========================
```

---

## 7. Final Assessment & Next Steps

### FINAL STATUS: **PASS**

### Summary of Accomplishments:
1. **Canonical Subgrid Formalized**: Defined standard $127 \times 121$ grid on $[6.50^\circ\text{N} \dots 38.00^\circ\text{N}, 68.00^\circ\text{E} \dots 98.00^\circ\text{E}]$ in `src/ingestion/spatial.py`.
2. **Pure Indexing Verified**: Direct slicing without interpolation confirmed to yield $0.0000^\circ$ coordinate difference across IMD, ERA5, and GFS.
3. **Automated Guard**: 5 new automated test cases in `tests/phase2/test_spatial_alignment.py` passing with 100% success rate.

### Next Exact Step (Phase 2C):
Proceed to **Phase 2C: Multi-Source Preprocessing & Aligned Feature Tensor Pipeline**:
- Combine Phase 2A (temporal differencing $A_{\text{f027}} - A_{\text{f003}}$) and Phase 2B (spatial subgrid index slicing) into a unified dataset builder generating aligned $(N, 127, 121)$ feature and target tensors with standardized units.
