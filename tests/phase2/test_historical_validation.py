"""Tests for Phase 2F Historical IMD and ERA5 Data Validation (2021-2023 vs 2024 Baseline).

Validates:
- IMD:
  - June 1 to September 30 (122 daily observations per year)
  - 0.25° grid spacing (lat: 6.5 to 38.5, lon: 66.5 to 100.0)
  - Coordinates ordering (ascending lat, ascending lon)
  - Rainfall variable ('RAINFALL'), units ('mm')
  - Missing-value / land-sea mask pattern match with 2024 baseline
  - Spatial compatibility across 2021, 2022, 2023, 2024
- ERA5:
  - June 1 to September 30 (488 timesteps per year: 122 days * 4 cycles)
  - Cycles: 00, 06, 12, 18 UTC
  - 0.25° grid spacing (lat: 38.0 to 6.0 descending, lon: 68.0 to 98.0 ascending)
  - Variables: u10, v10, t2m, d2m, msl, tcwv
  - Units and physical realism
  - Zero NaNs and zero Infs in ERA5 domain
  - Spatial compatibility across 2021, 2022, 2023, 2024
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xarray as xr

from scripts.validate_historical_data import validate_imd_datasets, validate_era5_datasets


class TestHistoricalValidation:
    """Test suite validating 2021-2023 historical datasets against 2024 baseline."""

    @pytest.fixture(scope="class")
    def imd_validation_results(self):
        raw_dir = Path("data/raw")
        return validate_imd_datasets(raw_dir)

    @pytest.fixture(scope="class")
    def era5_validation_results(self):
        raw_dir = Path("data/raw")
        return validate_era5_datasets(raw_dir)

    @pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
    def test_imd_yearly_completeness_and_grid(self, imd_validation_results, year):
        res = imd_validation_results[year]
        assert res["status"] == "PASS", f"IMD validation failed for year {year}: {res}"
        assert bool(res["exists"]) is True
        assert res["time_count"] == 122
        assert bool(res["continuous_daily"]) is True
        assert res["time_start"] == f"{year}-06-01"
        assert res["time_end"] == f"{year}-09-30"
        assert res["lat_len"] == 129
        assert res["lon_len"] == 135
        assert np.isclose(res["lat_step"], 0.25)
        assert np.isclose(res["lon_step"], 0.25)
        assert bool(res["lat_ascending"]) is True
        assert bool(res["lon_ascending"]) is True
        assert bool(res["lat_exact_match_2024"]) is True
        assert bool(res["lon_exact_match_2024"]) is True

    @pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
    def test_imd_rainfall_variable_and_mask(self, imd_validation_results, year):
        res = imd_validation_results[year]
        assert res["rf_var"] == "RAINFALL"
        assert res["rf_units"] == "mm"
        assert res["rf_min"] >= 0.0
        assert res["rf_max"] > 100.0  # Realistic monsoon maximum
        assert bool(res["mask_matches_baseline"]) is True
        assert bool(res["mask_temporally_stable"]) is True
        assert res["valid_count"] == 605608  # Exact match for 122 days * 4964 land points

    @pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
    def test_era5_yearly_completeness_and_cycles(self, era5_validation_results, year):
        res = era5_validation_results[year]
        assert res["status"] == "PASS", f"ERA5 validation failed for year {year}: {res}"
        assert res["time_count"] == 488
        assert bool(res["exact_6h_times_match"]) is True
        assert res["hours_present"] == [0, 6, 12, 18]
        assert res["time_start"] == f"{year}-06-01 00:00:00"
        assert res["time_end"] == f"{year}-09-30 18:00:00"
        assert res["lat_len"] == 129
        assert res["lon_len"] == 121
        assert res["lat_ordering"] == "descending"
        assert res["lon_ordering"] == "ascending"
        assert np.isclose(res["lat_step"], 0.25)
        assert np.isclose(res["lon_step"], 0.25)
        assert bool(res["lat_exact_match_2024"]) is True
        assert bool(res["lon_exact_match_2024"]) is True

    @pytest.mark.parametrize("year", [2021, 2022, 2023, 2024])
    def test_era5_variables_and_cleanliness(self, era5_validation_results, year):
        res = era5_validation_results[year]
        assert bool(res["all_vars_present_and_clean"]) is True
        vars_dict = res["variables"]
        expected_vars = ["u10", "v10", "t2m", "d2m", "msl", "tcwv"]
        for var in expected_vars:
            assert var in vars_dict
            v_info = vars_dict[var]
            assert bool(v_info["present"]) is True
            assert v_info["nan_count"] == 0, f"NaNs found in ERA5 {year} {var}"
            assert v_info["inf_count"] == 0, f"Infs found in ERA5 {year} {var}"
            assert bool(v_info["is_clean"]) is True
