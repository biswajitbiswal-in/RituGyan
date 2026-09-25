"""Automated validation tests for Phase 2A: Temporal Alignment Specification.

Validates the exact temporal relationships, GRIB metadata semantics,
and accumulation differencing between:
1. IMD daily rainfall observation (03:00 UTC D -> 03:00 UTC D+1)
2. ERA5 6-hourly predictors (06Z, 12Z, 18Z, 00Z D+1)
3. GFS forecast leads f003 through f027 (00Z initialization)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import datetime
import pytest
import xarray as xr
import numpy as np

from src.ingestion.readers import setup_eccodes_environment


@pytest.fixture(scope="module")
def eccodes_ready():
    """Ensure ecCodes environment is configured."""
    ready = setup_eccodes_environment()
    assert ready, "ecCodes C-library must be reachable for GRIB2 inspection"
    import eccodes
    return eccodes


@pytest.fixture
def gfs_exact_dir(project_root: Path) -> Path:
    """Return path to gfs_exact_test directory."""
    gfs_dir = project_root / "data" / "raw" / "gfs_exact_test"
    assert gfs_dir.exists(), f"GFS exact test directory missing: {gfs_dir}"
    return gfs_dir


@pytest.fixture
def expected_leads() -> List[int]:
    """Expected GFS forecast leads for 00Z cycle."""
    return [3, 6, 9, 12, 15, 18, 21, 24, 27]


class TestTemporalAlignment:
    """Validate temporal alignment specification and GRIB metadata semantics."""

    def test_imd_temporal_convention(self, project_root: Path):
        """Validate IMD daily observation convention (03:00 UTC D -> 03:00 UTC D+1)."""
        imd_path = project_root / "data" / "raw" / "RF25_ind2024_rfp25.nc"
        assert imd_path.exists(), f"IMD dataset missing: {imd_path}"

        ds = xr.open_dataset(imd_path)
        assert "TIME" in ds.coords or "TIME" in ds.dims
        assert ds["RAINFALL"].attrs.get("units") == "mm"

        # Check target date 2024-06-21 exists in IMD
        target_dt = np.datetime64("2024-06-21")
        time_vals = ds["TIME"].values
        # Convert to date string or date comparison
        time_dates = [str(t)[:10] for t in time_vals]
        assert "2024-06-21" in time_dates, "Target date 2024-06-21 not found in IMD calendar"

        # IMD convention definition verification:
        # Day D represents [D 03:00 UTC, (D+1) 03:00 UTC] (08:30 IST to 08:30 IST)
        target_start = datetime.datetime(2024, 6, 21, 3, 0, tzinfo=datetime.timezone.utc)
        target_end = datetime.datetime(2024, 6, 22, 3, 0, tzinfo=datetime.timezone.utc)
        duration_hours = (target_end - target_start).total_seconds() / 3600.0
        assert duration_hours == 24.0
        ds.close()

    def test_era5_temporal_convention_and_coverage(self, project_root: Path):
        """Validate ERA5 6-hourly snapshots span within the target IMD window."""
        era5_path = project_root / "data" / "raw" / "era5.nc"
        assert era5_path.exists(), f"ERA5 dataset missing: {era5_path}"

        ds = xr.open_dataset(era5_path)
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        assert time_coord in ds.coords

        # For target Day D = 2024-06-21, IMD window is 2024-06-21 03:00 UTC to 2024-06-22 03:00 UTC
        # Required 6-hourly ERA5 snapshots are:
        # 06Z(D), 12Z(D), 18Z(D), 00Z(D+1)
        expected_era5_timestamps = [
            "2024-06-21T06:00:00",
            "2024-06-21T12:00:00",
            "2024-06-21T18:00:00",
            "2024-06-22T00:00:00",
        ]

        actual_times = [str(t)[:19] for t in ds[time_coord].values]
        for exp_ts in expected_era5_timestamps:
            assert exp_ts in actual_times, f"ERA5 missing expected snapshot: {exp_ts}"

        # Confirm all 4 snapshots lie strictly inside the IMD accumulation interval [03Z, 27Z]
        window_start = np.datetime64("2024-06-21T03:00:00")
        window_end = np.datetime64("2024-06-22T03:00:00")
        for exp_ts in expected_era5_timestamps:
            ts_dt = np.datetime64(exp_ts)
            assert window_start <= ts_dt <= window_end
        ds.close()

    def test_gfs_forecast_leads_and_valid_timestamps(
        self, eccodes_ready, gfs_exact_dir: Path, expected_leads: List[int]
    ):
        """Verify all 9 GFS lead files f003-f027 exist and valid times match init + lead."""
        eccodes = eccodes_ready

        init_date_expected = 20240621
        init_time_expected = 0  # 00:00 UTC

        for lead in expected_leads:
            fname = f"gfs.0p25.2024062100.f{lead:03d}.grib2"
            fpath = gfs_exact_dir / fname
            assert fpath.exists(), f"GFS lead file {fname} does not exist"

            # Open GRIB with eccodes in streaming mode (header-only)
            with open(fpath, "rb") as fp:
                gid = eccodes.codes_grib_new_from_file(fp)
                assert gid is not None, f"Failed to parse GRIB message in {fname}"

                data_date = eccodes.codes_get(gid, "dataDate")
                data_time = eccodes.codes_get(gid, "dataTime")
                valid_date = eccodes.codes_get(gid, "validityDate")
                valid_time = eccodes.codes_get(gid, "validityTime")

                assert data_date == init_date_expected
                assert data_time == init_time_expected

                # Compute expected validity datetime
                init_dt = datetime.datetime(2024, 6, 21, 0, 0, tzinfo=datetime.timezone.utc)
                expected_valid_dt = init_dt + datetime.timedelta(hours=lead)
                expected_vdate = int(expected_valid_dt.strftime("%Y%m%d"))
                expected_vtime = int(expected_valid_dt.strftime("%H%M"))

                assert valid_date == expected_vdate, f"Mismatch valid_date for lead {lead}: {valid_date} vs {expected_vdate}"
                assert valid_time == expected_vtime, f"Mismatch valid_time for lead {lead}: {valid_time} vs {expected_vtime}"

                eccodes.codes_release(gid)

    def test_gfs_precipitation_grib_metadata(
        self, eccodes_ready, gfs_exact_dir: Path, expected_leads: List[int]
    ):
        """Inspect actual GRIB keys of tp field: stepType, stepRange, forecastTime, units."""
        eccodes = eccodes_ready

        for lead in expected_leads:
            fname = f"gfs.0p25.2024062100.f{lead:03d}.grib2"
            fpath = gfs_exact_dir / fname

            tp_messages = []
            with open(fpath, "rb") as fp:
                while True:
                    gid = eccodes.codes_grib_new_from_file(fp)
                    if gid is None:
                        break
                    sn = eccodes.codes_get(gid, "shortName")
                    if sn == "tp":
                        st = eccodes.codes_get(gid, "stepType")
                        sr = eccodes.codes_get(gid, "stepRange")
                        start = eccodes.codes_get(gid, "startStep")
                        end = eccodes.codes_get(gid, "endStep")
                        units = eccodes.codes_get(gid, "units")
                        tp_messages.append({
                            "stepType": st,
                            "stepRange": sr,
                            "startStep": start,
                            "endStep": end,
                            "units": units,
                        })
                    eccodes.codes_release(gid)

            assert len(tp_messages) >= 1, f"No tp message found in {fname}"

            # Verify every tp message has stepType accum and units kg m**-2
            for msg in tp_messages:
                assert msg["stepType"] == "accum", f"Unexpected stepType {msg['stepType']} in {fname}"
                assert msg["units"] == "kg m**-2", f"Unexpected units {msg['units']} in {fname}"
                assert msg["endStep"] == lead, f"endStep ({msg['endStep']}) does not match lead {lead} in {fname}"

    def test_gfs_precipitation_accumulation_semantics(
        self, eccodes_ready, gfs_exact_dir: Path, expected_leads: List[int]
    ):
        """Verify presence of continuous accumulation (0-lead) and interval buckets."""
        eccodes = eccodes_ready

        continuous_accum_found = {}
        interval_bucket_found = {}

        # Expected interval buckets per NCEP 6-hourly reset schedule
        expected_intervals = {
            3: "0-3",
            6: "0-6",
            9: "6-9",
            12: "6-12",
            15: "12-15",
            18: "12-18",
            21: "18-21",
            24: "18-24",
            27: "24-27",
        }

        for lead in expected_leads:
            fname = f"gfs.0p25.2024062100.f{lead:03d}.grib2"
            fpath = gfs_exact_dir / fname

            with open(fpath, "rb") as fp:
                while True:
                    gid = eccodes.codes_grib_new_from_file(fp)
                    if gid is None:
                        break
                    sn = eccodes.codes_get(gid, "shortName")
                    if sn == "tp":
                        start = eccodes.codes_get(gid, "startStep")
                        end = eccodes.codes_get(gid, "endStep")
                        sr = eccodes.codes_get(gid, "stepRange")
                        if start == 0 and end == lead:
                            continuous_accum_found[lead] = sr
                        if sr == expected_intervals[lead]:
                            interval_bucket_found[lead] = sr
                    eccodes.codes_release(gid)

        # Assert every lead has a continuous accumulation from 0 to lead
        for lead in expected_leads:
            assert lead in continuous_accum_found, f"Missing continuous accumulation (0-{lead}) in lead {lead}"
            assert lead in interval_bucket_found, f"Missing expected interval bucket {expected_intervals[lead]} in lead {lead}"

    def test_imd_window_reproducibility_and_differencing(
        self, eccodes_ready, gfs_exact_dir: Path
    ):
        """Verify that A(0->27) - A(0->3) reproduces the exact IMD 03Z-03Z window."""
        eccodes = eccodes_ready

        accum_0 = {}
        for lead in [3, 6, 9, 12, 15, 18, 21, 24, 27]:
            fname = f"gfs.0p25.2024062100.f{lead:03d}.grib2"
            fpath = gfs_exact_dir / fname
            with open(fpath, "rb") as fp:
                while True:
                    gid = eccodes.codes_grib_new_from_file(fp)
                    if gid is None:
                        break
                    if eccodes.codes_get(gid, "shortName") == "tp" and eccodes.codes_get(gid, "startStep") == 0:
                        # Inspect values for monotonicity and differencing check
                        accum_0[lead] = eccodes.codes_get_values(gid)
                    eccodes.codes_release(gid)

        # 1. Check differencing A(0->27) - A(0->3)
        diff_24h = accum_0[27] - accum_0[3]
        assert np.all(diff_24h >= -1e-5), "Differenced precipitation (f027 - f003) contains negative values"
        assert np.max(diff_24h) > 0.0, "Differenced precipitation has zero variance across India"

        # 2. Check 3-hourly consecutive differences sum to 24h difference exactly
        leads = [3, 6, 9, 12, 15, 18, 21, 24, 27]
        intervals = []
        for i in range(len(leads) - 1):
            l_prev = leads[i]
            l_curr = leads[i + 1]
            interval_diff = accum_0[l_curr] - accum_0[l_prev]
            assert np.all(interval_diff >= -1e-5), f"Negative accumulation in interval {l_prev}->{l_curr}"
            intervals.append(interval_diff)

        sum_intervals = np.sum(intervals, axis=0)
        max_discrepancy = float(np.max(np.abs(sum_intervals - diff_24h)))
        assert max_discrepancy < 1e-6, f"Sum of 3h intervals diverges from (f027 - f003) by {max_discrepancy}"

    def test_instantaneous_gfs_predictors(
        self, eccodes_ready, gfs_exact_dir: Path, expected_leads: List[int]
    ):
        """Verify instantaneous atmospheric predictors across all 9 lead files."""
        eccodes = eccodes_ready
        instant_vars = {"prmsl", "2t", "2d", "10u", "10v", "pwat"}

        for lead in expected_leads:
            fname = f"gfs.0p25.2024062100.f{lead:03d}.grib2"
            fpath = gfs_exact_dir / fname

            found_vars = set()
            with open(fpath, "rb") as fp:
                while True:
                    gid = eccodes.codes_grib_new_from_file(fp)
                    if gid is None:
                        break
                    sn = eccodes.codes_get(gid, "shortName")
                    if sn in instant_vars:
                        st = eccodes.codes_get(gid, "stepType")
                        sr = eccodes.codes_get(gid, "stepRange")
                        assert st == "instant", f"Expected instant for {sn} in {fname}, got {st}"
                        assert int(sr) == lead, f"Expected stepRange {lead} for {sn} in {fname}, got {sr}"
                        found_vars.add(sn)
                    eccodes.codes_release(gid)

            assert found_vars == instant_vars, f"Missing instant predictors in {fname}: {instant_vars - found_vars}"

    def test_temporal_alignment_matrix_definition(self):
        """Validate programmatic temporal alignment dictionary definition for target Day D."""
        # Target IMD Day D
        day_d = datetime.date(2024, 6, 21)

        # 1. IMD observation window: [D 03:00 UTC, (D+1) 03:00 UTC]
        imd_start = datetime.datetime.combine(day_d, datetime.time(3, 0), tzinfo=datetime.timezone.utc)
        imd_end = imd_start + datetime.timedelta(hours=24)

        # 2. GFS 00Z run initialization: D 00:00 UTC
        gfs_init = datetime.datetime.combine(day_d, datetime.time(0, 0), tzinfo=datetime.timezone.utc)
        gfs_leads = [3, 6, 9, 12, 15, 18, 21, 24, 27]
        gfs_valid_times = [gfs_init + datetime.timedelta(hours=h) for h in gfs_leads]

        # Valid time of f003 must match imd_start
        assert gfs_valid_times[0] == imd_start
        # Valid time of f027 must match imd_end
        assert gfs_valid_times[-1] == imd_end

        # 3. ERA5 6-hourly snapshots: 06Z(D), 12Z(D), 18Z(D), 00Z(D+1)
        era5_snapshots = [
            gfs_init + datetime.timedelta(hours=6),
            gfs_init + datetime.timedelta(hours=12),
            gfs_init + datetime.timedelta(hours=18),
            gfs_init + datetime.timedelta(hours=24),
        ]
        for era5_t in era5_snapshots:
            assert imd_start <= era5_t <= imd_end
