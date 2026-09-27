"""Tests for Phase 2E Multi-Year Dataset Availability & Batch Pipeline.

Validates:
- Data availability audit across IMD, ERA5, and GFS
- Accurate reporting of available and missing years (2020-2024)
- Expected JJAS day count calculation (610 days for 5 years)
- Identification of overlapping aligned days without data fabrication
- Required download dependency listing
"""

from pathlib import Path
import pytest

from src.features.batch_pipeline import DataAvailabilityAudit, MultiYearDatasetBuilder


class TestMultiYearDataset:
    """Test suite for historical dataset availability auditing and batch builder."""

    def test_data_availability_audit_structure(self):
        """Verify that the availability auditor generates structured metrics for JJAS 2020-2024."""
        builder = MultiYearDatasetBuilder(raw_data_dir="data/raw")
        audit = builder.audit_availability(target_years=[2020, 2021, 2022, 2023, 2024])

        assert isinstance(audit, DataAvailabilityAudit)
        assert audit.expected_jjas_days_total == 610  # 122 days * 5 years
        assert len(audit.expected_days_per_year) == 5
        for yr in [2020, 2021, 2022, 2023, 2024]:
            assert audit.expected_days_per_year[yr] == 122

        # Verify IMD audit results
        for yr in [2021, 2022, 2023, 2024]:
            assert yr in audit.imd_available_years
            assert audit.imd_available_years[yr] == 122  # Complete JJAS in RF25_ind<yr>_rfp25.nc
        assert audit.imd_available_years.get(2020, 0) == 0

        # Verify ERA5 audit results
        for yr in [2021, 2022, 2023, 2024]:
            assert yr in audit.era5_available_years
            assert audit.era5_available_years[yr] == 122  # Complete JJAS in era5 files
        assert audit.era5_available_years.get(2020, 0) == 0

        # Verify GFS audit results
        assert "2024-06-21" in audit.gfs_available_dates
        assert "2024-06-21" in audit.common_overlapping_dates

        # Verify multi-year completeness status and missing download listings
        assert audit.is_multiyear_complete is False
        assert len(audit.required_additional_downloads) > 0
        assert any("2020" in req for req in audit.required_additional_downloads)
        assert any("2021" in req for req in audit.required_additional_downloads)
        assert any("2022" in req for req in audit.required_additional_downloads)
        assert any("2023" in req for req in audit.required_additional_downloads)

    def test_missing_data_statistics(self):
        """Verify exact statistics for available vs missing JJAS days."""
        builder = MultiYearDatasetBuilder(raw_data_dir="data/raw")
        audit = builder.audit_availability(target_years=[2020, 2021, 2022, 2023, 2024])

        assert audit.usable_aligned_days_count >= 1
        assert audit.missing_dates_by_year[2020] == 122
        assert audit.missing_dates_by_year[2021] == 122
        assert audit.missing_dates_by_year[2022] == 122
        assert audit.missing_dates_by_year[2023] == 122
        # For 2024, exactly 1 aligned day (2024-06-21) exists in current raw test fixtures
        assert audit.missing_dates_by_year[2024] == 121
