"""
Unit tests for Helper modules (ISO_Parser and Calc_Months).
Tests date parsing and month calculation functionality.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after adding to path
from Helpers.ISO_Parser import _parse_iso8601  # noqa: E402
from Helpers.Calc_Months import _months_between  # noqa: E402


class TestISOParser:
    """Test cases for ISO 8601 date parsing."""

    def test_parse_iso8601_basic(self):
        """Test parsing basic ISO 8601 date."""
        date_str = "2023-01-15T10:30:00Z"
        result = _parse_iso8601(date_str)
        
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso8601_with_timezone(self):
        """Test parsing ISO 8601 date with timezone offset."""
        date_str = "2023-06-20T14:45:30+05:00"
        result = _parse_iso8601(date_str)
        
        assert result is not None
        assert result.year == 2023
        assert result.month == 6
        assert result.day == 20

    def test_parse_iso8601_date_only(self):
        """Test parsing ISO 8601 date without time."""
        date_str = "2023-12-25"
        result = _parse_iso8601(date_str)
        
        assert result is not None
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25

    def test_parse_iso8601_microseconds(self):
        """Test parsing ISO 8601 date with microseconds."""
        date_str = "2023-03-10T08:15:30.123456Z"
        result = _parse_iso8601(date_str)
        
        assert result is not None
        assert result.year == 2023
        assert result.month == 3
        assert result.day == 10
        assert result.microsecond == 123456

    def test_parse_iso8601_invalid_format(self):
        """Test parsing invalid ISO 8601 format."""
        invalid_dates = [
            "invalid-date",
            "2023/01/15",  # Wrong separator
            "2023-13-01",  # Invalid month
            "2023-01-32",  # Invalid day
            "",
            None
        ]
        
        for invalid_date in invalid_dates:
            result = _parse_iso8601(invalid_date)
            assert result is None

    def test_parse_iso8601_edge_cases(self):
        """Test parsing edge cases."""
        # Leap year
        result = _parse_iso8601("2024-02-29T12:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 29
        
        # End of year
        result = _parse_iso8601("2023-12-31T23:59:59Z")
        assert result is not None
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 31


class TestCalcMonths:
    """Test cases for month calculation between dates."""

    def test_months_between_same_date(self):
        """Test months between same date."""
        date = datetime(2023, 6, 15, tzinfo=timezone.utc)
        result = _months_between(date, date)
        assert result == 0

    def test_months_between_one_month(self):
        """Test months between dates one month apart."""
        date1 = datetime(2023, 1, 15, tzinfo=timezone.utc)
        date2 = datetime(2023, 2, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 1

    def test_months_between_one_year(self):
        """Test months between dates one year apart."""
        date1 = datetime(2023, 6, 15, tzinfo=timezone.utc)
        date2 = datetime(2024, 6, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 12

    def test_months_between_reverse_order(self):
        """Test months between dates in reverse order."""
        date1 = datetime(2023, 6, 15, tzinfo=timezone.utc)
        date2 = datetime(2023, 3, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 3  # Absolute difference

    def test_months_between_partial_months(self):
        """Test months between dates with partial months."""
        # Different days in month
        date1 = datetime(2023, 1, 31, tzinfo=timezone.utc)
        date2 = datetime(2023, 2, 28, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 1  # Should round to nearest month

    def test_months_between_cross_year(self):
        """Test months between dates crossing year boundary."""
        date1 = datetime(2022, 10, 15, tzinfo=timezone.utc)
        date2 = datetime(2023, 3, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 5

    def test_months_between_multiple_years(self):
        """Test months between dates multiple years apart."""
        date1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        date2 = datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 36

    def test_months_between_different_timezones(self):
        """Test months between dates with different timezones."""
        # Note: This test assumes the function handles timezone conversion
        date1 = datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        # Create a timezone-aware datetime with different timezone
        from datetime import timedelta
        tz_offset = timezone(timedelta(hours=5))
        date2 = datetime(2023, 2, 15, 17, 0, 0, tzinfo=tz_offset)
        
        result = _months_between(date1, date2)
        assert result == 1

    def test_months_between_none_values(self):
        """Test months between with None values."""
        date = datetime(2023, 6, 15, tzinfo=timezone.utc)
        
        # Test with None as first argument
        try:
            result = _months_between(None, date)
            # Should either return None or raise an exception
            assert result is None or isinstance(result, (int, float))
        except (TypeError, AttributeError):
            # Exception is acceptable for None input
            pass
        
        # Test with None as second argument
        try:
            result = _months_between(date, None)
            assert result is None or isinstance(result, (int, float))
        except (TypeError, AttributeError):
            pass

    def test_months_between_precision(self):
        """Test months between calculation precision."""
        # Test with specific dates to verify calculation accuracy
        date1 = datetime(2023, 1, 1, tzinfo=timezone.utc)
        date2 = datetime(2023, 7, 1, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 6
        
        # Test with mid-month dates
        date1 = datetime(2023, 1, 15, tzinfo=timezone.utc)
        date2 = datetime(2023, 7, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 6

    def test_months_between_leap_year(self):
        """Test months between calculation with leap year."""
        # February in leap year
        date1 = datetime(2024, 1, 29, tzinfo=timezone.utc)
        date2 = datetime(2024, 2, 29, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 1
        
        # Cross leap year boundary
        date1 = datetime(2023, 12, 15, tzinfo=timezone.utc)
        date2 = datetime(2024, 3, 15, tzinfo=timezone.utc)
        result = _months_between(date1, date2)
        assert result == 3