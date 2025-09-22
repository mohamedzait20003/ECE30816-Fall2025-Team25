from datetime import datetime


def _months_between(a: datetime, b: datetime) -> float:
    """
    Calculate the approximate number of months between two datetime objects.
    Args:
        a: The first datetime (typically the later date)
        b: The second datetime (typically the earlier date)
    Returns:
        The approximate number of months between the two dates.
        Returns 0.0 if the result would be negative.
    """
    # Approx. months using average days per month (30.44)
    delta_days = (a - b).total_seconds() / 86400.0
    return max(0.0, delta_days / 30.44)
