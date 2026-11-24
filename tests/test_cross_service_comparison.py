"""Unit tests for small, synthetic service-aggregated counts (no TLC download needed)."""
import numpy as np
import pandas as pd
import pytest

from src.supplementary.cross_service_comparison import summarize


def fixture():
    lookup = pd.DataFrame({
        "LocationID": [1, 2, 3],
        "Borough": ["EWR", "Manhattan", "Queens"],
        "Zone": ["Newark Airport", "Midtown", "Astoria"],
    })
    sparse = pd.DataFrame([
        ("Yellow", 2, 0, 9, 4),
        ("Yellow", 2, 5, 22, 3),
        ("Uber",   2, 0, 9, 10),
        ("Uber",   3, 6, 23, 6),
        ("Lyft",   3, 5, 0, 2),
    ], columns=["service", "zone_id", "day_of_week", "hour", "pickup_count"])
    qa = pd.DataFrame({
        "service": ["Yellow", "Uber", "Lyft"],
        "n_month_rows": [9, 17, 3],
        "n_eligible_pickups": [7, 16, 2],
    })
    return sparse, qa, lookup


def test_reconciliation_and_168_profiles():
    sparse, qa, lookup = fixture()
    result = summarize(sparse, qa, lookup, "2025-06")
    assert len(result["temporal_profile"]) == 3 * 7 * 24
    assert len(result["zone_counts"]) == 3 * 3
    assert result["summary"].eligible_pickups_june.tolist() == [7, 16, 2]
    assert result["quality_checks"].n_excluded_unknown_or_special_zone.tolist() == [2, 1, 1]
    assert np.allclose(result["temporal_profile"].groupby("service").normalized_share.sum(), 1)
    assert result["zone_counts"].groupby("service").zone_share.sum().round(12).eq(1).all()


def test_weekday_exposure_normalization():
    sparse, qa, lookup = fixture()
    result = summarize(sparse, qa, lookup, "2025-06")
    t = result["temporal_profile"]
    # June 2025: Monday occurs 5 times, Saturday occurs 4 times.
    ymon = t.query("service == 'Yellow' and day_of_week == 0 and hour == 9").iloc[0]
    ysat = t.query("service == 'Yellow' and day_of_week == 5 and hour == 22").iloc[0]
    assert ymon.n_calendar_days == 5
    assert ysat.n_calendar_days == 4
    assert ymon.mean_pickups_per_day_hour == pytest.approx(4/5)
    assert ysat.mean_pickups_per_day_hour == pytest.approx(3/4)


def test_reject_inconsistent_qa():
    sparse, qa, lookup = fixture()
    qa.loc[0, "n_eligible_pickups"] = 8
    with pytest.raises(AssertionError, match="reconciliation"):
        summarize(sparse, qa, lookup, "2025-06")
