"""Run: python -m pytest tests/test_network_features.py -q"""
import pandas as pd
import pytest
from src.network.network_features import make_history_panel, load_zone_ids


def test_24h_shift_excludes_current_and_older_than_24h():
    hourly = pd.DataFrame({
        "zone_id": [1, 1, 1],
        "completion_hour": pd.to_datetime([
            "2025-01-01 00:00", "2025-01-01 23:00", "2025-01-02 00:00"
        ]),
        "in_completed": [1, 2, 50],
        "out_completed": [3, 4, 70],
        "interzone_in_completed": [1, 0, 20],
        "interzone_out_completed": [3, 0, 30],
    })
    output = make_history_panel(
        hourly, zone_ids=[1, 2],
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-02 03:00"), window_hours=24, carryover_hours=0
    )
    assert len(output) == 2 * 27
    def get(zone, hour, col):
        return output.set_index(["zone_id", "feature_as_of"]).loc[
            (zone, pd.Timestamp(hour)), col
        ]
    assert pd.isna(get(1,"2025-01-01 23:00", "in_strength_24h"))
    assert get(1,"2025-01-02 00:00", "in_strength_24h") == 3
    assert get(1,"2025-01-02 00:00", "out_strength_24h") == 7
    assert get(1,"2025-01-02 00:00", "interzone_in_strength_24h") == 1
    assert get(1,"2025-01-02 01:00", "in_strength_24h") == 52
    assert get(1,"2025-01-02 01:00", "out_strength_24h") == 74
    assert get(2,"2025-01-02 00:00", "in_strength_24h") == 0
    assert get(1,"2025-01-02 00:00", "has_full_history_24h")
    assert not get(1,"2025-01-01 23:00", "has_full_history_24h")


def test_boundary_carryover_is_not_falsely_complete():
    empty = pd.DataFrame(columns=["zone_id", "completion_hour", "in_completed",
                                  "out_completed", "interzone_in_completed",
                                  "interzone_out_completed"])
    p = make_history_panel(empty, [1], start=pd.Timestamp("2025-01-01"),
                           end=pd.Timestamp("2025-01-03"), window_hours=24)
    p = p.set_index("feature_as_of")
    assert not bool(p.loc["2025-01-02 05:00", "has_full_history_24h"])
    assert pd.isna(p.loc["2025-01-02 05:00", "in_strength_24h"])
    assert bool(p.loc["2025-01-02 06:00", "has_full_history_24h"])
    assert p.loc["2025-01-02 06:00", "in_strength_24h"] == 0


def test_input_validation():
    hourly = pd.DataFrame({
        "zone_id": [1,1], "completion_hour": pd.to_datetime(["2025-01-01"]*2),
        "in_completed": [1,1], "out_completed": [1,1],
        "interzone_in_completed": [1,1], "interzone_out_completed": [1,1]
    })
    with pytest.raises(ValueError, match="duplicate"):
        make_history_panel(hourly, [1], end=pd.Timestamp("2025-01-02"))


def test_dst_flags_model_overlap():
    cols = {"zone_id": [], "completion_hour": [],
            "in_completed": [], "out_completed": [],
            "interzone_in_completed": [], "interzone_out_completed": []}
    empty = pd.DataFrame(cols)
    result = make_history_panel(empty, [1],
                                start=pd.Timestamp("2025-03-08"),
                                end=pd.Timestamp("2025-03-11"), window_hours=24)
    by_time = result.set_index("feature_as_of")
    assert not bool(by_time.loc["2025-03-09 02:00", "dst_safe_24h"])
    assert not bool(by_time.loc["2025-03-10 02:00", "dst_safe_24h"])
    assert bool(by_time.loc["2025-03-10 03:00", "dst_safe_24h"])


def test_sql_dropoff_not_available_until_next_hour(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from src.network.network_features import extract_completed_hourly_od
    trips = pd.DataFrame({
        "PULocationID": [1,1,1,1],
        "DOLocationID": [2,2,1,2],
        "tpep_pickup_datetime": pd.to_datetime([
            "2025-01-01 08:45", "2025-01-01 08:10",
            "2025-01-01 09:00", "2025-01-01 08:00"
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2025-01-01 09:20", "2025-01-01 08:30",
            "2025-01-01 09:15", "2025-01-01 09:00"
        ]),
        "valid_od": [True, True, True, False],
    })
    source = tmp_path / "tiny.parquet"
    trips.to_parquet(source, index=False)
    hourly = extract_completed_hourly_od([source], [1,2])
    result = make_history_panel(hourly, [1,2],
                                start=pd.Timestamp("2025-01-01"),
                                end=pd.Timestamp("2025-01-02 02:00"), window_hours=1, carryover_hours=0)
    def v(zone, time, col):
        return result.set_index(["zone_id", "feature_as_of"]).loc[
            (zone, pd.Timestamp(time)), col]
    assert v(2,"2025-01-01 09:00", "in_strength_1h") == 1
    assert v(2,"2025-01-01 10:00", "in_strength_1h") == 1
    assert v(1,"2025-01-01 10:00", "in_strength_1h") == 1
    assert v(1,"2025-01-01 10:00", "interzone_in_strength_1h") == 0
