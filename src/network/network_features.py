"""Historical OD network-strength features for time-safe SI 671 forecasting.

Run from repository root:
    python -m src.network.network_features

For target pickups during [t, t+1h), features at `feature_as_of=t` use
only valid OD trips with dropoff timestamps in [t-window_hours, t).
All timestamps are source naive local wall-clock times (NYC); DST windows
are flagged, not reconstructed. Offline assumption: completed trip records
are available immediately at trip completion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data/processed/cleaned_trips"
ZONE_FILE = ROOT / "data/external/taxi_zone_lookup.csv"
OUTPUT_DIR = ROOT / "data/processed/network_features"
START = pd.Timestamp("2025-01-01 00:00:00")
END = pd.Timestamp("2026-01-01 00:00:00")
VALUE_COLUMNS = (
    "in_completed", "out_completed", "interzone_in_completed", "interzone_out_completed"
)
DST_BAD_HOURS = (
    pd.Timestamp("2025-03-09 02:00:00"),  # nonexistent spring-forward hour
    pd.Timestamp("2025-11-02 01:00:00"),  # ambiguous fall-back hour
)


def load_zone_ids(zone_file: Path = ZONE_FILE) -> list[int]:
    """Use the same 263-zone universe as RQ1 and the forecasting panel."""
    zones = pd.read_csv(zone_file)
    if "LocationID" not in zones:
        raise ValueError("Zone lookup must contain LocationID")
    zone_ids = sorted(set(zones.LocationID.dropna().astype(int)) - {264, 265})
    if not zone_ids:
        raise ValueError("No usable taxi zones")
    return zone_ids


def extract_completed_hourly_od(
    parquet_files: list[Path], zone_ids: list[int], threads: int = 4
) -> pd.DataFrame:
    """Aggregate complete trips by dropoff hour and origin/destination zone.

    Two SQL scans avoid loading trip-level rows into Pandas. Columns hold
    integer counts for each zone-hour of *completed* OD trips, including
    loops for total strength and excluding them for inter-zone strength.
    """
    import duckdb  # local dependency; pandas-only tests can run without it

    parquet_files = [Path(p) for p in parquet_files]
    if not parquet_files or any(not p.is_file() for p in parquet_files):
        raise FileNotFoundError("Missing cleaned monthly Parquet files")
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_ids must be unique")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in parquet_files)
        con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet([{paths}])")
        schema = set(con.execute("DESCRIBE trips").df()["column_name"])
        needed = {
            "PULocationID", "DOLocationID", "tpep_pickup_datetime",
            "tpep_dropoff_datetime", "valid_od"
        }
        if not needed.issubset(schema):
            raise ValueError(f"Cleaned Parquet missing columns: {sorted(needed - schema)}")
        con.register("valid_zones", pd.DataFrame({"LocationID": zone_ids}))
        # RQ1's OD-valid and geographic filters; dropoff hour is used to
        # prevent knowledge of destination before the trip is completed.
        base = """
            FROM trips t
            JOIN valid_zones p ON t.PULocationID = p.LocationID
            JOIN valid_zones d ON t.DOLocationID = d.LocationID
            WHERE t.valid_od IS TRUE
              AND t.tpep_pickup_datetime >= TIMESTAMP '2025-01-01'
              AND t.tpep_pickup_datetime < TIMESTAMP '2026-01-01'
              AND t.tpep_dropoff_datetime >= TIMESTAMP '2025-01-01'
              AND t.tpep_dropoff_datetime < TIMESTAMP '2026-01-01'
              AND t.tpep_dropoff_datetime > t.tpep_pickup_datetime
        """
        outbound = con.execute("""
            SELECT CAST(t.PULocationID AS INTEGER) AS zone_id,
                   date_trunc('hour', t.tpep_dropoff_datetime) AS completion_hour,
                   COUNT(*)::BIGINT AS out_completed,
                   COUNT(*) FILTER (WHERE t.PULocationID <> t.DOLocationID)::BIGINT
                       AS interzone_out_completed
        """ + base + " GROUP BY 1, 2").df()
        inbound = con.execute("""
            SELECT CAST(t.DOLocationID AS INTEGER) AS zone_id,
                   date_trunc('hour', t.tpep_dropoff_datetime) AS completion_hour,
                   COUNT(*)::BIGINT AS in_completed,
                   COUNT(*) FILTER (WHERE t.PULocationID <> t.DOLocationID)::BIGINT
                       AS interzone_in_completed
        """ + base + " GROUP BY 1, 2").df()
    finally:
        con.close()
    result = inbound.merge(outbound, on=["zone_id", "completion_hour"],
                           how="outer", validate="one_to_one")
    for col in VALUE_COLUMNS:
        result[col] = result[col].fillna(0).astype("int64")
    result["completion_hour"] = pd.to_datetime(result["completion_hour"])
    if result.duplicated(["zone_id", "completion_hour"]).any():
        raise AssertionError("Duplicate completion-hour zone records")
    if result["in_completed"].sum() != result["out_completed"].sum():
        raise AssertionError("Incoming/outgoing completed-trip counts differ")
    if (result["interzone_in_completed"].sum() !=
            result["interzone_out_completed"].sum()):
        raise AssertionError("Incoming/outgoing inter-zone counts differ")
    return result


def make_history_panel(
    hourly: pd.DataFrame,
    zone_ids: list[int],
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    window_hours: int = 24,
    carryover_hours: int = 6,
) -> pd.DataFrame:
    """Dense zone-hour table. At target t sum complete hours [t-w, t).

    Row-level completion buckets at hour h contain dropoffs in [h,h+1).
    Shift(1) excludes the current incomplete hour, preventing leakage.
    """
    if not 1 <= window_hours <= 168:
        raise ValueError("window_hours must be between 1 and 168")
    if carryover_hours < 0:
        raise ValueError("carryover_hours must not be negative")
    if len(set(zone_ids)) != len(zone_ids):
        raise ValueError("zone_ids must be unique")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    if hours.empty:
        raise ValueError("Empty hour range")
    required = {"zone_id", "completion_hour", *VALUE_COLUMNS}
    if not required.issubset(hourly):
        raise ValueError(f"Hourly OD input missing {sorted(required - set(hourly))}")
    if hourly.duplicated(["zone_id", "completion_hour"]).any():
        raise ValueError("Input has duplicate zone-hour keys")
    if hourly["zone_id"].isna().any() or hourly["completion_hour"].isna().any():
        raise ValueError("Null zone ID or completion hour")
    if not hourly.empty and not hourly["zone_id"].isin(zone_ids).all():
        raise ValueError("Hourly OD has unknown zone IDs")
    if not hourly.empty and (hourly[list(VALUE_COLUMNS)].fillna(0) < 0).any().any():
        raise ValueError("Negative completed-trip counts")
    if not hourly.empty:
        completed = pd.to_datetime(hourly["completion_hour"])
        if (completed.dt.floor("h") != completed).any():
            raise ValueError("completion_hour must be truncated to whole hours")
        if not completed.between(start, end, inclusive="left").all():
            raise ValueError("Hourly OD contains hours outside the requested window")
    idx = pd.MultiIndex.from_product(
        [sorted(zone_ids), hours], names=["zone_id", "completion_hour"]
    )
    dense = hourly.set_index(["zone_id", "completion_hour"])[list(VALUE_COLUMNS)]
    dense = dense.reindex(idx, fill_value=0).reset_index()
    for col in VALUE_COLUMNS:
        dense[col] = pd.to_numeric(dense[col], errors="raise").fillna(0).astype("int64")
    grouped = dense.groupby("zone_id", sort=False)
    mapping = {
        "in_completed": "in_strength",
        "out_completed": "out_strength",
        "interzone_in_completed": "interzone_in_strength",
        "interzone_out_completed": "interzone_out_strength",
    }
    result = dense[["zone_id", "completion_hour"]].rename(
        columns={"completion_hour": "feature_as_of"}
    ).copy()
    feature_cols = []
    for source, name in mapping.items():
        target = f"{name}_{window_hours}h"
        result[target] = grouped[source].transform(
            lambda x: x.shift(1).rolling(window_hours, min_periods=window_hours).sum()
        ).astype("Int64")
        feature_cols.append(target)
    # The 2025 source files omit pickups before Jan 1, even if a pre-2025
    # trip drops off on Jan 1. valid_od requires trip duration <= 360 min;
    # suppress the first (window + 6) hours to avoid that boundary artifact.
    # Tests using a complete synthetic history may set carryover_hours=0.
    result[f"has_full_history_{window_hours}h"] = (
        result["feature_as_of"] >= start + pd.Timedelta(
            hours=window_hours + carryover_hours
        )
    )
    # Local source timestamps have no UTC offset. Mark every forecast hour
    # whose current hour/history may overlap an ambiguous DST transition.
    dst_unsafe = pd.Series(False, index=result.index)
    for hour in DST_BAD_HOURS:
        dst_unsafe |= result["feature_as_of"].between(
            hour, hour + pd.Timedelta(hours=window_hours), inclusive="both"
        )
    result[f"dst_safe_{window_hours}h"] = ~dst_unsafe
    result["source_clock"] = "America/New_York_wall_clock_naive"
    assert result[["zone_id", "feature_as_of"]].duplicated().sum() == 0
    assert len(result) == len(zone_ids) * len(hours)
    full = result[f"has_full_history_{window_hours}h"]
    assert result.loc[full, feature_cols].notna().all().all()
    # The first w rows have no rolling history; the next carryover_hours
    # rows are withheld because historical trips from 2024 may be missing.
    result.loc[~full, feature_cols] = pd.NA
    assert result.loc[~full, feature_cols].isna().all().all()
    for field in feature_cols:
        assert result[field].dropna().ge(0).all()
    return result


def build_network_features(
    clean_dir: Path = CLEAN_DIR,
    zone_file: Path = ZONE_FILE,
    output_dir: Path = OUTPUT_DIR,
    window_hours: int = 24,
    threads: int = 4,
) -> Path:
    paths = sorted(Path(clean_dir).glob("cleaned_2025-*.parquet"))
    if len(paths) != 12:
        raise FileNotFoundError(f"Expected 12 cleaned Parquet files, found {len(paths)}")
    zone_ids = load_zone_ids(zone_file)
    print(f"Reading {len(paths)} cleaned monthly files; {len(zone_ids)} taxi zones")
    hourly = extract_completed_hourly_od(paths, zone_ids, threads=threads)
    print(f"Aggregated completion zone-hours: {len(hourly):,}")
    features = make_history_panel(hourly, zone_ids, window_hours=window_hours)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"historical_network_features_{window_hours}h.parquet"
    features.to_parquet(target, index=False)
    print(f"Saved: {target}")
    print(f"Rows: {len(features):,}; zone IDs: {features.zone_id.nunique()}")
    print(f"Full-history rows: {features[f'has_full_history_{window_hours}h'].sum():,}")
    print(f"DST-unsafe rows: {(~features[f'dst_safe_{window_hours}h']).sum():,}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--zone-file", type=Path, default=ZONE_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    a = parser.parse_args()
    build_network_features(a.clean_dir, a.zone_file, a.output_dir,
                           a.window_hours, a.threads)
