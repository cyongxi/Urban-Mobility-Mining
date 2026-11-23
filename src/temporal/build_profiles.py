"""SI 671 RQ2: pickup-valid 2025 TLC trips -> zone x 168 temporal profiles.

From repo root: python -m src.temporal.build_profiles
Does NOT use valid_od: a valid pickup may have invalid dropoff/OD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data/processed/cleaned_trips"
LOOKUP_FILE = ROOT / "data/external/taxi_zone_lookup.csv"
OUTPUT_DIR = ROOT / "results/temporal/tables"
EXCLUDE_DATES = ("2025-03-09", "2025-11-02")  # NYC DST switch days
FEATURE_COLS = [f"d{d}_h{h:02d}" for d in range(7) for h in range(24)]


def exposure_days():
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    dates = dates[~dates.isin(pd.to_datetime(EXCLUDE_DATES))]
    counts = pd.Series(dates.day_of_week).value_counts()
    result = {d: int(counts.get(d, 0)) for d in range(7)}
    assert sum(result.values()) == 363
    return result


def profile_from_counts(sparse, zone_ids, n_days):
    """Pure-Pandas transformation, independently testable with tiny data.

    sparse keys: zone_id, day_of_week (Mon=0), hour, pickup_count.
    All non-observed zone x day-hour cells are structurally zero.
    """
    required = {"zone_id", "day_of_week", "hour", "pickup_count"}
    if not required.issubset(sparse.columns):
        raise ValueError(f"Missing columns: {required - set(sparse.columns)}")
    if len(zone_ids) != len(set(zone_ids)) or not zone_ids:
        raise ValueError("zone_ids must be a nonempty list of unique zone IDs")
    if any(n_days.get(d, 0) < 1 for d in range(7)):
        raise ValueError("Every weekday must have at least one observation day")
    if not sparse.empty:
        if sparse[list(required)].isna().any().any():
            raise ValueError("Nulls in sparse counts")
        if not sparse["zone_id"].isin(zone_ids).all():
            raise ValueError("Unknown zone IDs")
        if not sparse["day_of_week"].between(0, 6).all() or not sparse["hour"].between(0, 23).all():
            raise ValueError("Invalid weekday/hour")
        if not sparse.pickup_count.ge(0).all():
            raise ValueError("Negative pickup count")
        if sparse.duplicated(["zone_id", "day_of_week", "hour"]).any():
            raise ValueError("Duplicated zone/day/hour keys")
    complete = pd.MultiIndex.from_product(
        [sorted(zone_ids), range(7), range(24)],
        names=["zone_id", "day_of_week", "hour"],
    )
    dense = (sparse.set_index(["zone_id", "day_of_week", "hour"])["pickup_count"]
             .reindex(complete, fill_value=0).reset_index())
    dense["pickup_count"] = dense.pickup_count.astype("int64")
    dense["n_days"] = dense.day_of_week.map(n_days).astype("int16")
    dense["mean_pickups"] = dense.pickup_count / dense.n_days
    dense["profile_col"] = [f"d{d}_h{h:02d}" for d,h in zip(dense.day_of_week, dense.hour)]
    raw = dense.pivot(index="zone_id", columns="profile_col", values="mean_pickups")
    raw = raw.reindex(columns=FEATURE_COLS)
    assert raw.shape == (len(zone_ids), 168) and not raw.isna().any().any()
    mass = raw.sum(axis=1)
    normalized = raw.div(mass.where(mass > 0), axis=0).fillna(0.0)
    nonzero = mass > 0
    assert np.allclose(normalized.loc[nonzero].sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(normalized.loc[~nonzero].sum(axis=1), 0.0, atol=1e-10)
    diag = dense.groupby("zone_id", as_index=False).agg(
        total_pickups=("pickup_count", "sum"),
        active_slots=("pickup_count", lambda x: int((x > 0).sum())),
    )
    diag["mean_pickups_per_calendar_day"] = diag.total_pickups / sum(n_days.values())
    diag["status"] = np.where(diag.total_pickups > 0, "eligible", "no_pickups")
    assert int(dense.pickup_count.sum()) == int(sparse.pickup_count.sum())
    return (dense.drop(columns="profile_col"), raw.reset_index(),
            normalized.reset_index(), diag)


def aggregate_with_duckdb(clean_dir, lookup_file, threads=4):
    import duckdb
    files = sorted(Path(clean_dir).glob("cleaned_2025-*.parquet"))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 cleaned_2025-MM.parquet files; found {len(files)}")
    lookup = pd.read_csv(lookup_file)
    if "LocationID" not in lookup.columns:
        raise ValueError("Taxi zone lookup has no LocationID")
    zone_ids = sorted(set(lookup.LocationID.dropna().astype(int)) - {264, 265})
    if len(zone_ids) != 263:
        raise ValueError(f"Expected 263 geographic TLC zones, found {len(zone_ids)}")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        quoted = ",".join("'" + str(p).replace("'", "''") + "'" for p in files)
        con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet([{quoted}])")
        cols = set(con.execute("DESCRIBE trips").df().column_name)
        needed = {"tpep_pickup_datetime", "PULocationID"}
        if not needed.issubset(cols):
            raise ValueError(f"Missing columns in cleaned Parquet: {needed-cols}")
        con.register("valid_zones", pd.DataFrame({"LocationID": zone_ids}))
        sql = """
          SELECT CAST(t.PULocationID AS INTEGER) AS zone_id,
                 CAST(date_part('isodow', t.tpep_pickup_datetime)-1 AS INTEGER) AS day_of_week,
                 CAST(date_part('hour', t.tpep_pickup_datetime) AS INTEGER) AS hour,
                 COUNT(*)::BIGINT AS pickup_count
          FROM trips t
          INNER JOIN valid_zones v ON t.PULocationID=v.LocationID
          WHERE t.tpep_pickup_datetime >= TIMESTAMP '2025-01-01'
            AND t.tpep_pickup_datetime < TIMESTAMP '2026-01-01'
            AND CAST(t.tpep_pickup_datetime AS DATE) NOT IN
                (DATE '2025-03-09', DATE '2025-11-02')
          GROUP BY 1,2,3
          ORDER BY 1,2,3
        """
        return con.execute(sql).df(), zone_ids
    finally:
        con.close()


def main(clean_dir=CLEAN_DIR, lookup_file=LOOKUP_FILE, output=OUTPUT_DIR, threads=4):
    sparse, zones = aggregate_with_duckdb(clean_dir, lookup_file, threads)
    days = exposure_days()
    dense, raw, norm, diag = profile_from_counts(sparse, zones, days)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "zone_weekday_hour_counts.csv": dense,
        "zone_profiles_raw.csv": raw,
        "zone_profiles_normalized.csv": norm,
        "zone_profile_diagnostics.csv": diag,
    }.items():
        value.to_csv(output/name, index=False, float_format="%.12g")
    metadata = {
        "year": 2025,
        "n_zones": len(zones),
        "n_day_hour_features": 168,
        "total_included_pickups": int(dense.pickup_count.sum()),
        "days_by_weekday_monday_0": days,
        "excluded_dst_dates": list(EXCLUDE_DATES),
        "filters": "cleaned pickup-valid; 2025 pickup timestamps; TLC zones except 264,265; no valid_od condition",
        "raw": "mean pickup count per occurrence of given weekday/hour",
        "normalized": "divide each 168D raw vector by its sum; zero vectors stay zero and are not clustered",
        "time": "NYC naive local wall clock; DST dates excluded for descriptive clustering",
    }
    (output/"profile_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Raw / normalized shapes:", raw.shape, norm.shape)
    print("Included pickup records:", f"{metadata['total_included_pickups']:,}")
    print("Zero-pickup zones:", int((diag.total_pickups == 0).sum()))
    print("Saved:", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--lookup-file", type=Path, default=LOOKUP_FILE)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    main(args.clean_dir, args.lookup_file, args.output, args.threads)
