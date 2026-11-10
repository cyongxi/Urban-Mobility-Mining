"""Historical 24h OD connectivity and HHI for SI 671 RQ1 -> RQ3.

Run from repository root after `python -m src.network.network_features`:
    python -m src.network.network_topology_features --window-hours 24 --threads 4

Uses EXACTLY the same completed-trip validity/zone/time filters as network_features.py.
Each feature at forecast t uses completed trips with dropoffs in [t-24h, t).
Source timestamps are naive NYC wall clock. DST ambiguous windows are flagged and
must be excluded from one-step-ahead comparisons. Offline instant-availability
assumption for completed rides; this is not a real-time deployment claim.

Output: two local Parquets under data/processed/network_features/
    historical_network_topology_24h.parquet          (four new features)
    historical_network_features_24h_augmented.parquet (old + new features)

HHI excludes all self-loops; when no cross-zone traffic, HHI=NaN (undefined)
and a corresponding has_interzone_*_flow_24h flag is False. Degree=0 then.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .network_features import (
    CLEAN_DIR, ZONE_FILE, OUTPUT_DIR, START, END,
    load_zone_ids,
)

TOPOLOGY_COLUMNS = (
    "in_degree_24h", "out_degree_24h", "in_hhi_24h", "out_hhi_24h"
)


def iter_completed_hourly_pairs(
    paths: list[Path], zone_ids: list[int], threads: int = 4
) -> Iterable[pd.DataFrame]:
    """Stream ordered aggregated (completion_hour, origin, destination, trips) chunks.

    We aggregate in DuckDB instead of loading trip-level data or millions of
    unique OD-hour pairs into Pandas. Order-by may spill to DuckDB temp disk.
    """
    import duckdb

    if not paths or any(not Path(p).is_file() for p in paths):
        raise FileNotFoundError("Missing cleaned monthly Parquet files")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        quoted = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
        con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet([{quoted}])")
        needed = {"PULocationID", "DOLocationID", "tpep_pickup_datetime",
                  "tpep_dropoff_datetime", "valid_od"}
        schema = set(con.execute("DESCRIBE trips").df().column_name)
        if not needed.issubset(schema):
            raise ValueError(f"Missing cleaned columns: {sorted(needed - schema)}")
        con.register("valid_zones", pd.DataFrame({"LocationID": zone_ids}))
        con.execute("""
            SELECT date_trunc('hour', t.tpep_dropoff_datetime) AS completion_hour,
                   CAST(t.PULocationID AS INTEGER) AS origin,
                   CAST(t.DOLocationID AS INTEGER) AS destination,
                   COUNT(*)::BIGINT AS trip_count
            FROM trips t
            INNER JOIN valid_zones p ON t.PULocationID = p.LocationID
            INNER JOIN valid_zones d ON t.DOLocationID = d.LocationID
            WHERE t.valid_od IS TRUE
              AND t.tpep_pickup_datetime >= TIMESTAMP '2025-01-01'
              AND t.tpep_pickup_datetime < TIMESTAMP '2026-01-01'
              AND t.tpep_dropoff_datetime >= TIMESTAMP '2025-01-01'
              AND t.tpep_dropoff_datetime < TIMESTAMP '2026-01-01'
              AND t.tpep_dropoff_datetime > t.tpep_pickup_datetime
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
        """)
        while True:
            frame = con.fetch_df_chunk(vectors_per_chunk=32)
            if frame.empty:
                break
            yield frame
    finally:
        con.close()


def iter_hour_groups(chunks: Iterable[pd.DataFrame]):
    """Merge hourly group boundaries that may cross DuckDB chunk boundaries."""
    current_hour = None
    pending = []
    for chunk in chunks:
        required = {"completion_hour", "origin", "destination", "trip_count"}
        if not required.issubset(chunk.columns):
            raise ValueError("Missing hourly-pair columns")
        for hour, part in chunk.groupby("completion_hour", sort=False):
            hour = pd.Timestamp(hour)
            if current_hour is not None and hour < current_hour:
                raise ValueError("Hourly OD source must be sorted by completion_hour")
            if current_hour is not None and hour != current_hour:
                yield current_hour, pd.concat(pending, ignore_index=True)
                pending = []
            current_hour = hour
            pending.append(part)
    if current_hour is not None:
        yield current_hour, pd.concat(pending, ignore_index=True)


def degree_hhi(matrix: np.ndarray):
    """Directional cross-zone degree and HHI for one 24h OD matrix.

    Entry matrix[i,j] = completed trips from zone i to zone j. The diagonal
    is excluded before calculating both degrees and HHI.

    Returns (in_degree, out_degree, in_hhi, out_hhi,
             interzone_in_strength, interzone_out_strength).
    """
    cross = matrix.copy()
    np.fill_diagonal(cross, 0)
    in_degree = np.count_nonzero(cross, axis=0).astype(np.int16)
    out_degree = np.count_nonzero(cross, axis=1).astype(np.int16)
    in_flow = cross.sum(axis=0, dtype=np.int64)
    out_flow = cross.sum(axis=1, dtype=np.int64)
    squared = np.square(cross.astype(np.float64))
    in_hhi = np.divide(
        squared.sum(axis=0), np.square(in_flow.astype(float)),
        out=np.full(len(in_flow), np.nan), where=in_flow > 0
    )
    out_hhi = np.divide(
        squared.sum(axis=1), np.square(out_flow.astype(float)),
        out=np.full(len(out_flow), np.nan), where=out_flow > 0
    )
    return in_degree, out_degree, in_hhi, out_hhi, in_flow, out_flow


def rolling_topology_from_hour_groups(
    hour_groups: Iterable[tuple[pd.Timestamp, pd.DataFrame]],
    zone_ids: list[int], start: pd.Timestamp = START,
    end: pd.Timestamp = END, window_hours: int = 24,
    carryover_hours: int = 6,
) -> pd.DataFrame:
    """Streaming dense rolling OD adjacency; O(window * zone_count^2) memory.

    At forecast t, rotate in the completion bucket for [t-1h,t).
    Matrix is then *exactly* sum of hours [t-window_hours, t).
    This does not aggregate destination information from trips still in progress.
    """
    if not 1 <= window_hours <= 168:
        raise ValueError("window_hours must be between 1 and 168")
    if carryover_hours < 0:
        raise ValueError("carryover_hours must be >=0")
    if not zone_ids or len(zone_ids) != len(set(zone_ids)):
        raise ValueError("zone IDs must be nonempty and unique")
    ids = np.array(sorted(zone_ids), dtype=np.int32)
    n = len(ids)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    if len(hours) == 0:
        raise ValueError("Empty forecast calendar")
    id_to_idx = {int(z): i for i, z in enumerate(ids)}
    ring = np.zeros((window_hours, n, n), dtype=np.int64)
    matrix = np.zeros((n, n), dtype=np.int64)
    shape = (len(hours), n)
    deg_in = np.zeros(shape, dtype=np.int16)
    deg_out = np.zeros(shape, dtype=np.int16)
    hhi_in = np.full(shape, np.nan, dtype=np.float64)
    hhi_out = np.full(shape, np.nan, dtype=np.float64)
    flow_in = np.zeros(shape, dtype=np.int64)
    flow_out = np.zeros(shape, dtype=np.int64)
    groups = iter(hour_groups)
    upcoming = next(groups, None)
    for i, target in enumerate(hours):
        if i > 0:
            preceding_hour = hours[i-1]
            slot = (i-1) % window_hours
            # Expire hour t-window_hours-1, then add the now-finished hour t-1.
            matrix -= ring[slot]
            ring[slot].fill(0)
            if upcoming is not None and upcoming[0] < preceding_hour:
                raise ValueError(f"Unexpected stale hourly OD data: {upcoming[0]}")
            if upcoming is not None and upcoming[0] == preceding_hour:
                _, chunk = upcoming
                if chunk[["origin", "destination", "trip_count"]].isna().any().any():
                    raise ValueError("Null origin, destination or OD counts")
                if (chunk.trip_count <= 0).any():
                    raise ValueError("Nonpositive pair weight")
                try:
                    origins = np.fromiter((id_to_idx[int(v)] for v in chunk.origin), dtype=np.intp)
                    dests = np.fromiter((id_to_idx[int(v)] for v in chunk.destination), dtype=np.intp)
                except KeyError as exc:
                    raise ValueError(f"Unknown taxi zone: {exc}") from exc
                np.add.at(ring[slot], (origins, dests), chunk.trip_count.to_numpy(np.int64))
                upcoming = next(groups, None)
            matrix += ring[slot]
        d_in, d_out, h_in, h_out, f_in, f_out = degree_hhi(matrix)
        deg_in[i], deg_out[i] = d_in, d_out
        hhi_in[i], hhi_out[i] = h_in, h_out
        flow_in[i], flow_out[i] = f_in, f_out
    # The final completion hour [end-1h, end) is intentionally not used: it
    # only becomes available at the forecast origin `end`, outside this panel.
    if upcoming is not None and upcoming[0] < hours[-1]:
        raise ValueError(f"Unused stale OD events: {upcoming[0]}")
    if not np.array_equal(flow_in.sum(axis=1), flow_out.sum(axis=1)):
        raise AssertionError("Completed cross-zone OD flow not conserved within forecast hour")
    if deg_in.max() >= n or deg_out.max() >= n:
        raise AssertionError("Inter-zone degrees cannot exceed number of other zones")
    # Flatten hour-major; key uniqueness is exact by construction.
    result = pd.DataFrame({
        "zone_id": np.tile(ids, len(hours)),
        "feature_as_of": np.repeat(hours.to_numpy(), n),
        "in_degree_24h": deg_in.ravel(),
        "out_degree_24h": deg_out.ravel(),
        "in_hhi_24h": hhi_in.ravel(),
        "out_hhi_24h": hhi_out.ravel(),
        "has_interzone_in_flow_24h": (flow_in.ravel() > 0),
        "has_interzone_out_flow_24h": (flow_out.ravel() > 0),
        # Hold these for independent reconciliation to existing strength features.
        "_check_interzone_in": flow_in.ravel(),
        "_check_interzone_out": flow_out.ravel(),
    })
    # Initial 24h window + 6h potential 2024-started trips: missing history.
    full = result.feature_as_of >= pd.Timestamp(start) + pd.Timedelta(hours=window_hours+carryover_hours)
    result["has_full_history_24h"] = full.to_numpy(dtype=bool)
    for col in ("in_degree_24h", "out_degree_24h"):
        result[col] = result[col].astype("Int16").mask(~full)
    for col in ("in_hhi_24h", "out_hhi_24h"):
        result.loc[~full, col] = np.nan
    result.loc[~full, ["has_interzone_in_flow_24h", "has_interzone_out_flow_24h"]] = False
    assert not result.duplicated(["zone_id", "feature_as_of"]).any()
    return result


def merge_check_and_save(
    topology: pd.DataFrame, strength_file: Path,
    output_dir: Path = OUTPUT_DIR, window_hours: int=24
) -> tuple[Path, Path]:
    """Reconcile adjacency flow sums against existing time-safe strength table.

    Writes standalone topology and complete augmented features in two Parquets.
    """
    if window_hours != 24:
        raise ValueError("This schema and output contract currently require 24h")
    strength_file = Path(strength_file)
    if not strength_file.is_file():
        raise FileNotFoundError(f"Run network_features.py first: {strength_file}")
    strength = pd.read_parquet(strength_file)
    keys = ["zone_id", "feature_as_of"]
    if topology.duplicated(keys).any() or strength.duplicated(keys).any():
        raise ValueError("Duplicate feature keys")
    strength["feature_as_of"] = pd.to_datetime(strength["feature_as_of"])
    topology["feature_as_of"] = pd.to_datetime(topology["feature_as_of"])
    original_n = len(strength)
    check_cols = keys + ["in_degree_24h", "out_degree_24h", "in_hhi_24h", "out_hhi_24h",
                          "has_interzone_in_flow_24h", "has_interzone_out_flow_24h",
                          "_check_interzone_in", "_check_interzone_out"]
    missing = set(check_cols) - set(topology)
    if missing:
        raise ValueError(f"Missing topology output columns: {sorted(missing)}")
    combined = strength.merge(topology[check_cols], on=keys, how="left", validate="one_to_one", indicator=True)
    if len(combined) != original_n or not combined._merge.eq("both").all():
        raise AssertionError("Unexpected key mismatch between strength and topology")
    full = combined["has_full_history_24h"].astype(bool)
    if combined.loc[full, ["in_degree_24h", "out_degree_24h"]].isna().any().any():
        raise AssertionError("Missing topology degree in a full-history row")
    for old_col, new_col in (
        ("interzone_in_strength_24h", "_check_interzone_in"),
        ("interzone_out_strength_24h", "_check_interzone_out")
    ):
        observed = combined.loc[full, old_col].astype(np.int64).to_numpy()
        predicted = combined.loc[full, new_col].astype(np.int64).to_numpy()
        if not np.array_equal(observed, predicted):
            bad_index = int(np.flatnonzero(observed != predicted)[0])
            raise AssertionError(f"Pairwise reconciliation FAILED: {old_col} vs {new_col} at full row {bad_index}")
    # Mathematical invariants: HHI in [0,1] when flow>0; undefined (NaN) when zero.
    for col, flow_flag in (
        ("in_hhi_24h", "has_interzone_in_flow_24h"),
        ("out_hhi_24h", "has_interzone_out_flow_24h")
    ):
        has_flow = full & combined[flow_flag]
        zero_flow = full & ~combined[flow_flag]
        if not combined.loc[has_flow, col].between(0, 1).all():
            raise AssertionError(f"Invalid positive-flow HHI: {col}")
        if not combined.loc[zero_flow, col].isna().all():
            raise AssertionError(f"HHI must be NaN when no cross-zone flow: {col}")
        degree_col = "in_degree_24h" if col.startswith("in_") else "out_degree_24h"
        degree = combined.loc[has_flow, degree_col].astype(float)
        if degree.eq(0).any():
            raise AssertionError(f"Flow present but {degree_col} is zero")
        if (combined.loc[has_flow, col] + 1e-12 < 1.0 / degree).any():
            raise AssertionError(f"Invalid HHI below theoretical 1/degree: {col}")
    combined = combined.drop(columns=["_check_interzone_in", "_check_interzone_out", "_merge"])
    standalone = combined[keys + ["in_degree_24h", "out_degree_24h", "in_hhi_24h", "out_hhi_24h",
                                 "has_interzone_in_flow_24h", "has_interzone_out_flow_24h",
                                 "has_full_history_24h", "dst_safe_24h"]].copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stand_path = out / "historical_network_topology_24h.parquet"
    augmented_path = out / "historical_network_features_24h_augmented.parquet"
    standalone.to_parquet(stand_path, index=False)
    combined.to_parquet(augmented_path, index=False)
    print(f"PASS: {len(combined):,} aligned rows; historical inter-zone strengths exactly reconciled")
    print(f"Saved: {stand_path}")
    print(f"Saved: {augmented_path}")
    return stand_path, augmented_path


def main(clean_dir: Path = CLEAN_DIR, zone_file: Path = ZONE_FILE,
         output_dir: Path = OUTPUT_DIR, window_hours: int = 24,
         threads: int = 4):
    if window_hours != 24:
        raise ValueError("Current output contract supports only --window-hours 24")
    paths = sorted(Path(clean_dir).glob("cleaned_2025-*.parquet"))
    if len(paths) != 12:
        raise FileNotFoundError(f"Expected 12 cleaned_2025-*.parquet files; got {len(paths)}")
    zone_ids = load_zone_ids(Path(zone_file))
    print(f"Processing {len(paths)} months, {len(zone_ids)} zones (inter-zone pairs only for topology)")
    streaming_rows = iter_completed_hourly_pairs(paths, zone_ids, threads)
    groups = iter_hour_groups(streaming_rows)
    try:
        topology = rolling_topology_from_hour_groups(groups, zone_ids,
            window_hours=window_hours)
    finally:
        groups.close()
        streaming_rows.close()
    return merge_check_and_save(topology,
        Path(output_dir) / "historical_network_features_24h.parquet",
        Path(output_dir), window_hours=window_hours)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--zone-file", type=Path, default=ZONE_FILE)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--window-hours", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    main(a.clean_dir, a.zone_file, a.output_dir, a.window_hours, a.threads)
