"""Create EDA summary tables from the shared 2025 cleaned taxi data.

Run from the repository root:
    python src/eda/generate_summaries.py

All individual trip records remain local. Only aggregated
EDA summaries are generated.
"""

from pathlib import Path
import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CLEAN_DIR = ROOT / "data/processed/cleaned_trips"
ZONE_FILE = ROOT / "data/external/taxi_zone_lookup.csv"

OUT = ROOT / "results/eda/tables"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(CLEAN_DIR.glob("cleaned_2025-*.parquet"))

if len(files) != 12:
    raise FileNotFoundError(
        f"Expected 12 cleaned files; found {len(files)}"
    )

if not ZONE_FILE.exists():
    raise FileNotFoundError(ZONE_FILE)

con = duckdb.connect()
con.execute("SET threads = 4")

file_paths = [str(p) for p in files]

# Paths come from files discovered in the local project directory.
quoted_paths = ", ".join(
    "'" + path.replace("'", "''") + "'"
    for path in file_paths
)

con.execute(
    f"CREATE VIEW trips AS "
    f"SELECT * FROM read_parquet([{quoted_paths}])"
)

zones = pd.read_csv(ZONE_FILE)
valid_ids = zones.loc[
    ~zones["LocationID"].isin([264, 265]),
    "LocationID",
].astype(int).tolist()

# Build temporary lookup used by all summary queries.
con.register("zone_lookup", zones)

# -------------------------------------------------
# 1. Monthly data quality and demand
# -------------------------------------------------

monthly = con.execute("""
    SELECT
        strftime(tpep_pickup_datetime, '%Y-%m') AS month,
        COUNT(*) AS pickup_count,
        COUNT(*) FILTER (WHERE valid_od) AS valid_od_count,
        COUNT(*) FILTER (WHERE zero_duration) AS zero_duration_count,
        COUNT(*) FILTER (WHERE negative_duration) AS negative_duration_count,
        COUNT(*) FILTER (WHERE NOT valid_distance) AS invalid_distance_count
    FROM trips
    GROUP BY 1
    ORDER BY 1
""").df()

monthly["valid_od_pct"] = (
    monthly["valid_od_count"] / monthly["pickup_count"] * 100
)

monthly.to_csv(OUT / "monthly_summary.csv", index=False)

# -------------------------------------------------
# 2. Daily pickup demand
# -------------------------------------------------

daily = con.execute("""
    SELECT
        CAST(tpep_pickup_datetime AS DATE) AS date,
        COUNT(*) AS pickup_count
    FROM trips
    WHERE PULocationID NOT IN (264, 265)
    GROUP BY 1
    ORDER BY 1
""").df()

daily.to_csv(OUT / "daily_demand.csv", index=False)

# -------------------------------------------------
# 3. Hour × weekday patterns
# -------------------------------------------------

hourly = con.execute("""
    SELECT
        CAST(strftime(tpep_pickup_datetime, '%w') AS INTEGER)
            AS weekday,
        CAST(date_part('hour', tpep_pickup_datetime) AS INTEGER)
            AS hour,
        COUNT(*) AS pickup_count
    FROM trips
    WHERE PULocationID NOT IN (264, 265)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()

# Aggregate total volume divided by number of
# occurrences of each weekday in calendar year 2025.
calendar = pd.date_range("2025-01-01", "2025-12-31")

weekday_counts = (
    pd.Series((calendar.dayofweek + 1) % 7)
    .value_counts()
    .to_dict()
)

hourly["n_days"] = hourly["weekday"].map(weekday_counts)
hourly["avg_pickups"] = (
    hourly["pickup_count"] / hourly["n_days"]
)

hourly.to_csv(OUT / "hour_weekday.csv", index=False)

# -------------------------------------------------
# 4. Pickup demand by taxi zone
# -------------------------------------------------

zone_demand = con.execute("""
    SELECT
        t.PULocationID AS LocationID,
        z.Borough,
        z.Zone,
        COUNT(*) AS pickup_count
    FROM trips t
    JOIN zone_lookup z
      ON t.PULocationID = z.LocationID
    WHERE t.PULocationID NOT IN (264, 265)
    GROUP BY 1, 2, 3
    ORDER BY pickup_count DESC
""").df()

zone_demand.to_csv(OUT / "zone_demand.csv", index=False)

# -------------------------------------------------
# 5. OD flow aggregation
# -------------------------------------------------

od = con.execute("""
    SELECT
        PULocationID AS origin,
        DOLocationID AS destination,
        COUNT(*) AS trip_count
    FROM trips
    WHERE valid_od
      AND PULocationID NOT IN (264, 265)
      AND DOLocationID NOT IN (264, 265)
    GROUP BY 1, 2
    ORDER BY trip_count DESC
""").df()

od.to_csv(OUT / "od_flows.csv", index=False)

# -------------------------------------------------
# 6. OD network diagnostics
# -------------------------------------------------

od_total = int(od["trip_count"].sum())
same_zone = int(
    od.loc[
        od["origin"] == od["destination"],
        "trip_count"
    ].sum()
)

network_stats = pd.DataFrame([{
    "od_valid_trips": od_total,
    "unique_od_pairs": len(od),
    "same_zone_trips": same_zone,
    "same_zone_pct": 100 * same_zone / od_total,
    "top_10_flow_share_pct": (
        100 * od.head(10)["trip_count"].sum() / od_total
    ),
    "top_100_flow_share_pct": (
        100 * od.head(100)["trip_count"].sum() / od_total
    ),
}])

network_stats.to_csv(
    OUT / "network_summary.csv", index=False
)

con.close()

print("EDA summary generation completed.")
print("\nMonthly statistics:")
print(monthly.to_string(index=False))
print("\nNetwork statistics:")
print(network_stats.to_string(index=False))
print("\nOutput directory:", OUT)
