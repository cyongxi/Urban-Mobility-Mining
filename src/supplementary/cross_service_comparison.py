"""June 2025 NYC TLC Yellow Taxi vs Uber/Lyft pickup comparisons.

Run from the repository root:
    python -m src.supplementary.cross_service_comparison

Inputs: cleaned Yellow Taxi month; raw HVFHV month; official taxi zone lookup.
Uses only month-matched pickups; never feeds HVFHV features into forecasting.
The large Parquet datasets remain local, and DuckDB aggregates them before
anything is loaded into pandas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERVICES = ("Yellow", "Uber", "Lyft")
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ROOT = Path(__file__).resolve().parents[2]


def sql_literal(path: Path) -> str:
    """Quote one local path for a DuckDB SQL string literal."""
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def read_lookup(path: Path) -> pd.DataFrame:
    lookup = pd.read_csv(path)
    required = {"LocationID", "Borough", "Zone"}
    if not required.issubset(lookup.columns):
        raise ValueError(f"Zone lookup missing: {required - set(lookup.columns)}")
    lookup = lookup[["LocationID", "Borough", "Zone"]].copy()
    if lookup.LocationID.isna().any():
        raise ValueError("Null LocationID in zone lookup")
    lookup["LocationID"] = pd.to_numeric(lookup.LocationID, errors="raise").astype("int32")
    if lookup.LocationID.duplicated().any():
        raise ValueError("Duplicate LocationID in zone lookup")
    # Same area scope as the existing RQ2 pipeline: all TLC geographic
    # zones except special-purpose IDs 264 and 265. Zone 1 (EWR) is kept
    # and labeled separately; it is NOT one of NYC's five boroughs.
    return lookup.loc[~lookup.LocationID.isin((264, 265))].sort_values("LocationID")


def aggregate_pickups(yellow: Path, hvfhv: Path, lookup: pd.DataFrame,
                      month: str, threads: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return small grouped counts plus per-service QA; scan Parquet in DuckDB."""
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError("Install DuckDB first: python -m pip install duckdb") from e

    start = pd.Period(month, freq="M").start_time
    end = (pd.Period(month, freq="M") + 1).start_time
    lo, hi = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        con.register("valid_zones", lookup[["LocationID"]])
        # HVFHV codes are from the official TLC HVFHV data dictionary:
        # HV0003 = Uber, HV0005 = Lyft. Others are excluded from scope.
        # Both services are compared by pickup (not request) timestamp.
        con.execute(f"""
            CREATE TEMP VIEW service_pickups AS
            SELECT 'Yellow' AS service,
                   TRY_CAST(PULocationID AS INTEGER) AS zone_id,
                   tpep_pickup_datetime AS pickup_dt
              FROM read_parquet({sql_literal(yellow)})
            UNION ALL
            SELECT CASE hvfhs_license_num
                       WHEN 'HV0003' THEN 'Uber'
                       WHEN 'HV0005' THEN 'Lyft'
                   END AS service,
                   TRY_CAST(PULocationID AS INTEGER) AS zone_id,
                   pickup_datetime AS pickup_dt
              FROM read_parquet({sql_literal(hvfhv)})
             WHERE hvfhs_license_num IN ('HV0003', 'HV0005')
        """)
        # All raw month-matched records in the selected services are counted;
        # only records matched to eligible TLC zone IDs contribute to results.
        qa = con.execute(f"""
            SELECT p.service,
                   COUNT(*)::BIGINT AS n_month_rows,
                   COUNT(v.LocationID)::BIGINT AS n_eligible_pickups
              FROM service_pickups p
              LEFT JOIN valid_zones v ON p.zone_id = v.LocationID
             WHERE p.pickup_dt >= TIMESTAMP '{lo}'
               AND p.pickup_dt < TIMESTAMP '{hi}'
             GROUP BY p.service
        """).df()
        counts = con.execute(f"""
            SELECT p.service, p.zone_id,
                   (CAST(EXTRACT(ISODOW FROM p.pickup_dt) AS INTEGER) - 1)
                       AS day_of_week,
                   CAST(EXTRACT(HOUR FROM p.pickup_dt) AS INTEGER) AS hour,
                   COUNT(*)::BIGINT AS pickup_count
              FROM service_pickups p
              JOIN valid_zones v ON p.zone_id = v.LocationID
             WHERE p.pickup_dt >= TIMESTAMP '{lo}'
               AND p.pickup_dt < TIMESTAMP '{hi}'
             GROUP BY 1, 2, 3, 4
             ORDER BY 1, 2, 3, 4
        """).df()
    finally:
        con.close()
    return counts, qa


def summarize(counts: pd.DataFrame, qa: pd.DataFrame,
              lookup: pd.DataFrame, month: str) -> dict[str, pd.DataFrame]:
    """Generate validated, service-comparable summaries (testable without DuckDB)."""
    keys = ["service", "zone_id", "day_of_week", "hour"]
    if not {"pickup_count", *keys}.issubset(counts):
        raise ValueError("Grouped counts are missing required columns")
    if counts.duplicated(keys).any():
        raise ValueError("Duplicate service-zone-weekday-hour keys")
    if not counts.pickup_count.ge(0).all():
        raise ValueError("Negative pickup counts")
    if not counts.zone_id.isin(lookup.LocationID).all():
        raise ValueError("Unknown/special zone in grouped counts")
    if not counts.day_of_week.between(0, 6).all() or not counts.hour.between(0, 23).all():
        raise ValueError("Weekday/hour out of range")
    if set(qa.service) != set(SERVICES):
        raise ValueError("Expected Yellow, Uber, and Lyft in month-level QA")
    if qa.service.duplicated().any():
        raise ValueError("Duplicate service in QA")
    qa = qa.copy()
    qa["n_excluded_unknown_or_special_zone"] = qa.n_month_rows - qa.n_eligible_pickups
    if (qa.n_excluded_unknown_or_special_zone < 0).any():
        raise ValueError("Inconsistent QA counts")
    recon = counts.groupby("service").pickup_count.sum().reindex(SERVICES, fill_value=0)
    expected = qa.set_index("service").n_eligible_pickups.reindex(SERVICES)
    if not np.array_equal(recon.to_numpy(), expected.to_numpy()):
        raise AssertionError("QA reconciliation failed: grouped counts != eligible trips")
    if (expected <= 0).any():
        raise ValueError("A required service has zero eligible pickups; check codes/file")

    # Complete zero-filled 263-zone x 168-hour grid for EACH service.
    complete = pd.MultiIndex.from_product(
        [SERVICES, lookup.LocationID.to_list(), range(7), range(24)], names=keys
    )
    dense = (counts.set_index(keys).pickup_count.reindex(complete, fill_value=0)
             .astype("int64").rename("pickup_count").reset_index())
    dense = dense.merge(
        lookup.rename(columns={"LocationID": "zone_id"}),
        on="zone_id", how="left", validate="many_to_one"
    )
    if dense["Borough"].isna().any():
        raise AssertionError("Unmapped zone IDs")
    zone = dense.groupby(["service", "zone_id", "Borough", "Zone"], as_index=False).pickup_count.sum()
    service_totals = zone.groupby("service").pickup_count.sum().reindex(SERVICES)
    zone["zone_share"] = zone.pickup_count / zone.service.map(service_totals)

    borough = zone.groupby(["service", "Borough"], as_index=False).pickup_count.sum()
    borough["pickup_share"] = borough.pickup_count / borough.service.map(service_totals)

    monthly_days = pd.date_range(
        pd.Period(month, "M").start_time,
        pd.Period(month, "M").end_time.normalize(), freq="D"
    )
    exposures = (
        pd.Series(monthly_days.day_of_week)
        .value_counts()
        .reindex(range(7), fill_value=0)
    )
    if (exposures == 0).any():
        raise AssertionError("Selected month must contain all weekdays")
    temporal = dense.groupby(["service", "day_of_week", "hour"], as_index=False).pickup_count.sum()
    temporal["n_calendar_days"] = temporal.day_of_week.map(exposures).astype(int)
    temporal["mean_pickups_per_day_hour"] = temporal.pickup_count / temporal.n_calendar_days
    # RQ2-compatible typical-week shape (equal weekday exposure, not raw month total share).
    denom = temporal.groupby("service").mean_pickups_per_day_hour.transform("sum")
    temporal["normalized_share"] = temporal.mean_pickups_per_day_hour / denom
    if not np.allclose(
        temporal.groupby("service").normalized_share.sum().reindex(SERVICES).to_numpy(),
        np.ones(3), atol=1e-10
    ):
        raise AssertionError("Normalized typical-week profiles do not sum to 1")

    wide = zone.pivot(index="zone_id", columns="service", values="zone_share")
    wide = wide.reindex(index=lookup.LocationID, columns=SERVICES, fill_value=0).fillna(0)
    spearman = {service: wide["Yellow"].corr(wide[service], method="spearman")
                for service in SERVICES}
    top10 = (zone.sort_values("pickup_count", ascending=False)
             .groupby("service").head(10)
             .groupby("service").pickup_count.sum().div(service_totals))
    manhattan = (borough.loc[borough.Borough.eq("Manhattan")]
                 .set_index("service").pickup_share.reindex(SERVICES, fill_value=0))
    ewr = (borough.loc[borough.Borough.eq("EWR")]
           .set_index("service").pickup_share.reindex(SERVICES, fill_value=0))
    weekend = (temporal.loc[temporal.day_of_week.isin((5, 6))]
               .groupby("service").normalized_share.sum().reindex(SERVICES))
    summary = pd.DataFrame({
        "service": SERVICES,
        "eligible_pickups_june": service_totals.to_numpy(dtype="int64"),
        "active_zones": zone.loc[zone.pickup_count.gt(0)].groupby("service").zone_id.nunique().reindex(SERVICES,fill_value=0).to_numpy(),
        "manhattan_share": manhattan.to_numpy(),
        "ewr_share": ewr.to_numpy(),
        "top10_zone_share": top10.reindex(SERVICES).to_numpy(),
        "weekend_share_typical_week": weekend.to_numpy(),
        "zone_share_spearman_vs_yellow": [spearman[s] for s in SERVICES],
    })
    return {"quality_checks": qa, "zone_counts": zone,
            "borough_share": borough, "temporal_profile": temporal,
            "summary": summary}


def save_outputs(tables: dict[str, pd.DataFrame], out: Path,
                 month: str) -> None:
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")

    borough = tables["borough_share"]
    order = [x for x in ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR")
             if x in set(borough.Borough)]
    order += sorted(set(borough.Borough) - set(order))
    b = borough.pivot(index="Borough", columns="service", values="pickup_share")
    b = b.reindex(index=order, columns=SERVICES, fill_value=0).fillna(0)
    ax = b.plot.barh(figsize=(10, 5.2), width=.78)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of service's eligible June pickups")
    ax.set_ylabel("")
    ax.set_title(f"TLC Pickup Distribution by Borough / Area — {month}")
    ax.legend(title="Service", loc="best")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.figure.tight_layout()
    ax.figure.savefig(figs / "borough_pickup_share.png", dpi=200)
    plt.close(ax.figure)

    temporal = tables["temporal_profile"]
    vmax = temporal.normalized_share.max()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    for ax, service in zip(axs, SERVICES):
        data = temporal.loc[temporal.service.eq(service)].pivot(
            index="day_of_week", columns="hour", values="normalized_share"
        ).reindex(index=range(7), columns=range(24)).fillna(0)
        im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(service)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour of day (NYC local time)")
        ax.set_yticks(range(7), DAYS)
    axs[0].set_ylabel("Day of week")
    fig.suptitle(f"Normalized Typical-Week Pickup Patterns — {month}")
    fig.colorbar(im, ax=axs, label="Share of normalized typical week", shrink=.82)
    fig.subplots_adjust(left=.055, right=.91, top=.83, bottom=.17, wspace=.16)
    fig.savefig(figs / "weekday_hour_heatmaps.png", dpi=200)
    plt.close(fig)

    meta = {
        "month": month,
        "services": SERVICES,
        "uber_license": "HV0003",
        "lyft_license": "HV0005",
        "eligible_zones": "TLC lookup except IDs 264 and 265; EWR retained and labeled separately",
        "yellow_input": "Existing pickup-valid cleaned Yellow Taxi monthly parquet; valid_od not required",
        "hvfhv_input": "Official raw HVFHV monthly parquet, only HV0003/HV0005",
        "time_aggregation": "pickup local datetime; Mon=0; hour=0..23; average by count of that weekday in month",
        "normalization": "each service's 7x24 mean pickup vector divided by its sum",
        "scope": "descriptive comparison only; not causal, annual, or forecast feature engineering",
    }
    (out / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", default="2025-06", help="YYYY-MM (default 2025-06)")
    p.add_argument("--yellow", type=Path, default=ROOT / "data/processed/cleaned_trips/cleaned_2025-06.parquet")
    p.add_argument("--hvfhv", type=Path, default=ROOT / "data/external/hvfhv/fhvhv_tripdata_2025-06.parquet")
    p.add_argument("--lookup", type=Path, default=ROOT / "data/external/taxi_zone_lookup.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results/cross_service")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if len(a.month) != 7 or a.month[4] != "-":
        p.error("--month must be YYYY-MM")
    for name, file in (("yellow", a.yellow), ("hvfhv", a.hvfhv), ("lookup", a.lookup)):
        if not file.is_file():
            p.error(f"Missing {name} file: {file}")
    lookup = read_lookup(a.lookup)
    print("Loading DuckDB summaries; full trip data stays on disk...")
    sparse, qa = aggregate_pickups(a.yellow, a.hvfhv, lookup, a.month, a.threads)
    tables = summarize(sparse, qa, lookup, a.month)
    save_outputs(tables, a.out, a.month)
    print("\nData-quality reconciliation:")
    print(tables["quality_checks"].to_string(index=False))
    print("\nCross-service summary:")
    print(tables["summary"].to_string(index=False))
    print("\nSaved CSV/PNG outputs to", a.out)


if __name__ == "__main__":
    main()
