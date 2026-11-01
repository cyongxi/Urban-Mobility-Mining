"""Generate EDA figures from aggregated summary tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = ROOT / "results/eda/tables"
FIGURE_DIR = ROOT / "results/eda/figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / name,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURE_DIR / name}")


# -------------------------------------------------
# EDA 01 — Monthly pickup demand
# -------------------------------------------------

monthly = pd.read_csv(TABLE_DIR / "monthly_summary.csv")

fig, ax = plt.subplots(figsize=(11, 5))

sns.lineplot(
    data=monthly,
    x="month",
    y="pickup_count",
    marker="o",
    linewidth=2,
    ax=ax,
)

ax.set_title("NYC Yellow Taxi Pickup Demand — 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Pickups")
ax.tick_params(axis="x", rotation=45)

save_fig("01_monthly_demand.png")


# -------------------------------------------------
# EDA 02 — Average demand by hour of day
# -------------------------------------------------

hourly = pd.read_csv(TABLE_DIR / "hour_weekday.csv")

hour_avg = (
    hourly.groupby("hour", as_index=False)["pickup_count"]
    .sum()
)

hour_avg["avg_pickups"] = (
    hour_avg["pickup_count"] / 365
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(
    data=hour_avg,
    x="hour",
    y="avg_pickups",
    marker="o",
    ax=ax,
)

ax.set_xticks(range(0, 24, 2))
ax.set_title("Average NYC Taxi Demand by Hour")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Hourly Pickups")

save_fig("02_hourly_demand.png")


# -------------------------------------------------
# EDA 03 — Weekday-hour demand heatmap
# -------------------------------------------------

weekday_labels = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

heatmap = hourly.pivot(
    index="weekday",
    columns="hour",
    values="avg_pickups",
)

heatmap = heatmap.reindex(range(7))
heatmap.index = [
    weekday_labels[i] for i in heatmap.index
]

fig, ax = plt.subplots(figsize=(13, 5))

sns.heatmap(
    heatmap,
    cmap="YlOrRd",
    ax=ax,
    cbar_kws={"label": "Average Pickups"},
)

ax.set_title("NYC Taxi Demand by Day and Hour")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")

save_fig("03_weekday_hour_heatmap.png")


# -------------------------------------------------
# EDA 04 — Top 20 pickup zones
# -------------------------------------------------

zones = pd.read_csv(TABLE_DIR / "zone_demand.csv")

top_zones = (
    zones.nlargest(20, "pickup_count")
    .sort_values("pickup_count")
    .copy()
)

top_zones["label"] = (
    top_zones["Zone"]
    + " ("
    + top_zones["Borough"].fillna("Unknown")
    + ")"
)

fig, ax = plt.subplots(figsize=(11, 8))

sns.barplot(
    data=top_zones,
    x="pickup_count",
    y="label",
    color="steelblue",
    ax=ax,
)

ax.set_title("Top 20 NYC Taxi Pickup Zones — 2025")
ax.set_xlabel("Annual Pickup Count")
ax.set_ylabel("Taxi Zone")

save_fig("04_top_pickup_zones.png")


# -------------------------------------------------
# EDA 05 — Borough demand composition
# -------------------------------------------------

borough = (
    zones.groupby("Borough", as_index=False)
    ["pickup_count"].sum()
    .sort_values("pickup_count", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=borough,
    x="pickup_count",
    y="Borough",
    color="teal",
    ax=ax,
)

ax.set_title("Taxi Pickup Demand by Borough — 2025")
ax.set_xlabel("Annual Pickup Count")
ax.set_ylabel("Borough")

save_fig("05_borough_demand.png")


# -------------------------------------------------
# EDA 06 — Top 20 OD flows
# -------------------------------------------------

od = pd.read_csv(TABLE_DIR / "od_flows.csv")
lookup = zones.set_index("LocationID")["Zone"].to_dict()

# Remove self-loops from this inter-zone ranking.
od = od[od["origin"] != od["destination"]].copy()

top_od = (
    od.nlargest(20, "trip_count")
    .sort_values("trip_count")
    .copy()
)

top_od["route"] = (
    top_od["origin"].map(lookup).fillna(
        top_od["origin"].astype(str)
    )
    + " → "
    + top_od["destination"].map(lookup).fillna(
        top_od["destination"].astype(str)
    )
)

fig, ax = plt.subplots(figsize=(12, 9))

sns.barplot(
    data=top_od,
    x="trip_count",
    y="route",
    color="slateblue",
    ax=ax,
)

ax.set_title("Top 20 Inter-Zone Taxi OD Flows — 2025")
ax.set_xlabel("Number of Trips")
ax.set_ylabel("Origin → Destination")

save_fig("06_top_od_flows.png")

print("All EDA figures generated.")
