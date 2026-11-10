"""Create RQ1 figures from the saved aggregate tables.

Run from repository root after community_detection:
    python -m src.network.plot_network
Maps are optional and require the official TLC taxi_zones.zip in data/external/.
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .community_detection import align_labels

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/network/tables"
FIG_DIR = ROOT / "results/network/figures"
SHAPEFILE = ROOT / "data/external/taxi_zones/taxi_zones.shp"
PERIODS = ("annual", "weekday", "weekend")


def save(fig, fig_dir, filename):
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_dir / filename)


def plot_all(table_dir=TABLE_DIR, fig_dir=FIG_DIR, shape_file=SHAPEFILE):
    table_dir, fig_dir, shape_file = map(Path, (table_dir, fig_dir, shape_file))
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(table_dir / "network_metrics.csv")

    # 1. Core Louvain metrics
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(metrics.period, metrics.modularity, color="#4C78A8")
    axes[0].set(title="Weighted Louvain Modularity", ylabel="Modularity Q")
    axes[1].bar(metrics.period, metrics.within_community_interzone_share, color="#72B7B2")
    axes[1].set(title="Inter-zone Flow Within Communities", ylabel="Share of inter-zone trips", ylim=(0, 1))
    save(fig, fig_dir, "community_metrics.png")

    # 2. Annual community sizes (the old CSV had 'index', not 'community_id').
    summary = pd.read_csv(table_dir / "community_summary_annual.csv")
    if "community_id" not in summary.columns:
        raise ValueError("Re-run community_detection: community_summary_annual.csv needs community_id")
    summary = summary.sort_values("n_zones", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(summary.community_id.astype(str), summary.n_zones, color="#4C78A8")
    ax.set(xlabel="Annual community ID (arbitrary)", ylabel="Number of taxi zones",
           title="Annual Mobility Community Sizes")
    save(fig, fig_dir, "annual_community_sizes.png")

    # 3. Community composition by borough; more informative than just "dominant_borough".
    comp = pd.read_csv(table_dir / "community_borough_composition_annual.csv")
    comp_wide = comp.pivot(index="community_id", columns="Borough", values="n_zones").fillna(0)
    ordered = [x for x in ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]
               if x in comp_wide.columns]
    ordered += [x for x in comp_wide.columns if x not in ordered]
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_wide[ordered].plot(kind="bar", stacked=True, ax=ax, width=0.7)
    ax.set(title="Annual Community Composition by Borough", xlabel="Annual Community ID",
           ylabel="Number of Taxi Zones")
    ax.legend(title="Borough", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, fig_dir, "annual_borough_composition.png")

    # 4. Major-flows concentration: all denominator are inter-zone trips.
    labels = ["Top 10", "Top 100", "Top 1% edges"]
    cols = ["top10_interzone_edge_share", "top100_interzone_edge_share",
            "top1pct_interzone_edge_share"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(metrics)); width = 0.23
    for idx, (label, key) in enumerate(zip(labels, cols)):
        ax.bar(x + (idx - 1) * width, 100 * metrics[key], width=width, label=label)
    ax.set_xticks(x, metrics.period)
    ax.set(title="Directed Inter-zone OD Flow Concentration",
           ylabel="Share of inter-zone trips (%)", xlabel="Time period")
    ax.legend()
    save(fig, fig_dir, "od_flow_concentration.png")

    # 5. Annual directed community-to-community flow matrix.
    flow = pd.read_csv(table_dir / "community_flow_matrix_annual.csv")
    mat = flow.pivot(index="origin_community", columns="destination_community",
                     values="share_of_interzone_trips").fillna(0).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(mat.to_numpy() * 100, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(mat.columns)), [f"C{i}" for i in mat.columns])
    ax.set_yticks(range(len(mat.index)), [f"C{i}" for i in mat.index])
    ax.set(xlabel="Destination community", ylabel="Origin community",
           title="Annual Inter-community OD Flows (% of all inter-zone trips)")
    for (i, j), val in np.ndenumerate(mat.to_numpy() * 100):
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, color="white" if val > (mat.to_numpy().max() * 100) / 2 else "black")
    fig.colorbar(im, ax=ax, label="Share of inter-zone trips (%)")
    save(fig, fig_dir, "annual_community_flow_matrix.png")

    # 6. Fixed-graph seed stability and weekday/weekend overlap.
    compared = pd.read_csv(table_dir / "temporal_community_ari.csv").iloc[0]
    seed = pd.read_csv(table_dir / "louvain_seed_stability.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(seed.period, seed.mean_pairwise_seed_ari, color="#72B7B2")
    ax.set(title=f"Louvain Seed Stability (weekday/weekend ARI = {compared.ari:.3f})",
           xlabel="Network period", ylabel="Mean pairwise seed ARI", ylim=(-0.1, 1.05))
    save(fig, fig_dir, "louvain_seed_stability.png")

    comparison = pd.read_csv(table_dir / "comparable_zone_assignments.csv")
    cross = pd.crosstab(comparison.weekday_community, comparison.weekend_community)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cross.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cross.columns)), cross.columns)
    ax.set_yticks(range(len(cross.index)), cross.index)
    ax.set(xlabel="Weekend community (raw arbitrary ID)", ylabel="Weekday community (raw arbitrary ID)",
           title="Weekday/Weekend Community Overlap")
    for (i, j), v in np.ndenumerate(cross.to_numpy()):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > cross.to_numpy().max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Number of Zones")
    save(fig, fig_dir, "weekday_weekend_overlap.png")

    pairs = pd.read_csv(table_dir / "temporal_ari_seed_pairs.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pairs.ari, bins=np.linspace(-0.05, 1.05, 23), color="#72B7B2", edgecolor="white")
    ax.axvline(float(compared.ari), color="#E45756", linestyle="--",
               label=f"Chosen seed ARI = {compared.ari:.3f}")
    ax.set(title="Weekday/Weekend ARI Across Louvain Seed Pairs",
           xlabel="Adjusted Rand Index", ylabel="Count of seed pairs", xlim=(-0.05, 1.05))
    ax.legend()
    save(fig, fig_dir, "weekday_weekend_seed_sensitivity.png")

    # 7. Optional spatial maps. Colors are aligned to annual communities by maximum overlap.
    if not shape_file.exists():
        print(f"Map skipped: {shape_file} not found. Download official TLC taxi_zones.zip.")
    else:
        import geopandas as gpd
        try:
            geometry = gpd.read_file(shape_file)
            key = "LocationID" if "LocationID" in geometry.columns else (
                "OBJECTID" if "OBJECTID" in geometry.columns else None)
            if key is None:
                raise ValueError(f"Geometry has no LocationID/OBJECTID: {list(geometry.columns)}")
            geometry[key] = geometry[key].astype(int)
            labels_by_period = {}
            for period in PERIODS:
                labels_df = pd.read_csv(table_dir / f"communities_{period}.csv")
                labels_by_period[period] = dict(zip(labels_df.zone_id, labels_df.community_id))
            baseline = labels_by_period["annual"]
            max_label = max(baseline.values())
            for period in PERIODS:
                label_map = labels_by_period[period]
                if period != "annual":
                    nodes = sorted(set(baseline) & set(label_map))
                    remap = align_labels(baseline, label_map, nodes)
                else:
                    remap = {i: i for i in set(label_map.values())}
                lab = pd.DataFrame({"zone_id": list(label_map),
                                    "plot_community": [remap[x] for x in label_map.values()]})
                gdf = geometry.merge(lab, left_on=key, right_on="zone_id", how="left",
                                     validate="one_to_one")
                gdf = gdf.loc[~gdf[key].isin([264, 265])].copy()
                fig, ax = plt.subplots(figsize=(9, 9))
                gdf.plot(column="plot_community", categorical=True, legend=True,
                         cmap="tab20", vmin=0, vmax=max(max_label, max(remap.values())),
                         linewidth=0.3, edgecolor="white",
                         missing_kwds={"color": "lightgrey", "label": "No observed OD trips"}, ax=ax)
                ax.set_title(f"2025 {period.title()} Louvain Communities\n"
                             "Colors aligned to annual community partition")
                ax.set_axis_off()
                save(fig, fig_dir, f"community_map_{period}.png")
        except Exception as exc:
            raise RuntimeError(f"Could not plot zone maps from {shape_file}: {exc}") from exc
    print("RQ1 figures saved to:", fig_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--shape-file", type=Path, default=SHAPEFILE)
    args = p.parse_args()
    plot_all(args.table_dir, args.fig_dir, args.shape_file)
