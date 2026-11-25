"""RQ1: network diagnostics, weighted Louvain, and temporal stability.

Run from repository root, AFTER build_od_network:
    python -m src.network.community_detection

Uses OD aggregates only; does not read trip-level Parquet files.
Full-year outputs are retrospective and must NOT be used to forecast 2025 hours.
"""
from pathlib import Path
import argparse
from itertools import combinations
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/network/tables"
PERIODS = ("annual", "weekday", "weekend")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def build_graphs(edges: pd.DataFrame):
    """Build directed G (loops retained) and symmetric weighted H (loops removed)."""
    required = {"origin", "destination", "trip_count"}
    if not required.issubset(edges.columns):
        raise ValueError(f"OD table missing columns: {required - set(edges.columns)}")
    G, H = nx.DiGraph(), nx.Graph()
    for row in edges.itertuples(index=False):
        a, b, wt = int(row.origin), int(row.destination), int(row.trip_count)
        if wt <= 0:
            raise ValueError("OD weights must be positive")
        if G.has_edge(a, b):
            raise ValueError(f"Repeated OD pair ({a}, {b}); aggregate before graph construction")
        G.add_edge(a, b, weight=wt)
        H.add_nodes_from([a, b])
        if a != b:
            previous = H.get_edge_data(a, b, default={}).get("weight", 0)
            H.add_edge(a, b, weight=previous + wt)
    assert sum(d["weight"] for _, _, d in G.edges(data=True)) == int(edges.trip_count.sum())
    return G, H


def zone_metrics(G, H, zones):
    rows = []
    for node in sorted(G.nodes):
        rows.append({
            "zone_id": node,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            # Includes same-zone trips by design.
            "in_strength": G.in_degree(node, weight="weight"),
            "out_strength": G.out_degree(node, weight="weight"),
            # These two explicitly exclude self-loops.
            "interzone_weighted_degree": H.degree(node, weight="weight"),
            "interzone_neighbors": H.degree(node),
        })
    return (pd.DataFrame(rows)
            .merge(zones[["LocationID", "Borough", "Zone"]], left_on="zone_id",
                   right_on="LocationID", how="left", validate="one_to_one")
            .drop(columns="LocationID"))


def louvain_partition(H, seed=42, resolution=1.0):
    """Weighted Louvain on symmetrized inter-zone edges."""
    if H.number_of_nodes() == 0:
        raise ValueError("Empty graph")
    communities = nx.community.louvain_communities(
        H, weight="weight", resolution=resolution, seed=seed
    )
    # IDs are deterministic for a fixed partition, but not semantically matched across periods.
    communities = sorted(communities, key=lambda s: min(s))
    labels = {int(node): int(i) for i, group in enumerate(communities) for node in group}
    q = nx.community.modularity(H, communities, weight="weight", resolution=resolution)
    return labels, float(q)


def summarize_communities(edges, labels, zones):
    """Return zone assignments, one-row-per-community summary, and interzone within share."""
    inter = edges.loc[edges.origin != edges.destination].copy()
    inter["c_origin"] = inter.origin.map(labels)
    inter["c_destination"] = inter.destination.map(labels)
    assert inter[["c_origin", "c_destination"]].notna().all().all()
    within = inter.loc[inter.c_origin == inter.c_destination]
    total = int(inter.trip_count.sum())
    within_share = float(within.trip_count.sum() / total) if total else np.nan
    sizes = pd.Series(labels).rename_axis("zone_id").reset_index(name="community_id")
    sizes = (sizes.merge(zones[["LocationID", "Borough", "Zone"]],
                         left_on="zone_id", right_on="LocationID", how="left",
                         validate="one_to_one")
                  .drop(columns="LocationID"))
    counts = sizes.groupby("community_id").size().rename("n_zones")
    dominant = sizes.groupby("community_id")["Borough"].agg(
        lambda x: x.fillna("Unknown").value_counts().index[0]
    ).rename("dominant_borough")
    in_flow = within.groupby("c_origin").trip_count.sum().rename("within_interzone_trips")
    summary = pd.concat([counts, dominant, in_flow], axis=1).fillna({"within_interzone_trips": 0})
    # FIX: concatenation can lose the index name; make it explicit before reset_index.
    summary.index.name = "community_id"
    summary = summary.reset_index().sort_values("community_id").reset_index(drop=True)
    summary["within_interzone_trips"] = summary["within_interzone_trips"].astype("int64")
    assert "community_id" in summary.columns
    assert summary.n_zones.sum() == len(labels)
    return sizes, summary, within_share


def active_nodes(H):
    """Zones with at least one positive-weight inter-zone connection."""
    return {v for v in H.nodes if H.degree(v) > 0}


def flow_concentration(inter):
    """Concentration of directed inter-zone OD edge weights; shares in [0,1]."""
    weights = np.sort(inter["trip_count"].to_numpy(dtype=np.float64))[::-1]
    if len(weights) == 0 or weights.sum() == 0:
        return dict(top10_interzone_edge_share=np.nan, top100_interzone_edge_share=np.nan,
                    top1pct_interzone_edge_share=np.nan, interzone_edge_hhi=np.nan)
    total = weights.sum()
    top1_count = max(1, math.ceil(len(weights) * 0.01))
    return {
        "top10_interzone_edge_share": float(weights[:10].sum() / total),
        "top100_interzone_edge_share": float(weights[:100].sum() / total),
        "top1pct_interzone_edge_share": float(weights[:top1_count].sum() / total),
        "interzone_edge_hhi": float(((weights / total) ** 2).sum()),
    }


def save_community_details(inter, labels, assigned, zones, table_dir, period):
    """Save borough membership, inter/intra community OD matrix, and top directed OD flows."""
    members = assigned.copy()
    members["Borough"] = members["Borough"].fillna("Unknown")
    boroughs = (members.groupby(["community_id", "Borough"], as_index=False)
                      .size().rename(columns={"size": "n_zones"}))
    boroughs["community_size"] = boroughs.groupby("community_id").n_zones.transform("sum")
    boroughs["share_of_community"] = boroughs.n_zones / boroughs.community_size
    boroughs.to_csv(table_dir / f"community_borough_composition_{period}.csv", index=False)

    mapped = inter.copy()
    mapped["origin_community"] = mapped.origin.map(labels)
    mapped["destination_community"] = mapped.destination.map(labels)
    flows = (mapped.groupby(["origin_community", "destination_community"], as_index=False)
                   .trip_count.sum())
    # Include zero-flow community pairs, so the matrix shape is always C x C.
    community_ids = sorted(set(labels.values()))
    all_pairs = pd.MultiIndex.from_product(
        [community_ids, community_ids], names=["origin_community", "destination_community"]
    ).to_frame(index=False)
    flows = all_pairs.merge(flows, on=["origin_community", "destination_community"],
                            how="left", validate="one_to_one")
    flows["trip_count"] = flows.trip_count.fillna(0).astype("int64")
    total = flows.trip_count.sum()
    flows["share_of_interzone_trips"] = flows.trip_count / total if total else np.nan
    flows["is_within_community"] = flows.origin_community == flows.destination_community
    flows.to_csv(table_dir / f"community_flow_matrix_{period}.csv", index=False)
    assert total == int(inter.trip_count.sum())

    z_lookup = zones.set_index("LocationID")["Zone"].to_dict()
    top = inter.nlargest(20, "trip_count").copy()
    top["origin_name"] = top.origin.map(z_lookup)
    top["destination_name"] = top.destination.map(z_lookup)
    top.to_csv(table_dir / f"major_od_flows_{period}.csv", index=False)


def align_labels(reference, target, zone_ids):
    """Align arbitrary target partition IDs to reference by maximum zone overlap.

    Returns target ID -> aligned ID. Extra target communities get fresh IDs.
    This alignment is for interpreting differences, not for calculating ARI.
    """
    rows = sorted({reference[z] for z in zone_ids})
    cols = sorted({target[z] for z in zone_ids})
    contingency = pd.crosstab(
        pd.Categorical([reference[z] for z in zone_ids], categories=rows),
        pd.Categorical([target[z] for z in zone_ids], categories=cols),
        dropna=False,
    ).to_numpy()
    rr, cc = linear_sum_assignment(-contingency)
    mapping = {cols[c]: rows[r] for r, c in zip(rr, cc)}
    next_id = max(rows) + 1
    for c in cols:
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
    return mapping


def analyze(table_dir=TABLE_DIR, seeds=DEFAULT_SEEDS, resolution=1.0):
    table_dir = Path(table_dir)
    if not seeds:
        raise ValueError("Provide at least one random seed")
    zones = pd.read_csv(table_dir / "zone_lookup_used.csv")
    all_labels, active, all_repeats = {}, {}, {}
    metric_rows, stability_rows = [], []
    for period in PERIODS:
        edges = pd.read_csv(table_dir / f"od_{period}_all.csv")
        if edges.empty:
            raise ValueError(f"{period} OD edges are empty")
        G, H = build_graphs(edges)
        labels, q = louvain_partition(H, seed=seeds[0], resolution=resolution)
        active[period] = active_nodes(H)
        all_labels[period] = labels
        assigned, summary, within_share = summarize_communities(edges, labels, zones)
        assigned.to_csv(table_dir / f"communities_{period}.csv", index=False)
        summary.to_csv(table_dir / f"community_summary_{period}.csv", index=False)
        zone_metrics(G, H, zones).to_csv(table_dir / f"zone_metrics_{period}.csv", index=False)
        inter = edges.loc[edges.origin != edges.destination].copy()
        save_community_details(inter, labels, assigned, zones, table_dir, period)
        q_inter = int(inter.trip_count.sum())
        n_nodes = G.number_of_nodes()
        density = len(inter) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else np.nan
        metric_rows.append({
            "period": period, "n_nodes": n_nodes,
            "n_directed_edges_including_loops": G.number_of_edges(),
            "n_directed_interzone_edges": len(inter),
            "directed_interzone_density": float(density),
            "n_undirected_interzone_edges": H.number_of_edges(),
            "n_active_interzone_nodes": len(active[period]),
            "total_trips": int(edges.trip_count.sum()),
            "self_loop_trips": int(edges.loc[edges.origin == edges.destination, "trip_count"].sum()),
            "interzone_trips": q_inter,
            "n_communities": len(set(labels.values())),
            "modularity": q,
            "within_community_interzone_share": within_share,
            **flow_concentration(inter),
        })
        repeats = {s: louvain_partition(H, seed=s, resolution=resolution)[0] for s in seeds}
        all_repeats[period] = repeats
        active_sorted = sorted(active[period])
        aris = [adjusted_rand_score([a[n] for n in active_sorted], [b[n] for n in active_sorted])
                for a, b in combinations(repeats.values(), 2)] if len(active_sorted) >= 2 else []
        stability_rows.append({"period": period, "n_seeds": len(seeds),
                               "n_active_nodes": len(active_sorted),
                               "mean_pairwise_seed_ari": float(np.mean(aris)) if aris else np.nan,
                               "min_pairwise_seed_ari": float(np.min(aris)) if aris else np.nan})
        print(f"{period:8s}: nodes={n_nodes}, communities={len(set(labels.values()))}, "
              f"modularity={q:.3f}, within-flow={within_share:.1%}")
    pd.DataFrame(metric_rows).to_csv(table_dir / "network_metrics.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(table_dir / "louvain_seed_stability.csv", index=False)

    comparable = sorted(active["weekday"] & active["weekend"])
    if len(comparable) < 2:
        raise ValueError("Insufficient active nodes shared by weekday/weekend")
    w = [all_labels["weekday"][n] for n in comparable]
    e = [all_labels["weekend"][n] for n in comparable]
    ari = float(adjusted_rand_score(w, e))
    mapping = align_labels(all_labels["weekday"], all_labels["weekend"], comparable)
    changes = pd.DataFrame({
        "zone_id": comparable,
        "weekday_community": w,
        "weekend_community": e,
        "weekend_aligned_to_weekday": [mapping[v] for v in e],
    })
    changes["changed_community"] = (
        changes.weekday_community != changes.weekend_aligned_to_weekday
    )
    changes = changes.merge(zones[["LocationID", "Borough", "Zone"]],
                            left_on="zone_id", right_on="LocationID", how="left",
                            validate="one_to_one").drop(columns="LocationID")
    changes.to_csv(table_dir / "zone_changes_weekday_weekend.csv", index=False)
    changes[["zone_id", "weekday_community", "weekend_community"]].to_csv(
        table_dir / "comparable_zone_assignments.csv", index=False)
    pd.DataFrame([{"comparison": "weekday_vs_weekend",
                   "n_common_active_zones": len(comparable),
                   "ari": ari,
                   "changed_zone_count_after_alignment": int(changes.changed_community.sum())}]
                 ).to_csv(table_dir / "temporal_community_ari.csv", index=False)
    # Interpretation robustness: 25 seed-pair comparisons (5 by 5 at default).
    seed_aris = []
    for s_w, l_w in all_repeats["weekday"].items():
        for s_e, l_e in all_repeats["weekend"].items():
            seed_aris.append({"weekday_seed": s_w, "weekend_seed": s_e,
                              "ari": float(adjusted_rand_score(
                                  [l_w[z] for z in comparable], [l_e[z] for z in comparable]
                              ))})
    pd.DataFrame(seed_aris).to_csv(table_dir / "temporal_ari_seed_pairs.csv", index=False)
    print(f"Weekday/weekend ARI={ari:.3f} across {len(comparable)} common active zones; "
          f"{int(changes.changed_community.sum())} zones change after label alignment")
    print(f"Time-comparison ARI across seed pairs: {min(d['ari'] for d in seed_aris):.3f} "
          f"to {max(d['ari'] for d in seed_aris):.3f}")
    return pd.DataFrame(metric_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    p.add_argument("--resolution", type=float, default=1.0)
    args = p.parse_args()
    analyze(args.table_dir, resolution=args.resolution)
