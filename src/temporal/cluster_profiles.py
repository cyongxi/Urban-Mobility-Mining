"""SI 671 RQ2: PCA, K-means, silhouette, stability, RQ1 community agreement.

Run after build_profiles: python -m src.temporal.cluster_profiles
Labels are retrospective for 2025; never use as 2025 forecasting covariates.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score,
                             adjusted_rand_score, silhouette_score)

from .build_profiles import FEATURE_COLS, OUTPUT_DIR, LOOKUP_FILE, ROOT

NETWORK_COMMUNITIES = ROOT / "results/network/tables/communities_annual.csv"
SEEDS = (42, 43, 44, 45, 46)


def run_clustering(raw, norm, diag, min_pickups=1, variance=0.90,
                   k_min=2, k_max=8, min_cluster_size=5):
    """Compute full cluster analysis using only normalized shapes for main K-means."""
    for table in (raw, norm, diag):
        if table.zone_id.duplicated().any():
            raise ValueError("Duplicate zone IDs")
    if set(raw.zone_id) != set(norm.zone_id) or set(raw.zone_id) != set(diag.zone_id):
        raise ValueError("Zone ID sets disagree")
    if not set(FEATURE_COLS).issubset(raw.columns) or not set(FEATURE_COLS).issubset(norm.columns):
        raise ValueError("Expected 168 hourly columns")
    if not 0 < variance <= 1 or min_pickups < 1:
        raise ValueError("Invalid PCA variance or minimum pickups")
    ids = sorted(set(diag.zone_id))
    raw = raw.set_index("zone_id").reindex(ids)
    norm = norm.set_index("zone_id").reindex(ids)
    diag = diag.set_index("zone_id").reindex(ids)
    x_raw = raw[FEATURE_COLS].to_numpy(float)
    x = norm[FEATURE_COLS].to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or not np.isfinite(x_raw).all():
        raise ValueError("Bad or missing profile values")
    total = diag.total_pickups.to_numpy()
    if not np.allclose(x[total>0].sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Nonzero profile rows must sum to 1")
    if not np.allclose(x[total==0], 0):
        raise ValueError("Zero-demand rows must remain zero")
    eligible = total >= min_pickups
    if eligible.sum() < 10:
        raise ValueError("Too few eligible zones for RQ2 clustering")
    X = x[eligible]
    zone_ids = np.asarray(ids)[eligible]
    pca = PCA(svd_solver="full", random_state=42)
    scores = pca.fit_transform(X)  # PCA is on the normalized shape, not raw counts
    explained = pca.explained_variance_ratio_
    if not np.isfinite(explained).all():
        raise ValueError("PCA is undefined: all eligible profiles may be identical")
    cumulative = np.cumsum(explained)
    n_pc = int(np.searchsorted(cumulative, variance) + 1)
    reduced = scores[:, :n_pc]
    pca_table = pd.DataFrame({"component": np.arange(1, len(explained)+1),
                              "explained_variance_ratio": explained,
                              "cumulative_variance_ratio": cumulative})
    k_report = []
    models = {}
    for k in range(k_min, min(k_max, len(reduced)-1) + 1):
        if k < 2:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(reduced)
        labels = model.labels_
        sizes = np.bincount(labels, minlength=k)
        silhouette = float(silhouette_score(reduced, labels))
        k_report.append({"k": k, "silhouette": silhouette, "inertia": model.inertia_,
                         "min_cluster_size": int(sizes.min()),
                         "max_cluster_size": int(sizes.max()),
                         "passes_size_rule": bool(sizes.min() >= min_cluster_size)})
        models[k] = model
    report = pd.DataFrame(k_report)
    if report.empty:
        raise ValueError("No admissible K")
    candidates = report.loc[report.passes_size_rule]
    selection_rule = "highest silhouette with min cluster size >=5"
    if candidates.empty:
        candidates = report
        selection_rule = "highest silhouette (no K satisfied min size 5)"
    selected_k = int(candidates.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)
    report["selected"] = report.k.eq(selected_k)
    primary_model = models[selected_k]
    labels = primary_model.labels_
    assigned = pd.DataFrame({"zone_id": ids,
                             "total_pickups": total,
                             "cluster_id": pd.Series(pd.NA, index=range(len(ids)), dtype="Int64")})
    assigned.loc[eligible, "cluster_id"] = labels
    assigned["cluster_status"] = np.where(eligible, "clustered",
                                         np.where(total==0, "no_pickups", "low_volume_excluded"))
    projection = pd.DataFrame({"zone_id": zone_ids, "PC1": scores[:,0],
                               "PC2": scores[:,1] if scores.shape[1]>1 else 0,
                               "cluster_id": labels})
    shape = pd.DataFrame(X, columns=FEATURE_COLS)
    shape["cluster_id"] = labels
    raw_shape = pd.DataFrame(x_raw[eligible], columns=FEATURE_COLS)
    raw_shape["cluster_id"] = labels
    cluster_mean = shape.groupby("cluster_id")[FEATURE_COLS].mean()
    cluster_mean_raw = raw_shape.groupby("cluster_id")[FEATURE_COLS].mean()
    mean_long = []
    for c in sorted(cluster_mean.index):
        for d in range(7):
            for h in range(24):
                key = f"d{d}_h{h:02d}"
                mean_long.append({"cluster_id": int(c), "day_of_week": d, "hour": h,
                                  "mean_normalized_share": float(cluster_mean.loc[c, key]),
                                  "mean_raw_pickups": float(cluster_mean_raw.loc[c, key])})
    curves = pd.DataFrame(mean_long)
    summary = (assigned.loc[assigned.cluster_status.eq("clustered")]
               .groupby("cluster_id", as_index=False)
               .agg(n_zones=("zone_id", "size"),
                    median_annual_pickups=("total_pickups", "median"),
                    mean_annual_pickups=("total_pickups", "mean")))
    for c in sorted(cluster_mean.index):
        row = cluster_mean.loc[c]
        peak = row.idxmax()
        summary.loc[summary.cluster_id == c, "peak_day_of_week"] = int(peak[1])
        summary.loc[summary.cluster_id == c, "peak_hour"] = int(peak[4:])
        weekend_cols = [f"d{d}_h{h:02d}" for d in (5, 6) for h in range(24)]
        summary.loc[summary.cluster_id == c, "weekend_share"] = float(row[weekend_cols].sum())
    reps = pd.DataFrame({"zone_id": zone_ids, "cluster_id": labels,
                         "distance_to_centroid": np.linalg.norm(
                             reduced-primary_model.cluster_centers_[labels], axis=1)})
    reps = (reps.sort_values(["cluster_id", "distance_to_centroid"])
            .groupby("cluster_id", as_index=False).head(3).reset_index(drop=True))
    seed_labels = [KMeans(n_clusters=selected_k, n_init=20, random_state=seed)
                   .fit_predict(reduced) for seed in SEEDS]
    seed_table = pd.DataFrame({"seed": SEEDS,
                               "ari_vs_primary": [adjusted_rand_score(labels,l) for l in seed_labels]})
    seed_pairs = [adjusted_rand_score(a,b) for a,b in combinations(seed_labels, 2)]
    # Raw PCA + same K is an exploratory sensitivity check, not the main result.
    raw_pca = PCA(svd_solver="full")
    raw_scores = raw_pca.fit_transform(x_raw[eligible])
    raw_n_pc = int(np.searchsorted(np.cumsum(raw_pca.explained_variance_ratio_), variance) + 1)
    raw_reduced = raw_scores[:, :raw_n_pc]
    raw_labels = KMeans(n_clusters=selected_k, n_init=20, random_state=42).fit_predict(raw_reduced)
    raw_comparison = pd.DataFrame({"zone_id": zone_ids, "normalized_cluster_id": labels,
                                   "raw_cluster_id": raw_labels})
    metadata = {"chosen_k": selected_k, "k_selection_rule": selection_rule,
                "pca_n_components": n_pc, "pca_cumulative_variance": float(cumulative[n_pc-1]),
                "n_zones_clustered": int(eligible.sum()),
                "n_zones_unassigned": int((~eligible).sum()),
                "min_pickups": min_pickups,
                "seed_ari_pairwise_mean": float(np.mean(seed_pairs)),
                "seed_ari_pairwise_min": float(np.min(seed_pairs)),
                "raw_vs_normalized_ari": float(adjusted_rand_score(labels, raw_labels)),
                "interpretation": "Full-year descriptive 2025 clusters, NOT time-safe 2025 forecasts"}
    return ({"zone_temporal_clusters":assigned, "pca_explained_variance":pca_table,
             "pca_zone_scores":projection, "k_selection":report, "cluster_profiles":curves,
             "cluster_summary":summary, "cluster_representatives":reps,
             "kmeans_seed_stability":seed_table,
             "raw_vs_normalized_comparison":raw_comparison}, metadata)


def compare_rq1(rq2, community_file):
    """Compare the same zones; community IDs are arbitrary categorical labels."""
    if not Path(community_file).exists():
        return None
    communities = pd.read_csv(community_file)[["zone_id","community_id"]]
    df = rq2[["zone_id", "cluster_id"]].merge(communities,
             on="zone_id", how="inner", validate="one_to_one").dropna()
    if df.empty:
        return None
    cross = pd.crosstab(df.community_id, df.cluster_id).rename_axis(
        index="community_id", columns="cluster_id")
    long = cross.stack().rename("n_zones").reset_index()
    long["pct_within_community"] = long.n_zones / long.groupby("community_id").n_zones.transform("sum")
    metrics = {"n_shared_zones":len(df),
               "ARI":float(adjusted_rand_score(df.community_id,df.cluster_id)),
               "AMI":float(adjusted_mutual_info_score(df.community_id,df.cluster_id))}
    return long,metrics


def main(table_dir=OUTPUT_DIR, community_file=NETWORK_COMMUNITIES,
         lookup_file=LOOKUP_FILE, min_pickups=1, variance=0.90):
    table_dir = Path(table_dir)
    raw = pd.read_csv(table_dir/"zone_profiles_raw.csv")
    norm = pd.read_csv(table_dir/"zone_profiles_normalized.csv")
    diag = pd.read_csv(table_dir/"zone_profile_diagnostics.csv")
    tables, metadata = run_clustering(raw, norm, diag, min_pickups, variance)
    if Path(lookup_file).exists():
        lookup = pd.read_csv(lookup_file)[["LocationID", "Borough", "Zone"]]
        lookup = lookup.rename(columns={"LocationID":"zone_id"})
        for name in ("zone_temporal_clusters", "cluster_representatives"):
            tables[name] = tables[name].merge(lookup, on="zone_id", how="left", validate="one_to_one")
    comparison = compare_rq1(tables["zone_temporal_clusters"], community_file)
    if comparison:
        tables["community_cluster_crosstab"], metadata["rq1_rq2_agreement"] = comparison
    for name, table in tables.items():
        table.to_csv(table_dir/f"{name}.csv", index=False)
    (table_dir/"clustering_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Chosen K:", metadata["chosen_k"], "PCA components:", metadata["pca_n_components"])
    print("Silhouette candidate table:\n", tables["k_selection"].to_string(index=False))
    print("Cluster sizes:\n", tables["cluster_summary"].to_string(index=False))
    print("RQ1/RQ2 agreement:", metadata.get("rq1_rq2_agreement", "not available"))
    print("Saved:", table_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--community-file", type=Path, default=NETWORK_COMMUNITIES)
    ap.add_argument("--lookup-file", type=Path, default=LOOKUP_FILE)
    ap.add_argument("--min-pickups", type=int, default=1)
    ap.add_argument("--variance", type=float, default=0.90)
    args=ap.parse_args()
    main(args.table_dir,args.community_file,args.lookup_file,args.min_pickups,args.variance)
