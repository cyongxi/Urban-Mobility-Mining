"""Non-destructive RQ2 sensitivity to sparse-zone filtering.

Run from repository root after building the original RQ2 168D profiles:
    python -m src.temporal.rq2_sparsity_sensitivity

It never rewrites the primary `results/temporal/tables/` outputs.
This analysis is *exploratory*: choose the final threshold using stability,
coverage, cluster sizes and interpretability, not only maximum silhouette.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .build_profiles import OUTPUT_DIR, ROOT
from .cluster_profiles import NETWORK_COMMUNITIES, run_clustering, compare_rq1

OUT = ROOT / "results/temporal/sensitivity"


def run_sensitivity(table_dir=OUTPUT_DIR, output_dir=OUT,
                    thresholds=(1, 100, 500, 1000, 5000), variance=0.90):
    table_dir, output_dir = Path(table_dir), Path(output_dir)
    raw = pd.read_csv(table_dir / "zone_profiles_raw.csv")
    norm = pd.read_csv(table_dir / "zone_profiles_normalized.csv")
    diag = pd.read_csv(table_dir / "zone_profile_diagnostics.csv")
    if not {"zone_id", "total_pickups"}.issubset(diag):
        raise ValueError("Missing zone_id or total_pickups in diagnostics")
    if not thresholds or any(t < 1 for t in thresholds):
        raise ValueError("Thresholds must be positive integers")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for threshold in thresholds:
        folder = output_dir / f"min{threshold}"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== RQ2 sensitivity, min_pickups={threshold} ===")
        tables, meta = run_clustering(raw, norm, diag,
                                      min_pickups=int(threshold), variance=variance)
        if Path(NETWORK_COMMUNITIES).exists():
            comparison = compare_rq1(tables["zone_temporal_clusters"], NETWORK_COMMUNITIES)
            if comparison is not None:
                tables["community_cluster_crosstab"], meta["rq1_rq2_agreement"] = comparison
        for name, tab in tables.items():
            tab.to_csv(folder / f"{name}.csv", index=False)
        k_result = tables["k_selection"].loc[lambda t: t.selected].iloc[0]
        valid_cluster_sizes = bool(k_result.passes_size_rule)
        meta["passes_min_cluster_size_5"] = valid_cluster_sizes
        meta["sensitivity_threshold"] = int(threshold)
        (folder / "clustering_metadata.json").write_text(json.dumps(meta, indent=2))
        record = {
            "min_pickups": int(threshold),
            "n_clustered": meta["n_zones_clustered"],
            "n_unassigned": meta["n_zones_unassigned"],
            "chosen_k": meta["chosen_k"],
            "pca_n_components": meta["pca_n_components"],
            "silhouette": float(k_result.silhouette),
            "smallest_cluster": int(k_result.min_cluster_size),
            "largest_cluster": int(k_result.max_cluster_size),
            "passes_min_size_5": valid_cluster_sizes,
            "mean_pairwise_seed_ari": meta["seed_ari_pairwise_mean"],
            "raw_vs_normalized_ari": meta["raw_vs_normalized_ari"],
        }
        records.append(record)
        status = "OK" if valid_cluster_sizes else "REVIEW SINGLETON/TINY CLUSTERS"
        print(" ", status, ":", record)

    report = pd.DataFrame(records)
    report.to_csv(output_dir / "rq2_sparsity_sensitivity_summary.csv", index=False)
    print("\nSaved sensitivity summary:", output_dir / "rq2_sparsity_sensitivity_summary.csv")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--thresholds", nargs="+", type=int,
                   default=[1, 100, 500, 1000, 5000])
    p.add_argument("--variance", type=float, default=0.90)
    args = p.parse_args()
    run_sensitivity(args.table_dir, args.output_dir, args.thresholds, args.variance)
