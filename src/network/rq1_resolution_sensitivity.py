"""Non-destructive RQ1 robustness check.

Run from repo root:
    python -m src.network.rq1_resolution_sensitivity

Reads existing OD aggregates, never calls `analyze()` (which overwrites RQ1 CSVs).
"""
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .community_detection import build_graphs, louvain_partition

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/network/tables"
FIGURES = ROOT / "results/network/figures"
RESOLUTIONS = (0.5, 1.0, 1.5, 2.0)
SEEDS = (42, 43, 44, 45, 46)


def compute_sensitivity(edges, resolutions=RESOLUTIONS, seeds=SEEDS):
    _, H = build_graphs(edges)
    node_ids = sorted(H.nodes)
    baseline, _ = louvain_partition(H, seed=42, resolution=1.0)
    rows = []
    for gamma in resolutions:
        per_seed = {s:louvain_partition(H, seed=s, resolution=gamma)[0] for s in seeds}
        labels = per_seed[seeds[0]]
        communities = [set(n for n,v in labels.items() if v==c)
                       for c in sorted(set(labels.values()))]
        # Gamma-adjusted Q values are optimized for each gamma;
        # compare Q evaluated at a FIXED gamma=1 for an apples-to-apples diagnostic.
        modularity_fixed = nx.community.modularity(H, communities, weight="weight", resolution=1.0)
        q_gamma = nx.community.modularity(H, communities, weight="weight", resolution=gamma)
        sizes = sorted((len(c) for c in communities), reverse=True)
        comparisons = [adjusted_rand_score([a[z] for z in node_ids], [b[z] for z in node_ids])
                       for a,b in combinations(per_seed.values(), 2)]
        rows.append({"resolution":gamma, "n_communities":len(communities),
                     "largest_community_n":sizes[0],
                     "largest_community_share":sizes[0]/len(node_ids),
                     "modularity_at_own_gamma":q_gamma,
                     "modularity_at_gamma_1":modularity_fixed,
                     "ari_vs_gamma_1":adjusted_rand_score(
                         [baseline[z] for z in node_ids], [labels[z] for z in node_ids]),
                     "mean_pairwise_seed_ari":float(np.mean(comparisons)),
                     "min_pairwise_seed_ari":float(np.min(comparisons))})
    return pd.DataFrame(rows)


def main(table_dir=TABLES, fig_dir=FIGURES):
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    for period in ("annual", "weekday", "weekend"):
        edges = pd.read_csv(table_dir/f"od_{period}_all.csv")
        report = compute_sensitivity(edges)
        report.insert(0, "period", period)
        report.to_csv(table_dir/f"rq1_resolution_sensitivity_{period}.csv", index=False)
        print(period, "\n", report.to_string(index=False))
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(report.resolution, report.n_communities, marker="o")
        ax.set(xlabel="Louvain resolution gamma", ylabel="Number of communities",
               title=f"RQ1 {period}: Louvain resolution sensitivity")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir/f"rq1_resolution_sensitivity_{period}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
