# RQ1 / RQ2 — Mobility Communities and Temporal Demand Profiles

**Owner:** Yiwu Gu. **Downstream use:** historical OD features for RQ3; RQ1 communities and RQ2 clusters as evaluation groupings in RQ4.

All commands run from the repository root in the `si671-urban` environment (`environment.yml`).

## Required local inputs (not in Git)

- `data/processed/cleaned_trips/cleaned_2025-01.parquet` … `cleaned_2025-12.parquet`
- `data/external/taxi_zone_lookup.csv`
- `data/external/**/taxi_zones.shp` (optional; only needed for the zone maps)

## Tests

```bash
python -m pytest tests/test_network_rq1.py tests/test_rq2_profiles.py -q
```

## RQ1 — Weighted OD networks and Louvain communities

```bash
python -m src.network.build_od_network
python -m src.network.community_detection --resolution 1.0
python -m src.network.plot_network
python -m src.network.rq1_resolution_sensitivity   # optional robustness
```

**Design.** Directed graphs G_t = (V, E_t, W_t) for three predefined periods (annual, weekday, weekend), with w_ij,t = quality-filtered trips from zone i to zone j. Zones 264/265 are excluded. Network statistics (weighted in/out strength, density, top-k and top-1% edge shares, edge HHI) use the directed graph. For Louvain, flows are symmetrized as w_ij + w_ji and self-loops are dropped; the main resolution is γ = 1.0.

**Robustness.** Louvain seed stability (5 seeds); resolution sensitivity for γ ∈ {0.5, 1, 1.5, 2}, reporting both Q at the chosen γ and Q re-evaluated at γ = 1 so values are comparable. Sensitivity runs write separate files and never overwrite the main outputs.

**Key results** (`results/network/tables/`):

| Period | Interzone trips | Communities | Modularity | Within-community share | Top-1% edge share |
|---|---|---|---|---|---|
| Annual | 45.6M | 3 | 0.220 | 61.4% | 47.1% |
| Weekday | 32.5M | 3 | 0.216 | 61.4% | 48.3% |
| Weekend | 13.1M | 3 | 0.225 | 61.3% | 40.6% |

Weekday and weekend partitions are nearly identical (ARI = 0.97, 2 of 260 zones change), while weekend flows are less concentrated. Annual communities: two Manhattan communities (34 and 33 zones) and one large outer-borough community (194 zones).

Main files: `network_metrics.csv`, `communities_*.csv`, `community_summary_*.csv`, `community_flow_matrix_*.csv`, `major_od_flows_*.csv`, `zone_metrics_*.csv`, `temporal_community_ari.csv`, `louvain_seed_stability.csv`, and figures in `results/network/figures/`.

## RQ2 — 168-dimensional weekday-hour profiles

```bash
python -m src.temporal.build_profiles --threads 4
python -m src.temporal.cluster_profiles --min-pickups 1000 --variance 0.90
python -m src.temporal.plot_results
python -m src.temporal.rq2_sparsity_sensitivity     # optional robustness
```

**Profile construction.** Pickup-valid cleaned trips (no `valid_od` filter), 263 TLC geographic zones. The two DST switch dates (2025-03-09, 2025-11-02) are excluded entirely. `raw[zone, weekday, hour] = count / number of retained calendar days of that weekday` (Monday = 0). The normalized profile divides each zone's 168 raw values by their sum, so clustering captures temporal *shape* rather than volume.

**Clustering.** PCA on normalized profiles, keeping the fewest components reaching ≥ 90% variance; K-means for K = 2…8 evaluated by silhouette in PCA space. K is chosen as the highest silhouette among solutions whose smallest cluster has ≥ 5 zones (an interpretability heuristic). Zones with < 1,000 annual pickups are left unassigned. Diagnostics: seed stability (ARI over seeds 42–46), raw-vs-normalized clustering ARI, and RQ1–RQ2 agreement (ARI/AMI).

**Key results** (`results/temporal/tables/`):

- 14 principal components (90.2% variance); K = 3, silhouette 0.299; seed ARI = 1.0.
- 227 zones clustered, 36 low-volume zones unassigned.

| Cluster | Zones | Median annual pickups | Peak | Weekend share | Label |
|---|---|---|---|---|---|
| C0 | 128 | 9,574 | Tue 07:00 | 30.7% | Commuter (weekday AM peak) |
| C1 | 79 | 244,266 | Sat 23:00 | 30.9% | Core activity (afternoon–evening) |
| C2 | 20 | 69,122 | Sun 00:00 | 47.5% | Nightlife (weekend late night) |

- Raw vs normalized ARI = 0.22: level and shape give different groupings.
- RQ1 vs RQ2 ARI = 0.33, AMI = 0.27: spatial-flow communities and temporal-shape clusters capture different notions of similarity.

**Sensitivity.** Alternative volume thresholds (1, 100, 500, 1000, 5000 pickups) are under `results/temporal/sensitivity/`; min1000 is the primary specification.

## Historical network features for RQ3

```bash
python -m src.network.network_features --window-hours 24 --threads 4
python -m src.network.network_topology_features --window-hours 24 --threads 4
```

Features are keyed by `zone_id` and `feature_as_of`; for target hour t only trips completed before t contribute. See `NETWORK_FEATURE_HANDOFF.md` and `NETWORK_TOPOLOGY_HANDOFF.md`.

## Leakage note

RQ1 communities and RQ2 clusters are full-year 2025 descriptive outputs. They are used in RQ4 only to slice evaluation rows and are never model inputs; using them as predictors in a 2025 forecasting experiment would leak future information.
