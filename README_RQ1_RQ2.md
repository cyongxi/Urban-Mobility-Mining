# SI 671 — Yiwu RQ1 + RQ2 continuation (2025 NYC Yellow Taxi)

This patch is meant to be **unzipped at the existing `Urban-Mobility-Mining/` repo root**. It adds new files; it does **not** replace existing RQ1 or RQ3 code.

## What was confirmed in GitHub

`feature/network-topology-enhancement` contains the completed RQ1 annual/weekday/weekend OD graph, Louvain outputs, static features and time-safe historical features. No remote RQ2 implementation was visible as of 2026-11-19. Main branch is not yet fully integrated. The source of truth is the final submitted proposal: RQ2 uses `7*24=168` zone-specific weekday-hour pickup profiles, raw/normalized comparisons, PCA, K-means, silhouette and interpretable cluster means.

## 0. New branch from the existing RQ1/network topology branch

```bash
conda activate si671-urban
git status --short
git fetch origin
git switch feature/network-topology-enhancement
git pull --ff-only origin feature/network-topology-enhancement
git switch -c feature/temporal-clustering
```

**Stop if you have uncommitted changes** and commit/stash them first; if `feature/temporal-clustering` already exists locally, switch to it instead of creating it. The RQ2 branch will contain RQ1 as an ancestor. Submit a PR against the topology branch first, or first merge the topology branch to main and then retarget this RQ2 PR to main to avoid a confusing diff.

Unzip the distribution archive at repo root (`-d .`), preserving paths.

## 1. Check data & run tests

Inputs expected (from your existing local files):

- `data/processed/cleaned_trips/cleaned_2025-01.parquet` through `cleaned_2025-12.parquet`
- `data/external/taxi_zone_lookup.csv`
- `results/network/tables/od_annual_all.csv`, `od_weekday_all.csv`, `od_weekend_all.csv`, `communities_annual.csv` (from RQ1)
- `data/external/**/taxi_zones.shp` (optional, for geographic map)

```bash
python -m pip install duckdb pyarrow pandas numpy scipy scikit-learn matplotlib networkx geopandas pytest
python -m pytest tests/test_rq2_profiles.py tests/test_network_rq1.py -q
```

## 2. Build the 168D profiles

```bash
python -m src.temporal.build_profiles --threads 4
```

**Input definition:** 2025 Pickup-valid cleaned data; do **not** filter `valid_od`. Use the 263 TLC geographic zones (exclude 264 and 265). For ambiguous naive NYC wall-clock timestamps, exclude the *two entire DST switch dates* (March 9 and Nov 2, 2025) and divide each weekday-hour count by the number of retained calendar days of that weekday (not necessarily 52). `Monday=0` and `Sunday=6`.

**Raw formula:** `raw[zone,weekday,hour] = count / (# retained dates of that weekday)`.

**Normalized shape:** each zone row divided by the sum of its 168 raw values. If zero pickups, the row remains all zero and the zone is marked `no_pickups`; it is not assigned an artificial cluster. This row normalization removes *level*, so main clusters concern *temporal shape*.

Outputs in `results/temporal/tables/`:

- `zone_weekday_hour_counts.csv`: 263×168 = 44,184 zone/weekday/hour rows.
- `zone_profiles_raw.csv`: 263 rows with `zone_id` + 168 hourly columns.
- `zone_profiles_normalized.csv`: same shape, row sums = 1 for nonzero zones.
- `zone_profile_diagnostics.csv`: included pickups, annual volume and sparsity.
- `profile_metadata.json`: filters, DST, denominator, totals.

Check:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path('results/temporal/tables')
x=pd.read_csv(p/'zone_profiles_raw.csv')
y=pd.read_csv(p/'zone_profiles_normalized.csv')
z=pd.read_csv(p/'zone_weekday_hour_counts.csv')
assert x.shape == (263,169) and y.shape == (263,169)
assert len(z) == 263*168
assert x.zone_id.is_unique and y.zone_id.is_unique
assert not z.duplicated(['zone_id','day_of_week','hour']).any()
sums=y.drop(columns='zone_id').sum(axis=1)
assert (sums.sub(1).abs().lt(1e-7)|sums.eq(0)).all()
print('PASS profile integrity; included pickups:',int(z.pickup_count.sum()))
PY
```

## 3. PCA + K-means (main RQ2 results)

```bash
python -m src.temporal.cluster_profiles --min-pickups 1000 --variance 0.90
```

The method fits PCA on **row-normalized 168D shapes**, retains the minimum number of PCs with >=90% explained variance, then evaluates K-means K=2,...,8 using Silhouette **in the retained PCA space**. It prefers K with no clusters smaller than 5 zones; if none, uses highest silhouette and documents this. This is a heuristic for interpretability, not an official proposal requirement. It also measures K-means seed stability using ARI for seeds 42,...,46; outputs a raw-profile clustering as sensitivity; and compares annual RQ1 communities with RQ2 clusters using ARI/AMI and a cross-tab.

Main artifacts in `results/temporal/tables/`:

- `k_selection.csv`: candidate K, silhouette, cluster size and selected indicator.
- `pca_explained_variance.csv`: PCA cumulative variance.
- `zone_temporal_clusters.csv`: 263 zones, nullable cluster IDs for zero/low-volume zones, plus names.
- `cluster_profiles.csv`: average normalized 168D profiles, plus average raw pickups.
- `cluster_summary.csv`: sizes, median volume, peak weekday/hour and weekend share.
- `cluster_representatives.csv`: 3 zones closest to each centroid (look up zone names).
- `kmeans_seed_stability.csv`, `raw_vs_normalized_comparison.csv`.
- `community_cluster_crosstab.csv`: if RQ1 communities exist.
- `clustering_metadata.json`: K, retained PC count, ARI/AMI, thresholds, etc.

**No leakage claim needed for RQ2 as this is retrospective, but it DOES become future leakage if 2025 full-year cluster IDs are fed into a 2025 historical forecasting experiment.** Keep RQ2 labels descriptive only.

## 4. Make final figures

```bash
python -m src.temporal.plot_results
```

Outputs to `results/temporal/figures/`: silhouette curve, PCA cumulative variance/scatter, 168D cluster heatmaps, weekday/weekend curves, cluster sizes, RQ1/RQ2 cross-tab plot, and taxi-zone cluster map if the official `.shp` file is found. The script searches recursively under `data/external/`, supporting your existing nested `taxi_zones/taxi_zones` structure.

**How to interpret:** use `cluster_summary.csv` with `rq2_cluster_168d_heatmaps.png`; describe actual peak timing and weekday/weekend distribution. Use `cluster_representatives.csv` to name representative real zones. Discuss whether spatial RQ1 communities and behavioral RQ2 clusters coincide; a low ARI/AMI is a meaningful result (different notions of similarity).

## 5. Optional RQ1 robustness without overwriting original Louvain outputs

```bash
python -m src.network.rq1_resolution_sensitivity
```

For annual/weekday/weekend OD aggregates, it tests `gamma ∈ {0.5,1,1.5,2}` with five seeds and saves separate `rq1_resolution_sensitivity_*.csv` and figures. It never overwrites existing `communities_annual.csv` etc. Do not compare different gamma-adjusted modularity values as if optimized under one objective: the output includes both Q evaluated at each chosen gamma and Q re-evaluated at a fixed gamma=1 for comparability.

## 6. Optional sensitivity: omit extremely sparse zones

**Do NOT overwrite the main result**. Copy just the input profiles to a separate folder:

```bash
mkdir -p results/temporal/sensitivity_min100
cp results/temporal/tables/zone_profiles_{raw,normalized}.csv results/temporal/sensitivity_min100/
cp results/temporal/tables/zone_profile_diagnostics.csv results/temporal/sensitivity_min100/
python -m src.temporal.cluster_profiles \
  --table-dir results/temporal/sensitivity_min100 --min-pickups 100
```

Compare chosen K, cluster sizes and seed ARI. Threshold 100 is exploratory, not theoretically optimal.

## 7. Git and PR

```bash
python -m pytest tests/test_rq2_profiles.py tests/test_network_rq1.py -q
git status --short
git add src/temporal/ src/network/rq1_resolution_sensitivity.py \
        tests/test_rq2_profiles.py notebooks/03_temporal_clustering.ipynb \
        README_RQ1_RQ2.md results/temporal/
# Only if optional RQ1 sensitivity was run:
# git add results/network/tables/rq1_resolution_sensitivity_*.csv
# git add results/network/figures/rq1_resolution_sensitivity_*.png
git diff --cached --stat
git diff --cached --name-only
git commit -m 'Complete RQ2 weekday-hour PCA clustering and RQ1 sensitivity'
git push -u origin feature/temporal-clustering
```

Double-check that staged changes do not include `.parquet`, raw trip data, extracted NYC shapefile or machine-specific paths. In the final report report sampling rules, normalized-vs-raw rationale, PCA retained variance, K/silhouette, cluster shapes/names, community comparisons and limitations.
