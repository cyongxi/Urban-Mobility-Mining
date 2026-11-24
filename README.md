
# Urban Mobility Mining

SI 671 Data Mining — Fall 2026

## Project Overview

This project analyzes 2025 NYC Yellow Taxi trips to
study spatial mobility communities, temporal demand
patterns, and the incremental value of historical
network features for hourly demand forecasting.

## Research Questions

- RQ1: Mobility Communities
- RQ2: Temporal Demand Profiles
- RQ3: Incremental Predictive Value
- RQ4: Heterogeneity of Predictive Gains

## Data

Source: NYC Taxi and Limousine Commission (TLC).

The analysis uses 2025 Yellow Taxi trip records and
the official TLC taxi-zone lookup.

Raw and cleaned Parquet files are kept locally.

## RQ1: Network Mining

Run from the repository root:

```bash
python -m src.network.build_od_network
python -m src.network.community_detection --resolution 1.0
python -m src.network.plot_network
```

Main results:

- `results/network/tables/`
- `results/network/figures/`

The main Louvain resolution is gamma = 1.0.
Additional resolution sensitivity results are retained
as robustness diagnostics.

## RQ2: Temporal Pattern Mining

```bash
python -m src.temporal.build_profiles --threads 4
python -m src.temporal.cluster_profiles \
  --min-pickups 1000 --variance 0.90
python -m src.temporal.plot_results
```

Main results:

- `results/temporal/tables/`
- `results/temporal/figures/`

The primary clustering uses a minimum of 1,000 annual
pickups per zone. Alternative thresholds are available
under `results/temporal/sensitivity/`.

## Historical Network Features for RQ3

```bash
python -m src.network.network_features \
  --window-hours 24 --threads 4

python -m src.network.network_topology_features \
  --window-hours 24 --threads 4
```

Features are aligned by `zone_id` and `feature_as_of`.
For target hour t, only trips completed before t
contribute to historical OD features.

Generated Parquet files are stored locally under
`data/processed/network_features/`.

## Forecasting and Evaluation (RQ3–RQ4)

The final forecasting and heterogeneity analysis is in:

- `notebooks/09_final_rq3_rq4.ipynb`
- `README_RQ3_RQ4.md`

Earlier notebooks (05–08) document model development,
diagnostics, feature ablation, and exploratory analyses.

The final RQ4 analysis uses the RQ2 min1000 clusters
for retrospective subgroup evaluation.

RQ1/RQ2 full-year descriptive labels are not used
as forecasting predictors.

## Reproducibility

Run the analysis in the `si671-urban` Python environment.
Required inputs include the 12 cleaned monthly Parquet
files and the official taxi-zone lookup.

Run tests with:

```bash
python -m pytest tests/ -q
```

## Limitations

The analysis covers Yellow Taxi trips rather than
all urban mobility. Full-year clustering outputs are
retrospective and must not be used as historical
predictors for 2025 forecasting.
