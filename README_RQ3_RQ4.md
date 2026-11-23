# RQ3 / RQ4 — Forecasting and Heterogeneity

**Owner:** Yongxi Chen. **Inputs from:** Yiwu Gu (historical OD features, RQ1 communities, RQ2 clusters).

## Final notebook

`notebooks/09_final_rq3_rq4.ipynb` produces every RQ3/RQ4 number in the report. Earlier notebooks are kept as the development record:

| Notebook | Role |
| --- | --- |
| `05_forecasting_revised.ipynb` | First locked pipeline (9-feature M1, 3,000-round cap). Superseded. |
| `06_forecasting_model_diagnostics.ipynb` | Validation-only choice of temporal features and Poisson objective. |
| `07_enhanced_network_ablation.ipynb` | Validation-only M1′–M4′ ablation; all models hit the 6,000-round cap. |
| `08_forecasting_heterogeneity.ipynb` | Validation-only exploratory RQ4. |
| **`09_final_rq3_rq4.ipynb`** | **Final:** converged M1′–M4′, Test scored once, RQ4 with RQ1/RQ2 groupings. |
| `archive/05_forecasting_original.ipynb` | Original exploratory run (contained a DST-population bug fixed in 05_revised). |

## Design

- **Target:** Yellow Taxi pickups per zone-hour, 263 TLC zones, rolling one-hour-ahead.
- **Split:** Train Jan–Aug 2025, Validation Sep–Oct (early stopping only), Test Nov–Dec (scored once).
- **Sample:** 2,170,802 zone-hours after excluding DST-affected history and the first 30 hours without complete OD history. Identical rows for every model.
- **Models:** Poisson LightGBM, learning rate 0.05, 31 leaves, cap 20,000 rounds, early stopping after 200 rounds on validation MAE.
  - M1′ temporal: lags 1/2/3/24/168, rolling means 3/6/24/168, 3-vs-24 trend, hour, day of week, weekend, zone.
  - M2′ = M1′ + four 24h OD strengths. M3′ = M2′ + in/out degree. M4′ = M3′ + in/out HHI and flow flags.
- **Baselines:** daily (lag 24) and weekly (lag 168) seasonal naive.
- **Uncertainty:** paired bootstrap over calendar days (2,000 draws).
- **RQ4 groupings:** training-period demand tertiles, borough, RQ1 annual Louvain community, RQ2 min1000 temporal cluster, hour, weekday/weekend; validation-vs-test sign agreement for every subgroup.

RQ1/RQ2 labels are full-year descriptive groupings. They are used only to slice evaluation rows, never as model inputs, so they cannot leak future information into forecasts.

## Run

```bash
conda activate si671-urban
python -m pytest tests/test_forecasting_evaluation.py -q
jupyter lab notebooks/09_final_rq3_rq4.ipynb   # run all; ~20–40 min on an M-series Mac
```

Required local inputs (not in Git): `data/processed/cleaned_trips/cleaned_2025-MM.parquet`, `data/external/taxi_zone_lookup.csv`, and `data/processed/network_features/historical_network_features_24h_augmented.parquet` (from `python -m src.network.network_topology_features`).

## Outputs

`results/forecasting/final_rq3_rq4/`: metrics, bootstrap and subgroup CSVs, `figures/`, `experiment_manifest.json`, and `summary_for_report.md` with the key numbers generated from the run. Commit the CSV, PNG, JSON and MD files; prediction `.parquet` files are git-ignored.

Shared evaluation code lives in `src/forecasting/evaluation.py` (tested by `tests/test_forecasting_evaluation.py`).
