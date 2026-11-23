# SI 671 — RQ1 → RQ3 Historical Network Features

**Owner:** Yiwu (OD feature extraction and correctness). **Consumer:** Yongxi (zone-hour forecasting, model comparison and evaluation).

## Purpose

The existing `results/network/tables/static_network_features_2025.csv` is a **retrospective full-year summary** for RQ1 and a merge prototype only. Do not use it as a 2025 validation/test model feature: it uses future data.

The new `src/network/network_features.py` creates a 2025 hourly, time-indexed **completed-OD** feature table. This is not a dynamic-Louvain implementation; it captures historical directed network-strength signals.

## Inputs and keys

- `data/processed/cleaned_trips/cleaned_2025-01.parquet` ... `cleaned_2025-12.parquet` (local; already created by cleaning pipeline)
- `data/external/taxi_zone_lookup.csv` (local)
- Filter `valid_od = TRUE`, both taxi zones in lookup, excluding 264 and 265, pickup in 2025 and dropoff in 2025. The source cleaning pipeline defines `valid_od` using valid duration, distance, and destination zone.
- **Feature key:** `(zone_id, feature_as_of)`; one row for every one of 263 taxi zones and 8,760 nominal 2025 wall-clock hours (2,303,880 rows).
- **Forecast key:** `(zone_id, pickup_hour)`; `feature_as_of == pickup_hour`.
- **Target:** pickups during `[pickup_hour, pickup_hour+1h)`.

## Strict time boundary

At forecast origin `t`, source trips must have `t-24h <= tpep_dropoff_datetime < t` (local wall clock outside DST caveat). Trips still in progress at `t` must not be used. OD inputs reflect the offline assumption of instant access to *completed-trip* data; public TLC dataset release is not real time.

**Interpretation:** `out_strength_24h` counts trips *completed* in the previous day **whose origin is zone i**. It is not identical to pickups in that previous day.

## Output

`data/processed/network_features/historical_network_features_24h.parquet` (local, gitignored).

| Column | Meaning |
|---|---|
| `zone_id` | TLC taxi zone ID (excluding 264/265) |
| `feature_as_of` | Hour start of prediction target; excludes trips completed in this hour |
| `in_strength_24h` | Completed OD trips reaching this zone in `[t-24,t)` including loops |
| `out_strength_24h` | Completed OD trips originating here in `[t-24,t)` including loops |
| `interzone_in_strength_24h` | Same, inter-zone arrivals only |
| `interzone_out_strength_24h` | Same, inter-zone origins only |
| `has_full_history_24h` | False for the first **30 hours** of 2025 (24h rolling window + up to 6h carryover of eligible trips that began in 2024, which are absent from 2025 pickup files); these values are null, not artificial zeros |
| `dst_safe_24h` | False if the forecast time is a DST transition wall-clock hour or the previous 24h may intersect one |
| `source_clock` | Explicit naive wall-clock time convention |

The 2025 source timestamps do not have UTC offsets. The duplicate fall-back hour cannot be reconstructed from these files. Align with Yongxi's **more conservative** existing 168-hour DST exclusion in `05_forecasting.ipynb` for M1/M2; do not quietly assume timezone-aware semantics.

## How to run (from repository root)

```bash
conda activate si671-urban
python -m pip install duckdb pandas pyarrow pytest
python -m pytest tests/test_network_features.py -q
python -m src.network.network_features --window-hours 24 --threads 4
```

Check locally:

```bash
python - <<'PY'
import pandas as pd
p = pd.read_parquet('data/processed/network_features/historical_network_features_24h.parquet')
print(p.shape, p.zone_id.nunique())
print(p[['zone_id','feature_as_of','in_strength_24h','out_strength_24h']].head(30))
assert len(p) == 263 * 8760
assert p[['zone_id','feature_as_of']].duplicated().sum() == 0
assert p.loc[p.has_full_history_24h, ['in_strength_24h','out_strength_24h']].notna().all().all()
PY
```

## M1 vs M2 rules

1. Merge by zone + hour using `examples/merge_network_into_forecasting.py`.
2. First fix `05_forecasting.ipynb`: after creating the DST-excluded `model_df`, a later cell **recreates** it from `forecast_df` without the exclusion. Preserve one canonical modeling population.
3. Use the **same** selected observations for M1 and M2 (including DST exclusions), the same chronological split dates, and consistent tuning/evaluation.
4. M1: lag/rolling/calendar + categorical zone ID. M2: M1 + historical OD features. MAE primary, RMSE secondary.
5. Initially compare **M1 + in/out** and **M1 + all four directional/inter-zone strengths**. All four are traffic-volume summaries; they are not independent network-topology features.
6. If later studying genuine topology, add time-safe neighbor-count/edge-diversity features, calculated only from completed OD trips in the history window.

## Pitfalls

- Do **not** substitute the Annual Louvain `community_id` from the full-year table: it leaks future data.
- Do **not** group history by pickup timestamp and use destinations for unfinished rides.
- Missing `in_strength_24h` during the first 30 hours means **no complete source history**, not zero completed trips. The extra 6h accounts for eligible trips begun before Jan 1 that could be completed after midnight.
- Historical OD may be less complete than the pickup-valid forecast target by design. This is not an error.
- In a live system, having an observation with dropoff < t is not the same as operational data being delivered by t; document the offline availability assumption.
