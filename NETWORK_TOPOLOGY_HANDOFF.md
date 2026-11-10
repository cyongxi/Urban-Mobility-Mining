# SI 671 — RQ1 → RQ3: Historical Degree and OD-Flow HHI

**Author:** Yiwu Gu (Network & Pattern Mining). **Consumer:** Yongxi Chen (Forecasting & Evaluation).  
**Source:** The existing `feature/historical-network-features` branch.  
**Purpose:** Evaluate whether connectivity diversity and OD-flow concentration add predictive information beyond historical demand and OD strength.

## 1. What this patch changes (and what it does not)

- **Does not change** `src/network/network_features.py`, `src/network/build_od_network.py`, annual Louvain, or any RQ1 result.
- **Adds** `src/network/network_topology_features.py` (time-safe historical OD pair aggregation and rolling adjacency computation).
- **Updates** `examples/merge_network_into_forecasting.py` to accept `include_topology=True` while retaining the original default, `False`.
- **Adds** `tests/test_network_topology_features.py` with synthetic boundary and algebra tests.
- **Outputs locally (not in Git):**
  - `data/processed/network_features/historical_network_topology_24h.parquet` — new four metrics and availability flags.
  - `data/processed/network_features/historical_network_features_24h_augmented.parquet` — **all preexisting strength fields unchanged** plus the four new features and traffic-availability flags.

This is an extension to the current offline **rolling one-step-ahead forecasting** protocol, not proof of deployable real-time TLC data access.

## 2. Mathematical definitions

For target pickup hour `[t,t+1h)`, use only valid trips with **dropoff** in `[t-24h, t)` and a pickup within 2025. Let

`A_ij(t) = number of valid completed inter-zone trips from i to j over [t-24h,t)`, and `A_ii=0` for structural metrics.

Then:

- `out_degree_24h(i,t) = # {j != i : A_ij(t) > 0}`
- `in_degree_24h(i,t) = # {j != i : A_ji(t) > 0}`
- `out_hhi_24h(i,t) = sum_{j != i} [A_ij(t) / sum_{k != i} A_ik(t)]^2`
- `in_hhi_24h(i,t) = sum_{j != i} [A_ji(t) / sum_{k != i} A_ki(t)]^2`

**HHI denominator zero:** `NaN`, not zero or one, because concentration is undefined when no cross-zone trips exist. An explicitly named `has_interzone_in_flow_24h` / `has_interzone_out_flow_24h` field identifies this case. The corresponding degree is **0**.

Example: if one origin has three trips to zone B and one trip to zone C, its out-degree is 2 and out-HHI = `(3/4)^2 + (1/4)^2 = 0.625`. One destination only → HHI=1. A more even and diverse flow distribution → smaller HHI (for equal K destinations, HHI=1/K).

## 3. Time safety / matching contract

Use the exact same OD filter as the current historical strength pipeline:

```
valid_od IS TRUE
origin and destination in taxi_zone_lookup.csv except 264 and 265
2025-01-01 <= pickup_datetime < 2026-01-01
2025-01-01 <= dropoff_datetime < 2026-01-01
dropoff_datetime > pickup_datetime
```

Source timestamps are naive New York wall-clock times. The existing `dst_safe_24h` and forecasting's more conservative `dst_history_affected` are preserved, **not reconstructed into UTC**.

Forecast feature keys: `(zone_id, feature_as_of)`; require `feature_as_of == pickup_hour`. The rolling adjacency matrix is updated only with completed-dropoff-hour buckets **strictly before** `feature_as_of`. Early unavailable 30 hours remain null, as in `network_features.py`.

**Reconciliation guard:** for all full-history zone-hours,

- `sum_{j != i} A_ij(t)` MUST equal `interzone_out_strength_24h(i,t)`, and
- `sum_{j != i} A_ji(t)` MUST equal `interzone_in_strength_24h(i,t)`.

The script raises `AssertionError` if either fails. This is a stronger real-data test than just checking that the parquet file exists.

## 4. Recommended branch and execution order

If the working directory has uncommitted RQ2 changes, save them first. For example, after all work is committed:

```bash
git status
git fetch origin
git switch feature/historical-network-features
git switch -c feature/network-topology-enhancement
```

Unzip the patch in the **repository root**, keeping the existing `network_features.py`:

```bash
unzip -o ~/Downloads/SI671_Network_Topology_Enhancement.zip -d .
conda activate si671-urban
python -m pytest tests/test_network_topology_features.py -q
```

**First ensure the original file still exists:**

```bash
ls -lh data/processed/network_features/historical_network_features_24h.parquet
```

If missing, regenerate it with:

```bash
python -m src.network.network_features --window-hours 24 --threads 4
```

Then build the new topology features:

```bash
python -m src.network.network_topology_features --window-hours 24 --threads 4
```

The DuckDB process scans monthly cleaned Parquet files once and sorts hourly OD pairs. It streams batches to Python and maintains a ring buffer of 24 matrices (263 × 263), avoiding a huge 47M-row Pandas DataFrame. Sorting may use disk temp space.

**Source-code quick check:**

```bash
python - <<'PY'
import pandas as pd
p='data/processed/network_features/historical_network_features_24h_augmented.parquet'
df=pd.read_parquet(p)
print(df.shape,df.zone_id.nunique(),df.feature_as_of.min(),df.feature_as_of.max())
assert len(df)==263*8760
assert not df.duplicated(['zone_id','feature_as_of']).any()
valid=df.has_full_history_24h & df.dst_safe_24h
for column in ['in_degree_24h','out_degree_24h']:
    assert df.loc[valid,column].between(0,262).all()
for hhi,flag in [('in_hhi_24h','has_interzone_in_flow_24h'),
                 ('out_hhi_24h','has_interzone_out_flow_24h')]:
    assert df.loc[valid&df[flag],hhi].between(0,1).all()
    assert df.loc[valid&~df[flag],hhi].isna().all()
print('PASS: shape, keys, coverage, degree bounds, HHI bounds and zero flow')
PY
```

The topology generator prints `PASS: ... historical inter-zone strengths exactly reconciled` only after confirming *every* full-history zone-hour's interzone counts match the original strength file.

## 5. RQ3 integration / ablation

Yongxi can load the augmented file through the backward-compatible helper:

```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from examples.merge_network_into_forecasting import merge_history
common_df, network_cols = merge_history(model_df, PROJECT_ROOT, include_topology=True)

strength_cols = ['in_strength_24h', 'out_strength_24h',
                 'interzone_in_strength_24h', 'interzone_out_strength_24h']
topology_cols = ['in_degree_24h', 'out_degree_24h',
                 'in_hhi_24h', 'out_hhi_24h']

m1_features = temporal_cols
m2_features = temporal_cols + strength_cols
m3_features = temporal_cols + strength_cols + topology_cols
```

**Important:** retrain/refit **M1/M2/M3 on `common_df`**, keeping identical chronological split, preprocessing, tuning/evaluation protocols and selected rows. LightGBM accepts missing HHI; DO NOT globally `dropna()` and accidentally change the comparison population. The boolean traffic-availability flags are included in the output and can be tested as an additional control, but keep the four numeric columns as the requested primary experiment.

Suggested evaluation table: test MAE, RMSE, absolute/relative M3−M2 gains, per-zone improvements, and feature importance. A small or negative M3 gain is a valid result. New degree is correlated with demand, too; inspect within-zone correlation with historical lag/strength and avoid causal interpretations.

## 6. GitHub commit

```bash
python -m pytest tests/test_network_topology_features.py -q
git add src/network/network_topology_features.py \
        tests/test_network_topology_features.py \
        examples/merge_network_into_forecasting.py \
        NETWORK_TOPOLOGY_HANDOFF.md
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add time-safe historical OD degree and HHI features"
git push -u origin feature/network-topology-enhancement
```

Do not commit raw or cleaned trips, `.parquet` outputs, or extracted shapefiles. The feature-generating script + tests + schema document is the reproducible handoff.

## 7. Interpretation and limitations

- Strength is largely a historical traffic-volume metric; Degree counts distinct cross-zone neighbors, HHI measures traffic concentration. Their incremental predictive value is an empirical question.
- Full-year annual static Louvain and static HHI are not included in the 2025 historical prediction experiments.
- Source TLC data are published with delay. The experiment assumes that *completed* trip details are instantly available in the **offline replay**, not in actual live TLC data.
- The ambiguous DST repeated hour is not reconstructed; always use an identical DST-safe evaluation population.
- Per-window HHI is *undefined* when total interzone flow is zero; this is a missing structural quantity, not a measurement failure.
