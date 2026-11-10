"""RQ3 integration helper. Run with `model_df` from 05_forecasting.ipynb.

Usage in Notebook (from repo root or notebooks/):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from examples.merge_network_into_forecasting import merge_history
    common_df, network_cols = merge_history(model_df, PROJECT_ROOT)

Use `common_df` as the SAME comparison rows for M1 and M2.
"""
from pathlib import Path
import pandas as pd


def merge_history(model_df: pd.DataFrame, project_root: Path):
    path = (Path(project_root) / "data/processed/network_features/"
            "historical_network_features_24h.parquet")
    net = pd.read_parquet(path)
    left = model_df.copy()
    left["zone_id"] = left["zone_id"].astype(int)
    left["pickup_hour"] = pd.to_datetime(left["pickup_hour"])
    net["feature_as_of"] = pd.to_datetime(net["feature_as_of"])
    if left.duplicated(["zone_id", "pickup_hour"]).any():
        raise ValueError("Forecast panel is not unique on zone_id + pickup_hour")
    if net.duplicated(["zone_id", "feature_as_of"]).any():
        raise ValueError("Network features have duplicate keys")
    merged = left.merge(
        net, left_on=["zone_id", "pickup_hour"],
        right_on=["zone_id", "feature_as_of"],
        how="left", validate="one_to_one", indicator=True
    )
    if len(merged) != len(left) or not merged["_merge"].eq("both").all():
        raise ValueError("Network join lost or failed to match forecasting rows")
    if not merged["pickup_hour"].eq(merged["feature_as_of"]).all():
        raise ValueError("Network feature time not aligned to target-hour start")
    net_cols = [
        "in_strength_24h", "out_strength_24h",
        "interzone_in_strength_24h", "interzone_out_strength_24h"
    ]
    keep = merged["has_full_history_24h"] & merged["dst_safe_24h"]
    if "dst_history_affected" in merged:
        keep &= ~merged["dst_history_affected"].fillna(True)
    shared = merged.loc[keep].copy().drop(columns="_merge")
    if shared[net_cols].isna().any().any():
        raise ValueError("Missing network feature values among usable rows")
    if shared.empty:
        raise ValueError("No rows left for model comparison")
    print(f"Aligned zone-hour rows: {len(merged):,}")
    print(f"Common M1/M2 evaluation rows: {len(shared):,}")
    print("Network columns:", net_cols)
    return shared, net_cols
