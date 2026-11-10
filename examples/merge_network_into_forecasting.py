"""Backward-compatible RQ3 helper for three-model feature ablation.

From 05_forecasting.ipynb, after building one canonical model_df:

    from examples.merge_network_into_forecasting import merge_history
    common_df, all_network_cols = merge_history(model_df, PROJECT_ROOT, include_topology=True)
    strength_cols = ["in_strength_24h", "out_strength_24h",
                     "interzone_in_strength_24h", "interzone_out_strength_24h"]
    topology_cols = ["in_degree_24h", "out_degree_24h", "in_hhi_24h", "out_hhi_24h"]
    # M1 = temporal_cols; M2 = temporal_cols+strength_cols;
    # M3 = temporal_cols+strength_cols+topology_cols.

NaN HHI is a legitimate 'no cross-zone traffic' observation. LightGBM handles it.
"""
from pathlib import Path
import pandas as pd

STRENGTH_COLUMNS = [
    "in_strength_24h", "out_strength_24h",
    "interzone_in_strength_24h", "interzone_out_strength_24h",
]
TOPOLOGY_COLUMNS = ["in_degree_24h", "out_degree_24h", "in_hhi_24h", "out_hhi_24h"]


def merge_history(model_df: pd.DataFrame, project_root: Path, include_topology: bool = False):
    """Return one common sample for M1/M2 or M1/M2/M3, and chosen network cols.

    Existing `merge_history(model_df, PROJECT_ROOT)` behavior is preserved.
    """
    suffix = "historical_network_features_24h_augmented.parquet" if include_topology else "historical_network_features_24h.parquet"
    file = Path(project_root) / "data/processed/network_features" / suffix
    net = pd.read_parquet(file)
    left = model_df.copy()
    left["zone_id"] = left["zone_id"].astype(int)
    left["pickup_hour"] = pd.to_datetime(left["pickup_hour"])
    net["feature_as_of"] = pd.to_datetime(net["feature_as_of"])
    if left.duplicated(["zone_id", "pickup_hour"]).any():
        raise ValueError("Forecast panel has duplicate zone-hour keys")
    if net.duplicated(["zone_id", "feature_as_of"]).any():
        raise ValueError("Network table has duplicate zone-hour keys")
    merged = left.merge(
        net, left_on=["zone_id", "pickup_hour"],
        right_on=["zone_id", "feature_as_of"],
        how="left", validate="one_to_one", indicator=True
    )
    if len(merged) != len(left) or not merged["_merge"].eq("both").all():
        raise ValueError("Network join lost or failed to match forecasting rows")
    if not merged["pickup_hour"].eq(merged["feature_as_of"]).all():
        raise ValueError("Network feature time not aligned to target hour")
    keep = merged["has_full_history_24h"].fillna(False).astype(bool) & merged["dst_safe_24h"].fillna(False).astype(bool)
    if "dst_history_affected" in merged:
        keep &= ~merged["dst_history_affected"].fillna(True).astype(bool)
    shared = merged.loc[keep].copy().drop(columns="_merge")
    if shared.empty:
        raise ValueError("No full-history, DST-safe zone hours to compare")
    if shared[STRENGTH_COLUMNS].isna().any().any():
        raise ValueError("Missing OD strength in a valid zone hour")
    chosen = STRENGTH_COLUMNS.copy()
    if include_topology:
        if shared[["in_degree_24h", "out_degree_24h"]].isna().any().any():
            raise ValueError("Degree missing in valid zone hours")
        for col, flag in (("in_hhi_24h", "has_interzone_in_flow_24h"),
                          ("out_hhi_24h", "has_interzone_out_flow_24h")):
            has_flow = shared[flag].astype(bool)
            if shared.loc[has_flow, col].isna().any():
                raise ValueError(f"Missing {col} despite observed cross-zone flow")
            if shared.loc[~has_flow, col].notna().any():
                raise ValueError(f"Undefined {col} must be NaN when cross-zone flow is zero")
        chosen += TOPOLOGY_COLUMNS
    print(f"Aligned rows: {len(merged):,}; common comparison rows: {len(shared):,}")
    return shared, chosen
