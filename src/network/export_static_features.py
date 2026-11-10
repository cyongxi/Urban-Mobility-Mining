"""Export one-row-per-zone static RQ1 features for downstream integration.

Run from repository root after `python -m src.network.community_detection`:
    python -m src.network.export_static_features

Output: results/network/tables/static_network_features_2025.csv
Full-year 2025 retrospective features: NOT time-safe for forecasting 2025 hours.
"""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results/network/tables"
OUTPUT_NAME = "static_network_features_2025.csv"


def export_static_features(table_dir: Path = TABLE_DIR) -> pd.DataFrame:
    table_dir = Path(table_dir)
    lookup = pd.read_csv(table_dir / "zone_lookup_used.csv")
    metrics = pd.read_csv(table_dir / "zone_metrics_annual.csv")
    communities = pd.read_csv(table_dir / "communities_annual.csv")

    required = {
        "lookup": (lookup, {"LocationID"}),
        "metrics": (metrics, {"zone_id", "in_degree", "out_degree", "in_strength", "out_strength", "interzone_weighted_degree", "interzone_neighbors"}),
        "communities": (communities, {"zone_id", "community_id"}),
    }
    for name, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    zone_ids = lookup.loc[~lookup["LocationID"].isin([264, 265]), ["LocationID"]].rename(columns={"LocationID": "zone_id"}).copy()
    zone_ids["zone_id"] = zone_ids["zone_id"].astype(int)
    if any(not frame[col].is_unique for frame, col in ((zone_ids,"zone_id"),(metrics,"zone_id"),(communities,"zone_id"))):
        raise ValueError("Duplicate zone_id detected in lookup, metrics or communities")
    if metrics["zone_id"].isna().any() or communities["zone_id"].isna().any():
        raise ValueError("Null zone_id found in source tables")

    cols = ["zone_id", "in_degree", "out_degree", "in_strength", "out_strength", "interzone_weighted_degree", "interzone_neighbors"]
    metrics = metrics[cols].copy()
    communities = communities[["zone_id", "community_id"]].copy()
    allowed = set(zone_ids["zone_id"])
    for name, frame in (("metrics", metrics), ("communities", communities)):
        unknown = set(frame["zone_id"]) - allowed
        if unknown:
            raise ValueError(f"{name} contains zone IDs absent from valid lookup: {sorted(unknown)[:10]}")

    out = zone_ids.merge(metrics, on="zone_id", how="left", validate="one_to_one")
    out = out.merge(communities, on="zone_id", how="left", validate="one_to_one")
    # Absent graph nodes have no observed OD activity; zero network counts are appropriate.
    # Their Louvain membership is undefined: keep community_id missing (not an invented label).
    numeric = [c for c in cols if c != "zone_id"]
    out[numeric] = out[numeric].fillna(0).astype("int64")
    out["community_id"] = out["community_id"].astype("Int64")
    out["has_observed_od"] = out["zone_id"].isin(metrics["zone_id"]).astype("int8")
    out["feature_period"] = "2025_full_year"
    out = out.sort_values("zone_id").reset_index(drop=True)
    assert out["zone_id"].is_unique
    assert len(out) == len(zone_ids)

    target = table_dir / OUTPUT_NAME
    out.to_csv(target, index=False)
    print(f"Saved: {target}")
    print(f"Rows: {len(out)}, zones with OD: {out['has_observed_od'].sum()}, missing community: {out['community_id'].isna().sum()}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    args = parser.parse_args()
    export_static_features(args.table_dir)
