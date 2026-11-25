"""Add a lag-1 persistence baseline to the final RQ3 results.

The proposal promises "seasonal-naive and lag-based benchmarks". Notebook 09
reports daily (lag 24) and weekly (lag 168) seasonal naive baselines; this
script adds the simplest lag-based benchmark, last-hour persistence
(y_hat[t] = y[t-1]), without re-training any model.

It reads the prediction files written by notebooks/09_final_rq3_rq4.ipynb
(git-ignored, local only):
    results/forecasting/final_rq3_rq4/validation_predictions.parquet
    results/forecasting/final_rq3_rq4/test_predictions.parquet

y[t-1] is taken from the observed pickups of the same zone exactly one hour
earlier. Validation (Sep-Oct) and Test (Nov-Dec) are contiguous, so the first
Test hour uses the last Validation hour. Rows whose previous hour is not in
the evaluation sample (e.g. around the excluded DST dates) are dropped, and
every model is re-scored on exactly the same rows so comparisons stay paired.

Output: results/forecasting/final_rq3_rq4/rq3_baselines_with_persistence.csv

Run from the repository root:
    python -m scripts.add_persistence_baseline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/forecasting/final_rq3_rq4")
TARGET_CANDIDATES = ("pickups", "pickup_count", "y", "target")
NAMES = {
    "b0p": "B0 Persistence (lag 1)",
    "b0d": "B0 Daily Naive (lag 24)",
    "b0w": "B0 Weekly Naive (lag 168)",
    "m1": "M1 Temporal",
    "m2": "M2 + OD Strength",
    "m3": "M3 + Degree",
    "m4": "M4 + HHI & Flow Flags",
}


def find_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    rest = [c for c in df.columns if c not in ("zone_id", "pickup_hour") and not c.startswith("pred_")]
    if len(rest) != 1:
        raise ValueError(f"Cannot identify target column among {rest}")
    return rest[0]


def add_lag1(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.sort_values(["zone_id", "pickup_hour"]).reset_index(drop=True)
    prev_hour = df.groupby("zone_id")["pickup_hour"].shift(1)
    prev_y = df.groupby("zone_id")[target].shift(1)
    exact = (df["pickup_hour"] - prev_hour) == pd.Timedelta(hours=1)
    df["pred_b0p"] = prev_y.where(exact).astype(float)
    return df


def score(df: pd.DataFrame, target: str, split: str, codes: list[str]) -> pd.DataFrame:
    rows = []
    y = df[target].to_numpy(dtype=float)
    for c in codes:
        e = df[f"pred_{c}"].to_numpy(dtype=float) - y
        rows.append({"split": split, "code": c, "model": NAMES[c], "n_obs": len(df),
                     "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))})
    return pd.DataFrame(rows)


def main() -> None:
    val = pd.read_parquet(OUT / "validation_predictions.parquet").assign(split="Validation")
    test = pd.read_parquet(OUT / "test_predictions.parquet").assign(split="Test")
    target = find_target(val)
    both = add_lag1(pd.concat([val, test], ignore_index=True), target)

    codes = ["b0p"] + [c for c in ("b0d", "b0w", "m1", "m2", "m3", "m4") if f"pred_{c}" in both.columns]
    out = []
    for split in ("Validation", "Test"):
        part = both[both["split"] == split]
        kept = part.dropna(subset=["pred_b0p"])
        print(f"{split}: lag-1 available for {len(kept):,}/{len(part):,} rows "
              f"({len(kept) / len(part):.2%}); all models re-scored on these rows")
        out.append(score(kept, target, split, codes))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT / "rq3_baselines_with_persistence.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
