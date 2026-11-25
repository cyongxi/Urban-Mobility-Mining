"""Paired forecast-error evaluation helpers for RQ3 (ablation) and RQ4 (heterogeneity).

All functions work on a *prediction frame*: one row per zone-hour with the
observed target and one ``pred_<code>`` column per model. Every comparison is
paired: all models are scored on exactly the same rows.

Sign convention used everywhere: ``gain = MAE(old) - MAE(new)``; positive means
the newer model (more network features) has lower error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TARGET = "pickup_count"


def add_errors(pred: pd.DataFrame, codes: Sequence[str], target: str = TARGET) -> pd.DataFrame:
    """Add absolute- and squared-error columns ``ae_<code>``/``se_<code>`` in place."""
    y = pred[target].to_numpy(dtype=float)
    for code in codes:
        p = pred[f"pred_{code}"].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            raise ValueError(f"Non-finite predictions in pred_{code}")
        pred[f"ae_{code}"] = np.abs(y - p)
        pred[f"se_{code}"] = (y - p) ** 2
    return pred


def overall_metrics(pred: pd.DataFrame, codes: Sequence[str], names: dict | None = None,
                    split: str | None = None) -> pd.DataFrame:
    """MAE/RMSE per model plus absolute/relative change vs previous model and vs the first."""
    rows = []
    for code in codes:
        rows.append({
            "code": code,
            "model": (names or {}).get(code, code.upper()),
            "n_obs": len(pred),
            "MAE": float(pred[f"ae_{code}"].mean()),
            "RMSE": float(np.sqrt(pred[f"se_{code}"].mean())),
        })
    out = pd.DataFrame(rows)
    if split is not None:
        out.insert(0, "split", split)
    for metric in ["MAE", "RMSE"]:
        prev = out[metric].shift(1)
        first = out[metric].iloc[0]
        out[f"{metric}_reduction_vs_prev"] = prev - out[metric]
        out[f"{metric}_reduction_vs_prev_pct"] = 100 * (prev - out[metric]) / prev
        out[f"{metric}_reduction_vs_first"] = first - out[metric]
        out[f"{metric}_reduction_vs_first_pct"] = 100 * (first - out[metric]) / first
    return out


def compare_subgroups(pred: pd.DataFrame, group_cols, old: str = "m1", new: str = "m2",
                      target: str = TARGET) -> pd.DataFrame:
    """Observation-weighted paired MAE comparison within each subgroup.

    ``share_of_net_gain_pct`` is each group's share of the *total* error reduction;
    values above 100% mean other groups are offsetting it with losses.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    x = (pred.groupby(group_cols, observed=True, dropna=False)
         .agg(n=(target, "size"),
              n_zones=("zone_id", "nunique"),
              mean_actual_pickups=(target, "mean"),
              old_MAE=(f"ae_{old}", "mean"),
              new_MAE=(f"ae_{new}", "mean"),
              old_MSE=(f"se_{old}", "mean"),
              new_MSE=(f"se_{new}", "mean"))
         .reset_index())
    x["old_RMSE"] = np.sqrt(x.pop("old_MSE"))
    x["new_RMSE"] = np.sqrt(x.pop("new_MSE"))
    x["abs_MAE_reduction"] = x["old_MAE"] - x["new_MAE"]
    x["relative_MAE_reduction_pct"] = np.where(
        x["old_MAE"] > 0, 100 * x["abs_MAE_reduction"] / x["old_MAE"], np.nan)
    x["RMSE_reduction"] = x["old_RMSE"] - x["new_RMSE"]
    total = (x["abs_MAE_reduction"] * x["n"]).sum()
    x["share_of_net_gain_pct"] = np.where(
        abs(total) > 1e-12, 100 * x["abs_MAE_reduction"] * x["n"] / total, np.nan)
    x.insert(0, "comparison", f"{old.upper()}->{new.upper()}")
    return x


def daily_block_bootstrap(pred: pd.DataFrame, old: str, new: str, n_boot: int = 2000,
                          seed: int = 671, group_col: str | None = None,
                          time_col: str = "pickup_hour") -> pd.DataFrame:
    """Paired bootstrap that resamples calendar days (all zone-hours of a day together).

    Returns observed gain, 95% percentile interval and the fraction of positive draws,
    overall (``group='ALL'``) and, if ``group_col`` is given, within each group.
    Each group uses its own resampling of days from a shared seed.
    """
    rng = np.random.default_rng(seed)
    work = pred.assign(_date=pred[time_col].dt.normalize(),
                       _gain=pred[f"ae_{old}"] - pred[f"ae_{new}"])
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", work)]
    if group_col is not None:
        groups += [(str(k), g) for k, g in work.groupby(group_col, observed=True, sort=True)]
    rows = []
    for name, frame in groups:
        by_day = frame.groupby("_date").agg(n=("_gain", "size"), s=("_gain", "sum"))
        n = by_day["n"].to_numpy(dtype=float)
        s = by_day["s"].to_numpy(dtype=float)
        ix = rng.integers(0, len(by_day), size=(n_boot, len(by_day)))
        draws = s[ix].sum(axis=1) / n[ix].sum(axis=1)
        rows.append({
            "comparison": f"{old.upper()}->{new.upper()}",
            "group": name,
            "n_obs": int(n.sum()),
            "n_days": len(by_day),
            "observed_abs_MAE_reduction": float(s.sum() / n.sum()),
            "ci_lower_2.5pct": float(np.quantile(draws, 0.025)),
            "ci_upper_97.5pct": float(np.quantile(draws, 0.975)),
            "fraction_positive": float((draws > 0).mean()),
            "n_boot": n_boot,
        })
    out = pd.DataFrame(rows)
    out.insert(1, "grouping", group_col or "ALL")
    return out


def training_demand_groups(train_counts: pd.Series, n_train_hours: int,
                           labels: Iterable[str] = ("Low", "Medium", "High")) -> pd.DataFrame:
    """Equal-sized ranked demand groups from *training-period* pickups only.

    ``train_counts`` is indexed by zone_id (zones with no pickups must be present as 0).
    Ties are broken deterministically by zone_id.
    """
    labels = list(labels)
    z = pd.DataFrame({"zone_id": train_counts.index.astype(int),
                      "train_total_pickups": train_counts.to_numpy(dtype="int64")})
    z["train_mean_pickups_per_hour"] = z["train_total_pickups"] / n_train_hours
    z = z.sort_values(["train_mean_pickups_per_hour", "zone_id"]).reset_index(drop=True)
    z["demand_group"] = pd.Categorical(
        pd.qcut(np.arange(len(z)), len(labels), labels=labels).astype(str),
        categories=labels, ordered=True)
    return z


def load_rq1_communities(path: Path, label_prefix: str = "Community") -> pd.DataFrame:
    """Read an RQ1 community table (zone_id, community_id[, ...]) as RQ4 grouping labels.

    Labels include the dominant borough and its share of each community to make tables readable.
    Zones absent from the table (no OD trips) get ``'No OD trips'``.
    """
    c = pd.read_csv(path)
    missing = {"zone_id", "community_id"} - set(c.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    c = c[["zone_id", "community_id"] + (["Borough"] if "Borough" in c.columns else [])].copy()
    c["zone_id"] = c["zone_id"].astype(int)
    if "Borough" in c.columns:
        # e.g. "Community 1 (Queens 36%, 194 zones)": the share shows when a community is mixed
        top = c.groupby("community_id")["Borough"].agg(
            lambda s: (s.value_counts().index[0], s.value_counts(normalize=True).iloc[0]))
        sizes = c.groupby("community_id").size()
        c["rq1_community"] = c["community_id"].map(
            lambda k: f"{label_prefix} {int(k)} ({top[k][0]} {top[k][1]:.0%}, {sizes[k]} zones)")
    else:
        c["rq1_community"] = c["community_id"].map(lambda k: f"{label_prefix} {int(k)}")
    return c[["zone_id", "rq1_community"]]


def load_rq2_clusters(cluster_path: Path, summary_path: Path | None = None,
                      names: dict | None = None) -> pd.DataFrame:
    """Read RQ2 temporal-cluster assignments as RQ4 grouping labels.

    Zones without a cluster (``cluster_status != 'clustered'``, e.g. below the
    minimum-pickup threshold) get ``'Unclustered (low volume)'``.

    ``names`` optionally maps cluster_id -> (expected_peak_hour, descriptive name).
    If given, each expected peak hour is checked against ``summary_path`` so that a
    re-run that permutes arbitrary cluster IDs fails loudly instead of mislabeling.
    """
    z = pd.read_csv(cluster_path)
    z["zone_id"] = z["zone_id"].astype(int)
    clustered = z["cluster_id"].notna()
    if "cluster_status" in z.columns:
        clustered &= z["cluster_status"].eq("clustered")
    label = pd.Series("Unclustered (low volume)", index=z.index, dtype=object)
    ids = z.loc[clustered, "cluster_id"].astype(int)
    if names:
        if summary_path is None:
            raise ValueError("summary_path is required to validate cluster names")
        s = pd.read_csv(summary_path).set_index("cluster_id")
        for cid, (peak_hour, _) in names.items():
            got = int(s.loc[cid, "peak_hour"])
            if got != int(peak_hour):
                raise ValueError(f"Cluster {cid}: expected peak hour {peak_hour}, found {got}. "
                                 "Cluster IDs changed; update the name map.")
        label.loc[clustered] = ids.map(lambda k: f"C{k} {names[k][1]}")
    else:
        label.loc[clustered] = ids.map(lambda k: f"C{k}")
    return pd.DataFrame({"zone_id": z["zone_id"], "rq2_cluster": label.to_numpy()})


def direction_agreement(val_table: pd.DataFrame, test_table: pd.DataFrame, key) -> pd.DataFrame:
    """Join validation and test subgroup tables and flag whether the gain sign agrees."""
    if isinstance(key, str):
        key = [key]
    cols = key + ["n", "abs_MAE_reduction", "relative_MAE_reduction_pct"]
    m = val_table[cols].merge(test_table[cols], on=key, suffixes=("_val", "_test"), how="outer")
    m["same_sign"] = np.sign(m["abs_MAE_reduction_val"]) == np.sign(m["abs_MAE_reduction_test"])
    return m
