from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation import (
    add_errors, compare_subgroups, daily_block_bootstrap, direction_agreement,
    load_rq1_communities, load_rq2_clusters, overall_metrics, training_demand_groups,
)

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    hours = pd.date_range("2025-11-01", periods=48, freq="h")
    df = pd.DataFrame([(z, h) for z in [1, 2, 3] for h in hours], columns=["zone_id", "pickup_hour"])
    rng = np.random.default_rng(0)
    df["pickup_count"] = rng.poisson(5, len(df))
    df["pred_m1"] = df["pickup_count"] + 2.0          # always off by 2
    df["pred_m2"] = df["pickup_count"] + np.where(df.zone_id == 1, 1.0, 2.5)
    df["group"] = np.where(df.zone_id == 1, "A", "B")
    return add_errors(df, ["m1", "m2"])


def test_overall_metrics_and_signs():
    df = _toy()
    out = overall_metrics(df, ["m1", "m2"], split="Test")
    assert out.loc[0, "MAE"] == pytest.approx(2.0)
    expected_m2 = (1.0 + 2.5 + 2.5) / 3
    assert out.loc[1, "MAE"] == pytest.approx(expected_m2)
    assert out.loc[1, "MAE_reduction_vs_prev"] == pytest.approx(2.0 - expected_m2)
    assert np.isnan(out.loc[0, "MAE_reduction_vs_prev"])


def test_compare_subgroups_weighted_and_share():
    df = _toy()
    g = compare_subgroups(df, "group")
    a = g.set_index("group")
    assert a.loc["A", "abs_MAE_reduction"] == pytest.approx(1.0)
    assert a.loc["B", "abs_MAE_reduction"] == pytest.approx(-0.5)
    # A contributes 48*1, B contributes 96*-0.5 -> total 0 -> share undefined
    assert a["share_of_net_gain_pct"].isna().all()
    # subgroup MAE re-weights to overall MAE
    overall = (g["new_MAE"] * g["n"]).sum() / g["n"].sum()
    assert overall == pytest.approx(df["ae_m2"].mean())


def test_bootstrap_deterministic_and_bounds():
    df = _toy()
    b1 = daily_block_bootstrap(df, "m1", "m2", n_boot=200, group_col="group")
    b2 = daily_block_bootstrap(df, "m1", "m2", n_boot=200, group_col="group")
    pd.testing.assert_frame_equal(b1, b2)
    a = b1.set_index("group").loc["A"]
    assert a["observed_abs_MAE_reduction"] == pytest.approx(1.0)
    assert a["ci_lower_2.5pct"] == pytest.approx(1.0) and a["fraction_positive"] == 1.0
    assert set(b1["group"]) == {"ALL", "A", "B"}


def test_training_demand_groups_equal_sizes():
    counts = pd.Series([0, 5, 5, 10, 100, 7], index=[6, 5, 4, 3, 2, 1])
    z = training_demand_groups(counts, n_train_hours=10)
    assert z["demand_group"].value_counts().tolist() == [2, 2, 2]
    assert z.loc[z.zone_id == 2, "demand_group"].item() == "High"
    assert z.loc[z.zone_id == 6, "demand_group"].item() == "Low"


def test_direction_agreement():
    v = pd.DataFrame({"g": ["A", "B"], "n": [1, 1], "abs_MAE_reduction": [0.1, -0.1],
                      "relative_MAE_reduction_pct": [1, -1]})
    t = v.assign(abs_MAE_reduction=[0.2, 0.1])
    m = direction_agreement(v, t, "g").set_index("g")
    assert bool(m.loc["A", "same_sign"]) and not bool(m.loc["B", "same_sign"])


@pytest.mark.skipif(not (ROOT / "results/network/tables/communities_annual.csv").exists(),
                    reason="RQ1 outputs not present")
def test_real_rq1_labels():
    c = load_rq1_communities(ROOT / "results/network/tables/communities_annual.csv")
    assert c["zone_id"].is_unique and c["rq1_community"].notna().all()


MIN1000 = ROOT / "results/temporal/sensitivity/min1000"


@pytest.mark.skipif(not (MIN1000 / "zone_temporal_clusters.csv").exists(),
                    reason="RQ2 min1000 outputs not present")
def test_real_rq2_labels_and_name_guard():
    names = {0: (7, "commuter"), 1: (23, "Manhattan"), 2: (0, "nightlife")}
    z = load_rq2_clusters(MIN1000 / "zone_temporal_clusters.csv", MIN1000 / "cluster_summary.csv", names)
    assert len(z) == 263
    assert (z["rq2_cluster"] == "Unclustered (low volume)").sum() == 36
    with pytest.raises(ValueError):
        load_rq2_clusters(MIN1000 / "zone_temporal_clusters.csv", MIN1000 / "cluster_summary.csv",
                          {0: (12, "wrong")})
