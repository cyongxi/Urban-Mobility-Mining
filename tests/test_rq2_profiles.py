"""Run from repo root: python -m pytest tests/test_rq2_profiles.py -q"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.temporal.build_profiles import (FEATURE_COLS, exposure_days,
                                         profile_from_counts)
from src.temporal.cluster_profiles import run_clustering, compare_rq1


def test_exposure_dates_and_dst():
    d = exposure_days()
    assert sum(d.values()) == 363
    assert d[2] == 53  # Wednesday: Jan 1 2025
    assert d[6] == 50  # Sunday: two transition dates excluded


def test_profile_denominators_zero_fill_and_normalization():
    src = pd.DataFrame({"zone_id":[1,1,2], "day_of_week":[0,0,6],
                        "hour":[8,9,22], "pickup_count":[100,50,40]})
    dense, raw, norm, diag = profile_from_counts(src, [1,2,3], {d:10 for d in range(7)})
    assert len(dense) == 504
    assert dense.pickup_count.sum() == 190
    assert raw.shape == (3,169) and norm.shape == (3,169)
    a = raw.set_index("zone_id")
    b = norm.set_index("zone_id")
    assert a.loc[1,"d0_h08"] == 10 and a.loc[1,"d0_h09"] == 5
    assert np.isclose(b.loc[1,"d0_h08"],2/3)
    assert np.isclose(b.loc[1,FEATURE_COLS].sum(),1)
    assert b.loc[3,FEATURE_COLS].sum() == 0
    assert diag.set_index("zone_id").loc[3,"status"] == "no_pickups"


def _make_profiles():
    rng = np.random.default_rng(42)
    raws, norms, drows = [], [], []
    for i in range(31):
        if i==30:
            arr=np.zeros(168)
        else:
            h = np.arange(168)%24
            wd=np.arange(168)//24
            peak=[8,15,22][i//10]
            circ=np.minimum(abs(h-peak),24-abs(h-peak))
            arr=(.05+np.exp(-.5*(circ/2.2)**2)*(1+(.6 if i//10==2 else 0)*(wd>=5))
                 +.015*rng.random(168))*(5+3*(i%10))
        n=arr/arr.sum() if arr.sum() else arr
        raws.append({"zone_id":i+1,**dict(zip(FEATURE_COLS,arr))})
        norms.append({"zone_id":i+1,**dict(zip(FEATURE_COLS,n))})
        drows.append({"zone_id":i+1,"total_pickups":int(round(arr.sum()*52))})
    return pd.DataFrame(raws),pd.DataFrame(norms),pd.DataFrame(drows)


def test_kmeans_core_and_unassigned_zero_zones():
    raw,norm,diag=_make_profiles()
    results, meta=run_clustering(raw,norm,diag,k_min=2,k_max=5,min_cluster_size=3)
    labels=results["zone_temporal_clusters"]
    assert meta["n_zones_clustered"]==30
    assert meta["n_zones_unassigned"]==1
    assert labels.loc[labels.zone_id==31,"cluster_id"].isna().all()
    assert 2<=meta["chosen_k"]<=5
    assert results["k_selection"].selected.sum()==1
    assert len(results["cluster_profiles"])==meta["chosen_k"]*168
    for _,sub in results["cluster_profiles"].groupby("cluster_id"):
        assert np.isclose(sub.mean_normalized_share.sum(),1)
    assert results["kmeans_seed_stability"].ari_vs_primary.between(-1,1).all()


def test_cluster_input_rejects_bad_normalization():
    raw,norm,diag=_make_profiles()
    norm.loc[0,"d0_h00"]=99
    with pytest.raises(ValueError,match="sum to 1"):
        run_clustering(raw,norm,diag)


def test_rq1_rq2_overlap(tmp_path):
    assigned=pd.DataFrame({"zone_id":[1,2,3,4],
                           "cluster_id":pd.Series([0,0,1,pd.NA],dtype="Int64")})
    f=tmp_path/"communities_annual.csv"
    pd.DataFrame({"zone_id":[1,2,3,4],"community_id":[5,5,6,6]}).to_csv(f,index=False)
    result=compare_rq1(assigned,f)
    long,metrics=result
    assert metrics["n_shared_zones"]==3
    assert metrics["ARI"]==1
    assert long.groupby("community_id").pct_within_community.sum().eq(1).all()


def test_no_od_valid_filter_on_pickups():
    from src.temporal import build_profiles
    import inspect
    s=inspect.getsource(build_profiles.aggregate_with_duckdb)
    # RQ2 must NOT use the OD-valid subset in SQL WHERE.
    assert "WHERE t.valid_od" not in s
    assert "INNER JOIN valid_zones" in s
