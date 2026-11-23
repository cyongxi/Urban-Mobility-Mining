"""Run from repository root: python -m pytest tests/test_network_topology_features.py -q"""

import numpy as np
import pandas as pd
import pytest
from src.network.network_topology_features import (
    degree_hhi, iter_hour_groups, rolling_topology_from_hour_groups,
    merge_check_and_save,
)


def test_directed_degree_and_hhi_and_self_loops():
    # zones 1,2,3,4; 1->2:3, 1->3:1, 4->2:4, 1->1:5 loop
    m = np.zeros((4,4), dtype=np.int64)
    m[0,1]=3; m[0,2]=1; m[3,1]=4; m[0,0]=5
    di, do, hi, ho, fi, fo = degree_hhi(m)
    assert do.tolist() == [2,0,0,1]
    assert di.tolist() == [0,2,1,0]
    assert np.isclose(ho[0], (3/4)**2 + (1/4)**2)
    assert np.isclose(hi[1], (3/7)**2 + (4/7)**2)
    assert fo[0] == 4 # excludes the five same-zone trips
    assert fi[1] == 7
    assert np.isnan(hi[0]) and np.isnan(ho[1])


def test_split_completion_hour_across_input_chunks():
    cols = ["completion_hour", "origin", "destination", "trip_count"]
    a = pd.DataFrame([("2025-01-01 08:00", 1,2,3)], columns=cols)
    b = pd.DataFrame([("2025-01-01 08:00", 1,3,1),
                      ("2025-01-01 09:00", 2,1,4)], columns=cols)
    groups = list(iter_hour_groups([a,b]))
    assert len(groups) == 2 and len(groups[0][1]) == 2
    assert int(groups[0][1].trip_count.sum()) == 4


def test_rolling_window_uses_only_completed_previous_hours():
    # Validates expiration and that the current-hour event is never used.
    groups = [
        (pd.Timestamp("2025-01-01 08:00"), pd.DataFrame({
            "origin":[1,1,4,1], "destination":[2,3,2,1],
            "trip_count":[3,1,4,5]})),
        (pd.Timestamp("2025-01-01 09:00"), pd.DataFrame({
            "origin":[1,3], "destination":[2,2], "trip_count":[2,2]})),
        (pd.Timestamp("2025-01-02 08:00"), pd.DataFrame({
            "origin":[1], "destination":[4], "trip_count":[50]})),
        (pd.Timestamp("2025-01-02 09:00"), pd.DataFrame({
            "origin":[4], "destination":[1], "trip_count":[999]})),
    ]
    out = rolling_topology_from_hour_groups(
        groups, [1,2,3,4], start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-02 10:00"), window_hours=24, carryover_hours=0
    ).set_index(["feature_as_of", "zone_id"])
    # At 2025-01-02 08:00, window is [Jan 1 08:00, Jan 2 08:00).
    before_expiry = out.loc[("2025-01-02 08:00",1)]
    assert before_expiry.out_degree_24h == 2  # origins to zones 2 and 3
    assert before_expiry._check_interzone_out == 6 # 3+1+2
    # At 09:00 next day, Jan 1 08 bucket expired and Jan 2 08 bucket enters.
    after_expiry = out.loc[("2025-01-02 09:00",1)]
    assert after_expiry.out_degree_24h == 2  # now zones 2,4
    assert after_expiry._check_interzone_out == 52 # Jan1 09: 2 + Jan2 08:50
    assert np.isclose(after_expiry.out_hhi_24h,(2/52)**2+(50/52)**2)
    # At target 09, a trip completed at 09:00–10:00 is still unavailable.
    assert after_expiry._check_interzone_in == 0
    # At target 10, 999 trip for 4->1 enters (not available at 09).
    # We don't produce 10:00 in this range. By construction, it never leaked.


def test_initial_missing_history_is_not_zero_flow():
    out = rolling_topology_from_hour_groups(
        [], [1,2], start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-02 08:00"), window_hours=24, carryover_hours=6
    )
    p = out.set_index("feature_as_of")
    assert p.loc["2025-01-02 05:00", "in_degree_24h"].isna().all()
    assert p.loc["2025-01-02 06:00", "in_degree_24h"].eq(0).all()
    assert p.loc["2025-01-02 06:00", "in_hhi_24h"].isna().all()
    assert not p.loc["2025-01-02 06:00", "has_interzone_in_flow_24h"].any()


def test_reconciliation_can_detect_wrong_strength_counts(tmp_path):
    pytest.importorskip("pyarrow")
    p = pd.Timestamp("2025-01-03 00:00")
    topo = pd.DataFrame({
        "zone_id":[1], "feature_as_of":[p],
        "in_degree_24h":[1], "out_degree_24h":[1],
        "in_hhi_24h":[1.0], "out_hhi_24h":[1.0],
        "has_interzone_in_flow_24h":[True], "has_interzone_out_flow_24h":[True],
        "_check_interzone_in":[3], "_check_interzone_out":[3],
    })
    base = pd.DataFrame({
        "zone_id":[1], "feature_as_of":[p],
        "in_strength_24h":[3], "out_strength_24h":[3],
        "interzone_in_strength_24h":[3], "interzone_out_strength_24h":[100],
        "has_full_history_24h":[True], "dst_safe_24h":[True]
    })
    path = tmp_path / "base.parquet"
    base.to_parquet(path,index=False)
    with pytest.raises(AssertionError, match="reconciliation FAILED"):
        merge_check_and_save(topo,path,tmp_path)


def test_merge_reconciliation_and_nan_hhi_without_pyarrow(monkeypatch, tmp_path):
    """Check merge/invariants using an in-memory Parquet stand-in."""
    p = pd.Timestamp('2025-02-01 12:00')
    base = pd.DataFrame({
        "zone_id":[1,2], "feature_as_of":[p,p],
        "in_strength_24h":[5,4], "out_strength_24h":[3,6],
        "interzone_in_strength_24h":[3,0],
        "interzone_out_strength_24h":[3,0],
        "has_full_history_24h":[True,True], "dst_safe_24h":[True,True]
    })
    topo = pd.DataFrame({
        "zone_id":[1,2], "feature_as_of":[p,p],
        "in_degree_24h":pd.Series([1,0],dtype='Int16'),
        "out_degree_24h":pd.Series([1,0],dtype='Int16'),
        "in_hhi_24h":[1.0,float('nan')],
        "out_hhi_24h":[1.0,float('nan')],
        "has_interzone_in_flow_24h":[True,False],
        "has_interzone_out_flow_24h":[True,False],
        "_check_interzone_in":[3,0],
        "_check_interzone_out":[3,0],
    })
    stored = {}
    import src.network.network_topology_features as mod
    monkeypatch.setattr(pd, 'read_parquet', lambda path: base.copy())
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, path, index=False: stored.update({str(path): self.copy()}))
    stub = tmp_path/'strength.parquet'
    stub.touch()
    out1,out2 = mod.merge_check_and_save(topo, stub, tmp_path)
    assert str(out1) in stored and str(out2) in stored
    assert stored[str(out1)].shape[0] == 2
    assert stored[str(out2)].loc[1,'in_degree_24h'] == 0
    assert np.isnan(stored[str(out2)].loc[1,'in_hhi_24h'])
    # An incorrect flow column must cause a hard error, not a silent export.
    wrong = topo.copy()
    wrong.loc[0,'_check_interzone_out'] = 999
    with pytest.raises(AssertionError, match='reconciliation FAILED'):
        mod.merge_check_and_save(wrong,stub,tmp_path)


def test_merge_helper_keeps_nan_hhi_but_common_rows(monkeypatch, tmp_path):
    from examples.merge_network_into_forecasting import merge_history
    hours = pd.to_datetime(['2025-07-01 10:00','2025-07-01 11:00'])
    panel = pd.DataFrame({'zone_id':[1,1], 'pickup_hour':hours,
                          'pickup_count':[5,7], 'lag_1':[4,5]})
    net = pd.DataFrame({
        'zone_id':[1,1], 'feature_as_of':hours,
        'in_strength_24h':[3,0], 'out_strength_24h':[4,0],
        'interzone_in_strength_24h':[2,0], 'interzone_out_strength_24h':[4,0],
        'in_degree_24h':[1,0], 'out_degree_24h':[2,0],
        'in_hhi_24h':[1.0,np.nan], 'out_hhi_24h':[0.625,np.nan],
        'has_interzone_in_flow_24h':[True,False],
        'has_interzone_out_flow_24h':[True,False],
        'has_full_history_24h':[True,True], 'dst_safe_24h':[True,True],
    })
    monkeypatch.setattr(pd, 'read_parquet', lambda *args,**kwargs: net.copy())
    shared, cols = merge_history(panel,tmp_path,include_topology=True)
    assert len(shared)==2 and len(cols)==8
    assert shared.loc[shared.pickup_hour == hours[1], 'out_hhi_24h'].isna().all()
    assert not shared.duplicated(['zone_id','pickup_hour']).any()


def test_duckdb_od_aggregation_uses_completion_not_pickup(tmp_path):
    pytest.importorskip('duckdb')
    pytest.importorskip('pyarrow')
    from src.network.network_topology_features import iter_completed_hourly_pairs
    trips = pd.DataFrame({
        'PULocationID':[1,1,2,1,1],
        'DOLocationID':[2,1,1,2,264],
        'tpep_pickup_datetime':pd.to_datetime([
            '2025-01-01 08:45', '2025-01-01 08:00',
            '2025-01-01 09:00', '2025-01-01 09:50', '2025-01-01 08:00']),
        'tpep_dropoff_datetime':pd.to_datetime([
            '2025-01-01 09:20','2025-01-01 08:15',
            '2025-01-01 09:30','2025-01-01 10:20','2025-01-01 08:30']),
        'valid_od':[True,True,False,True,True],
    })
    parquet=tmp_path/'tiny.parquet'
    trips.to_parquet(parquet,index=False)
    chunks = list(iter_completed_hourly_pairs([parquet],[1,2]))
    observed = pd.concat(chunks).sort_values(['completion_hour','origin','destination'])
    assert int(observed.trip_count.sum()) == 3  # invalid and zone264 excluded
    cross = observed.loc[observed.origin != observed.destination]
    assert set(cross.completion_hour) == {
        pd.Timestamp('2025-01-01 09:00'),pd.Timestamp('2025-01-01 10:00')}
    assert len(cross)==2
