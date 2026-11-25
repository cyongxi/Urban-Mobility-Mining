"""Run in repo root: python -m pytest tests/test_network_rq1.py -q"""
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from src.network.community_detection import build_graphs, louvain_partition, summarize_communities


def test_edge_conservation_and_symmetrization():
    od = pd.DataFrame({'origin':[1,2,1,2,3], 'destination':[2,1,1,3,2],
                       'trip_count':[8,3,4,2,5]})
    G, H = build_graphs(od)
    assert G.number_of_edges() == 5
    assert H.number_of_edges() == 2
    assert H[1][2]['weight'] == 11
    assert H[2][3]['weight'] == 7
    assert not H.has_edge(1,1)
    assert G.out_degree(1, weight='weight') == 12
    assert sum(d['weight'] for _,_,d in G.edges(data=True)) == 22


def test_louvain_and_share():
    od = pd.DataFrame({'origin':[1,2,3,4], 'destination':[2,1,4,3],
                       'trip_count':[100,100,80,80]})
    zones = pd.DataFrame({'LocationID':[1,2,3,4], 'Borough':['A']*4, 'Zone':['a','b','c','d']})
    _, H = build_graphs(od)
    labels, q = louvain_partition(H, seed=42)
    assert len(labels) == 4
    assert isinstance(q, float)
    _, summary, share = summarize_communities(od, labels, zones)
    assert summary.n_zones.sum() == 4
    assert 0 <= share <= 1
    # Label permutations have no effect on ARI.
    assert adjusted_rand_score([0,0,1,1], [5,5,9,9]) == 1


def test_community_summary_column_name_and_flow_concentration():
    from src.network.community_detection import flow_concentration
    od = pd.DataFrame({
        'origin': [1, 2, 3, 4],
        'destination': [2, 1, 4, 3],
        'trip_count': [100, 50, 20, 10],
    })
    lookup = pd.DataFrame({
        'LocationID': [1, 2, 3, 4],
        'Borough': ['A'] * 4,
        'Zone': ['One', 'Two', 'Three', 'Four'],
    })
    labels = {1: 0, 2: 0, 3: 1, 4: 1}
    _, summary, share = summarize_communities(od, labels, lookup)
    assert 'community_id' in summary.columns
    assert sorted(summary.community_id.tolist()) == [0, 1]
    assert summary.n_zones.sum() == 4
    assert share == 1.0
    c = flow_concentration(od)
    assert abs(c['top10_interzone_edge_share'] - 1.0) < 1e-12
    assert abs(c['top1pct_interzone_edge_share'] - 100 / 180) < 1e-12


def test_align_label_permutation_and_one_changed_zone():
    from src.network.community_detection import align_labels
    ref = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    new = {1: 9, 2: 9, 3: 7, 4: 7, 5: 9}
    alignment = align_labels(ref, new, sorted(ref))
    assert alignment[9] == 0
    assert alignment[7] == 1
    assert sum(ref[z] != alignment[new[z]] for z in ref) == 1


def test_static_export_preserves_unobserved_zones(tmp_path):
    from src.network.export_static_features import export_static_features
    pd.DataFrame({'LocationID': [1, 2, 3]}).to_csv(tmp_path / 'zone_lookup_used.csv', index=False)
    pd.DataFrame([{'zone_id': 1, 'in_degree': 1, 'out_degree': 1,
                   'in_strength': 6, 'out_strength': 7,
                   'interzone_weighted_degree': 13, 'interzone_neighbors': 1}]).to_csv(
        tmp_path / 'zone_metrics_annual.csv', index=False)
    pd.DataFrame({'zone_id': [1], 'community_id': [0]}).to_csv(
        tmp_path / 'communities_annual.csv', index=False)
    features = export_static_features(tmp_path)
    assert len(features) == 3
    assert features.zone_id.is_unique
    assert features.has_observed_od.tolist() == [1, 0, 0]
    assert features.loc[features.zone_id == 2, 'in_strength'].iloc[0] == 0
    assert pd.isna(features.loc[features.zone_id == 3, 'community_id'].iloc[0])
