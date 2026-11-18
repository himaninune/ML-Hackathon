import networkx as nx
import pandas as pd
from src.hmm.map_matcher import decode


def _candidate(arc, source, target, lon, probability):
    return {"candidate_arc_id": arc, "from_node_id": source, "to_node_id": target,
            "obs_lon": lon, "obs_lat": 0.0, "timestamp": float(arc // 2),
            "remaining_m": 0.0, "offset_m": 0.0,
            "gps_to_road_distance_m": 0.0, "pred_prob": probability}


def test_connected_transition_beats_disconnected_emission_peak():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, length=10.0)
    left = pd.DataFrame([_candidate(0, 0, 1, 0.0, 0.2), _candidate(0, 5, 6, 0.0, 0.8)])
    right = pd.DataFrame([_candidate(2, 2, 3, 0.0001, 0.2), _candidate(2, 8, 9, 0.0001, 0.8)])
    gps = pd.DataFrame([{"longitude": 0.0, "latitude": 0.0, "timestamp": 0.0},
                        {"longitude": 0.0001, "latitude": 0.0, "timestamp": 1.0}])
    assert decode([left, right], gps, graph) == [0, 0]


def test_fully_disconnected_network_still_decodes():
    left = pd.DataFrame([_candidate(0, 0, 1, 0.0, 0.7)])
    right = pd.DataFrame([_candidate(1, 8, 9, 0.0001, 0.7)])
    gps = pd.DataFrame([{"longitude": 0.0, "latitude": 0.0, "timestamp": 0.0},
                        {"longitude": 0.0001, "latitude": 0.0, "timestamp": 1.0}])
    assert decode([left, right], gps, nx.DiGraph()) == [0, 0]
