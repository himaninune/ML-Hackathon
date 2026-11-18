import pandas as pd
import networkx as nx
from src.config import ARC_ID_MODE
from src.preprocessing.io_utils import read_nodes, read_arcs, read_route

def load_network(nodes_path, arcs_path):
    nodes = read_nodes(nodes_path).reset_index().rename(columns={"index": "node_id"})
    arcs = read_arcs(arcs_path).reset_index().rename(columns={"index": "arc_row_id"})
    # Validate node references before indexing.
    valid = (
        arcs["from_node_id"].between(0, len(nodes)-1)
        & arcs["to_node_id"].between(0, len(nodes)-1)
    )
    arcs = arcs[valid].copy()
    arcs["arc_id"] = arcs["arc_row_id"]
    if ARC_ID_MODE == "one_based":
        arcs["arc_id"] = arcs["arc_row_id"] + 1

    arcs["from_lon"] = arcs["from_node_id"].map(nodes["longitude"])
    arcs["from_lat"] = arcs["from_node_id"].map(nodes["latitude"])
    arcs["to_lon"] = arcs["to_node_id"].map(nodes["longitude"])
    arcs["to_lat"] = arcs["to_node_id"].map(nodes["latitude"])
    return nodes, arcs

def make_graph(nodes, arcs):
    graph = nx.DiGraph()
    for r in nodes.itertuples():
        graph.add_node(int(r.node_id), longitude=float(r.longitude), latitude=float(r.latitude))
    for r in arcs.itertuples():
        graph.add_edge(int(r.from_node_id), int(r.to_node_id), arc_id=int(r.arc_id))
    return graph

def route_arc_ids(route_path):
    route = read_route(route_path)
    vals = pd.to_numeric(route["arc_id"], errors="coerce").dropna().astype(int)
    if ARC_ID_MODE == "one_based":
        vals = vals - 1
    return set(vals.tolist())
