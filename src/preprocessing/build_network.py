from src.config import DATA_DIR, PARTITIONS, RESULTS_DIR
from src.preprocessing.io_utils import trip_files
from src.preprocessing.network_utils import load_network, make_graph

def main():
    rows = []
    for partition in PARTITIONS:
        for folder in sorted(p for p in (DATA_DIR / partition).iterdir() if p.is_dir()):
            tid = folder.name
            p = trip_files(partition, tid)
            nodes, arcs = load_network(p["nodes"], p["arcs"])
            graph = make_graph(nodes, arcs)
            rows.append({
                "partition": partition, "trip_id": tid,
                "nodes": len(nodes), "valid_arcs": len(arcs),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "weakly_connected_components": __import__("networkx").number_weakly_connected_components(graph)
            })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "network_summary.csv", index=False)
    print(pd.DataFrame(rows).groupby("partition").sum(numeric_only=True))
    print("Saved network_summary.csv")

if __name__ == "__main__":
    main()
