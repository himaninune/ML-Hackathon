from pathlib import Path
import pandas as pd
from src.config import DATA_DIR, PARTITIONS, RESULTS_DIR
from src.preprocessing.io_utils import trip_files, read_track, read_nodes, read_arcs, read_route

def main():
    rows = []
    for partition in PARTITIONS:
        root = DATA_DIR / partition
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            tid = folder.name
            paths = trip_files(partition, tid)
            row = {"partition": partition, "trip_id": tid}
            try:
                track = read_track(paths["track"])
                nodes = read_nodes(paths["nodes"])
                arcs = read_arcs(paths["arcs"])
                route = read_route(paths["route"])
                row.update({
                    "gps_rows": len(track), "node_rows": len(nodes),
                    "arc_rows": len(arcs), "route_rows": len(route),
                    "invalid_gps_rows": int((
                        track[["longitude", "latitude", "timestamp"]].isna().any(axis=1)
                        | ~track["longitude"].between(-180, 180)
                        | ~track["latitude"].between(-90, 90)
                    ).sum()),
                    "duplicate_gps_rows": int(track.duplicated().sum()),
                    "backwards_timestamp_steps": int((track["timestamp"].diff() < 0).sum()),
                    "gaps_over_2s": int((track["timestamp"].diff() > 2).sum()),
                    "gaps_over_30s": int((track["timestamp"].diff() > 30).sum()),
                    "invalid_node_ids": int((
                        ~arcs["from_node_id"].between(0, len(nodes)-1)
                        | ~arcs["to_node_id"].between(0, len(nodes)-1)
                    ).sum()),
                    "route_min_arc_id": route["arc_id"].min() if len(route) else None,
                    "route_max_arc_id": route["arc_id"].max() if len(route) else None,
                })
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)
    print(out.groupby("partition").sum(numeric_only=True).to_string())
    print("Saved:", RESULTS_DIR / "dataset_audit.csv")

if __name__ == "__main__":
    main()
