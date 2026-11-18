import folium
import pandas as pd
from src.config import DATA_DIR, PROCESSED_DIR, VISUALIZATIONS_DIR, PARTITIONS, MAP_TRIPS_PER_PARTITION
from src.preprocessing.io_utils import trip_files, read_nodes
from src.preprocessing.network_utils import load_network

def main():
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for partition in PARTITIONS:
        folders = sorted(p for p in (DATA_DIR / partition).iterdir() if p.is_dir())[:MAP_TRIPS_PER_PARTITION]
        for folder in folders:
            tid = folder.name
            gps = pd.read_csv(PROCESSED_DIR / partition / f"{tid}.csv")
            paths = trip_files(partition, tid)
            nodes, arcs = load_network(paths["nodes"], paths["arcs"])
            center = [gps["latitude"].mean(), gps["longitude"].mean()]
            m = folium.Map(location=center, zoom_start=14)
            coords = gps[["latitude", "longitude"]].values.tolist()
            folium.PolyLine(coords, color="red", weight=3, tooltip="GPS trajectory").add_to(m)
            folium.CircleMarker(coords[0], radius=5, color="green", tooltip="Start").add_to(m)
            folium.CircleMarker(coords[-1], radius=5, color="red", tooltip="End").add_to(m)
            for r in arcs.itertuples():
                folium.PolyLine(
                    [[r.from_lat, r.from_lon], [r.to_lat, r.to_lon]],
                    color="blue", weight=1, opacity=0.45
                ).add_to(m)
            out = VISUALIZATIONS_DIR / f"{partition}_{tid}_trajectory_network.html"
            m.save(str(out))
            print("Saved", out)

if __name__ == "__main__":
    main()
