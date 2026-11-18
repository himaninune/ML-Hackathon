from pathlib import Path
import pandas as pd
from src.config import DATA_DIR

def trip_files(partition, trip_id):
    folder = DATA_DIR / partition / str(trip_id)
    return {
        "track": folder / f"{trip_id}.track",
        "nodes": folder / f"{trip_id}.nodes",
        "arcs": folder / f"{trip_id}.arcs",
        "route": folder / f"{trip_id}.route",
    }

def read_track(path):
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["longitude", "latitude", "timestamp"])
    for col in ["longitude", "latitude", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def read_nodes(path):
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=["longitude", "latitude"])

def read_arcs(path):
    return pd.read_csv(path, sep=r"\s+", header=None,
                       names=["from_node_id", "to_node_id"])

def read_route(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])
