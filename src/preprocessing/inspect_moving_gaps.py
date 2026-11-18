
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

TARGETS = {
    "00000055": 184,
    "00000070": 152
}

def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


for trip_id, target_gap in TARGETS.items():

    path = (
        DATA_DIR / "train" / trip_id
        / f"{trip_id}.track"
    )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    df["gap"] = df["timestamp"].diff()

    idx = (df["gap"] - target_gap).abs().idxmin()

    start = max(0, idx - 5)
    end = min(len(df), idx + 6)

    print(f"\n===== TRIP {trip_id} =====")
    print(f"Investigating gap: {df.loc[idx, 'gap']} seconds")

    section = df.loc[start:end-1].copy()

    section["distance_from_previous_m"] = 0.0

    for i in section.index:
        if i > 0:
            previous = df.loc[i - 1]
            current = df.loc[i]

            section.loc[i, "distance_from_previous_m"] = haversine(
                previous["longitude"],
                previous["latitude"],
                current["longitude"],
                current["latitude"]
            )

    print(section.to_string(index=True))
