
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

# Trips with unusually large gaps
TARGET_TRIPS = {
    "train": ["00000090", "00000078", "00000069",
              "00000021", "00000011"],
    "student_test": ["00000065", "00000076"]
}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate geographical distance in metres."""

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


results = []

for partition, trip_ids in TARGET_TRIPS.items():

    for trip_id in trip_ids:

        path = (
            DATA_DIR / partition / trip_id / f"{trip_id}.track"
        )

        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        df["gap_seconds"] = df["timestamp"].diff()

        gap_indices = df.index[
            df["gap_seconds"] > 2
        ]

        for idx in gap_indices:

            previous = df.loc[idx - 1]
            current = df.loc[idx]

            distance = haversine(
                previous["longitude"],
                previous["latitude"],
                current["longitude"],
                current["latitude"]
            )

            gap = current["timestamp"] - previous["timestamp"]

            speed = distance / gap

            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "previous_timestamp": previous["timestamp"],
                "current_timestamp": current["timestamp"],
                "gap_seconds": gap,
                "previous_longitude": previous["longitude"],
                "previous_latitude": previous["latitude"],
                "current_longitude": current["longitude"],
                "current_latitude": current["latitude"],
                "distance_metres": distance,
                "implied_speed_mps": speed,
                "implied_speed_kmh": speed * 3.6
            })


output = pd.DataFrame(results)

output = output.sort_values(
    "gap_seconds", ascending=False
)

save_dir = ROOT / "experiments" / "results"
save_dir.mkdir(parents=True, exist_ok=True)

output.to_csv(
    save_dir / "large_gap_investigation.csv",
    index=False
)

print("\n===== LARGE GAP INVESTIGATION =====")

print(output.head(20).to_string(index=False))

print("\nSaved large_gap_investigation.csv")
