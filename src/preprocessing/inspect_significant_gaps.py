
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

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


results = []

for partition in ["train", "student_test"]:
    partition_dir = DATA_DIR / partition

    for trip_dir in partition_dir.iterdir():
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name
        track_file = trip_dir / f"{trip_id}.track"

        if not track_file.exists():
            continue

        df = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        df["gap"] = df["timestamp"].diff()

        for idx in df.index[df["gap"] > 30]:
            previous = df.loc[idx - 1]
            current = df.loc[idx]

            distance = haversine(
                previous["longitude"],
                previous["latitude"],
                current["longitude"],
                current["latitude"]
            )

            gap = current["gap"]

            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "gap_seconds": gap,
                "distance_metres": distance,
                "implied_speed_kmh": distance / gap * 3.6
            })


output = pd.DataFrame(results)

output = output.sort_values(
    "gap_seconds",
    ascending=False
)

print("\n===== SIGNIFICANT GAP INVESTIGATION =====")
print(output.to_string(index=False))

save_path = ROOT / "experiments" / "results"
save_path.mkdir(parents=True, exist_ok=True)

output.to_csv(
    save_path / "significant_gap_investigation.csv",
    index=False
)

print("\nTotal gaps investigated:", len(output))
print("Saved significant_gap_investigation.csv")
