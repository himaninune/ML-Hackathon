
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

records = []

for partition in ["train", "student_test"]:

    folder = DATA_DIR / partition

    for trip_dir in sorted(folder.iterdir()):

        if not trip_dir.is_dir():
            continue

        track_file = trip_dir / f"{trip_dir.name}.track"

        track = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        timestamps = pd.to_numeric(
            track["timestamp"], errors="coerce"
        )

        gaps = timestamps.diff().dropna()
        gaps = gaps[gaps >= 0]

        if len(gaps) == 0:
            continue

        records.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "observations": len(track),
            "median_gap": gaps.median(),
            "mean_gap": gaps.mean(),
            "max_gap": gaps.max(),
            "gaps_over_2s": (gaps > 2).sum(),
            "gaps_over_5s": (gaps > 5).sum(),
            "gaps_over_10s": (gaps > 10).sum(),
            "gaps_over_30s": (gaps > 30).sum(),
            "gap_95th_percentile": gaps.quantile(0.95)
        })

report = pd.DataFrame(records)

output = ROOT / "experiments" / "results"
output.mkdir(parents=True, exist_ok=True)

report.to_csv(
    output / "timestamp_gap_report.csv",
    index=False
)

print("\n===== TIMESTAMP GAP ANALYSIS =====")

for partition, group in report.groupby("partition"):

    print(f"\n{partition.upper()}")

    print(group[[
        "median_gap",
        "mean_gap",
        "max_gap",
        "gaps_over_2s",
        "gaps_over_5s",
        "gaps_over_10s",
        "gaps_over_30s"
    ]].describe())

    print("\nTrips with largest gaps:")

    print(
        group.nlargest(5, "max_gap")[
            ["trip_id", "max_gap", "gaps_over_2s"]
        ].to_string(index=False)
    )

print("\nSaved timestamp_gap_report.csv")
