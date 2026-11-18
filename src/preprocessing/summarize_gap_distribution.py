
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

all_gaps = []

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

        gaps = df["timestamp"].diff().dropna()

        for gap in gaps:
            if gap > 2:
                all_gaps.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    "gap_seconds": gap
                })

gaps_df = pd.DataFrame(all_gaps)

bins = [2, 5, 30, 300, 1800, float("inf")]
labels = [
    "2–5 sec",
    "5–30 sec",
    "30 sec–5 min",
    "5–30 min",
    "Over 30 min"
]

gaps_df["gap_category"] = pd.cut(
    gaps_df["gap_seconds"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=False
)

summary = (
    gaps_df.groupby(
        ["partition", "gap_category"],
        observed=False
    )
    .size()
    .unstack(fill_value=0)
)

print("\n===== GAP DISTRIBUTION =====")
print(summary.to_string())

print("\n===== OVERALL GAP COUNTS =====")
print(gaps_df["gap_category"].value_counts().sort_index())

output_path = ROOT / "experiments" / "results" / "gap_distribution.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_path)

print(f"\nSaved report to {output_path}")
