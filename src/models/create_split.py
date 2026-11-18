
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "experiments/results/train_features.csv"
SPLIT_PATH = ROOT / "experiments/splits/trip_split.csv"

print("Loading feature dataset...")
df = pd.read_csv(DATA_PATH)

# Get unique trip IDs
trip_ids = sorted(df["trip_id"].unique().tolist())

# Split trips, not individual candidate rows
train_trips, val_trips = train_test_split(
    trip_ids,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# Create shared split file
split_df = pd.DataFrame({
    "trip_id": list(train_trips) + list(val_trips),
    "split": (
        ["train"] * len(train_trips)
        + ["validation"] * len(val_trips)
    )
})

SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
split_df.to_csv(SPLIT_PATH, index=False)

print("\n===== TRIP-WISE SPLIT COMPLETE =====")
print("Total trips:", len(trip_ids))
print("Training trips:", len(train_trips))
print("Validation trips:", len(val_trips))
print("Saved to:", SPLIT_PATH)
print("\nSplit distribution:")
print(split_df["split"].value_counts())
