import pandas as pd
from src.config import RESULTS_DIR, PARTITIONS

# Explicitly exclude candidate_is_route_member: it is used to construct
# the heuristic target and would leak the target into model inputs.
MODEL_FEATURES = [
    "gps_to_road_distance_m",
    "heading_difference_deg",
    "gps_heading_deg",
    "road_bearing_deg",
    "gps_speed_kmh",
    "dt_seconds",
    "step_distance_m",
    "gap_flag",
    "segment_id",
    "road_length_m",
    "road_connectivity_degree",
    "candidate_rank",
]

def main():
    for partition in PARTITIONS:
        path = RESULTS_DIR / f"{partition}_candidates.csv"
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"No candidate rows found in {path}")
        df["label"] = df["label"].astype(int)
        # Keep metadata + features + target for the model team.
        columns = [
            "partition", "trip_id", "observation_index",
            "timestamp", "candidate_arc_id", "label"
        ] + MODEL_FEATURES
        out = df[columns].copy()
        out.to_csv(RESULTS_DIR / f"{partition}_features.csv", index=False)
        print(partition, "rows:", len(out), "positive:", int(out["label"].sum()))
    print("Feature tables saved.")

if __name__ == "__main__":
    main()
