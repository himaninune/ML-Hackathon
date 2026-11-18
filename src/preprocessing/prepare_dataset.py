
import pandas as pd
from pathlib import Path

from src.config import (
    DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PARTITIONS,
    GAP_FLAG_SECONDS,
    SEGMENT_BREAK_SECONDS
)

from src.preprocessing.io_utils import trip_files, read_track

from src.preprocessing.geo_utils import (
    haversine,
    bearing_deg
)


def process_trip(track_path, output_path):

    # Read original GPS observations
    df = read_track(track_path)

    original_count = len(df)

    # Remove invalid observations
    df = df.dropna(
        subset=["longitude", "latitude", "timestamp"]
    ).copy()

    df = df[
        df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
    ].copy()

    # Remove exact duplicate observations
    df = df.drop_duplicates(
        subset=["longitude", "latitude", "timestamp"]
    ).reset_index(drop=True)

    # Ensure timestamps are chronological
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"Non-monotonic timestamps in {track_path}"
        )

    # Unique observation index within each trip
    df["observation_index"] = range(len(df))

    # Time difference
    df["dt_seconds"] = df["timestamp"].diff()

    # Distance between consecutive GPS points
    df["step_distance_m"] = 0.0

    if len(df) > 1:

        df.loc[1:, "step_distance_m"] = haversine(
            df["longitude"].iloc[:-1].to_numpy(),
            df["latitude"].iloc[:-1].to_numpy(),
            df["longitude"].iloc[1:].to_numpy(),
            df["latitude"].iloc[1:].to_numpy()
        )

    # GPS speed in km/h
    valid_dt = df["dt_seconds"].replace(0, float("nan"))

    df["gps_speed_kmh"] = (
        df["step_distance_m"] / valid_dt
    ) * 3.6

    # GPS heading/direction
    df["gps_heading_deg"] = float("nan")

    if len(df) > 1:

        df.loc[1:, "gps_heading_deg"] = bearing_deg(
            df["longitude"].iloc[:-1].to_numpy(),
            df["latitude"].iloc[:-1].to_numpy(),
            df["longitude"].iloc[1:].to_numpy(),
            df["latitude"].iloc[1:].to_numpy()
        )

    # Flag timestamp gaps above 30 seconds
    df["gap_flag"] = (
        df["dt_seconds"] > GAP_FLAG_SECONDS
    )

    # Split trajectory after gaps above 5 minutes
    df["segment_break"] = (
        df["dt_seconds"] > SEGMENT_BREAK_SECONDS
    )

    df["segment_id"] = (
        df["segment_break"]
        .fillna(False)
        .astype(int)
        .cumsum()
    )

    # Save processed GPS observations
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    return {
        "original_rows": original_count,
        "processed_rows": len(df),
        "removed_rows": original_count - len(df),
        "flagged_gaps": int(df["gap_flag"].sum()),
        "segment_breaks": int(df["segment_break"].sum()),
        "segments": int(df["segment_id"].nunique())
    }


def main():

    summaries = []

    for partition in PARTITIONS:

        partition_dir = DATA_DIR / partition

        print(f"\nProcessing {partition}...")

        for folder in sorted(
            p for p in partition_dir.iterdir()
            if p.is_dir()
        ):

            trip_id = folder.name

            paths = trip_files(
                partition,
                trip_id
            )

            output_path = (
                PROCESSED_DIR
                / partition
                / f"{trip_id}.csv"
            )

            summary = process_trip(
                paths["track"],
                output_path
            )

            summary["partition"] = partition
            summary["trip_id"] = trip_id

            summaries.append(summary)

    # Save preprocessing report
    report = pd.DataFrame(summaries)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        RESULTS_DIR / "preprocessing_summary.csv",
        index=False
    )

    print("\n===== PREPROCESSING COMPLETE =====")

    print("Trips processed:", len(report))

    print(
        "Original GPS rows:",
        report["original_rows"].sum()
    )

    print(
        "Processed GPS rows:",
        report["processed_rows"].sum()
    )

    print(
        "Removed rows:",
        report["removed_rows"].sum()
    )

    print(
        "Flagged gaps:",
        report["flagged_gaps"].sum()
    )

    print(
        "Trajectory breaks:",
        report["segment_breaks"].sum()
    )

    print("\nPartition summary:")

    print(
        report.groupby("partition")[
            [
                "original_rows",
                "processed_rows",
                "removed_rows",
                "flagged_gaps",
                "segment_breaks"
            ]
        ].sum()
    )

    print("\nSaved processed data to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
