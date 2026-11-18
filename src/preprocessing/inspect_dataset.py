
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Student folder"

PARTITIONS = ["train", "student_test"]

REQUIRED_EXTENSIONS = [
    ".track",
    ".nodes",
    ".arcs",
    ".route"
]


def read_file(path, columns):
    """Read whitespace-separated files without headers."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
        engine="python"
    )


def inspect_trip(trip_dir, partition):
    """Inspect all four files belonging to one trip."""

    trip_id = trip_dir.name

    paths = {
        ext: trip_dir / f"{trip_id}{ext}"
        for ext in REQUIRED_EXTENSIONS
    }

    result = {
        "partition": partition,
        "trip_id": trip_id,
        "files_complete": all(p.exists() for p in paths.values()),
    }

    if not result["files_complete"]:
        result["error"] = "Missing one or more files"
        return result

    try:
        track = read_file(
            paths[".track"],
            ["longitude", "latitude", "timestamp"]
        )

        nodes = read_file(
            paths[".nodes"],
            ["longitude", "latitude"]
        )

        arcs = read_file(
            paths[".arcs"],
            ["from_node_id", "to_node_id"]
        )

        route = read_file(
            paths[".route"],
            ["arc_id"]
        )

        # Convert all columns to numeric values
        track = track.apply(pd.to_numeric, errors="coerce")
        nodes = nodes.apply(pd.to_numeric, errors="coerce")
        arcs = arcs.apply(pd.to_numeric, errors="coerce")
        route = route.apply(pd.to_numeric, errors="coerce")

        # GPS checks
        invalid_gps = (
            track.isna().any(axis=1)
            | ~np.isfinite(track).all(axis=1)
            | ~track["longitude"].between(-180, 180)
            | ~track["latitude"].between(-90, 90)
        )

        duplicate_gps = track.duplicated().sum()

        # Timestamp checks
        time_diff = track["timestamp"].diff().dropna()

        backwards_timestamps = (time_diff < 0).sum()
        duplicate_timestamps = (time_diff == 0).sum()
        large_gaps = (time_diff > 2).sum()

        # Node checks
        invalid_nodes = (
            nodes.isna().any(axis=1)
            | ~np.isfinite(nodes).all(axis=1)
            | ~nodes["longitude"].between(-180, 180)
            | ~nodes["latitude"].between(-90, 90)
        )

        # Arc checks
        invalid_arcs = (
            arcs.isna().any(axis=1)
            | (arcs < 0).any(axis=1)
            | (arcs >= len(nodes)).any(axis=1)
        )

        # Route checks
        invalid_routes = (
            route.isna().any(axis=1)
            | (route["arc_id"] < 0)
            | (route["arc_id"] >= len(arcs))
        )

        # Summary
        result.update({
            "gps_observations": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "route_arcs": len(route),

            "invalid_gps_rows": int(invalid_gps.sum()),
            "duplicate_gps_rows": int(duplicate_gps),

            "backwards_timestamps": int(backwards_timestamps),
            "duplicate_timestamps": int(duplicate_timestamps),
            "gaps_over_2_seconds": int(large_gaps),

            "invalid_node_rows": int(invalid_nodes.sum()),
            "invalid_arc_rows": int(invalid_arcs.sum()),
            "invalid_route_rows": int(invalid_routes.sum()),

            "trajectory_duration_seconds": (
                track["timestamp"].max() - track["timestamp"].min()
                if len(track) > 0 else np.nan
            ),

            "average_sampling_interval": (
                time_diff.mean() if len(time_diff) > 0 else np.nan
            ),

            "one_second_interval_percentage": (
                (time_diff == 1).mean() * 100
                if len(time_diff) > 0 else np.nan
            ),

            "error": ""
        })

    except Exception as e:
        result["error"] = str(e)

    return result


def main():

    all_results = []

    for partition in PARTITIONS:

        partition_dir = DATA_DIR / partition

        if not partition_dir.exists():
            print(f"Missing partition: {partition}")
            continue

        trip_dirs = sorted(
            p for p in partition_dir.iterdir()
            if p.is_dir()
        )

        print(f"\nInspecting {partition}: {len(trip_dirs)} trips")

        for trip_dir in trip_dirs:
            result = inspect_trip(trip_dir, partition)
            all_results.append(result)

    report = pd.DataFrame(all_results)

    output_dir = ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dataset_audit.csv"

    report.to_csv(output_file, index=False)

    print("\n========== DATASET AUDIT ==========")

    print(
        report.groupby("partition")["trip_id"]
        .count()
    )

    print("\nTotal GPS observations:")

    print(
        report.groupby("partition")["gps_observations"]
        .sum()
    )

    print("\nAudit report saved to:")
    print(output_file)

    print("\n===================================")


if __name__ == "__main__":
    main()
