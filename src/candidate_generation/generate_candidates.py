
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from src.config import (
    DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PARTITIONS,
    CANDIDATE_RADIUS_METRES,
    MAX_CANDIDATES_PER_POINT
)

from src.preprocessing.io_utils import trip_files
from src.preprocessing.network_utils import load_network, route_arc_ids

from src.preprocessing.geo_utils import (
    local_xy,
    point_to_segment_distance,
    bearing_deg
)


def candidates_for_trip(gps, nodes, arcs, route_ids):

    if arcs.empty:
        return pd.DataFrame()

    # Reference coordinates for local projection
    ref_lon = float(gps["longitude"].mean())
    ref_lat = float(gps["latitude"].mean())

    # Convert road endpoints into local metre coordinates
    ax, ay = local_xy(
        arcs["from_lon"],
        arcs["from_lat"],
        ref_lon,
        ref_lat
    )

    bx, by = local_xy(
        arcs["to_lon"],
        arcs["to_lat"],
        ref_lon,
        ref_lat
    )

    arc_work = arcs.copy()

    arc_work["ax"] = ax
    arc_work["ay"] = ay
    arc_work["bx"] = bx
    arc_work["by"] = by

    # Road segment length
    arc_work["road_length_m"] = np.hypot(
        bx - ax,
        by - ay
    )

    # Road direction
    arc_work["road_bearing_deg"] = bearing_deg(
        arcs["from_lon"].to_numpy(),
        arcs["from_lat"].to_numpy(),
        arcs["to_lon"].to_numpy(),
        arcs["to_lat"].to_numpy()
    )

    # Road connectivity degree
    degree = pd.concat([
        arcs["from_node_id"],
        arcs["to_node_id"]
    ]).value_counts().to_dict()

    arc_work["road_connectivity_degree"] = [
        int(degree.get(a, 0) + degree.get(b, 0))
        for a, b in zip(
            arcs["from_node_id"],
            arcs["to_node_id"]
        )
    ]

    # Build KD-tree using road segment midpoints
    midpoint_x = (
        arc_work["ax"].to_numpy()
        + arc_work["bx"].to_numpy()
    ) / 2

    midpoint_y = (
        arc_work["ay"].to_numpy()
        + arc_work["by"].to_numpy()
    ) / 2

    midpoint_tree = cKDTree(
        np.column_stack([midpoint_x, midpoint_y])
    )

    rows = []

    # Process every GPS observation
    for point in gps.itertuples():

        px, py = local_xy(
            point.longitude,
            point.latitude,
            ref_lon,
            ref_lat
        )

        # Find 32 nearest road-segment midpoints
        k = min(32, len(arc_work))

        _, nearest_indices = midpoint_tree.query(
            [px, py],
            k=k
        )

        nearest_indices = np.atleast_1d(
            nearest_indices
        ).astype(int)

        # Calculate exact point-to-segment distances
        distances = []

        for i in nearest_indices:

            arc = arc_work.iloc[i]

            distance = point_to_segment_distance(
                px,
                py,
                arc.ax,
                arc.ay,
                arc.bx,
                arc.by
            )

            distances.append((i, distance))

        # Keep candidates within the configured radius
        eligible = [
            (i, distance)
            for i, distance in distances
            if distance <= CANDIDATE_RADIUS_METRES
        ]

        # Always retain at least one candidate
        if not eligible:
            eligible = [
                min(distances, key=lambda item: item[1])
            ]

        # Sort by distance and limit candidate count
        eligible.sort(key=lambda item: item[1])

        eligible = eligible[
            :MAX_CANDIDATES_PER_POINT
        ]

        # Generate feature row for each candidate
        for rank, (i, distance) in enumerate(
            eligible,
            start=1
        ):

            arc = arc_work.iloc[i]

            # Heading difference
            if pd.notna(point.gps_heading_deg):

                heading_difference = abs(
                    (
                        point.gps_heading_deg
                        - arc.road_bearing_deg
                        + 180
                    ) % 360 - 180
                )

            else:
                heading_difference = np.nan

            # Route membership label
            is_route_member = int(
                int(arc.arc_id) in route_ids
            )

            rows.append({

                "partition": point.partition,

                "trip_id": point.trip_id,

                "observation_index": int(
                    point.observation_index
                ),

                "timestamp": float(
                    point.timestamp
                ),

                "segment_id": int(
                    point.segment_id
                ),

                "candidate_arc_id": int(
                    arc.arc_id
                ),

                "candidate_rank": rank,

                "gps_to_road_distance_m": float(
                    distance
                ),

                "road_length_m": float(
                    arc.road_length_m
                ),

                "road_bearing_deg": float(
                    arc.road_bearing_deg
                ),

                "road_connectivity_degree": int(
                    arc.road_connectivity_degree
                ),

                "gps_heading_deg": point.gps_heading_deg,

                "gps_speed_kmh": point.gps_speed_kmh,

                "dt_seconds": point.dt_seconds,

                "step_distance_m": point.step_distance_m,

                "gap_flag": int(
                    bool(point.gap_flag)
                ) if pd.notna(point.gap_flag) else 0,

                "heading_difference_deg": float(
                    heading_difference
                ) if pd.notna(heading_difference) else np.nan,

                "candidate_is_route_member": is_route_member,

                "label": is_route_member
            })

    return pd.DataFrame(rows)


def main():

    summaries = []

    all_by_partition = {
        partition: []
        for partition in PARTITIONS
    }

    for partition in PARTITIONS:

        print(f"\nProcessing partition: {partition}")

        partition_dir = DATA_DIR / partition

        for folder in sorted(
            p for p in partition_dir.iterdir()
            if p.is_dir()
        ):

            trip_id = folder.name

            print(f"Processing trip: {trip_id}")

            paths = trip_files(
                partition,
                trip_id
            )

            gps_path = (
                PROCESSED_DIR
                / partition
                / f"{trip_id}.csv"
            )

            gps = pd.read_csv(gps_path)

            # Ensure required observation index exists
            if "observation_index" not in gps.columns:
                gps["observation_index"] = range(len(gps))

            gps["partition"] = partition
            gps["trip_id"] = trip_id

            # Load road network
            nodes, arcs = load_network(
                paths["nodes"],
                paths["arcs"]
            )

            # Read ground-truth route arc IDs
            route_ids = route_arc_ids(
                paths["route"]
            )

            # Validate route-to-arc mapping
            valid_ids = set(
                arcs["arc_id"].astype(int)
            )

            unknown_ids = route_ids - valid_ids

            if unknown_ids:

                raise ValueError(
                    f"{partition}/{trip_id}: "
                    f"Route IDs not found in arc mapping. "
                    f"Examples: {sorted(unknown_ids)[:10]}. "
                    f"Check ARC_ID_MODE in src/config.py."
                )

            # Generate candidates
            candidates = candidates_for_trip(
                gps,
                nodes,
                arcs,
                route_ids
            )

            if not candidates.empty:

                all_by_partition[partition].append(
                    candidates
                )

                summaries.append({

                    "partition": partition,

                    "trip_id": trip_id,

                    "gps_observations": len(gps),

                    "candidate_rows": len(candidates),

                    "route_arc_count": len(route_ids),

                    "positive_candidate_rows": int(
                        candidates["label"].sum()
                    )

                })

    # Save candidate datasets
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for partition, frames in all_by_partition.items():

        if frames:

            combined = pd.concat(
                frames,
                ignore_index=True
            )

        else:

            combined = pd.DataFrame()

        output_path = (
            RESULTS_DIR
            / f"{partition}_candidates.csv"
        )

        combined.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nSaved {partition} candidates: "
            f"{len(combined)} rows"
        )

    # Save candidate summary
    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        RESULTS_DIR / "candidate_summary.csv",
        index=False
    )

    if not summary_df.empty:

        print("\n===== CANDIDATE SUMMARY =====")

        print(
            summary_df.groupby("partition")[
                [
                    "gps_observations",
                    "candidate_rows",
                    "route_arc_count",
                    "positive_candidate_rows"
                ]
            ].sum()
        )

    print("\nCandidate generation completed.")


if __name__ == "__main__":
    main()
