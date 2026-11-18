# PathSnap – Person 1 Data & Preprocessing Package

The FastAPI backend setup, endpoints, prediction pipeline, and two-laptop
deployment instructions are documented in [BACKEND.md](BACKEND.md).

This package provides the Person 1 pipeline for the GPS-to-road map-matching project.

## What it does

- Audits the 80 training and 10 student-test trips.
- Reads `.track`, `.nodes`, `.arcs`, and `.route`.
- Preserves original data and creates processed GPS CSVs.
- Flags timestamp gaps and creates trajectory segments.
- Builds a road-network graph and edge geometry.
- Generates nearby road-arc candidates for each GPS observation.
- Extracts distance, heading, speed, time-gap, direction, and connectivity features.
- Creates Folium trajectory/network maps.
- Produces a notebook for exploration.
- Exports candidate-level CSVs and summary reports.

## Important data assumptions

The package follows the schema observed in the project scripts so far:

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude
- `.arcs`: from_node_id, to_node_id
- `.route`: arc_id

**Route-label caution:** the `.route` file appears to contain an ordered sequence of route arc IDs, not necessarily one label per GPS observation. Therefore, the pipeline does not silently pretend that route rows align one-to-one with GPS rows. It creates a clearly marked *route-membership heuristic label* only when the arc-ID mapping is valid. Validate the dataset's arc-ID convention and label semantics with the team/organizer before treating these labels as definitive ground truth.

By default, arc IDs are assumed to be zero-based row positions in `.arcs`. If your dataset uses another convention, change `ARC_ID_MODE` in `config.py`.

## Install

From the project root, activate your venv and run:

```bash
pip install -r requirements-person1.txt
```

If you already have pandas and numpy, install the remaining packages with the same command.

## Expected folder layout

Keep the original dataset exactly as it is:

```text
data/
  Student folder/
    train/
      <trip_id>/
        <trip_id>.track
        <trip_id>.nodes
        <trip_id>.arcs
        <trip_id>.route
    student_test/
      <trip_id>/
        ...
```

Copy the package folders into your repository root, merging with existing folders when prompted. Do not replace your existing `src/preprocessing/prepare_dataset.py` unless you want the package version.

## Run the pipeline

Run from the repository root:

```bash
python -m src.preprocessing.run_person1_pipeline
```

Or run stages individually:

```bash
python -m src.preprocessing.audit_dataset
python -m src.preprocessing.prepare_dataset
python -m src.preprocessing.build_network
python -m src.candidate_generation.generate_candidates
python -m src.features.build_features
python -m src.preprocessing.make_visualizations
```

## Outputs

```text
data/processed/
  train/<trip_id>.csv
  student_test/<trip_id>.csv

experiments/results/
  dataset_audit.csv
  preprocessing_summary.csv
  network_summary.csv
  candidate_summary.csv
  feature_summary.csv
  train_features.csv
  student_test_features.csv

visualizations/
  <partition>_<trip_id>_trajectory.html
  <partition>_<trip_id>_network.html
```

The feature tables are **candidate-level**: one row per (GPS observation, candidate road arc). They include `trip_id`, `observation_index`, `candidate_arc_id`, `label`, and engineered features.

## Handoff contract for Person 2

Use:
- `experiments/results/train_features.csv`
- `experiments/results/student_test_features.csv`

Suggested features:
- `gps_to_road_distance_m`
- `heading_difference_deg`
- `gps_heading_deg`
- `road_heading_deg`
- `gps_speed_kmh`
- `dt_seconds`
- `step_distance_m`
- `gap_flag`
- `segment_id`
- `road_length_m`
- `road_bearing_deg`
- `road_connectivity_degree`
- `candidate_rank`
- `candidate_is_route_member`
- `label`

Do not use `candidate_is_route_member` as an input feature if `label` is derived from route membership. That would leak the target. The default feature exporter excludes it from `MODEL_FEATURES`.

## Known limitations to validate

1. Arc IDs may not equal row indices. Check `.route` values against `.arcs` row count and inspect the first few records.
2. Candidate generation uses local equirectangular projection around each trip and a configurable search radius.
3. Candidate labels are heuristic route-membership labels, not pointwise ground truth, unless the benchmark documentation confirms this mapping.
4. The nearest-candidate search is intentionally straightforward and CPU-based. It is suitable for the current small dataset; spatial indexing can be added later if needed.
