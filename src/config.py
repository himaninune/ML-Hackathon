from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Student folder"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "experiments" / "results"
VISUALIZATIONS_DIR = ROOT / "visualizations"

PARTITIONS = ("train", "student_test", "evaluator_test")
# Dataset convention assumed by this package:
# .route arc IDs are zero-based row positions in .arcs.
# Change to "one_based" if the dataset documentation says so.
ARC_ID_MODE = "zero_based"

# Gap handling policy based on the observed dataset audit.
GAP_FLAG_SECONDS = 30
SEGMENT_BREAK_SECONDS = 300

# Candidate generation
CANDIDATE_RADIUS_METRES = 100.0
MAX_CANDIDATES_PER_POINT = 8

# Avoid generating huge Folium files for every trip unless requested.
MAP_TRIPS_PER_PARTITION = 3
