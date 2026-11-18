from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Student folder"
DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", str(DATA_DIR))).resolve()
PROCESSED_DIR = Path(os.getenv("PATHSNAP_PROCESSED_DIR", str(ROOT / "data" / "processed"))).resolve()
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

# Backend deployment defaults. Paths can be overridden without changing code.
MODEL_PATH = Path(os.getenv(
    "XGBOOST_MODEL_PATH",
    str(ROOT / "experiments" / "model_results" / "xgboost" / "xgboost_baseline.json"),
)).resolve()
API_HOST = os.getenv("PATHSNAP_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PATHSNAP_PORT", "8000"))
CORS_ORIGINS = [v.strip() for v in os.getenv("PATHSNAP_CORS_ORIGINS", "http://localhost:3000").split(",") if v.strip()]

# Avoid generating huge Folium files for every trip unless requested.
MAP_TRIPS_PER_PARTITION = 3
