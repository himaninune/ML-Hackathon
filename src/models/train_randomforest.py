import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline


# train.py is inside src/models/random_forest/
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
MODEL_DIR = ROOT / "experiments" / "models" / "random_forest"
VALIDATION_MANIFEST = RESULTS_DIR / "label_validation.json"

TRAIN_FILE = RESULTS_DIR / "train_features.csv"
TEST_FILE = RESULTS_DIR / "student_test_features.csv"  # schema check only

RANDOM_STATE = 42
N_SPLITS = 5

# Metadata and IDs are deliberately excluded.
NUMERIC_FEATURES = [
    "gps_to_road_distance_m",
    "heading_difference_deg",
    "gps_speed_kmh",
    "dt_seconds",
    "step_distance_m",
    "gap_flag",
    "road_length_m",
    "road_connectivity_degree",
    "candidate_rank",
]

# Encode compass angles as sine/cosine to handle the 0°/360° boundary.
ANGLE_FEATURES = [
    "gps_heading_deg",
    "road_bearing_deg",
]

REQUIRED_METADATA = [
    "trip_id",
    "observation_index",
    "candidate_arc_id",
    "label",
]


def validate_labels():
    """Require evidence that labels came from validated observation alignment."""
    if not VALIDATION_MANIFEST.exists():
        raise RuntimeError(
            f"Missing {VALIDATION_MANIFEST}. Do not train until labels have "
            "been regenerated from validated GPS-to-route alignment."
        )

    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))

    if manifest.get("status") != "validated":
        raise RuntimeError("Label manifest status is not 'validated'.")

    if manifest.get("method") != "route_sequence_alignment_v1":
        raise RuntimeError(
            "Expected labels from route_sequence_alignment_v1."
        )

    if manifest.get("target_column") != "label":
        raise RuntimeError("Manifest must identify 'label' as the target.")

    if manifest.get("uncertain_observations_excluded") is not True:
        raise RuntimeError(
            "Manifest must confirm uncertain observations were excluded."
        )

    if manifest.get("candidate_misses_reported") is not True:
        raise RuntimeError(
            "Manifest must confirm candidate misses were audited."
        )


def load_training_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    required = set(REQUIRED_METADATA + NUMERIC_FEATURES + ANGLE_FEATURES)

    for name, frame in (("training", train), ("student-test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {missing}")

    # Student-test features are checked for schema only; never used for fitting.
    if train["label"].isna().any():
        raise ValueError("Training labels contain missing values.")

    labels = set(train["label"].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0 and 1; got {labels}.")

    return train


def make_features(frame):
    X = frame[NUMERIC_FEATURES].copy()

    for column in ANGLE_FEATURES:
        radians = np.deg2rad(pd.to_numeric(frame[column], errors="coerce"))
        X[f"{column}_sin"] = np.sin(radians)
        X[f"{column}_cos"] = np.cos(radians)

    # Identifiers, timestamp, segment_id, and labels are not predictors.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def main():
    validate_labels()
    train = load_training_data()

    X = make_features(train)
    y = train["label"].astype(int)
    groups = train["trip_id"].astype(str)

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} independent trips for GroupKFold; "
            f"found {groups.nunique()}."
        )

    # Check each class appears in enough trips for the requested grouped folds.
    class_group_counts = train.groupby("label")["trip_id"].nunique()
    if (class_group_counts < N_SPLITS).any():
        raise ValueError(
            "Insufficient trips containing each class for this GroupKFold: "
            f"{class_group_counts.to_dict()}"
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score, zero_division=0
        ),
        "recall": make_scorer(
            recall_score, zero_division=0
        ),
        "macro_f1": make_scorer(
            f1_score, average="macro", zero_division=0
        ),
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    cv_summary = {}
    for metric in scoring:
        scores = cv_result[f"test_{metric}"]
        cv_summary[metric] = {
            "fold_scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }

    # Fit final model on training trips only.
    start = time.perf_counter()
    pipeline.fit(X, y)
    training_seconds = time.perf_counter() - start

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cv_output = RESULTS_DIR / "random_forest"
    cv_output.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_DIR / "random_forest_model.joblib")

    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(list(X.columns), indent=2),
        encoding="utf-8",
    )

    (MODEL_DIR / "model_configuration.json").write_text(
        json.dumps({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_splits": N_SPLITS,
            "group_column": "trip_id",
            "training_rows": len(train),
            "training_trips": int(groups.nunique()),
            "training_seconds": training_seconds,
            "features": list(X.columns),
        }, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"metric": metric, **values}
        for metric, values in cv_summary.items()
    ]).to_csv(
        cv_output / "random_forest_cv_results.csv",
        index=False,
    )

    print("Grouped cross-validation:")
    print(json.dumps(cv_summary, indent=2))
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Saved model to: {MODEL_DIR}")


if __name__ == "__main__":
    main()