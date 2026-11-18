
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "experiments/results/train_features.csv"
SPLIT_PATH = ROOT / "experiments/splits/trip_split.csv"

OUTPUT_DIR = ROOT / "experiments/model_results/extra_trees"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading training feature dataset...")

df = pd.read_csv(DATA_PATH)
split_df = pd.read_csv(SPLIT_PATH)

df["trip_id"] = df["trip_id"].astype(str)
split_df["trip_id"] = split_df["trip_id"].astype(str)

df = df.merge(
    split_df,
    on="trip_id",
    how="inner",
    validate="many_to_one"
)

print("Total candidate rows:", len(df))
print("Total trips:", df["trip_id"].nunique())

# Verify that the split has no overlapping trips
assert split_df["trip_id"].is_unique
assert set(df["split"].unique()) == {"train", "validation"}

# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

FEATURES = [
    "gps_to_road_distance_m",
    "heading_difference_deg",
    "gps_speed_kmh",
    "dt_seconds",
    "step_distance_m",
    "gap_flag",
    "road_length_m",
    "road_connectivity_degree",
    "candidate_rank"
]

# Convert compass angles into circular features.
# This handles the 359-degree versus 1-degree issue.

for col in ["gps_heading_deg", "road_bearing_deg"]:
    radians = np.deg2rad(df[col])
    df[f"{col}_sin"] = np.sin(radians)
    df[f"{col}_cos"] = np.cos(radians)

FEATURES += [
    "gps_heading_deg_sin",
    "gps_heading_deg_cos",
    "road_bearing_deg_sin",
    "road_bearing_deg_cos"
]

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
y = df["label"].astype(int)

# --------------------------------------------------
# 4. TRAIN / VALIDATION SPLIT
# --------------------------------------------------

train_mask = df["split"] == "train"
val_mask = df["split"] == "validation"

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

print("\n===== DATASET DETAILS =====")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training trips:", df.loc[train_mask, "trip_id"].nunique())
print("Validation trips:", df.loc[val_mask, "trip_id"].nunique())
print("Number of features:", len(FEATURES))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nValidation label distribution:")
print(y_val.value_counts())

# --------------------------------------------------
# 5. HANDLE MISSING VALUES
# --------------------------------------------------

print("\nHandling missing values...")

imputer = SimpleImputer(strategy="median")

# Fit ONLY on training data
X_train = imputer.fit_transform(X_train)

# Apply the same learned medians to validation
X_val = imputer.transform(X_val)

# --------------------------------------------------
# 6. TRAIN EXTRA TREES
# --------------------------------------------------

print("\n===== TRAINING EXTRA TREES =====")

model = ExtraTreesClassifier(
    n_estimators=150,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# --------------------------------------------------
# 7. VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

start_time = time.time()

val_probabilities = model.predict_proba(X_val)[:, 1]

inference_time = time.time() - start_time

val_predictions = (val_probabilities >= 0.5).astype(int)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

metrics = {
    "accuracy": accuracy_score(y_val, val_predictions),
    "precision": precision_score(
        y_val, val_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, val_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, val_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_val, val_probabilities),
    "average_precision": average_precision_score(
        y_val, val_probabilities
    ),
    "training_time_seconds": training_time,
    "validation_inference_time_seconds": inference_time
}

print("\n========== EXTRA TREES RESULTS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification report:")
print(classification_report(
    y_val,
    val_predictions,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_val, val_predictions))

# --------------------------------------------------
# 9. SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "features": FEATURES
    },
    OUTPUT_DIR / "extra_trees_model.joblib"
)

prediction_df = df.loc[
    val_mask,
    [
        "trip_id",
        "observation_index",
        "candidate_arc_id",
        "label"
    ]
].copy()

prediction_df["predicted_probability"] = val_probabilities
prediction_df["predicted_label"] = val_predictions

prediction_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

with open(OUTPUT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nSaved all results to:", OUTPUT_DIR)
print("\nEXTRA TREES TRAINING COMPLETE!")
