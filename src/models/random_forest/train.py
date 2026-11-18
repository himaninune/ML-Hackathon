import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "train_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "random_forest"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "random_forest"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

required_columns = [
    "partition",
    "trip_id",
    "observation_index",
    "timestamp",
    "candidate_arc_id",
    "label",
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
    "candidate_rank"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. SELECT TRAINING PARTITION
# ============================================================

print("\nSelecting training partition...")

print("Available partitions:")
print(df["partition"].value_counts())

df = df[
    df["partition"].astype(str).str.lower() == "train"
].copy()

if df.empty:
    raise ValueError(
        "No rows found with partition='train'. "
        "Check your partition column values."
    )

print("Training partition shape:", df.shape)


# ============================================================
# 4. CHECK LABELS
# ============================================================

print("\nChecking labels...")

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"])

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Labels must contain only 0 and 1.")

df["label"] = df["label"].astype(int)

if df["label"].nunique() != 2:
    raise ValueError(
        "Training data must contain both positive (1) "
        "and negative (0) labels."
    )

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

# Excluded:
# partition             -> dataset metadata
# trip_id               -> used only for grouped validation
# observation_index     -> identifier
# timestamp             -> not directly used
# candidate_arc_id      -> candidate identifier, not a feature
# segment_id            -> segment metadata

FEATURE_COLUMNS = [
    "gps_to_road_distance_m",
    "heading_difference_deg",
    "gps_heading_deg",
    "road_bearing_deg",
    "gps_speed_kmh",
    "dt_seconds",
    "step_distance_m",
    "gap_flag",
    "road_length_m",
    "road_connectivity_degree",
    "candidate_rank"
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "trip_id"

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()
groups = df[GROUP_COLUMN].copy()


# ============================================================
# 6. FEATURE PREPROCESSING
# ============================================================

print("\nPreprocessing features...")

# Convert all features to numeric values.
for col in FEATURE_COLUMNS:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

# Replace infinite values with NaN.
X = X.replace([np.inf, -np.inf], np.nan)

# Convert angular features into sine/cosine representations.
# This handles the circular nature of angles.

angular_columns = [
    "heading_difference_deg",
    "gps_heading_deg",
    "road_bearing_deg"
]

for col in angular_columns:
    radians = np.deg2rad(X[col])

    X[f"{col}_sin"] = np.sin(radians)
    X[f"{col}_cos"] = np.cos(radians)

# Remove the original angular columns.
X = X.drop(columns=angular_columns)

# Remove columns that contain no usable values.
all_missing_columns = X.columns[X.isna().all()].tolist()

if all_missing_columns:
    print("Removing completely empty columns:", all_missing_columns)
    X = X.drop(columns=all_missing_columns)

if X.shape[1] == 0:
    raise ValueError("No usable features remain after preprocessing.")

print("Final features:")
print(list(X.columns))

print("Number of features:", X.shape[1])


# ============================================================
# 7. GROUP-BASED TRAIN / VALIDATION SPLIT
# ============================================================

print("\nSplitting data by trip_id...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, val_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_val = X.iloc[val_indices]

y_train = y.iloc[train_indices]
y_val = y.iloc[val_indices]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

print(
    "Training trips:",
    groups.iloc[train_indices].nunique()
)

print(
    "Validation trips:",
    groups.iloc[val_indices].nunique()
)

if y_train.nunique() != 2 or y_val.nunique() != 2:
    raise ValueError(
        "The selected split does not contain both classes "
        "in training and validation. Try another random_state."
    )


# ============================================================
# 8. BUILD RANDOM FOREST MODEL
# ============================================================

print("\nBuilding Random Forest model...")

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "random_forest",
        random_forest
    )
])


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 10. VALIDATION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_val)

y_prob = model.predict_proba(X_val)

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

loss = log_loss(
    y_val,
    y_prob,
    labels=[0, 1]
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Log Loss  : {loss:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# ============================================================
# 11. SAVE VALIDATION METRICS
# ============================================================

metrics = {
    "model": "Random Forest",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "log_loss": float(loss),
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "training_trips": int(
        groups.iloc[train_indices].nunique()
    ),
    "validation_trips": int(
        groups.iloc[val_indices].nunique()
    )
}

metrics_path = RESULT_DIR / "evaluation_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved:", metrics_path)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

rf_model = model.named_steps["random_forest"]

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_path = RESULT_DIR / "feature_importance.csv"

importance_df.to_csv(
    importance_path,
    index=False
)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

print("\nFeature importance saved:", importance_path)


# ============================================================
# 13. TRAIN FINAL MODEL ON ALL TRAINING DATA
# ============================================================

print("\nTraining final model on complete training partition...")

final_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    )
])

final_model.fit(X, y)

print("Final model training completed!")


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

model_path = MODEL_DIR / "random_forest_model.joblib"

joblib.dump(final_model, model_path)

print("\nModel saved:", model_path)


# ============================================================
# 15. SAVE MODEL CONFIGURATION
# ============================================================

config = {
    "model_name": "Random Forest",
    "algorithm": "RandomForestClassifier",
    "features": list(X.columns),
    "target": TARGET_COLUMN,
    "number_of_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42
}

config_path = MODEL_DIR / "model_config.json"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n==========================================")
print("RANDOM FOREST TRAINING COMPLETED")
print("==========================================")

print("\nGenerated files:")

print("1. Trained model:")
print(model_path)

print("\n2. Evaluation metrics:")
print(metrics_path)

print("\n3. Feature importance:")
print(importance_path)

print("\n4. Model configuration:")
print(config_path)

print("\n==========================================")