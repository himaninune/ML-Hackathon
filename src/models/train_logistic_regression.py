
import os
import time
import json
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
DATA_PATH = "experiments/results/train_features (1).csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# 2. Select features
FEATURES = [
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

TARGET = "label"
GROUP = "trip_id"

# Use only the training partition if present
if "partition" in df.columns:
    df = df[df["partition"] == "train"].copy()

# Remove rows with missing target or trip ID
df = df.dropna(subset=[TARGET, GROUP])

X = df[FEATURES]
y = df[TARGET].astype(int)
groups = df[GROUP]

print("\nClass distribution:")
print(y.value_counts())

# 3. Split by trip ID to avoid data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Build Logistic Regression pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    ))
])

# 5. Train model
print("\nTraining Logistic Regression...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed!")
print("Training time:", round(training_time, 2), "seconds")

# 6. Make predictions
start_time = time.time()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

inference_time = time.time() - start_time

# 7. Evaluate model
metrics = {
    "model": "Logistic Regression",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_macro": precision_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "training_time_seconds": training_time,
    "inference_time_seconds": inference_time
}

print("\nMODEL RESULTS")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save model and results
os.makedirs("models", exist_ok=True)
os.makedirs("experiments/results", exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")

with open(
    "experiments/results/logistic_regression_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

# 9. Save predictions
predictions = pd.DataFrame({
    "trip_id": groups.iloc[test_idx].values,
    "candidate_arc_id": df.iloc[test_idx]["candidate_arc_id"].values,
    "actual_label": y_test.values,
    "predicted_label": y_pred,
    "probability_route_member": y_prob
})

predictions.to_csv(
    "experiments/results/logistic_regression_predictions.csv",
    index=False
)

print("\nModel and results saved successfully!")
