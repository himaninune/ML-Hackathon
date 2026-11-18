import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss, accuracy_score, precision_score, recall_score

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TRAIN_CSV = "experiments/results/train_features.csv"
TEST_CSV = "experiments/results/student_test_features.csv"
RESULTS_DIR = "experiments/results"
MODEL_OUTPUT_DIR = "experiments/model_results/mlp"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

TRIP_ID_COL = "trip_id"
OBS_ID_COL = "observation_index"
ARC_ID_COL = "candidate_arc_id"
TARGET_COL = "label"

FEATURE_COLS = [
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
    "candidate_rank",
]

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,                 # L2 regularization
    "batch_size": 2048,            # Fast mini-batch updates on 1.35M rows
    "learning_rate_init": 1e-3,
    "max_iter": 40,
    "early_stopping": True,        # Uses 10% internal validation split for early stopping
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "random_state": 42,
    "verbose": True,
}

# ==============================================================================
# 2. METRICS HELPER
# ==============================================================================
def compute_observation_top1_accuracy(df: pd.DataFrame, probs: np.ndarray) -> float:
    eval_df = df[[TRIP_ID_COL, OBS_ID_COL, TARGET_COL]].copy()
    eval_df["pred_prob"] = probs

    idx_top1 = eval_df.groupby([TRIP_ID_COL, OBS_ID_COL])["pred_prob"].idxmax()
    top1_rows = eval_df.loc[idx_top1]

    return float((top1_rows[TARGET_COL] == 1).mean())

def evaluate_predictions(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = df[TARGET_COL].values
    y_pred = (probs >= threshold).astype(int)

    return {
        "top1_accuracy": compute_observation_top1_accuracy(df, probs),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probs),
        "binary_accuracy": accuracy_score(y_true, y_pred),
    }

def build_mlp_pipeline() -> Pipeline:
    # Imputer + StandardScaler fitted strictly on training folds (no leakage)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MLP_PARAMS)),
    ])

# ==============================================================================
# 3. MAIN TRAINING & GROUP-AWARE CV PIPELINE
# ==============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print(f"Loading training data from {TRAIN_CSV}...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loading student-test data from {TEST_CSV}...")
    test_df = pd.read_csv(TEST_CSV)

    for col in FEATURE_COLS:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values
    groups = train_df[TRIP_ID_COL].values
    X_test = test_df[FEATURE_COLS]

    # --- 5-Fold GroupKFold Cross-Validation ---
    gkf = GroupKFold(n_splits=5)
    cv_macro_f1s = []
    cv_top1_accs = []
    cv_log_losses = []

    print("\nRunning 5-Fold GroupKFold Cross-Validation for MLP...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        print(f"\n--- Fold {fold}/5 ---")
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val = X_train.iloc[val_idx]
        val_meta = train_df.iloc[val_idx]

        fold_pipe = build_mlp_pipeline()
        fold_pipe.fit(X_tr, y_tr)

        val_probs = fold_pipe.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(val_meta, val_probs)

        cv_macro_f1s.append(metrics["macro_f1"])
        cv_top1_accs.append(metrics["top1_accuracy"])
        cv_log_losses.append(metrics["log_loss"])
        print(
            f"Fold {fold} Summary | Top-1 Acc: {metrics['top1_accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Log Loss: {metrics['log_loss']:.4f}"
        )

    # --- Fit Final MLP Model on Full Training Set ---
    print("\nTraining final MLP model on all 80 training trips...")
    t0 = time.perf_counter()
    final_pipe = build_mlp_pipeline()
    final_pipe.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0

    # Train set check (overfitting diagnosis)
    train_probs = final_pipe.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_predictions(train_df, train_probs)

    # Student-test evaluation & inference timing
    t1 = time.perf_counter()
    test_probs = final_pipe.predict_proba(X_test)[:, 1]
    inference_time_sec = time.perf_counter() - t1
    test_metrics = evaluate_predictions(test_df, test_probs)

    # --- Print Report-Ready Summary ---
    print("\n" + "=" * 68)
    print("MLP NEURAL NETWORK RESULTS (FOR REPORT SECTION 3.1 & 3.2)")
    print("=" * 68)
    print(f"Train Top-1 Acc:         {train_metrics['top1_accuracy']:.4f} | Train Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"5-Fold CV Top-1 Acc:     {np.mean(cv_top1_accs):.4f} (+/- {np.std(cv_top1_accs):.4f})")
    print(f"5-Fold CV Macro F1:      {np.mean(cv_macro_f1s):.4f} (+/- {np.std(cv_macro_f1s):.4f})")
    print(f"5-Fold CV Log Loss:      {np.mean(cv_log_losses):.4f} (+/- {np.std(cv_log_losses):.4f})")
    print("-" * 68)
    print(f"Student-Test Top-1 Acc:  {test_metrics['top1_accuracy']:.4f}")
    print(f"Student-Test Macro F1:   {test_metrics['macro_f1']:.4f}")
    print(f"Student-Test Precision:  {test_metrics['precision_macro']:.4f}")
    print(f"Student-Test Recall:     {test_metrics['recall_macro']:.4f}")
    print(f"Student-Test Log Loss:   {test_metrics['log_loss']:.4f}")
    print(f"Training Time (s):       {train_time_sec:.3f}s")
    print(f"Student-Test Inf (s):    {inference_time_sec:.4f}s")
    print("=" * 68)

    # 1. Save trained pipeline (includes imputer + scaler + MLP weights)
    model_path = os.path.join(MODEL_OUTPUT_DIR, "mlp_baseline.joblib")
    joblib.dump(final_pipe, model_path)

    # 2. Save real epoch-wise training curve (for Report Section 3.3)
    mlp_step = final_pipe.named_steps["mlp"]
    history_df = pd.DataFrame({
        "epoch": range(1, len(mlp_step.loss_curve_) + 1),
        "train_loss": mlp_step.loss_curve_,
        "val_accuracy": mlp_step.validation_scores_,
    })
    history_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "training_history.csv"), index=False)

    # 3. Save student-test predictions
    preds_df = test_df[[TRIP_ID_COL, OBS_ID_COL, "timestamp", ARC_ID_COL, TARGET_COL]].copy()
    preds_df["pred_prob"] = test_probs
    preds_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "student_test_predictions.csv"), index=False)

    # 4. Append results to consolidated comparison table
    serializable_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MLP_PARAMS.items() if k != "verbose"}
    log_row = {
        "Model": "MLP Classifier (128, 64)",
        "Algorithm_Family": "Neural Networks",
        "Key_Configuration": json.dumps(serializable_params),
        "CV_Macro_F1": round(float(np.mean(cv_macro_f1s)), 4),
        "CV_Macro_F1_Std": round(float(np.std(cv_macro_f1s)), 4),
        "Student_Test_Top1_Accuracy": round(test_metrics["top1_accuracy"], 4),
        "Student_Test_Macro_F1": round(test_metrics["macro_f1"], 4),
        "Student_Test_Log_Loss": round(test_metrics["log_loss"], 4),
        "Train_Time_Sec": round(train_time_sec, 3),
        "Inference_Time_Sec": round(inference_time_sec, 4),
    }

    log_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    log_df = pd.DataFrame([log_row])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing["Model"] != "MLP Classifier (128, 64)"]
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nSaved MLP pipeline to '{model_path}', epoch history to 'mlp_training_history.csv', and logged metrics to '{log_path}'.")

if __name__ == "__main__":
    main()