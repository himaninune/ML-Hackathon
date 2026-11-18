import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TRAIN_CSV = "experiments/results/train_features.csv"
TEST_CSV = "experiments/results/student_test_features.csv"
RESULTS_DIR = "experiments/results"
MODELS_DIR = "models"

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

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# ==============================================================================
# 2. METRICS HELPER (OBSERVATION-LEVEL TOP-1 & CANDIDATE METRICS)
# ==============================================================================
def compute_observation_top1_accuracy(df: pd.DataFrame, probs: np.ndarray) -> float:
    """
    Groups candidates by (trip_id, observation_index), selects the candidate
    with the highest predicted probability, and checks if label == 1.
    """
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

# ==============================================================================
# 3. MAIN TRAINING & GROUP-AWARE CV PIPELINE
# ==============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading training data from {TRAIN_CSV}...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loading student-test data from {TEST_CSV}...")
    test_df = pd.read_csv(TEST_CSV)

    # Convert boolean columns (like gap_flag) cleanly to float/int for XGBoost
    for col in FEATURE_COLS:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)

    print(
        f"Train rows: {len(train_df):,} ({train_df[TRIP_ID_COL].nunique()} trips) | "
        f"Positive rate: {train_df[TARGET_COL].mean():.2%}"
    )
    print(
        f"Test rows:  {len(test_df):,} ({test_df[TRIP_ID_COL].nunique()} trips) | "
        f"Positive rate: {test_df[TARGET_COL].mean():.2%}"
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values
    groups = train_df[TRIP_ID_COL].values

    X_test = test_df[FEATURE_COLS]

    # --- 5-Fold GroupKFold Cross-Validation ---
    gkf = GroupKFold(n_splits=5)
    cv_macro_f1s = []
    cv_top1_accs = []
    cv_log_losses = []

    print("\nRunning 5-Fold GroupKFold Cross-Validation on Training Trips...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups), 1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val = X_train.iloc[val_idx]
        val_meta = train_df.iloc[val_idx]

        fold_model = XGBClassifier(**XGB_PARAMS)
        fold_model.fit(X_tr, y_tr)

        val_probs = fold_model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(val_meta, val_probs)

        cv_macro_f1s.append(metrics["macro_f1"])
        cv_top1_accs.append(metrics["top1_accuracy"])
        cv_log_losses.append(metrics["log_loss"])
        print(
            f"  Fold {fold} | Top-1 Acc: {metrics['top1_accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Log Loss: {metrics['log_loss']:.4f}"
        )

    # --- Fit Final Baseline Model on All Training Trips ---
    print("\nTraining final XGBoost baseline on full training set...")
    t0 = time.perf_counter()
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t0

    # Train set evaluation (to diagnose overfitting/underfitting in report)
    train_probs = final_model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_predictions(train_df, train_probs)

    # Student-test evaluation & inference timing
    t1 = time.perf_counter()
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inference_time_sec = time.perf_counter() - t1
    test_metrics = evaluate_predictions(test_df, test_probs)

    # --- Print Report-Ready Summary ---
    print("\n" + "=" * 68)
    print("XGBOOST BASELINE RESULTS (FOR REPORT SECTION 3.1 & 3.2)")
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

    # 1. Save trained model artifact
    model_path = os.path.join(MODELS_DIR, "xgboost_baseline.json")
    final_model.save_model(model_path)

    # 2. Save feature importances
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_feature_importance.csv"), index=False)

    # 3. Save student-test predictions (needed later for HMM/Viterbi & Route Continuity)
    preds_df = test_df[[TRIP_ID_COL, OBS_ID_COL, "timestamp", ARC_ID_COL, TARGET_COL]].copy()
    preds_df["pred_prob"] = test_probs
    preds_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_student_test_preds.csv"), index=False)

    # 4. Log metrics into consolidated comparison table
    log_row = {
        "Model": "XGBoost (Baseline)",
        "Algorithm_Family": "Boosting",
        "Key_Configuration": json.dumps(XGB_PARAMS),
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
        existing = existing[existing["Model"] != "XGBoost (Baseline)"]
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nSaved model to '{model_path}', predictions to 'xgboost_student_test_preds.csv', and logged metrics to '{log_path}'.")

if __name__ == "__main__":
    main()