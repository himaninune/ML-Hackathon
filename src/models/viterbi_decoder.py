import json
import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
TRAIN_CSV = "experiments/results/train_features.csv"
TEST_CSV = "experiments/results/student_test_features.csv"
RESULTS_DIR = "experiments/results"
MODELS_DIR = "models"

# Exact directory containing zero-padded 8-digit trip folders (e.g. 00000000/00000000.arcs)
RAW_TEST_DIR = os.path.join("data", "Student folder", "student_test")

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

BASELINE_XGB_PARAMS = {
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

# HMM Transition Log-Penalties (physical road-network priors)
LOG_TRANS_SAME_ARC = 0.0        # Stay on current road segment
LOG_TRANS_CONNECTED = -0.5      # Valid turn: prev_arc.to_node == next_arc.from_node
LOG_TRANS_2HOP = -2.0           # Short intermediate segment skipped within 1s
LOG_TRANS_DISCONNECTED = -7.5   # Unconnected jump (prevented unless emission is overwhelming)
LOG_TRANS_GAP = -1.0            # Relaxed transition when gap_flag == 1 (>30s GPS dropout)


# ==============================================================================
# 2. ROAD NETWORK GRAPH LOADER (.arcs)
# ==============================================================================
def load_trip_arcs(trip_id: int | str, raw_dir: str) -> pd.DataFrame:
    """Load tab-separated [from_node_id, to_node_id] .arcs file where row index == arc_id."""
    trip_str = f"{int(trip_id):08d}"
    arcs_path = os.path.join(raw_dir, trip_str, f"{trip_str}.arcs")
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Expected .arcs file not found at: {arcs_path}")

    df = pd.read_csv(arcs_path, sep="\t", header=None, names=["from_node_id", "to_node_id"])
    df.insert(0, "arc_id", np.arange(len(df), dtype=int))
    return df


def build_adjacency_lookup(arcs_df: pd.DataFrame) -> tuple[set, set]:
    """Build fast lookup sets for 1-hop and 2-hop directed road connectivity."""
    arc_to_nodes = {}
    from_node_to_arcs = {}
    for row in arcs_df.itertuples(index=False):
        arc_to_nodes[int(row.arc_id)] = (int(row.from_node_id), int(row.to_node_id))
        from_node_to_arcs.setdefault(int(row.from_node_id), set()).add(int(row.arc_id))

    direct_edges = set()
    two_hop_edges = set()
    for a1, (_, mid_node) in arc_to_nodes.items():
        next_arcs = from_node_to_arcs.get(mid_node, set())
        for a2 in next_arcs:
            direct_edges.add((a1, a2))
            _, end_node = arc_to_nodes[a2]
            for a3 in from_node_to_arcs.get(end_node, set()):
                two_hop_edges.add((a1, a3))

    return direct_edges, two_hop_edges


# ==============================================================================
# 3. VITERBI SEQUENCE DECODER (PER TRIP)
# ==============================================================================
def viterbi_decode_trip(
    trip_df: pd.DataFrame,
    direct_edges: set,
    two_hop_edges: set,
) -> pd.DataFrame:
    """Decode the highest-probability connected road sequence for a single trip."""
    trip_df = trip_df.sort_values([OBS_ID_COL, "candidate_rank"]).reset_index(drop=True)
    obs_groups = [g for _, g in trip_df.groupby(OBS_ID_COL, sort=True)]
    T = len(obs_groups)

    obs_arcs = []
    obs_log_emiss = []
    obs_gap_flags = []

    for g in obs_groups:
        arcs = g[ARC_ID_COL].astype(int).values
        probs = np.clip(g["pred_prob"].values, 1e-6, 1.0 - 1e-6)
        probs = probs / probs.sum()
        obs_arcs.append(arcs)
        obs_log_emiss.append(np.log(probs))
        obs_gap_flags.append(int(g["gap_flag"].iloc[0]) if "gap_flag" in g.columns else 0)

    dp = [obs_log_emiss[0].copy()]
    backpointers = [np.full(len(obs_arcs[0]), -1, dtype=int)]

    for t in range(1, T):
        prev_arcs = obs_arcs[t - 1]
        curr_arcs = obs_arcs[t]
        prev_dp = dp[t - 1]
        curr_emiss = obs_log_emiss[t]
        is_gap = obs_gap_flags[t] == 1

        n_prev = len(prev_arcs)
        n_curr = len(curr_arcs)

        # Build small (n_prev x n_curr) transition matrix for timestamp t
        trans_mat = np.full((n_prev, n_curr), LOG_TRANS_GAP if is_gap else LOG_TRANS_DISCONNECTED)
        for i in range(n_prev):
            a_prev = int(prev_arcs[i])
            for j in range(n_curr):
                a_curr = int(curr_arcs[j])
                if a_prev == a_curr:
                    trans_mat[i, j] = LOG_TRANS_SAME_ARC
                elif not is_gap:
                    if (a_prev, a_curr) in direct_edges:
                        trans_mat[i, j] = LOG_TRANS_CONNECTED
                    elif (a_prev, a_curr) in two_hop_edges:
                        trans_mat[i, j] = LOG_TRANS_2HOP

        scores = prev_dp[:, None] + trans_mat
        best_i = np.argmax(scores, axis=0)
        curr_dp = scores[best_i, np.arange(n_curr)] + curr_emiss

        dp.append(curr_dp)
        backpointers.append(best_i)

    best_path_idx = np.zeros(T, dtype=int)
    best_path_idx[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 1, 0, -1):
        best_path_idx[t - 1] = backpointers[t][best_path_idx[t]]

    records = []
    for t, g in enumerate(obs_groups):
        chosen_idx = best_path_idx[t]
        chosen_arc = int(obs_arcs[t][chosen_idx])
        pointwise_idx = int(np.argmax(g["pred_prob"].values))
        pointwise_arc = int(obs_arcs[t][pointwise_idx])

        true_rows = g[g[TARGET_COL] == 1]
        true_arc = int(true_rows[ARC_ID_COL].iloc[0]) if len(true_rows) > 0 else -1

        records.append({
            TRIP_ID_COL: int(g[TRIP_ID_COL].iloc[0]),
            OBS_ID_COL: int(g[OBS_ID_COL].iloc[0]),
            "timestamp": g["timestamp"].iloc[0] if "timestamp" in g.columns else t,
            "gap_flag": obs_gap_flags[t],
            "true_arc_id": true_arc,
            "pointwise_xgb_arc_id": pointwise_arc,
            "viterbi_arc_id": chosen_arc,
            "pointwise_correct": int(pointwise_arc == true_arc),
            "viterbi_correct": int(chosen_arc == true_arc),
        })

    return pd.DataFrame(records)


def compute_route_continuity(
    seq_arcs: np.ndarray,
    gap_flags: np.ndarray,
    direct_edges: set,
    two_hop_edges: set,
) -> float:
    """Fraction of consecutive non-gap transitions that follow a valid road edge."""
    if len(seq_arcs) < 2:
        return 1.0
    valid = 0
    total = 0
    for t in range(1, len(seq_arcs)):
        if gap_flags[t] == 1:
            continue
        total += 1
        a_prev, a_curr = int(seq_arcs[t - 1]), int(seq_arcs[t])
        if a_prev == a_curr or (a_prev, a_curr) in direct_edges or (a_prev, a_curr) in two_hop_edges:
            valid += 1
    return float(valid / total) if total > 0 else 1.0


# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading {TRAIN_CSV} and {TEST_CSV}...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for col in FEATURE_COLS:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)

    baseline_model_path = os.path.join(MODELS_DIR, "xgboost_baseline.json")
    model = XGBClassifier(**BASELINE_XGB_PARAMS)
    if os.path.exists(baseline_model_path):
        print(f"Loading existing baseline model from {baseline_model_path}...")
        model.load_model(baseline_model_path)
    else:
        print("Training Untuned Baseline XGBoost on all 80 training trips...")
        t0 = time.perf_counter()
        model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL].values)
        model.save_model(baseline_model_path)
        print(f"Saved baseline model to {baseline_model_path} ({time.perf_counter() - t0:.2f}s).")

    test_df["pred_prob"] = model.predict_proba(test_df[FEATURE_COLS])[:, 1]

    all_routes = []
    pw_continuities = []
    vit_continuities = []

    t_vit0 = time.perf_counter()
    for trip_id, trip_df in test_df.groupby(TRIP_ID_COL, sort=True):
        arcs_df = load_trip_arcs(trip_id, RAW_TEST_DIR)
        direct_edges, two_hop_edges = build_adjacency_lookup(arcs_df)
        print(f"Trip {int(trip_id):08d}: loaded {len(arcs_df)} road arcs, {len(direct_edges)} directed transitions.")

        route_df = viterbi_decode_trip(trip_df, direct_edges, two_hop_edges)
        all_routes.append(route_df)

        pw_cont = compute_route_continuity(
            route_df["pointwise_xgb_arc_id"].values, route_df["gap_flag"].values, direct_edges, two_hop_edges
        )
        vit_cont = compute_route_continuity(
            route_df["viterbi_arc_id"].values, route_df["gap_flag"].values, direct_edges, two_hop_edges
        )
        pw_continuities.append(pw_cont)
        vit_continuities.append(vit_cont)

    vit_time = time.perf_counter() - t_vit0
    full_routes_df = pd.concat(all_routes, ignore_index=True)

    pw_acc = float(full_routes_df["pointwise_correct"].mean())
    vit_acc = float(full_routes_df["viterbi_correct"].mean())
    pw_cont_mean = float(np.mean(pw_continuities))
    vit_cont_mean = float(np.mean(vit_continuities))

    print("\n" + "=" * 72)
    print("FINAL PIPELINE RESULTS: POINTWISE XGBOOST vs. XGBOOST + HMM/VITERBI")
    print("=" * 72)
    print(f"Pointwise Baseline XGBoost Top-1 Accuracy:  {pw_acc:.4f} ({pw_acc*100:.2f}%)")
    print(f"XGBoost + HMM/Viterbi Sequence Accuracy:    {vit_acc:.4f} ({vit_acc*100:.2f}%)")
    print(f"Sequence Accuracy Gain:                     {(vit_acc - pw_acc)*100:+.2f}%")
    print("-" * 72)
    print(f"Pointwise Route Continuity (Valid Edges):   {pw_cont_mean*100:.2f}%")
    print(f"HMM/Viterbi Route Continuity (Valid Edges): {vit_cont_mean*100:.2f}%")
    print(f"Route Continuity Gain:                      {(vit_cont_mean - pw_cont_mean)*100:+.2f}%")
    print(f"Viterbi Decoding Time (10 Trips):           {vit_time:.3f}s")
    print("=" * 72)

    out_path = os.path.join(RESULTS_DIR, "viterbi_student_test_routes.csv")
    full_routes_df.to_csv(out_path, index=False)

    metrics_path = os.path.join(RESULTS_DIR, "viterbi_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pointwise_xgb_top1_accuracy": round(pw_acc, 4),
                "viterbi_sequence_top1_accuracy": round(vit_acc, 4),
                "accuracy_gain": round(vit_acc - pw_acc, 4),
                "pointwise_route_continuity": round(pw_cont_mean, 4),
                "viterbi_route_continuity": round(vit_cont_mean, 4),
                "route_continuity_gain": round(vit_cont_mean - pw_cont_mean, 4),
                "viterbi_decoding_time_seconds": round(vit_time, 4),
                "transition_log_priors": {
                    "same_arc": LOG_TRANS_SAME_ARC,
                    "connected_1hop": LOG_TRANS_CONNECTED,
                    "connected_2hop": LOG_TRANS_2HOP,
                    "disconnected": LOG_TRANS_DISCONNECTED,
                    "gap_flag": LOG_TRANS_GAP,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved reconstructed routes to '{out_path}' and summary to '{metrics_path}'.")


if __name__ == "__main__":
    main()