from __future__ import annotations
from functools import lru_cache
import logging
import math
import time
import pandas as pd
from src.config import DATA_DIR, PROCESSED_DIR, GAP_FLAG_SECONDS, SEGMENT_BREAK_SECONDS, ARC_ID_MODE
from src.preprocessing.io_utils import trip_files, read_track, read_route
from src.preprocessing.network_utils import load_network, make_graph
from src.preprocessing.geo_utils import haversine, bearing_deg
from src.candidate_generation.generate_candidates import candidates_for_trip
from src.hmm.map_matcher import decode
from src.models.xgboost_loader import XGBoostScorer

log = logging.getLogger("pathsnap.prediction")


class PredictionService:
    def __init__(self, scorer: XGBoostScorer):
        self.scorer = scorer
        self._networks = {}

    @property
    def network_cache_size(self):
        return len(self._networks)

    def _paths(self, trip_id):
        # IDs are restricted at the API boundary; enforce path containment here too.
        root = (DATA_DIR / "student_test").resolve()
        folder = (root / trip_id).resolve()
        if folder.parent != root or not folder.is_dir():
            raise FileNotFoundError("Trip not found")
        return trip_files("student_test", trip_id)

    @lru_cache(maxsize=32)
    def load_gps(self, trip_id):
        path = PROCESSED_DIR / "student_test" / f"{trip_id}.csv"
        gps = pd.read_csv(path) if path.is_file() else read_track(self._paths(trip_id)["track"])
        gps = gps.dropna(subset=["longitude", "latitude", "timestamp"]).copy()
        gps = gps[gps.longitude.between(-180, 180) & gps.latitude.between(-90, 90)]
        gps = gps.sort_values("timestamp", kind="stable").drop_duplicates(["longitude", "latitude", "timestamp"]).reset_index(drop=True)
        if "observation_index" not in gps: gps["observation_index"] = range(len(gps))
        if "dt_seconds" not in gps:
            gps["dt_seconds"] = gps.timestamp.diff()
            gps["step_distance_m"] = 0.0
            if len(gps) > 1:
                gps.loc[1:, "step_distance_m"] = haversine(gps.longitude.iloc[:-1].to_numpy(), gps.latitude.iloc[:-1].to_numpy(), gps.longitude.iloc[1:].to_numpy(), gps.latitude.iloc[1:].to_numpy())
            gps["gps_speed_kmh"] = gps.step_distance_m / gps.dt_seconds.replace(0, float("nan")) * 3.6
            gps["gps_heading_deg"] = float("nan")
            if len(gps) > 1:
                gps.loc[1:, "gps_heading_deg"] = bearing_deg(gps.longitude.iloc[:-1].to_numpy(), gps.latitude.iloc[:-1].to_numpy(), gps.longitude.iloc[1:].to_numpy(), gps.latitude.iloc[1:].to_numpy())
            gps["gap_flag"] = gps.dt_seconds.gt(GAP_FLAG_SECONDS).fillna(False)
            gps["segment_id"] = gps.dt_seconds.gt(SEGMENT_BREAK_SECONDS).fillna(False).astype(int).cumsum()
        for col, default in (("segment_id", 0), ("gap_flag", False), ("gps_heading_deg", float("nan")), ("gps_speed_kmh", float("nan")), ("dt_seconds", float("nan")), ("step_distance_m", 0.0)):
            if col not in gps: gps[col] = default
        return gps

    def _network(self, trip_id):
        if trip_id not in self._networks:
            paths = self._paths(trip_id)
            nodes, arcs = load_network(paths["nodes"], paths["arcs"])
            # Add physical length weights to directed graph edges.
            graph = make_graph(nodes, arcs)
            for r in arcs.itertuples():
                graph[int(r.from_node_id)][int(r.to_node_id)]["length"] = float(haversine(r.from_lon, r.from_lat, r.to_lon, r.to_lat))
            self._networks[trip_id] = (nodes, arcs, graph)
            log.info("Loaded road network for %s (%d arcs)", trip_id, len(arcs))
        return self._networks[trip_id]

    def predict(self, trip_id):
        total_started = time.perf_counter()
        log.info("Prediction started for trip %s", trip_id)
        gps = self.load_gps(trip_id)
        nodes, arcs, graph = self._network(trip_id)
        stage = time.perf_counter()
        candidates = candidates_for_trip(gps.assign(partition="student_test", trip_id=trip_id), nodes, arcs, set())
        log.info("Candidate generation for %s: %d rows in %.2fs", trip_id, len(candidates), time.perf_counter() - stage)
        # Keep an empty candidate frame for observations excluded by invalid GPS rows.
        by_index = {int(k): g for k, g in candidates.groupby("observation_index")} if not candidates.empty else {}
        candidate_sets = []
        ordered_gps = gps.reset_index(drop=True).copy()
        for i, point in ordered_gps.iterrows():
            candidate_sets.append(by_index.get(int(point.observation_index), pd.DataFrame()))
        if any(not g.empty for g in candidate_sets):
            stage = time.perf_counter()
            flat = pd.concat([g for g in candidate_sets if not g.empty], ignore_index=True)
            probabilities = self.scorer.predict(flat)
            offset, scored = 0, []
            for group in candidate_sets:
                if group.empty:
                    scored.append(group)
                else:
                    count = len(group)
                    scored.append(group.assign(pred_prob=probabilities[offset:offset + count]))
                    offset += count
            candidate_sets = scored
            log.info("XGBoost inference for %s: %d rows in %.2fs", trip_id, len(flat), time.perf_counter() - stage)
        stage = time.perf_counter()
        selected, bridge_paths = decode(candidate_sets, ordered_gps, graph, return_bridges=True)
        log.info("HMM/Viterbi for %s: %d observations in %.2fs", trip_id, len(gps), time.perf_counter() - stage)
        records, route_ids, route_coords, route_segments = [], [], [], []
        active_segment, previous = [], None
        matched = 0
        for i, choice in enumerate(selected):
            if choice is None:
                records.append(None)
                if active_segment:
                    route_segments.append(active_segment)
                    active_segment = []
                previous = None
                continue
            row = candidate_sets[i].iloc[int(choice)]
            matched += 1
            aid = int(row.candidate_arc_id)
            if previous is not None and int(previous.candidate_arc_id) != aid:
                elapsed = float(row.timestamp - previous.timestamp)
                if 0 <= elapsed <= SEGMENT_BREAK_SECONDS:
                    node_path = bridge_paths[i]
                    if node_path is None:
                        if active_segment:
                            route_segments.append(active_segment)
                            active_segment = []
                    else:
                        for u, v in zip(node_path, node_path[1:]):
                            edge_id = int(graph[u][v]["arc_id"])
                            edge_coords = [[float(graph.nodes[u]["latitude"]), float(graph.nodes[u]["longitude"])],
                                           [float(graph.nodes[v]["latitude"]), float(graph.nodes[v]["longitude"])]]
                            if not route_ids or route_ids[-1] != edge_id:
                                route_ids.append(edge_id)
                                if not active_segment:
                                    active_segment.append(edge_coords[0])
                                if active_segment[-1] != edge_coords[0]:
                                    active_segment.append(edge_coords[0])
                                active_segment.append(edge_coords[1])
                else:
                    if active_segment:
                        route_segments.append(active_segment)
                        active_segment = []
            coords = [[float(row.from_lat), float(row.from_lon)], [float(row.to_lat), float(row.to_lon)]]
            if not route_ids or route_ids[-1] != aid:
                route_ids.append(aid)
                if not active_segment:
                    active_segment.append(coords[0])
                elif active_segment[-1] != coords[0]:
                    active_segment.append(coords[0])
                active_segment.append(coords[1])
            records.append({"observation_index": int(ordered_gps.iloc[i].observation_index), "arc_id": aid,
                            "probability": float(row.pred_prob), "coordinates": coords})
            previous = row
        if active_segment:
            route_segments.append(active_segment)
        route_coords = [coord for segment in route_segments for coord in segment]
        stage = time.perf_counter()
        evaluation = self._evaluate(trip_id, route_ids)
        log.info("Server-side evaluation for %s completed in %.3fs", trip_id, time.perf_counter() - stage)
        log.info("Prediction completed for %s in %.2fs (%d/%d matched)", trip_id, time.perf_counter() - total_started, matched, len(gps))
        return {"trip_id": trip_id, "status": "success", "prediction": {
            "matched_observations": matched, "total_observations": len(gps), "predicted_route": route_ids,
            "matched_arcs": [r for r in records if r], "predicted_coordinates": route_coords,
            "predicted_route_segments": route_segments,
            "gps_coordinates": gps[["latitude", "longitude"]].values.tolist(),
            "unmatched_observation_indices": [int(gps.iloc[i].observation_index) for i, r in enumerate(records) if r is None]},
            "evaluation": evaluation}

    def _evaluate(self, trip_id, predicted):
        # Evaluator references are not loaded by this implementation. Existing .route
        # values in student_test are heuristic membership labels, not verified alignment.
        path = self._paths(trip_id)["route"]
        if not path.is_file():
            return {"available": False, "metrics": {}, "reason": "Reference route unavailable"}
        truth = pd.to_numeric(read_route(path).arc_id, errors="coerce").dropna().astype(int).tolist()
        if ARC_ID_MODE == "one_based":
            truth = [value - 1 for value in truth]
        if not truth:
            return {"available": False, "metrics": {}, "reason": "Reference route is empty"}
        from difflib import SequenceMatcher
        p, t = list(map(int, predicted)), list(map(int, truth))
        pset, tset = set(p), set(t)
        precision = len(pset & tset) / len(pset) if pset else 0.0
        recall = len(pset & tset) / len(tset) if tset else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"available": True, "reference_type": "route_membership_heuristic", "metrics": {
            "arc_set_precision": precision, "arc_set_recall": recall, "arc_set_f1": f1,
            "sequence_similarity": SequenceMatcher(a=t, b=p, autojunk=False).ratio()}}
