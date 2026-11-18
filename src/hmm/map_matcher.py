"""Road-aware transition probabilities and Viterbi map matching."""
from __future__ import annotations
import heapq
import math
import numpy as np
from src.config import SEGMENT_BREAK_SECONDS
from src.preprocessing.geo_utils import haversine
from src.hmm.viterbi import viterbi


def decode(candidates, gps, graph, return_bridges=False):
    """Candidates are per-observation DataFrames with model probabilities."""
    emissions = []
    for rows in candidates:
        if rows.empty:
            emissions.append([])
        else:
            # XGBoost likelihood tempered with a Gaussian GPS error model.
            p = np.clip(rows.pred_prob.to_numpy(float), 1e-8, 1.0)
            d = rows.gps_to_road_distance_m.to_numpy(float)
            e = p * np.exp(-0.5 * (d / 25.0) ** 2)
            emissions.append((e / max(e.sum(), 1e-300)).tolist())

    adjacency = {}
    for u, v, data in graph.edges(data=True):
        adjacency.setdefault(int(u), []).append((int(v), float(data.get("length", 0.0))))

    def bounded_distances(source, cutoff):
        distances = {source: 0.0}
        parents = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node) or distance > cutoff:
                continue
            for neighbor, weight in adjacency.get(node, ()):
                candidate = distance + weight
                if candidate <= cutoff and candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        return distances, parents

    # One bounded Dijkstra per source node and transition window serves all
    # candidate destinations for that source. This avoids a shortest-path search
    # for every candidate pair on large per-trip networks.
    shortest_from = {}
    def matrix(left, right):
        out = np.full((len(left), len(right)), -25.0)
        valid = np.zeros((len(left), len(right)), dtype=bool)
        for i, a in enumerate(left.itertuples()):
            for j, b in enumerate(right.itertuples()):
                source, target = int(a.to_node_id), int(b.from_node_id)
                dt = max(float(b.timestamp - a.timestamp), 0.0)
                direct = float(haversine(a.obs_lon, a.obs_lat, b.obs_lon, b.obs_lat))
                cutoff = max(500.0, dt * 45.0 + direct + 100.0)
                bucket = int(math.ceil(cutoff / 100.0) * 100)
                key = (source, bucket)
                if key not in shortest_from:
                    try:
                        shortest_from[key] = bounded_distances(source, bucket)
                    except (KeyError, ValueError):
                        shortest_from[key] = ({}, {})
                network_gap = shortest_from[key][0].get(target, math.inf)
                # Allow skipped arcs, penalize detours and impossible travel speeds.
                route_dist = float(a.remaining_m) + network_gap + float(b.offset_m)
                excess = abs(route_dist - direct) / max(30.0, direct * 0.5 + 20.0)
                speed = route_dist / max(dt, 1.0)
                speed_penalty = max(0.0, speed - 45.0) / 10.0
                if math.isfinite(network_gap):
                    valid[i, j] = True
                    out[i, j] = -excess - speed_penalty - max(0.0, direct - route_dist - 30.0) / 100.0
        # Normalize reachable transitions only; disconnected states retain a
        # very small absolute probability rather than becoming uniform choices.
        for i in range(len(left)):
            if valid[i].any():
                out[i, valid[i]] -= np.logaddexp.reduce(out[i, valid[i]])
        return out

    transition = lambda t0, t1: matrix(candidates[t0], candidates[t1])
    breaks = [False] * len(gps)
    for i in range(1, len(gps)):
        breaks[i] = float(gps.iloc[i].timestamp - gps.iloc[i - 1].timestamp) > SEGMENT_BREAK_SECONDS
    path = viterbi(emissions, transition, breaks)
    if not return_bridges:
        return path
    bridges = [None] * len(gps)
    for i in range(1, len(gps)):
        if path[i - 1] is None or path[i] is None or breaks[i]:
            continue
        before = candidates[i - 1].iloc[int(path[i - 1])]
        after = candidates[i].iloc[int(path[i])]
        source, target = int(before.to_node_id), int(after.from_node_id)
        dt = max(float(after.timestamp - before.timestamp), 0.0)
        direct = float(haversine(before.obs_lon, before.obs_lat, after.obs_lon, after.obs_lat))
        bucket = int(math.ceil(max(500.0, dt * 45.0 + direct + 100.0) / 100.0) * 100)
        search = shortest_from.get((source, bucket))
        if search is None or target not in search[0]:
            continue
        parents = search[1]
        nodes = [target]
        while nodes[-1] != source and nodes[-1] in parents:
            nodes.append(parents[nodes[-1]])
        if nodes[-1] == source:
            bridges[i] = list(reversed(nodes))
    return path, bridges
