import numpy as np
from src.hmm.viterbi import viterbi


def test_viterbi_prefers_connected_route_over_local_peak():
    emissions = [[0.9, 0.1], [0.1, 0.9]]
    def transition(t0, t1):
        return np.log([[0.99, 0.01], [0.01, 0.99]])
    assert viterbi(emissions, transition) == [0, 0]


def test_empty_candidates_and_gps_gap_split_sequences():
    emissions = [[0.2, 0.8], [], [0.1, 0.9], [0.9, 0.1]]
    def transition(t0, t1):
        return np.log([[0.99, 0.01], [0.01, 0.99]])
    assert viterbi(emissions, transition, [False, False, True, False]) == [1, None, 0, 0]


def test_numerical_stability_long_sequence():
    emissions = [[1e-200, 1e-100] for _ in range(1500)]
    transition = lambda a, b: np.log(np.full((2, 2), 0.5))
    path = viterbi(emissions, transition)
    assert len(path) == 1500 and set(path) == {1}
