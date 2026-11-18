"""Log-space Viterbi decoding over time-varying candidate sets."""
from __future__ import annotations
import numpy as np


def viterbi(emissions, transition_logprob, break_before=None):
    """Return the best candidate index at each timestep; empty emissions split sequences."""
    if not emissions:
        return []
    result = [None] * len(emissions)
    break_before = break_before or [False] * len(emissions)
    # Decode independent contiguous non-empty runs, so candidate-free samples are unmatched.
    start = 0
    while start < len(emissions):
        while start < len(emissions) and len(emissions[start]) == 0:
            start += 1
        if start == len(emissions):
            break
        end = start
        while end + 1 < len(emissions) and len(emissions[end + 1]) > 0 and not break_before[end + 1]:
            end += 1
        scores = np.log(np.clip(np.asarray(emissions[start], dtype=float), 1e-300, 1.0))
        back = []
        for t in range(start + 1, end + 1):
            trans = np.asarray(transition_logprob(t - 1, t), dtype=float)
            step = scores[:, None] + trans
            ptr = np.argmax(step, axis=0)
            scores = step[ptr, np.arange(step.shape[1])] + np.log(np.clip(emissions[t], 1e-300, 1.0))
            back.append(ptr)
        state = int(np.argmax(scores))
        result[end] = state
        for offset in range(end - start - 1, -1, -1):
            state = int(back[offset][state])
            result[start + offset] = state
        start = end + 1
    return result
