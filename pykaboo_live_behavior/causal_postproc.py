"""Causal (online) replacements for the offline post-processing.

The offline pipeline applies, per class, in order:
    smooth_probs   : CENTERED moving average of width ``smooth_win``   (reads the future)
    threshold      : per-class decision threshold
    merge_gap      : bridge OFF gaps shorter than ``merge_gap`` bounded by ON  (reads the future)
    enforce_min_bout: delete ON runs shorter than ``min_bout``               (reads the future)

All three temporal steps are non-causal. For a closed loop we cannot wait for a run
to finish, so each is replaced by a streaming state machine with a known, bounded
latency:

    smooth   -> trailing box of width ``smooth_win`` (running sum). Lag ~ smooth_win//2.
    min_bout -> onset confirmation: emit ON only after the class has stayed above
                threshold for ``min_bout`` consecutive frames. EXACTLY reproduces
                enforce_min_bout for the streaming case. Onset latency = min_bout-1.
    merge_gap-> offset debounce: once ON, keep emitting ON across OFF dips up to
                ``merge_gap`` frames; commit OFF only after the dip exceeds it.
                Adds latency to OFF transitions only (irrelevant when you trigger on
                onset).

``min_bout`` and ``merge_gap`` are deploy-time knobs: lowering ``min_bout`` reduces
onset latency at the cost of more false positives. See PLAN.md for the latency budget.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class CausalBehaviorPostproc:
    """Per-class streaming post-processing for one probability stream [K]."""

    def __init__(
        self,
        num_classes: int,
        thresholds: np.ndarray,
        smooth_win: int = 9,
        min_bout_frames: int = 20,
        merge_gap_frames: int = 30,
    ):
        self.K = int(num_classes)
        self.thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if self.thr.shape[0] != self.K:
            raise ValueError("thresholds length != num_classes")
        self.smooth_win = max(int(smooth_win), 1)
        self.min_bout = max(int(min_bout_frames), 1)
        self.merge_gap = max(int(merge_gap_frames), 0)

        # trailing-box smoothing state
        self._ring: deque[np.ndarray] = deque(maxlen=self.smooth_win)
        self._run_sum = np.zeros(self.K, dtype=np.float64)

        # per-class debounce state machine
        self._on = np.zeros(self.K, dtype=bool)
        self._up_run = np.zeros(self.K, dtype=np.int64)
        self._down_run = np.zeros(self.K, dtype=np.int64)

    def reset(self) -> None:
        self._ring.clear()
        self._run_sum[:] = 0.0
        self._on[:] = False
        self._up_run[:] = 0
        self._down_run[:] = 0

    def smooth(self, prob: np.ndarray) -> np.ndarray:
        """Trailing box of width smooth_win via running sum. O(K) per frame."""
        prob = np.asarray(prob, dtype=np.float64).reshape(self.K)
        if len(self._ring) == self.smooth_win:
            self._run_sum -= self._ring[0]
        self._ring.append(prob)
        self._run_sum += prob
        return self._run_sum / len(self._ring)

    def step(self, prob: np.ndarray) -> dict:
        """Advance one frame.

        Returns dict with:
          smoothed  [K] float  - trailing-smoothed probabilities
          above     [K] bool   - smoothed >= threshold (instantaneous decision)
          binary    [K] int8   - debounced/confirmed ON state (the trigger signal)
          onsets    [K] bool   - classes that transitioned OFF->ON this frame
          offsets   [K] bool   - classes that transitioned ON->OFF this frame
        """
        smoothed = self.smooth(prob)
        above = smoothed >= self.thr

        prev_on = self._on.copy()
        for k in range(self.K):
            if not self._on[k]:
                if above[k]:
                    self._up_run[k] += 1
                    if self._up_run[k] >= self.min_bout:
                        self._on[k] = True
                        self._down_run[k] = 0
                else:
                    self._up_run[k] = 0
            else:
                if above[k]:
                    self._down_run[k] = 0
                else:
                    self._down_run[k] += 1
                    if self._down_run[k] > self.merge_gap:
                        self._on[k] = False
                        self._up_run[k] = 0

        binary = self._on.astype(np.int8)
        onsets = self._on & (~prev_on)
        offsets = (~self._on) & prev_on
        return {
            "smoothed": smoothed,
            "above": above,
            "binary": binary,
            "onsets": onsets,
            "offsets": offsets,
        }
