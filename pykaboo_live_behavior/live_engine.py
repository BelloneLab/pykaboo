"""OnlineBehaviorEngine: streaming free-interaction behavior inference for closed loop.

Wiring per frame:

    pykaboo LiveDetectionResult
       -> FrameRecord (via pykaboo_adapter)
       -> OnlineFeatureExtractor (rolling buffer -> [win, 432] per identity track)
       -> z-score normalize -> EmbTCN-AT forward (both tracks batched)
       -> read decision column (look-ahead L)
       -> CausalBehaviorPostproc per track (smooth -> threshold -> onset/offset debounce)
       -> scene reduction (OR of binaries, MAX of probs across the two mice)
       -> BehaviorEvent onset/offset callbacks  -> your TTL trigger

Latency model. The engine emits the decision for frame ``t - L`` when frame ``t`` arrives,
so the look-ahead ``L`` (frames) budgets BOTH the wavelet right-edge support and the
bidirectional model's in-window right context with a single knob. Total onset latency to
a trigger is approximately ``L/fps + (min_bout-1)/fps + smooth_win/2/fps`` (see PLAN.md).
For the shipped non-causal checkpoint, L in [8, 16] (0.27-0.53 s) recovers the fast
dynamics; a retrained ``causal=True`` checkpoint can run at L=0.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from causal_postproc import CausalBehaviorPostproc
from live_features import FrameRecord, OnlineFeatureExtractor
from model_runtime import LiveModel


@dataclass
class BehaviorEvent:
    behavior: str           # class name, e.g. "mounting"
    edge: str               # "onset" or "offset"
    frame_idx: int          # the decision frame this event refers to
    timestamp_s: float      # timestamp of the decision frame
    prob: float             # scene probability (max over mice) at this frame


@dataclass
class FrameDecision:
    frame_idx: int
    timestamp_s: float
    scene_prob: np.ndarray      # [K] max over the two mice (smoothed)
    scene_binary: np.ndarray    # [K] OR over the two mice (debounced)
    per_track_prob: dict        # {subject_id: smoothed prob [K]}
    per_track_binary: dict      # {subject_id: debounced binary [K]}
    events: list                # list[BehaviorEvent] fired this frame


class OnlineBehaviorEngine:
    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        lookahead: int = 8,
        identities: tuple[str, str] = ("1", "2"),
        trigger_behaviors: list[str] | None = None,
        on_event: Callable[[BehaviorEvent], None] | None = None,
        on_frame: Callable[[FrameDecision], None] | None = None,
        smooth_win: int | None = None,
        min_bout_frames: int | None = None,
        merge_gap_frames: int | None = None,
        history_pad: int = 64,
        min_window: int | None = None,
    ):
        self.model = LiveModel(ckpt_path, device=device)
        self.L = int(lookahead)
        self.identities = tuple(identities)
        self.K = self.model.num_classes
        self.labels = self.model.labels
        self.name_to_k = self.model.name_to_k
        self.background_id = self.model.background_id
        self.behavior_ids = [k for k in range(self.K) if k != self.background_id]

        self.on_event = on_event
        self.on_frame = on_frame
        if trigger_behaviors is None:
            self.trigger_ids = list(self.behavior_ids)
        else:
            self.trigger_ids = [self.name_to_k[b] for b in trigger_behaviors if b in self.name_to_k]

        sw = self.model.smooth_win if smooth_win is None else smooth_win
        mb = self.model.min_bout_frames if min_bout_frames is None else min_bout_frames
        mg = self.model.merge_gap_frames if merge_gap_frames is None else merge_gap_frames
        self._pp_cfg = dict(smooth_win=sw, min_bout_frames=mb, merge_gap_frames=mg)

        self.extractor = OnlineFeatureExtractor(
            feature_names=self.model.feature_names,
            frame_rate=self.model.frame_rate,
            window_frames=self.model.win,
            identities=self.identities,
            history_pad=history_pad,
            min_window=min_window,
        )
        self.postproc = {
            sid: CausalBehaviorPostproc(self.K, self.model.thresholds, **self._pp_cfg)
            for sid in self.identities
        }
        self._scene_on = np.zeros(self.K, dtype=bool)
        self._n_frames = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        for pp in self.postproc.values():
            pp.reset()
        self._scene_on[:] = False
        self.extractor = OnlineFeatureExtractor(
            feature_names=self.model.feature_names,
            frame_rate=self.model.frame_rate,
            window_frames=self.model.win,
            identities=self.identities,
            history_pad=self.extractor.buffer_len - self.model.win,
            min_window=self.extractor.min_window,
        )
        self._n_frames = 0

    # ------------------------------------------------------------------ #
    def push_only(self, frame: FrameRecord) -> None:
        """Append a frame to the rolling buffer WITHOUT scoring.

        Used by the off-thread worker to keep the temporal buffer contiguous when it
        must coalesce (skip scoring) under load: push every frame, score the newest.
        The postproc state machine only advances on scored frames (``score_latest``).
        """
        self.extractor.push(frame)
        self._n_frames += 1

    def on_detection(self, frame: FrameRecord) -> FrameDecision | None:
        """Feed one frame and score it. Returns a FrameDecision (for the look-ahead-
        delayed frame) once warmed up, else None."""
        self.push_only(frame)
        return self.score_latest()

    def score_latest(self) -> FrameDecision | None:
        """Score the current buffer (newest frame minus look-ahead L). Returns a
        FrameDecision once warmed up, else None. Assumes frames already pushed."""
        windows = self.extractor.compute_windows()
        if windows is None:
            return None

        # batch the two identity tracks into one forward pass
        sids = [s for s in self.identities if s in windows]
        if not sids:
            return None
        mats = [self.model.normalize(windows[s]) for s in sids]  # each [T, 432]
        Teff = min(m.shape[0] for m in mats)
        batch = np.stack([m[-Teff:] for m in mats], axis=0)       # [B, T, 432]
        probs = self.model.forward_window(batch)                  # [B, K, T]

        col = Teff - 1 - self.L
        if col < 0:
            col = 0
        decision_col = probs[:, :, col]                           # [B, K]

        per_track_prob = {}
        per_track_binary = {}
        for bi, sid in enumerate(sids):
            res = self.postproc[sid].step(decision_col[bi])
            per_track_prob[sid] = res["smoothed"]
            per_track_binary[sid] = res["binary"]

        scene_prob = np.zeros(self.K, dtype=np.float64)
        scene_bin = np.zeros(self.K, dtype=np.int8)
        for sid in sids:
            scene_prob = np.maximum(scene_prob, per_track_prob[sid])
            scene_bin |= per_track_binary[sid].astype(np.int8)

        # decision frame = newest buffered frame minus look-ahead L
        newest = self.extractor._buf[-1]
        # the decision frame is L frames before the newest in the window
        dframe = self.extractor._buf[-1 - self.L] if len(self.extractor._buf) > self.L else newest
        decision_frame_idx = dframe.frame_idx
        decision_ts = dframe.timestamp_s

        scene_on = scene_bin.astype(bool)
        onsets = scene_on & (~self._scene_on)
        offsets = (~scene_on) & self._scene_on
        self._scene_on = scene_on

        events: list[BehaviorEvent] = []
        for k in self.trigger_ids:
            if onsets[k]:
                events.append(BehaviorEvent(self.labels[k], "onset", decision_frame_idx,
                                            decision_ts, float(scene_prob[k])))
            if offsets[k]:
                events.append(BehaviorEvent(self.labels[k], "offset", decision_frame_idx,
                                            decision_ts, float(scene_prob[k])))

        decision = FrameDecision(
            frame_idx=decision_frame_idx,
            timestamp_s=decision_ts,
            scene_prob=scene_prob,
            scene_binary=scene_bin,
            per_track_prob=per_track_prob,
            per_track_binary=per_track_binary,
            events=events,
        )
        if self.on_event:
            for ev in events:
                self.on_event(ev)
        if self.on_frame:
            self.on_frame(decision)
        return decision

    def __repr__(self) -> str:
        return (f"OnlineBehaviorEngine({self.model!r}, L={self.L}, "
                f"triggers={[self.labels[k] for k in self.trigger_ids]}, "
                f"pp={self._pp_cfg})")
