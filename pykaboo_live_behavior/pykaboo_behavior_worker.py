"""LiveBehaviorWorker: run the streaming behavior engine OFF the GUI thread.

pykaboo's ``LiveInferenceWorker`` emits a ``LiveDetectionResult`` per detected frame
on the GUI thread. Behavior inference (feature extraction over a rolling 480-frame
buffer + EmbTCN forward) costs far more than a 33 ms frame budget, so it must not run
inline in the GUI slot. This QThread owns the ``OnlineBehaviorEngine`` and:

  * receives each ``LiveDetectionResult`` via ``submit_result`` (GUI thread, cheap);
  * on its own thread, PUSHES every queued frame into the rolling buffer so the
    temporal window stays contiguous, and SCORES only the newest frame (automatic
    coalescing when it falls behind -- the buffer never develops gaps, only the
    decision rate drops);
  * emits ``behavior_ready(BehaviorFrameState)`` with the scene-level debounced
    ON/OFF per behavior class + probabilities + onset/offset events + latency.

The main window feeds the ON/OFF state into ``LiveRuleEngine.set_behavior_state`` so
``behavior_class`` trigger rules fire TTL pulses exactly like the geometric rules.

Model loading (a heavy GPU checkpoint load) happens lazily on the worker thread, so
constructing the worker is instant and the GUI never blocks.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QThread, Signal

from pykaboo_adapter import IdentityMapper, result_to_framerecord


@dataclass
class BehaviorFrameState:
    """One scene-level behavior decision, emitted per scored frame."""

    frame_idx: int
    timestamp_s: float
    active: dict           # {behavior_name: bool}  debounced ON/OFF (scene OR over mice)
    probs: dict            # {behavior_name: float} scene probability (MAX over mice)
    events: list           # list[BehaviorEvent] onset/offset fired this frame
    latency_ms: float = 0.0
    dropped: int = 0       # frames dropped from the queue since the last decision
    labels: list = field(default_factory=list)
    # Per-mouse (directed track) decisions, keyed by subject id "1"/"2", for drawing
    # per-mouse subtitles: {sid: {"probs": {name: float}, "binary": {name: bool}}}.
    per_track: dict = field(default_factory=dict)


def read_checkpoint_labels(ckpt_path: str) -> dict:
    """Cheaply read class names / feature count from a checkpoint for UI population.

    Returns {"labels": [...], "num_features": int, "frame_rate": float, "win": int}.
    Loads onto CPU; does not build the model.
    """
    import numpy as np
    import torch

    p = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels = list(p.get("label_map", {}).get("names", []))
    fr = float(p.get("frame_rate", 30.0))
    win = max(int(round(float(p.get("window_seconds", 16.0)) * fr)), 1)
    return {
        "labels": labels,
        "num_features": int(len(p.get("feature_names", []))),
        "frame_rate": fr,
        "win": win,
        "thresholds": list(np.asarray(p.get("thresholds", []), dtype=float)),
    }


class LiveBehaviorWorker(QThread):
    """Drive the streaming behavior model on a background thread."""

    behavior_ready = Signal(object)     # BehaviorFrameState
    status_changed = Signal(str)
    error_occurred = Signal(str)
    labels_ready = Signal(object)       # list[str] once the engine is built

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cond = threading.Condition()
        self._running = True
        self._active = False
        self._queue: deque = deque()
        self._max_queue = 16
        self._dropped = 0
        self._engine = None
        self._rule_detector = None
        self._engine_sig: Optional[tuple] = None
        self._cfg: dict = {}
        self._mapper = IdentityMapper()
        self._kp_reorder: Optional[list] = None
        self._labels: Optional[list] = None

    # ------------------------------------------------------------------ #
    def configure(
        self,
        ckpt_path: str,
        *,
        backend: str = "rules",
        device: Optional[str] = None,
        lookahead: int = 8,
        trigger_behaviors: Optional[list] = None,
        smooth_win: Optional[int] = None,
        min_bout_frames: Optional[int] = None,
        merge_gap_frames: Optional[int] = None,
        kp_reorder: Optional[list] = None,
        max_queue: int = 16,
        min_window: Optional[int] = None,
    ) -> None:
        """Set / change the backend + model + post-processing. The engine is (re)built
        lazily on the worker thread the next time it processes a frame.

        backend: "rules" (fast geometric/kinematic detector, torch-free, real-time) or
        "ml" (EmbTCN-Attention temporal model; needs the checkpoint + GPU).
        """
        backend = "ml" if str(backend).lower() in ("ml", "model", "embtcn") else "rules"
        sig = (
            backend, str(ckpt_path), device, int(lookahead),
            tuple(trigger_behaviors) if trigger_behaviors else None,
            smooth_win, min_bout_frames, merge_gap_frames, min_window,
        )
        with self._cond:
            self._cfg = dict(
                backend=backend,
                ckpt_path=ckpt_path, device=device, lookahead=int(lookahead),
                trigger_behaviors=list(trigger_behaviors) if trigger_behaviors else None,
                smooth_win=smooth_win, min_bout_frames=min_bout_frames,
                merge_gap_frames=merge_gap_frames, min_window=min_window,
            )
            self._kp_reorder = list(kp_reorder) if kp_reorder else None
            self._max_queue = max(1, int(max_queue))
            if sig != self._engine_sig:
                self._engine = None          # force rebuild
                self._rule_detector = None
                self._pending_sig = sig
            self._cond.notify_all()

    def start_behavior(self) -> None:
        with self._cond:
            self._active = True
            self._mapper = IdentityMapper()
            self._queue.clear()
            self._dropped = 0
            self._cond.notify_all()
        if not self.isRunning():
            self.start()
        self.status_changed.emit("Behavior detection armed")

    def stop_behavior(self) -> None:
        with self._cond:
            self._active = False
            self._queue.clear()
            self._cond.notify_all()
        self.status_changed.emit("Behavior detection stopped")

    def submit_result(self, result: object) -> None:
        """Queue a pykaboo LiveDetectionResult (GUI thread, cheap)."""
        with self._cond:
            if not self._active or not self._running:
                return
            self._queue.append(result)
            while len(self._queue) > self._max_queue:
                self._queue.popleft()
                self._dropped += 1
            self._cond.notify_all()

    def shutdown(self) -> None:
        with self._cond:
            self._running = False
            self._active = False
            self._queue.clear()
            self._cond.notify_all()
        self.wait(5000)
        self._engine = None
        self._rule_detector = None

    def labels(self) -> Optional[list]:
        return list(self._labels) if self._labels else None

    def _backend(self) -> str:
        return str(self._cfg.get("backend", "rules"))

    # ------------------------------------------------------------------ #
    def _ensure_engine(self) -> None:
        if self._engine is not None or self._rule_detector is not None:
            return
        cfg = dict(self._cfg)
        if cfg.get("backend") == "ml":
            if not cfg.get("ckpt_path"):
                return
            from live_engine import OnlineBehaviorEngine  # heavy import (torch); worker thread

            self.status_changed.emit("Loading behavior model")
            engine = OnlineBehaviorEngine(
                ckpt_path=cfg["ckpt_path"],
                device=cfg.get("device"),
                lookahead=cfg.get("lookahead", 8),
                trigger_behaviors=cfg.get("trigger_behaviors"),
                smooth_win=cfg.get("smooth_win"),
                min_bout_frames=cfg.get("min_bout_frames"),
                merge_gap_frames=cfg.get("merge_gap_frames"),
                min_window=cfg.get("min_window"),
            )
            self._engine = engine
            self._engine_sig = getattr(self, "_pending_sig", None)
            self._labels = list(engine.labels)
            self.labels_ready.emit(list(engine.labels))
            self.status_changed.emit(
                f"Behavior model ready ({engine.model.num_features} feats, "
                f"{engine.K} classes, {engine.model.device}, win {engine.model.win}f)"
            )
        else:
            from rule_based_social import LABELS, RuleBasedSocialDetector  # torch-free, fast

            self._rule_detector = RuleBasedSocialDetector(identities=("1", "2"))
            self._engine_sig = getattr(self, "_pending_sig", None)
            self._labels = list(LABELS)
            self.labels_ready.emit(list(LABELS))
            self.status_changed.emit(f"Rule-based behavior detector ready ({len(LABELS)} behaviors)")

    def run(self) -> None:
        while True:
            with self._cond:
                while self._running and (not self._active or not self._queue):
                    self._cond.wait(timeout=0.2)
                if not self._running:
                    break
                if not self._active or not self._queue:
                    continue
                batch = list(self._queue)
                self._queue.clear()
                dropped = self._dropped
                self._dropped = 0

            try:
                self._ensure_engine()
                if self._engine is None and self._rule_detector is None:
                    self.error_occurred.emit("Behavior detector not configured")
                    with self._cond:
                        self._active = False
                    continue

                frames = [
                    result_to_framerecord(r, self._mapper, kp_reorder=self._kp_reorder)
                    for r in batch
                ]
                if self._rule_detector is not None:
                    state = self._process_rules(frames, dropped)
                else:
                    state = self._process_ml(frames, dropped)
                if state is not None:
                    self.behavior_ready.emit(state)
            except Exception as exc:  # never kill the thread on a single bad frame
                self.error_occurred.emit(f"Behavior inference error: {exc}")

    # ------------------------------------------------------------------ #
    def _process_ml(self, frames, dropped):
        # keep the buffer contiguous: push every frame, score only the newest
        for fr in frames[:-1]:
            self._engine.push_only(fr)
        t0 = time.perf_counter()
        decision = self._engine.on_detection(frames[-1])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if decision is None:
            return None
        labels = self._engine.labels
        K = self._engine.K
        active = {labels[k]: bool(decision.scene_binary[k]) for k in range(K)}
        probs = {labels[k]: float(decision.scene_prob[k]) for k in range(K)}
        from rule_based_social import pick_top_behavior  # torch-free helper
        per_track = {}
        for sid, pvec in (decision.per_track_prob or {}).items():
            bvec = (decision.per_track_binary or {}).get(sid)
            probs = {labels[k]: float(pvec[k]) for k in range(K)}
            binary = {labels[k]: bool(bvec[k]) for k in range(K)} if bvec is not None else {}
            # ML has no rule priority; pick the highest-prob active class, else background
            tname, tprob = pick_top_behavior(probs, binary, priority=[], none_label="background")
            per_track[str(sid)] = {"probs": probs, "binary": binary, "top": tname, "top_prob": tprob}
        return BehaviorFrameState(
            frame_idx=int(decision.frame_idx), timestamp_s=float(decision.timestamp_s),
            active=active, probs=probs, events=list(decision.events),
            latency_ms=latency_ms, dropped=int(dropped), labels=list(labels), per_track=per_track,
        )

    def _process_rules(self, frames, dropped):
        # the rule detector is fast & needs every frame for kinematics -> process all
        t0 = time.perf_counter()
        result = None
        for fr in frames:
            result = self._rule_detector.process(fr)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            return None
        return BehaviorFrameState(
            frame_idx=int(result.frame_idx), timestamp_s=float(result.timestamp_s),
            active=dict(result.active), probs=dict(result.probs), events=[],
            latency_ms=latency_ms, dropped=int(dropped), labels=list(result.labels),
            per_track=result.per_track,
        )
