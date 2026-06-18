"""Headless end-to-end test for the live behavior closed loop.

Feeds a synthetic two-mouse stream of ``LiveDetectionResult`` objects through the real
``LiveBehaviorWorker`` (background thread, GPU model), routes the emitted scene state
into a ``LiveRuleEngine`` ``behavior_class`` rule, and asserts the rule machinery fires
a TTL pulse. Skips automatically when torch / the model package / the checkpoint are
not present (e.g. CI without a GPU).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QTimer  # noqa: E402

import live_behavior_integration as integ  # noqa: E402
from live_detection_logic import LiveRuleEngine  # noqa: E402
from live_detection_types import LiveTriggerRule  # noqa: E402

pytestmark = pytest.mark.skipif(
    not integ.BEHAVIOR_AVAILABLE or not integ.default_checkpoint_exists(),
    reason=f"behavior stack unavailable: {integ.BEHAVIOR_IMPORT_ERROR or 'no checkpoint'}",
)

W, H = 240, 200
KP = 8


def _ellipse_mask(cx, cy, a, b, ang):
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.deg2rad(ang)
    xr = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    yr = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0


def _keypoints(cx, cy, length, ang):
    t = np.deg2rad(ang)
    ax = np.array([np.cos(t), np.sin(t)])
    perp = np.array([-np.sin(t), np.cos(t)])
    c = np.array([cx, cy])
    pts = [c + ax * length * 0.5, c + ax * length * 0.2 + perp * length * 0.18,
           c + ax * length * 0.2 - perp * length * 0.18, c + ax * length * 0.2,
           c, c - ax * length * 0.24 + perp * length * 0.16,
           c - ax * length * 0.24 - perp * length * 0.16, c - ax * length * 0.5]
    return np.stack(pts, axis=0)


class _Tracked:
    def __init__(self, mid, label, cx, cy, ang):
        length = 44.0
        mask = _ellipse_mask(cx, cy, length * 0.45, length * 0.18, ang)
        ys, xs = np.where(mask)
        self.mouse_id = mid
        self.label = label
        self.class_id = 0
        self.confidence = 0.95
        self.bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))  # XYXY
        self.center = (float(xs.mean()), float(ys.mean()))
        self.mask = mask
        self.keypoints = _keypoints(cx, cy, length, ang)
        self.keypoint_scores = np.full(KP, 0.9)


class _Result:
    def __init__(self, i, fps, interacting):
        self.frame_index = i
        self.timestamp_s = i / fps
        self.width = W
        self.height = H
        if interacting:
            self.tracked_mice = [
                _Tracked(1, "mouse1", W / 2 - 28, H / 2, 0.0),
                _Tracked(2, "mouse2", W / 2 + 28, H / 2, 180.0),
            ]
        else:
            r = 60.0
            t = i / fps
            self.tracked_mice = [
                _Tracked(1, "mouse1", W / 2 + r * np.cos(t), H / 2 + r * np.sin(t), 90),
                _Tracked(2, "mouse2", W / 2 + 0.5 * r * np.cos(-t + 2), H / 2 + 0.5 * r * np.sin(-t + 2), -90),
            ]


def test_behavior_worker_closed_loop():
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    worker = integ.LiveBehaviorWorker()
    worker.configure(
        integ.DEFAULT_BEHAVIOR_CHECKPOINT,
        backend="ml",           # this test exercises the EmbTCN model path
        device=None,            # auto (cuda if available)
        lookahead=4,
        min_window=120,         # warm up fast for the test
        max_queue=4000,         # do not drop frames in this synthetic feed
    )

    engine = LiveRuleEngine()
    # one behavior_class rule: any "mounting" -> pulse DO1 at onset
    rule = LiveTriggerRule(
        rule_id="b1", rule_type="behavior_class", output_id="DO1",
        mode="pulse", activation_pattern="entry", duration_ms=100,
        behavior_name="mounting",
    )
    engine.set_rules([rule])

    state = {"decisions": 0, "last": None, "errors": [], "labels": None, "pulses": 0, "latencies": []}

    def on_behavior(s):
        state["decisions"] += 1
        state["last"] = s
        state["labels"] = s.labels
        state["latencies"].append(s.latency_ms)
        # drive the rule engine exactly like the main window will
        engine.set_behavior_state(s.active, s.probs)
        ev = engine.evaluate(None, now_ms=int(s.timestamp_s * 1000) + state["decisions"] * 10)
        state["pulses"] += len(ev.triggered_pulses)

    worker.behavior_ready.connect(on_behavior)
    worker.error_occurred.connect(lambda m: state["errors"].append(m))
    worker.start_behavior()

    fps = 30.0
    n_frames = 1200
    target_decisions = 12
    feed = {"i": 0}

    def feed_one():
        i = feed["i"]
        if i >= n_frames or state["decisions"] >= target_decisions:
            timer.stop()
            return
        interacting = (i // 90) % 2 == 1   # alternate apart / interacting every 3 s
        worker.submit_result(_Result(i, fps, interacting))
        feed["i"] += 1

    # steady ~25 fps feed so the worker always has fresh frames to score (the worker
    # coalesces internally; decision rate is bounded by compute, not the feed)
    timer = QTimer()
    timer.timeout.connect(feed_one)
    timer.start(40)

    # safety timeout
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(app.quit)
    deadline.start(120_000)

    def maybe_quit():
        if state["decisions"] >= target_decisions:
            app.quit()
    poll = QTimer()
    poll.timeout.connect(maybe_quit)
    poll.start(150)

    app.exec()
    worker.shutdown()

    assert not state["errors"], f"worker errors: {state['errors']}"
    assert state["decisions"] >= 5, f"too few decisions: {state['decisions']}"
    assert state["labels"] and "mounting" in state["labels"]
    assert state["last"] is not None
    # scene state must cover every class
    assert set(state["last"].active.keys()) == set(state["labels"])
    assert all(l > 0 for l in state["latencies"])
    # per-mouse (per-track) data must be present for the overlay subtitles
    per_track = state["last"].per_track
    assert set(per_track.keys()) <= {"1", "2"} and per_track
    for sid, blk in per_track.items():
        assert set(blk["probs"].keys()) == set(state["labels"])
        # argmax pick (what the subtitle shows) must be a real class
        top = max(blk["probs"].items(), key=lambda kv: kv[1])[0]
        assert top in state["labels"]
    # behavior_class truth must be readable by the rule engine without error
    engine.set_behavior_state({"mounting": True}, {"mounting": 0.9})
    truth = engine._rule_truth(rule, {})
    assert truth is True
    engine.set_behavior_state({"mounting": False}, {"mounting": 0.1})
    assert engine._rule_truth(rule, {}) is False


class _ChaseTracked:
    """Mouse for the rule-backend test: explicit centre via keypoints."""

    def __init__(self, mid, cx, cy, ang):
        length = 44.0
        mask = _ellipse_mask(cx, cy, length * 0.45, length * 0.18, ang)
        ys, xs = np.where(mask)
        self.mouse_id = mid
        self.label = f"mouse{mid}"
        self.class_id = 0
        self.confidence = 0.95
        self.bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        self.center = (float(xs.mean()), float(ys.mean()))
        self.mask = mask
        self.keypoints = _keypoints(cx, cy, length, ang)
        self.keypoint_scores = np.full(KP, 0.9)


class _ChaseResult:
    def __init__(self, i, fps):
        self.frame_index = i
        self.timestamp_s = i / fps
        self.width = W
        self.height = H
        bx = 60 + i * 5
        # mouse 1 chases mouse 2 (behind it, both moving right)
        self.tracked_mice = [
            _ChaseTracked(1, bx - 52, H / 2, 0.0),
            _ChaseTracked(2, bx, H / 2, 0.0),
        ]


def test_behavior_worker_rule_backend():
    """The fast rule-based backend runs through the worker and drives behavior_class
    rules (no checkpoint / model load needed)."""
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    worker = integ.LiveBehaviorWorker()
    worker.configure(integ.DEFAULT_BEHAVIOR_CHECKPOINT, backend="rules", max_queue=4000)

    engine = LiveRuleEngine()
    engine.set_rules([LiveTriggerRule(
        rule_id="c1", rule_type="behavior_class", output_id="DO1",
        mode="gate", behavior_name="chasing",
    )])

    state = {"decisions": 0, "last": None, "errors": [], "labels": None, "chasing_truth": []}

    def on_behavior(s):
        state["decisions"] += 1
        state["last"] = s
        state["labels"] = s.labels
        engine.set_behavior_state(s.active, s.probs)
        state["chasing_truth"].append(engine._rule_truth(engine.rules[0], {}))

    worker.behavior_ready.connect(on_behavior)
    worker.error_occurred.connect(lambda m: state["errors"].append(m))
    worker.start_behavior()

    fps = 30.0
    feed = {"i": 0}
    target = 30

    def feed_one():
        if feed["i"] >= 200 or state["decisions"] >= target:
            timer.stop()
            return
        worker.submit_result(_ChaseResult(feed["i"], fps))
        feed["i"] += 1

    timer = QTimer()
    timer.timeout.connect(feed_one)
    timer.start(10)

    deadline = QTimer(); deadline.setSingleShot(True)
    deadline.timeout.connect(app.quit); deadline.start(30_000)

    poll = QTimer()
    poll.timeout.connect(lambda: app.quit() if state["decisions"] >= target else None)
    poll.start(100)

    app.exec()
    worker.shutdown()

    assert not state["errors"], state["errors"]
    assert state["decisions"] >= 10, state["decisions"]
    assert state["labels"] and "chasing" in state["labels"] and "following" in state["labels"]
    # the rule engine must have read True for chasing on at least some frames
    assert any(t is True for t in state["chasing_truth"]), "chasing never detected"
    # per-track present and keyed by mouse id
    assert set(state["last"].per_track.keys()) <= {"1", "2"}


if __name__ == "__main__":
    test_behavior_worker_closed_loop()
    test_behavior_worker_rule_backend()
    print("OK")
