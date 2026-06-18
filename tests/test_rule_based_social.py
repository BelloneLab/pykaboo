"""Unit tests for the rule-based social-behavior detector (torch-free, fast).

Drives the detector with synthetic two-mouse geometry and asserts the documented
rules fire (and don't fire) as expected. No GPU / torch / checkpoint required.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "pykaboo_live_behavior")
for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

rbs = pytest.importorskip("rule_based_social")
RuleBasedSocialDetector = rbs.RuleBasedSocialDetector
LABELS = rbs.LABELS

W, H = 320, 240


def _kp(cx, cy, length, ang):
    t = np.deg2rad(ang)
    ax = np.array([np.cos(t), np.sin(t)])
    perp = np.array([-np.sin(t), np.cos(t)])
    c = np.array([cx, cy])
    nose = c + ax * length * 0.5
    tail = c - ax * length * 0.5
    neck = c + ax * length * 0.2
    body = c
    lear = neck + perp * length * 0.18
    rear = neck - perp * length * 0.18
    lhip = c - ax * length * 0.24 + perp * length * 0.16
    rhip = c - ax * length * 0.24 - perp * length * 0.16
    return np.stack([nose, lear, rear, neck, body, lhip, rhip, tail], 0)


def _ellipse(cx, cy, a, b, ang):
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.deg2rad(ang)
    xr = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    yr = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0


def _frame(i, m1, m2, with_mask=True):
    mice = {}
    L = 60.0
    for sid, (cx, cy, ang) in (("1", m1), ("2", m2)):
        mask = _ellipse(cx, cy, L * 0.45, L * 0.18, ang) if with_mask else None
        if mask is not None:
            ys, xs = np.where(mask)
            bb = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        else:
            bb = (cx - 20, cy - 10, 40, 20)
        mice[sid] = {"present": True, "bbox_xywh": bb, "score": 0.95, "mask": mask,
                     "keypoints": _kp(cx, cy, L, ang), "keypoint_scores": np.full(8, 0.9)}
    return SimpleNamespace(frame_idx=i, timestamp_s=i / 30.0, width=W, height=H, mice=mice)


def _run(scenario, n=26):
    det = RuleBasedSocialDetector()
    last = None
    for i in range(n):
        st = det.process(_frame(i, *scenario(i)))
        if st is not None:
            last = st
    return last


def _active(st):
    return {k for k, v in st.active.items() if v}


def test_labels_count_and_none_synthetic():
    assert len(LABELS) == 13
    st = _run(lambda i: ((60, 60, 0), (260, 200, 90)))   # far apart
    assert _active(st) & set(LABELS) == set()             # no real behavior active
    assert st.active.get("none") is True
    # 'none' is not in the trigger label list
    assert "none" not in LABELS


def test_nose2nose_when_facing_and_close():
    # masks must actually overlap (the contact gate is mask contour overlap); two
    # mice nose-to-nose have touching bodies, so place them close enough to overlap.
    st = _run(lambda i: ((150, 120, 0), (198, 120, 180)))
    assert "nose2nose" in _active(st)


def test_far_apart_is_idle():
    st = _run(lambda i: ((60, 60, 0), (260, 200, 90)))
    assert "nose2nose" not in _active(st)
    assert "fighting" not in _active(st)


def test_chase_fires_following_and_chasing_directional():
    def chase(i):
        bx = 110 + i * 5
        return (bx - 70, 120, 0), (bx, 120, 0)   # mouse1 behind mouse2, both moving right
    st = _run(chase, 30)
    act = _active(st)
    assert "following" in act and "chasing" in act
    # mouse 1 is the actor (follower/chaser)
    assert st.per_track["1"]["binary"]["following"] is True
    assert st.per_track["2"]["binary"]["following"] is False


def test_approach_is_directional_and_subject_driven():
    def approach(i):
        return (60 + i * 4, 120, 0), (220, 120, 90)   # mouse1 moves toward stationary mouse2
    st = _run(approach, 24)
    assert "approach" in _active(st)
    assert st.per_track["1"]["binary"]["approach"] is True


def test_per_track_probs_cover_all_labels():
    st = _run(lambda i: ((150, 120, 0), (216, 120, 180)))
    for sid in ("1", "2"):
        assert set(st.probs.keys()) >= set(LABELS)
        assert set(st.per_track[sid]["probs"].keys()) >= set(LABELS)


def test_runs_without_masks():
    # detector must not crash when masks are absent (pose-only contact)
    st = _run(lambda i: ((150, 120, 0), (216, 120, 180)))
    assert st is not None
    det = RuleBasedSocialDetector()
    last = None
    for i in range(10):
        last = det.process(_frame(i, (150, 120, 0), (216, 120, 180), with_mask=False))
    assert last is not None
