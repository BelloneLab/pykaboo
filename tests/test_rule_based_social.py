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
    assert len(LABELS) == 15
    assert "rearing" in LABELS
    assert "passive" in LABELS
    st = _run(lambda i: ((60, 60, 0), (260, 200, 90)))   # far apart, constant posture
    assert _active(st) & set(LABELS) == set()             # no real behavior active
    assert st.active.get("none") is True
    # 'none' is not in the trigger label list
    assert "none" not in LABELS


def test_nose2nose_when_facing_and_close():
    # masks must actually overlap (the contact gate is mask contour overlap); two
    # mice nose-to-nose have touching bodies, so place them close enough to overlap.
    st = _run(lambda i: ((150, 120, 0), (198, 120, 180)))
    assert "nose2nose" in _active(st)


def test_nose2nose_when_masks_within_contact_pad_and_noses_closest():
    f0 = _frame(0, (150, 120, 0), (206, 120, 180))
    m1 = f0.mice["1"]["mask"]
    m2 = f0.mice["2"]["mask"]
    assert not np.any(m1 & m2)
    assert rbs._mask_contact_distance(m1, m2, tol=3.0) <= 3.0

    st = _run(lambda i: ((150, 120, 0), (206, 120, 180)))
    act = _active(st)
    assert "nose2nose" in act
    assert "oriented_toward" not in act
    assert st.per_track["1"].get("top") == "nose2nose"
    assert st.per_track["2"].get("top") == "nose2nose"


def test_nose2nose_when_nose_tips_under_segmented():
    # Real-world failure: a segmentation net under-segments the slender nose tips, so
    # the two mask contours keep a gap WIDER than the +/-5% band even though the nose
    # keypoints overlap. The mask gate alone then reports "no contact" and the frame
    # decays to oriented_toward. The keypoint-proximity OR must rescue nose2nose.
    # _ellipse(.., a=L*0.45, ..) ends short of the nose keypoint (nose = c + ax*L*0.5),
    # so facing mice naturally leave a contour gap while the nose kps sit ~5 px apart.
    f0 = _frame(0, (150, 120, 0), (215, 120, 180))
    m1, m2 = f0.mice["1"]["mask"], f0.mice["2"]["mask"]
    assert not np.any(m1 & m2)
    # masks are genuinely outside the +/-5% contact band (this is the regression)
    contact_pad = max(2.0, 0.05 * 60.0)
    assert rbs._mask_contact_distance(m1, m2, tol=contact_pad) > contact_pad

    st = _run(lambda i: ((150, 120, 0), (215, 120, 180)))
    act = _active(st)
    assert "nose2nose" in act
    assert "oriented_toward" not in act
    assert st.per_track["1"].get("top") == "nose2nose"
    assert st.per_track["2"].get("top") == "nose2nose"


def test_fast_nose2nose_investigation_is_not_fighting():
    # Fast, close, jittery nose-to-nose investigation used to be mislabeled as
    # fighting because axis jitter satisfied the old erratic-motion test.
    det = RuleBasedSocialDetector()
    last = None
    for i in range(14):
        y = 100 + i * 4
        jitter = 25 if i % 2 else -25
        last = det.process(_frame(i, (150, y, jitter), (198, y, 180 - jitter)))
    assert last is not None
    assert "nose2nose" in _active(last)
    assert "fighting" not in _active(last)


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


def _frame_lengths(i, m1, m2):
    """Frame with explicit per-mouse body length (cx, cy, ang, length), no masks."""
    mice = {}
    for sid, (cx, cy, ang, L) in (("1", m1), ("2", m2)):
        mice[sid] = {"present": True, "bbox_xywh": None, "score": 0.95, "mask": None,
                     "keypoints": _kp(cx, cy, L, ang), "keypoint_scores": np.full(8, 0.9)}
    return SimpleNamespace(frame_idx=i, timestamp_s=i / 30.0, width=W, height=H, mice=mice)


def test_rearing_when_body_foreshortens():
    # mouse1 keeps a normal extended body for the baseline window, then foreshortens
    # (vertical stand); mouse2 keeps a constant posture far away.
    det = RuleBasedSocialDetector()
    last = None
    for i in range(70):
        L1 = 70.0 if i < 50 else 28.0
        last = det.process(_frame_lengths(i, (90, 90, 0, L1), (260, 200, 90, 70.0)))
    assert last is not None
    assert last.per_track["1"]["binary"]["rearing"] is True
    assert last.per_track["2"]["binary"]["rearing"] is False
    assert "rearing" in _active(last)
    assert last.per_track["1"].get("top") == "rearing"


def test_rearing_suppressed_during_close_social_contact():
    # A close partner can hide the nose-tail extent in a top-down view. That is not
    # reliable evidence for a vertical stand, so rearing should stay off.
    det = RuleBasedSocialDetector()
    last = None
    for i in range(70):
        L1 = 70.0 if i < 50 else 28.0
        last = det.process(_frame_lengths(i, (90, 90, 0, L1), (150, 90, 180, 70.0)))
    assert last is not None
    assert last.per_track["1"]["binary"]["rearing"] is False
    assert "rearing" not in _active(last)


def test_no_rearing_for_constant_posture():
    det = RuleBasedSocialDetector()
    last = None
    for i in range(70):
        last = det.process(_frame_lengths(i, (90, 90, 0, 70.0), (260, 200, 90, 70.0)))
    assert last is not None
    assert last.per_track["1"]["binary"]["rearing"] is False
    assert last.per_track["2"]["binary"]["rearing"] is False


def _drive_label(det, sid, seq):
    """Feed a sequence of (candidate, active_set) and return the displayed labels."""
    out = []
    for cand, active in seq:
        binary = {n: (n in active) for n in set(list(active) + [cand])}
        probs = {n: (0.6 if n in active else 0.1) for n in binary}
        name, _ = det._sticky_top(sid, cand, probs.get(cand, 0.5), binary, probs)
        det._last_top[sid] = name
        out.append(name)
    return out


def test_sticky_label_ignores_short_blip():
    det = RuleBasedSocialDetector(identities=("1",))
    det.p.top_label_dwell_frames = 3
    sid = "1"
    # nose2body steady, with a single 1-frame blip of 'following' at index 3.
    seq = [("nose2body", {"nose2body"})] * 3 + [("following", {"following", "nose2body"})] \
        + [("nose2body", {"nose2body"})] * 3
    labels = _drive_label(det, sid, seq)
    assert labels[2] == "nose2body"
    assert labels[3] == "nose2body"   # the 1-frame blip is debounced away
    assert all(l == "nose2body" for l in labels)


def test_sticky_label_switches_after_dwell():
    det = RuleBasedSocialDetector(identities=("1",))
    det.p.top_label_dwell_frames = 3
    sid = "1"
    seq = [("nose2body", {"nose2body"})] * 2 + [("following", {"following"})] * 5
    labels = _drive_label(det, sid, seq)
    assert labels[:2] == ["nose2body", "nose2body"]
    # following must persist dwell(=3) frames before the label switches.
    assert labels[2] == "nose2body"
    assert labels[3] == "nose2body"
    assert labels[4] == "following"
    assert labels[-1] == "following"


def test_sticky_label_dwell_one_is_legacy_immediate():
    det = RuleBasedSocialDetector(identities=("1",))
    det.p.top_label_dwell_frames = 1
    sid = "1"
    seq = [("a", {"a"}), ("b", {"b"}), ("c", {"c"})]
    labels = _drive_label(det, sid, seq)
    assert labels == ["a", "b", "c"]  # no debounce when dwell == 1
