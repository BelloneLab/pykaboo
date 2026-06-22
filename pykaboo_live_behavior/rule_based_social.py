"""Rule-based social-behavior detection for real-time closed loop.

A fast, torch-free alternative to the EmbTCN temporal model: per-frame geometric and
kinematic tests over keypoints + mask contours, mirroring the rules in
``dlc_processor/core/social_behaviors.py``. Each rule is a boolean test over the two
mice; booleans are temporally smoothed with a trailing uniform window (online
equivalent of ``uniform_filter1d``, kept where smoothed > 0.4).

It produces the SAME scene-level + per-track behavior state the ML worker emits
(``active`` / ``probs`` / ``per_track``), so it plugs straight into the existing
``behavior_class`` TTL rules and the per-mouse preview subtitles. Latency is sub-ms,
so this is the backend to use for genuine closed-loop triggering.

Keypoint order (pykaboo, 8): nose, left_ear, right_ear, neck, body, left_hip,
right_hip, tail. Shared building blocks (rear anchor, body axis, relative heading,
close_tol, body-frame projection, likelihood gating, motion thresholds) follow the
spec. ``probs`` here are the trailing-smoothed fractions in [0, 1] (a natural
confidence), and a synthetic ``none`` class is added so the subtitle reads "none"
when nothing is active.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Keypoint indices (pykaboo order)
I_NOSE, I_LEAR, I_REAR_EAR, I_NECK, I_BODY, I_LHIP, I_RHIP, I_TAIL = range(8)

# Scene-level behavior names (base, direction-agnostic). Directional rules set the
# acting mouse in per_track; bidirectional rules set both mice.
LABELS = [
    "nose2nose", "sidebyside", "sidereside", "nose2anogenital", "nose2body",
    "oriented_toward", "following", "chasing", "approach",
    "withdrawal_from_partner", "escape", "withdrawal_after_contact", "fighting",
    "rearing", "passive",
]
BIDIRECTIONAL = {"nose2nose", "sidebyside", "sidereside", "fighting"}
# Single-mouse (non-social) behaviors: scored per subject, no partner required.
SOLO = {"rearing"}
NONE_LABEL = "none"

# Chip/label priority: when several behaviors are active at once, show the most
# specific / salient one (not whichever weak cue has the highest smoothed value).
# ``passive`` is intentionally last: it is the receiving role, shown only when the
# investigated mouse is doing nothing more specific itself.
PRIORITY = [
    "fighting", "chasing", "nose2anogenital", "nose2nose", "nose2body",
    "following", "sidebyside", "sidereside", "rearing", "escape",
    "withdrawal_after_contact", "approach", "withdrawal_from_partner", "oriented_toward",
    "passive",
]


def pick_top_behavior(probs: dict, binary: dict, priority=PRIORITY,
                      none_label: str = NONE_LABEL):
    """Choose the behavior to display: the highest-priority ACTIVE (debounced-on)
    behavior, breaking ties by probability. Falls back to ``none`` when nothing is
    on, so weak sub-threshold cues never show."""
    active = [n for n, on in (binary or {}).items()
              if on and str(n).lower() not in (none_label, "background")]
    if not active:
        return none_label, float((probs or {}).get(none_label, 0.0))

    def rank(n):
        return priority.index(n) if n in priority else 10_000
    active.sort(key=lambda n: (rank(n), -float((probs or {}).get(n, 0.0))))
    top = active[0]
    return top, float((probs or {}).get(top, 0.0))


@dataclass
class RuleParams:
    likelihood_threshold: float = 0.20
    smooth_frames: int = 6
    smooth_keep: float = 0.4
    # Tolerances scale with body length (px) for resolution/arena invariance. A
    # positive absolute value overrides the auto fraction; 0 means "auto".
    close_tol: float = 0.0            # auto = clip(0.30*body_len, floor, 1.5*body_len)
    side_tol: float = 0.0             # auto = max(side_tol_floor, side_tol_frac*body_len)
    side_tol_frac: float = 0.75
    side_tol_floor: float = 30.0
    follow_tol: float = 0.0           # auto = max(follow_tol_floor, follow_tol_frac*body_len)
    follow_tol_frac: float = 0.45
    follow_tol_floor: float = 20.0
    angle_tol_nose2nose: float = 60.0
    angle_tol_anogenital: float = 75.0
    angle_tol_oriented: float = 30.0
    follow_window: int = 6
    min_follow_frames: int = 6
    # Absolute px/frame degeneracy guard only; the real motion floor is 0.010*scale
    # (a fraction of body length), so motion gating is resolution-invariant.
    stationary_threshold: float = 0.1   # px/frame floor for motion thresholds
    contact_window: int = 12            # frames to remember recent contact
    use_mask_contact: bool = True
    # Social contact gate: masks within this fraction of body length count as touching
    # ("contour +/-5% overlap"). Contact TYPE is then read from the closest keypoints.
    mask_contact_frac: float = 0.05
    # Keypoint-proximity contact fallback (used even WHEN masks are available). A
    # segmentation network routinely under-segments the slender nose tips, so two mice
    # in a true nose-to-nose can keep a mask-contour gap wider than the +/-5% band
    # while their nose keypoints overlap. When the closest inter-animal keypoint pair
    # is within this fraction of body length, the gate counts contact too, so the
    # frame is classified from the closest keypoints (nose2nose) instead of decaying
    # to oriented_toward.
    kp_contact_frac: float = 0.15
    # An animal "leads with its nose" (active investigation) when its nose is within
    # this fraction of body length of being the closest of its keypoints to the partner.
    lead_margin_frac: float = 0.18
    # proximity gate (x body length) for approach / withdrawal / escape / oriented
    near_limit_frac: float = 2.5
    oriented_max_frac: float = 2.5
    # Minimum history before velocity or acceleration based rules may fire. This
    # prevents startup pose jitter from becoming fighting or escape.
    kinematic_warmup_frames: int = 6
    # body-frame zone fractions (of body length)
    rear_depth_frac: float = 0.15
    rear_width_frac: float = 0.55
    head_depth_frac: float = 0.15
    # Fighting is deliberately conservative. In top-down recordings, clean
    # nose-to-nose or nose-to-body investigation can look fast and erratic when
    # poses jitter, so aggression needs sustained unstructured contact plus
    # severe reciprocal motion.
    fighting_contact_frames: int = 4
    fighting_contact_frac: float = 0.75
    fighting_rel_speed_frac: float = 0.040
    fighting_turn_deg: float = 55.0
    fighting_accel_mult: float = 1.50
    # Rearing (single mouse, top-down): a vertical stand foreshortens the projected
    # body (nose-tail) and shrinks the mask area relative to the animal's OWN normal
    # extended posture. Baseline = a high percentile over a trailing window so a
    # sustained rear does not drag the baseline down. Mask-area check is applied
    # only when masks are available (it confirms the body-length shrink).
    # Calibrated on top-down recordings: a rear foreshortens the body markedly but
    # only shrinks the projected mask area ~20-30% (the silhouette stays a compact
    # blob), so the area gate is looser than the body gate. The body+area AND also
    # rejects keypoint-collapse artifacts (body_len near 0 while area stays normal).
    rearing_body_frac: float = 0.72      # rear when body_len < frac * baseline
    rearing_area_frac: float = 0.80      # ...and mask area < frac * baseline (if masks)
    rearing_baseline_frames: int = 150
    rearing_min_samples: int = 30
    rearing_baseline_pct: float = 70.0
    # Suppress posture-only rearing calls during close social contact. In a top-down
    # camera, contact and partial occlusion shorten the nose-tail projection in the
    # same way as a true rear.
    rearing_partner_suppression_frac: float = 1.35


@dataclass
class RuleFrameState:
    frame_idx: int
    timestamp_s: float
    active: dict                       # {name: bool}  scene-level (OR over directions)
    probs: dict                        # {name: float} scene-level (MAX smoothed)
    per_track: dict                    # {sid: {"probs": {...}, "binary": {...}}}
    labels: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _finite(p) -> bool:
    return p is not None and np.all(np.isfinite(p))


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-9:
        return None
    return np.array([v[0] / n, v[1] / n], dtype=np.float64)


def _angle_deg(u, v) -> float:
    """Angle between two vectors in degrees (0..180). NaN-safe -> 180 if undefined."""
    uu, vv = _unit(u), _unit(v)
    if uu is None or vv is None:
        return 180.0
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _perp(u):
    return np.array([-u[1], u[0]], dtype=np.float64)


@dataclass
class _Geom:
    present: bool
    nose: object
    tail: object
    centre: object
    rear: object
    axis: object       # unit vector rear->nose or None
    body_len: float


def _mouse_geom(payload: dict, params: RuleParams) -> _Geom:
    if payload is None or not payload.get("present", False):
        return _Geom(False, None, None, None, None, None, float("nan"))
    kps = payload.get("keypoints")
    scores = payload.get("keypoint_scores")
    bbox = payload.get("bbox_xywh")
    if kps is None:
        # no pose -> centre from bbox only, no axis
        if bbox is not None:
            cx = float(bbox[0] + bbox[2] / 2.0)
            cy = float(bbox[1] + bbox[3] / 2.0)
            return _Geom(True, None, None, np.array([cx, cy]), None, None, float("nan"))
        return _Geom(False, None, None, None, None, None, float("nan"))
    kp = np.asarray(kps, dtype=np.float64).copy()
    if scores is not None:
        sc = np.asarray(scores, dtype=np.float64)
        low = sc < params.likelihood_threshold
        kp[low] = np.nan

    def g(i):
        return kp[i] if _finite(kp[i]) else None

    nose = g(I_NOSE)
    tail = g(I_TAIL)
    lhip, rhip = g(I_LHIP), g(I_RHIP)
    if lhip is not None and rhip is not None:
        hips_mid = (lhip + rhip) / 2.0
    else:
        hips_mid = lhip if lhip is not None else rhip
    body = g(I_BODY)
    centre = hips_mid if hips_mid is not None else (body if body is not None else None)
    if centre is None and bbox is not None:
        centre = np.array([float(bbox[0] + bbox[2] / 2.0), float(bbox[1] + bbox[3] / 2.0)])
    # rear anchor = explicit tail, else hips nudged 25% toward where the tail would be
    rear = tail if tail is not None else hips_mid
    axis = None
    body_len = float("nan")
    if nose is not None and rear is not None:
        axis = _unit(nose - rear)
        body_len = float(np.hypot(*(nose - rear)))
    return _Geom(True, nose, tail, centre, rear, axis, body_len)


# --------------------------------------------------------------------------- #
# mask contact (segmentation contour distance, cheap & cropped)
# --------------------------------------------------------------------------- #
def _mask_contact_distance(mask_a, mask_b, tol: float) -> float:
    """Min surface distance (px) between two boolean masks; 0 if overlapping.

    Returns +inf quickly when the masks' bounding boxes are farther than ``tol``
    apart, so the (cropped) distance transform only runs when contact is plausible.
    """
    if mask_a is None or mask_b is None:
        return float("inf")
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape or not a.any() or not b.any():
        return float("inf")
    ya, xa = np.where(a)
    yb, xb = np.where(b)
    ax0, ax1, ay0, ay1 = xa.min(), xa.max(), ya.min(), ya.max()
    bx0, bx1, by0, by1 = xb.min(), xb.max(), yb.min(), yb.max()
    gap_x = max(0, bx0 - ax1, ax0 - bx1)
    gap_y = max(0, by0 - ay1, ay0 - by1)
    if min(gap_x, gap_y) > tol and math.hypot(gap_x, gap_y) > tol:
        return float("inf")
    if np.any(a & b):
        return 0.0
    import cv2

    m = int(math.ceil(tol)) + 2
    x0 = max(0, min(ax0, bx0) - m)
    y0 = max(0, min(ay0, by0) - m)
    x1 = min(a.shape[1], max(ax1, bx1) + m)
    y1 = min(a.shape[0], max(ay1, by1) + m)
    ca = a[y0:y1, x0:x1]
    cb = b[y0:y1, x0:x1]
    if not ca.any() or not cb.any():
        return float("inf")
    dist_to_b = cv2.distanceTransform((~cb).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist_to_b[ca].min())


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class RuleBasedSocialDetector:
    """Per-frame geometric social-behavior detector for two mice."""

    LABELS = LABELS

    def __init__(self, identities=("1", "2"), params: Optional[RuleParams] = None):
        self.identities = tuple(identities)
        self.p = params or RuleParams()
        # kinematic + posture history (deep enough for the rearing baseline window)
        self._hist = {sid: deque(maxlen=256) for sid in self.identities}
        self._smooth: dict = {}                                          # slot -> deque[bool]
        self._follow_inst = {sid: deque(maxlen=64) for sid in self.identities}
        self._chase_inst = {sid: deque(maxlen=64) for sid in self.identities}
        self._contact_hist = deque(maxlen=max(2, self.p.contact_window))
        # Robust body-length scale (px), median over a trailing window. Every
        # tolerance is a fraction of this, so the rules transfer across resolutions
        # and camera distances; the median keeps it stable when one frame's pose
        # collapses (occlusion / rearing) instead of letting tolerances blink.
        self._scale_hist: deque = deque(maxlen=150)
        self._scale: Optional[float] = None
        self._n = 0

    def reset(self) -> None:
        for d in self._hist.values():
            d.clear()
        for d in self._follow_inst.values():
            d.clear()
        for d in self._chase_inst.values():
            d.clear()
        self._smooth.clear()
        self._contact_hist.clear()
        self._scale_hist.clear()
        self._scale = None
        self._n = 0

    # -------------------- smoothing -------------------- #
    def _smoothed(self, slot, raw: bool) -> float:
        dq = self._smooth.get(slot)
        if dq is None:
            dq = deque(maxlen=self.p.smooth_frames)
            self._smooth[slot] = dq
        dq.append(1.0 if raw else 0.0)
        return float(sum(dq) / len(dq))

    # -------------------- main -------------------- #
    def process(self, frame) -> Optional[RuleFrameState]:
        self._n += 1
        a, b = self.identities[0], self.identities[1]
        ga = _mouse_geom(frame.mice.get(a), self.p)
        gb = _mouse_geom(frame.mice.get(b), self.p)
        ts = float(getattr(frame, "timestamp_s", 0.0))

        # kinematics from history (per mouse): velocity, speed, accel
        kin = {}
        for sid, g in ((a, ga), (b, gb)):
            h = self._hist[sid]
            prev = h[-1] if h else None
            vel = np.array([0.0, 0.0])
            speed = 0.0
            accel = 0.0
            if g.centre is not None and prev is not None and prev["centre"] is not None:
                vel = g.centre - prev["centre"]
                speed = float(np.hypot(*vel))
                accel = speed - prev.get("speed", 0.0)
            kin[sid] = {"vel": vel, "speed": speed, "accel": accel}
            m = frame.mice.get(sid, {}).get("mask")
            try:
                area = float(np.count_nonzero(m)) if m is not None else float("nan")
            except Exception:
                area = float("nan")
            h.append({"centre": g.centre, "nose": g.nose, "tail": g.tail, "rear": g.rear,
                      "axis": g.axis, "speed": speed, "vel": vel, "t": ts,
                      "body_len": g.body_len, "area": area})

        raw = self._eval_rules(a, b, ga, gb, kin, frame)

        # ---- smooth every slot, build scene + per-track ----
        scene_active = {n: False for n in LABELS}
        scene_probs = {n: 0.0 for n in LABELS}
        per_track = {sid: {"probs": {}, "binary": {}} for sid in self.identities}
        for sid in self.identities:
            for n in LABELS:
                per_track[sid]["probs"][n] = 0.0
                per_track[sid]["binary"][n] = False

        for (name, actor), val in raw.items():
            sm = self._smoothed((name, actor), bool(val))
            on = sm > self.p.smooth_keep
            scene_probs[name] = max(scene_probs[name], sm)
            scene_active[name] = scene_active[name] or on
            sids = self.identities if (actor is None) else (actor,)
            for sid in sids:
                if sm >= per_track[sid]["probs"][name]:
                    per_track[sid]["probs"][name] = sm
                    per_track[sid]["binary"][name] = on

        # synthetic 'none' so the subtitle shows something sensible when idle
        scene_probs[NONE_LABEL] = max(0.0, 1.0 - max(scene_probs.values()) if scene_probs else 1.0)
        scene_active[NONE_LABEL] = not any(scene_active.values())
        for sid in self.identities:
            top = max(per_track[sid]["probs"].values()) if per_track[sid]["probs"] else 0.0
            per_track[sid]["probs"][NONE_LABEL] = max(0.0, 1.0 - top)
            per_track[sid]["binary"][NONE_LABEL] = top <= self.p.smooth_keep
            # priority-based label for the chip / CSV (most specific active behavior)
            tname, tprob = pick_top_behavior(per_track[sid]["probs"], per_track[sid]["binary"])
            per_track[sid]["top"] = tname
            per_track[sid]["top_prob"] = tprob

        # need ~2 frames for kinematics before emitting
        if self._n < 2:
            return None
        return RuleFrameState(
            frame_idx=int(getattr(frame, "frame_idx", self._n)),
            timestamp_s=ts,
            active=scene_active,
            probs=scene_probs,
            per_track=per_track,
            labels=list(LABELS),
        )

    # -------------------- rules -------------------- #
    def _eval_rules(self, a, b, ga: _Geom, gb: _Geom, kin: dict, frame) -> dict:
        p = self.p
        raw: dict = {}

        def put(name, actor, val):
            raw[(name, actor)] = bool(val)

        # default everything False (so smoothing decays when geometry is missing)
        for name in LABELS:
            if name in BIDIRECTIONAL:
                put(name, None, False)
            else:
                put(name, a, False)
                put(name, b, False)

        # ---- rearing (single mouse): a vertical stand foreshortens the projected
        # body (nose-tail) and shrinks the mask area vs the animal's OWN baseline.
        # Solo, so it is scored even if the partner is missing this frame.
        def rearing(sid, g, socially_occluded=False):
            if socially_occluded:
                return False
            hist = self._hist[sid]
            bl_now = g.body_len
            if not np.isfinite(bl_now):
                return False
            window = list(hist)[-p.rearing_baseline_frames:]
            bls = [r["body_len"] for r in window if np.isfinite(r.get("body_len", np.nan))]
            if len(bls) < p.rearing_min_samples:
                return False
            base_bl = float(np.percentile(bls, p.rearing_baseline_pct))
            if base_bl <= 1e-6 or bl_now >= p.rearing_body_frac * base_bl:
                return False
            # Confirm with mask area when masks are available (rejects pose jitter).
            area_now = window[-1].get("area", np.nan) if window else np.nan
            areas = [r["area"] for r in window if np.isfinite(r.get("area", np.nan))]
            if len(areas) >= p.rearing_min_samples and np.isfinite(area_now):
                base_area = float(np.percentile(areas, p.rearing_baseline_pct))
                if base_area > 1e-6 and area_now >= p.rearing_area_frac * base_area:
                    return False
            return True

        if not (ga.present and gb.present and ga.centre is not None and gb.centre is not None):
            put("rearing", a, rearing(a, ga))
            put("rearing", b, rearing(b, gb))
            self._contact_hist.append(False)
            for sid in self.identities:
                self._follow_inst[sid].append(False)
                self._chase_inst[sid].append(False)
            return raw

        # body length + tolerances, all keyed off a RESOLUTION-INVARIANT scale.
        # ``scale`` is the robust (trailing-median) body length in px; every
        # tolerance below is a fraction of it, so a higher-res or closer camera just
        # scales the animal and the thresholds together. The instantaneous mean
        # feeds the median; tiny px guards only prevent a degenerate near-zero scale.
        lens = [g.body_len for g in (ga, gb) if np.isfinite(g.body_len) and g.body_len > 1.0]
        if lens:
            self._scale_hist.append(float(np.mean(lens)))
        if self._scale_hist:
            scale = float(np.median(self._scale_hist))
        elif self._scale is not None:
            scale = self._scale
        else:
            scale = float(np.mean(lens)) if lens else 40.0
        self._scale = scale
        body_len = scale
        guard = max(1.0, 0.02 * scale)
        if p.close_tol > 0:
            close_tol = p.close_tol
        else:
            close_tol = float(np.clip(0.30 * scale, guard, 1.5 * scale))
        side_tol = p.side_tol if p.side_tol > 0 else max(guard, p.side_tol_frac * scale)
        follow_tol = p.follow_tol if p.follow_tol > 0 else max(guard, p.follow_tol_frac * scale)
        # motion thresholds: px/frame as fractions of the scale. The noise floor is a
        # small fraction of the scale (not an absolute px value) so it is invariant.
        stat = max(p.stationary_threshold, 0.010 * scale)
        move_thr = 0.012 * scale
        fast_thr = 0.030 * scale
        accel_thr = 0.010 * scale
        # proximity gate: approach/withdrawal/escape/oriented only count when the mice
        # are within a few body lengths (not the whole arena).
        near_limit = max(2 * side_tol, 4 * close_tol, p.near_limit_frac * body_len)
        oriented_max = p.oriented_max_frac * body_len
        rear_depth = p.rear_depth_frac * body_len
        rear_width = p.rear_width_frac * body_len
        head_depth = p.head_depth_frac * body_len

        noseA, noseB = ga.nose, gb.nose
        tailA, tailB = ga.tail, gb.tail
        cA, cB = ga.centre, gb.centre
        rA, rB = ga.rear, gb.rear
        axA, axB = ga.axis, gb.axis

        def dist(u, v):
            if u is None or v is None:
                return float("inf")
            return float(np.hypot(*(u - v)))

        centre_dist = dist(cA, cB)
        rel_heading = _angle_deg(axA, axB) if (axA is not None and axB is not None) else 180.0

        # Social-contact gate: mask contours overlapping or within +/-5% of body length.
        contact_pad = max(2.0, p.mask_contact_frac * body_len)
        maskA = frame.mice.get(a, {}).get("mask")
        maskB = frame.mice.get(b, {}).get("mask")
        masks_available = maskA is not None and maskB is not None
        mask_dist = float("inf")
        if p.use_mask_contact and masks_available:
            mask_dist = _mask_contact_distance(maskA, maskB, contact_pad)
        mask_contact = mask_dist <= contact_pad

        # body-frame projection of point q into target (centre c, axis ax)
        def project(q, c, ax):
            if q is None or c is None or ax is None:
                return None
            d = q - c
            return float(np.dot(d, ax)), float(abs(np.dot(d, _perp(ax))))

        def in_rear_zone(q, c, ax):
            pr = project(q, c, ax)
            return pr is not None and pr[0] < -rear_depth and pr[1] < rear_width

        def in_head_zone(q, c, ax):
            pr = project(q, c, ax)
            return pr is not None and pr[0] > head_depth and pr[1] < rear_width

        # ---- contact: mask-overlap GATE, then closest-keypoint CLASSIFICATION ----
        # Body-part regions of the 8 keypoints (pykaboo order).
        HEAD = (I_NOSE, I_LEAR, I_REAR_EAR, I_NECK)   # nose, ears, neck
        BODY = (I_BODY, I_LHIP, I_RHIP)               # body, hips (flank)
        REAR = (I_TAIL,)                              # tail base (anogenital)

        def gated(sid):
            kps = frame.mice.get(sid, {}).get("keypoints")
            if kps is None:
                return None
            out = np.asarray(kps, dtype=np.float64).copy()
            sc = frame.mice.get(sid, {}).get("keypoint_scores")
            if sc is not None:
                out[np.asarray(sc, dtype=np.float64) < p.likelihood_threshold] = np.nan
            return out

        gkpsA, gkpsB = gated(a), gated(b)

        def nearest_kp(q, kT):
            """Nearest target keypoint index to point q -> (dist, idx)."""
            if q is None or kT is None:
                return (float("inf"), None)
            best = (float("inf"), None)
            for j in range(len(kT)):
                if _finite(kT[j]):
                    d = float(np.hypot(*(q - kT[j])))
                    if d < best[0]:
                        best = (d, j)
            return best

        def closest_pair(kA, kB):
            """Closest inter-animal keypoint pair -> (dist, iA, jB)."""
            if kA is None or kB is None:
                return (float("inf"), None, None)
            best = (float("inf"), None, None)
            for i in range(len(kA)):
                if not _finite(kA[i]):
                    continue
                for j in range(len(kB)):
                    if _finite(kB[j]):
                        d = float(np.hypot(*(kA[i] - kB[j])))
                        if d < best[0]:
                            best = (d, i, j)
            return best

        dpair, ip, jp = closest_pair(gkpsA, gkpsB)
        # GATE: mask contour overlap (within the +/-5% band) OR the closest
        # inter-animal keypoint pair near-touching. The keypoint OR rescues nose2nose:
        # thin nose tips are routinely under-segmented, so the two mask contours keep a
        # gap wider than 5% of body length even while the nose keypoints overlap.
        # Without it a true nose-to-nose loses the contact gate and collapses to
        # oriented_toward. With masks missing we rely on keypoint proximity alone (the
        # looser close_tol), exactly as before.
        kp_contact = dpair <= max(contact_pad, p.kp_contact_frac * body_len)
        if masks_available and p.use_mask_contact:
            in_contact = mask_contact or kp_contact
        else:
            in_contact = dpair <= close_tol

        nose2nose = sidebyside = sidereside = False
        anog_ab = anog_ba = n2b_ab = n2b_ba = False
        if in_contact:
            # Exact contact-location override: if the mask-contact gate passes and
            # the closest inter-animal keypoint pair is nose-nose, this is nose2nose.
            if ip == I_NOSE and jp == I_NOSE:
                nose2nose = True
            else:
                dA, jA = nearest_kp(noseA, gkpsB)   # A's nose -> nearest B keypoint
                dB, jB = nearest_kp(noseB, gkpsA)   # B's nose -> nearest A keypoint
                lead = p.lead_margin_frac * body_len
                # an animal "leads" (active nose investigation) when its nose is
                # close to the closest of its keypoints to the partner.
                a_leads = jA is not None and dA <= dpair + lead
                b_leads = jB is not None and dB <= dpair + lead

                if a_leads and b_leads and jA in HEAD and jB in HEAD:
                    nose2nose = True
                else:
                    if a_leads:
                        if jA in REAR:
                            anog_ab = True
                        else:
                            n2b_ab = True          # nose at flank/body/head, one-sided
                    if b_leads:
                        if jB in REAR:
                            anog_ba = True
                        else:
                            n2b_ba = True
                    if not (a_leads or b_leads):
                        # bodies touching, neither nose leads -> parallel / anti-parallel
                        if rel_heading < 60.0:
                            sidebyside = True
                        elif rel_heading > 120.0:
                            sidereside = True

        put("nose2nose", None, nose2nose)
        put("sidebyside", None, sidebyside)
        put("sidereside", None, sidereside)
        put("nose2anogenital", a, anog_ab)
        put("nose2anogenital", b, anog_ba)
        put("nose2body", a, n2b_ab)
        put("nose2body", b, n2b_ba)

        # oriented_toward only counts toward a reasonably near partner (else any
        # incidental alignment across the arena fires it).
        structured_contact = bool(
            nose2nose or anog_ab or anog_ba or n2b_ab or n2b_ba or sidebyside or sidereside
        )
        oriented_near = centre_dist < oriented_max
        oriented_ab = bool(
            not structured_contact
            and oriented_near
            and axA is not None
            and _angle_deg(axA, cB - cA) < p.angle_tol_oriented
        )
        oriented_ba = bool(
            not structured_contact
            and oriented_near
            and axB is not None
            and _angle_deg(axB, cA - cB) < p.angle_tol_oriented
        )
        put("oriented_toward", a, oriented_ab)
        put("oriented_toward", b, oriented_ba)

        # contact priority already applied inside nose2body/anogenital tests

        # ---- contact memory ----
        any_contact = bool(nose2nose or anog_ab or anog_ba or n2b_ab or n2b_ba
                           or sidebyside or sidereside or mask_contact)
        recent_prior_contact = any(self._contact_hist)
        self._contact_hist.append(any_contact)
        recent_contact = any(self._contact_hist)

        close_social_occlusion = bool(any_contact or centre_dist <= p.rearing_partner_suppression_frac * body_len)
        if close_social_occlusion:
            # Do not let the posture smoother carry a rearing label into contact.
            # Contact occlusion is precisely the case where this cue is unreliable.
            self._smooth.pop(("rearing", a), None)
            self._smooth.pop(("rearing", b), None)
        put("rearing", a, rearing(a, ga, close_social_occlusion))
        put("rearing", b, rearing(b, gb, close_social_occlusion))

        kinematic_ready = min(len(self._hist[a]), len(self._hist[b])) >= max(2, p.kinematic_warmup_frames)
        if not kinematic_ready:
            for sid in self.identities:
                self._follow_inst[sid].append(False)
                self._chase_inst[sid].append(False)
            return raw

        # ---- kinematics-based ----
        velA, velB = kin[a]["vel"], kin[b]["vel"]
        spA, spB = kin[a]["speed"], kin[b]["speed"]
        acA = kin[a]["accel"]
        acB_ = kin[b]["accel"]
        prevA = self._hist[a][-2] if len(self._hist[a]) >= 2 else None
        prevB = self._hist[b][-2] if len(self._hist[b]) >= 2 else None
        prev_centre_dist = (dist(prevA["centre"], prevB["centre"])
                            if prevA and prevB and prevA["centre"] is not None and prevB["centre"] is not None
                            else centre_dist)
        closing = centre_dist < prev_centre_dist - 1e-6
        separating = centre_dist > prev_centre_dist + 1e-6

        def toward_component(vel, src_c, dst_c):
            u = _unit(dst_c - src_c) if (src_c is not None and dst_c is not None) else None
            return float(np.dot(vel, u)) if u is not None else 0.0

        a_toward_b = toward_component(velA, cA, cB)
        b_toward_a = toward_component(velB, cB, cA)

        # following: follower nose near leader's recent rear positions
        def following_instant(follower, leader, ax_f, nose_f):
            # leader rear history
            lead_hist = self._hist[leader]
            if nose_f is None or not lead_hist:
                return False
            win = list(lead_hist)[-p.follow_window:]
            dmin = float("inf")
            for rec in win:
                if rec["rear"] is not None:
                    dmin = min(dmin, float(np.hypot(*(nose_f - rec["rear"]))))
            if not (dmin < follow_tol):
                return False
            if not (kin[follower]["speed"] > stat and kin[leader]["speed"] > 0.25 * stat):
                return False
            rear_l = self._hist[leader][-1]["rear"]
            if ax_f is None or rear_l is None:
                return False
            if _angle_deg(ax_f, rear_l - nose_f) >= p.angle_tol_anogenital:
                return False
            # follower behind leader (in leader rear zone)
            cl = self._hist[leader][-1]["centre"]
            axl = self._hist[leader][-1]["axis"]
            return in_rear_zone(nose_f, cl, axl)

        fi_ab = following_instant(a, b, axA, noseA)   # A follows B
        fi_ba = following_instant(b, a, axB, noseB)
        self._follow_inst[a].append(fi_ab)
        self._follow_inst[b].append(fi_ba)

        def sustained(dq, n, frac=0.6):
            w = list(dq)[-n:]
            return len(w) >= max(1, int(math.ceil(frac * n))) and (sum(w) / len(w)) >= frac

        following_ab = sustained(self._follow_inst[a], p.min_follow_frames)
        following_ba = sustained(self._follow_inst[b], p.min_follow_frames)
        put("following", a, following_ab)
        put("following", b, following_ba)

        def chasing_instant(follower, leader, following_now, ax_f, vel_f, vel_l, nose_f):
            if not following_now:
                return False
            if rel_heading >= 65.0:
                return False
            if _angle_deg(vel_f, vel_l) > 55.0:
                return False
            cl = self._hist[leader][-1]["centre"]
            axl = self._hist[leader][-1]["axis"]
            if not in_rear_zone(nose_f, cl, axl):
                return False
            if centre_dist >= 3.0 * body_len:
                return False
            if separating:
                return False
            if not (kin[follower]["speed"] > 1.25 * stat and kin[leader]["speed"] > 0.70 * stat):
                return False
            return True

        ci_ab = chasing_instant(a, b, fi_ab, axA, velA, velB, noseA)
        ci_ba = chasing_instant(b, a, fi_ba, axB, velB, velA, noseB)
        self._chase_inst[a].append(ci_ab)
        self._chase_inst[b].append(ci_ba)
        chasing_ab = sustained(self._chase_inst[a], p.min_follow_frames)
        chasing_ba = sustained(self._chase_inst[b], p.min_follow_frames)
        put("chasing", a, chasing_ab)
        put("chasing", b, chasing_ba)

        # fighting (needs both fast + erratic + close, not aligned pursuit)
        def turn(prev_axis, axis):
            if prev_axis is None or axis is None:
                return 0.0
            return _angle_deg(prev_axis, axis)
        axisturnA = turn(prevA["axis"] if prevA else None, axA)
        axisturnB = turn(prevB["axis"] if prevB else None, axB)
        severe_turn = axisturnA > p.fighting_turn_deg or axisturnB > p.fighting_turn_deg
        severe_accel = acA > p.fighting_accel_mult * accel_thr or acB_ > p.fighting_accel_mult * accel_thr
        erratic = severe_turn or severe_accel
        close_fight = in_contact or centre_dist <= max(close_tol, 1.1 * body_len)
        contact_window = list(self._contact_hist)[-max(1, p.fighting_contact_frames):]
        min_contact = int(math.ceil(p.fighting_contact_frac * max(1, p.fighting_contact_frames)))
        sustained_contact = len(contact_window) >= max(1, p.fighting_contact_frames) and sum(contact_window) >= min_contact
        rel_speed = float(np.hypot(*(velA - velB)))
        reciprocal_scramble = rel_speed > max(fast_thr, p.fighting_rel_speed_frac * body_len)
        structured_investigation = bool(nose2nose or anog_ab or anog_ba or n2b_ab or n2b_ba)
        aligned_pursuit = (rel_heading < 65.0 and _angle_deg(velA, velB) < 55.0)
        fighting = bool(close_fight and sustained_contact and not structured_investigation
                        and spA > fast_thr and spB > fast_thr
                        and erratic and reciprocal_scramble and not aligned_pursuit)
        put("fighting", None, fighting)

        # partner pressure helper (partner advancing/chasing toward subject)
        def pressure(partner, subj_toward_from_partner, chasing_partner):
            return bool(chasing_partner or subj_toward_from_partner > move_thr)

        pressure_on_a = pressure(b, b_toward_a, chasing_ba)
        pressure_on_b = pressure(a, a_toward_b, chasing_ab)

        # approach
        def approach(subj, subj_toward, other_toward, ax_s, c_s, c_o, is_chase, near):
            if not (kin[subj]["speed"] > move_thr and closing):
                return False
            if ax_s is None or _angle_deg(ax_s, c_o - c_s) >= 70.0:
                return False
            if subj_toward <= other_toward:        # subject-driven
                return False
            if centre_dist >= near:
                return False
            return not (is_chase or fighting)

        approach_ab = approach(a, a_toward_b, b_toward_a, axA, cA, cB, chasing_ab, near_limit)
        approach_ba = approach(b, b_toward_a, a_toward_b, axB, cB, cA, chasing_ba, near_limit)
        put("approach", a, approach_ab)
        put("approach", b, approach_ba)

        # withdrawal_from_partner
        def away(vel, c_s, c_o):
            u = _unit(c_s - c_o) if (c_s is not None and c_o is not None) else None
            return float(np.dot(vel, u)) > 0 if u is not None else False

        def withdrawal(subj, vel, c_s, c_o, partner_pressure):
            if not (kin[subj]["speed"] > move_thr and separating and away(vel, c_s, c_o)):
                return False
            if centre_dist >= near_limit:
                return False
            return not (partner_pressure or fighting)

        wd_ab = withdrawal(a, velA, cA, cB, pressure_on_a)  # A withdraws from B
        wd_ba = withdrawal(b, velB, cB, cA, pressure_on_b)
        put("withdrawal_from_partner", a, wd_ab)
        put("withdrawal_from_partner", b, wd_ba)

        # escape (pressured retreat)
        def escape(subj, vel, accel, c_s, c_o, partner_pressure):
            if not (kin[subj]["speed"] > fast_thr and separating and accel > accel_thr):
                return False
            if not away(vel, c_s, c_o):
                return False
            if not partner_pressure:
                return False
            if not (centre_dist < near_limit or recent_contact):
                return False
            return not fighting

        esc_ab = escape(a, velA, acA, cA, cB, pressure_on_a)
        esc_ba = escape(b, velB, acB_, cB, cA, pressure_on_b)
        put("escape", a, esc_ab)
        put("escape", b, esc_ba)

        # withdrawal_after_contact
        def wd_after(subj, vel, c_s, c_o, axisturn):
            if not recent_prior_contact or any_contact:
                return False
            if not (kin[subj]["speed"] > fast_thr and separating and away(vel, c_s, c_o)):
                return False
            turned = axisturn > 35.0 or recent_contact
            return bool(turned)

        wac_a = wd_after(a, velA, cA, cB, axisturnA)
        wac_b = wd_after(b, velB, cB, cA, axisturnB)
        put("withdrawal_after_contact", a, wac_a)
        put("withdrawal_after_contact", b, wac_b)

        # passive: the receiving role. The subject is being investigated by its
        # partner (partner's nose leads at the subject's body/anogenital) while the
        # subject itself is barely moving. Mutual nose2nose is not passive.
        put("passive", a, bool((anog_ba or n2b_ba) and spA < move_thr))
        put("passive", b, bool((anog_ab or n2b_ab) and spB < move_thr))

        return raw
