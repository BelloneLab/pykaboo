"""Geometry-only mouse keypoint extraction from binary instance masks.

The extractor is designed for live use: no torch model, no Qt dependency, and
only NumPy plus OpenCV. It converts one mouse mask into the same eight-point
pose layout used by the live overlay:

    nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail_base

The important orientation cue is the tail filament. A mask opening removes the
thin tail from the thick body core, then the filament attachment marks the rear
of the animal. When the tail cannot be recovered, the stateful wrapper falls
back to the previous frame direction before using a low-confidence width cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


KP_ORDER: tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "body",
    "left_hip",
    "right_hip",
    "tail_base",
)

EAR_FRAC = 0.18
NECK_FRAC = 0.30
HIP_FRAC = 0.74


@dataclass
class MaskGeom:
    mask: np.ndarray
    body_core: np.ndarray
    mask_points: np.ndarray
    body_points: np.ndarray
    centroid: np.ndarray
    body_radius: float
    tail_base: Optional[np.ndarray]
    tail_tip: Optional[np.ndarray]
    tail_confidence: float


@dataclass
class Skeleton:
    keypoints: np.ndarray
    scores: np.ndarray
    orientation_confidence: float
    method: str = "pca"
    # Orientation internals (filled by keypoints_pca) so the stateful wrapper can
    # apply a temporal lock without recomputing the PCA. ``evidence_signed`` is the
    # per-frame shape vote: > 0 favours the nose at the high-projection end of the
    # body axis. ``nose_at_high`` is the orientation actually used to build the pose.
    axis: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    body_length: float = 0.0
    evidence_signed: float = 0.0
    # Signed end-shape (taper) vote alone: > 0 favours the nose at the high end. It is
    # the dominant component of ``evidence_signed`` and seeds the per-frame orientation.
    # A genuine head<->tail swap on a MOVING mouse still requires real reorientation to
    # flip; a STATIONARY mouse only flips when this shape vote disagrees strongly and
    # continuously for a long window (a real backwards seed, not a brief artifact), or
    # when the user manually corrects it. Kept for diagnostics either way.
    taper_signed: float = 0.0
    # Raw signed anatomical-tail vote BEFORE the taper-disagreement damping; > 0 favours
    # the nose at the high end. The temporal lock trusts this (the tail filament is the
    # reliable anatomical cue) rather than the taper to decide a STATIONARY flip.
    tail_signed: float = 0.0
    # PCA elongation / axis dominance in [0, 1]. A low value means a round, compact
    # silhouette whose head/tail taper is ambiguous (a small motionless test object).
    axis_strength: float = 0.0
    nose_at_high: Optional[bool] = None

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(point[0]), float(point[1]))
            for name, point in zip(KP_ORDER, self.keypoints)
        }


# Orientation cue weights and flip hysteresis. The temporal lock dominates so a
# single bad tail/cable filament cannot invert head<->tail between frames; a real
# turn (sustained disagreement, or fast directed motion) still wins quickly.
# END SHAPE is the primary anchor (silhouette method: pointy/low-area end = nose,
# round/blunt end = rump). The tail filament only CONFIRMS the rump; a pointy snout
# is routinely stripped off by the body opening and mis-detected as a filament, so a
# filament that contradicts the end shape is damped (it is most likely the snout).
_ORIENT_TAPER_W = 0.90     # end shape: pointy end = nose, blunt end = rump (primary)
_ORIENT_TAIL_W = 1.0       # anatomical tail filament (scaled by its confidence)
_ORIENT_WIDTH_W = 0.30     # wider body cross-section = rear (weak fallback)
_ORIENT_HINT_W = 0.50      # previous-frame direction, per-frame nudge (standalone calls)
_ORIENT_MOTION_W = 0.80    # nose leads locomotion, per-frame nudge (standalone calls)
_TAIL_VS_SHAPE_DAMP = 0.35 # multiply a filament vote that disagrees with the end shape
_FLIP_MARGIN = 0.55        # disagreeing shape evidence must exceed this to count
_FLIP_CONFIRM = 3          # consecutive disagreeing frames to overturn the lock
_MOTION_OVERRIDE_FRAC = 0.16   # |motion|/body_len above this: motion decides immediately
# A debounced shape/tail flip is only considered when the animal is actually
# reorientating: it has translated at least this fraction of a body length, OR its
# body-axis LINE has rotated at least this many degrees from the lock. A stationary
# mouse with a steady axis cannot have swapped head<->tail, so a "tail" that appears
# at the locked nose end (snout protrusion / occluded-tail dropout during contact) is
# rejected as an artifact instead of freezing the orientation backwards.
_FLIP_MIN_MOTION_FRAC = 0.06
_FLIP_MIN_AXIS_DEG = 25.0
# Slow STATIONARY recovery from a mis-seeded orientation. The reorienting guard
# above deliberately refuses to flip a still mouse, which is correct for the
# transient artifacts it protects against (a snout protrusion, or the real tail
# dropping out under partner occlusion) because those last well under a second.
# But it also means an orientation that was seeded BACKWARDS on the first frame --
# e.g. a genuinely motionless subject such as a frozen mouse or a stationary test
# object whose snout taper is ambiguous -- can never recover. A real mis-seed is
# distinguishable from an artifact on the TIME axis: it produces a steady, strong
# shape disagreement that never goes away, whereas an artifact is brief. So a very
# strong end-shape vote (dominated by the reliable silhouette taper) that disagrees
# with the lock for a long, continuous window is treated as a real mis-seed and is
# allowed to flip a stationary animal. The strong margin leans on the taper cue: a
# passive-mouse occlusion only perturbs the (damped) filament, leaving the taper --
# and therefore the net evidence -- agreeing with the lock, so it stays below it.
_FLIP_MARGIN_STRONG = 0.85       # |evidence| must exceed this for stationary recovery
_FLIP_CONFIRM_STATIONARY = 45    # continuous strong-disagree frames (~1.5-2 s) to recover
# Stationary recovery must NEVER be driven by the silhouette taper alone: taper is the
# one cue that is systematically wrong for the failure case (a compact, round-headed
# motionless test object whose pointy end is the rear, or an unusual frozen pose). So a
# stationary flip additionally requires the anatomical TAIL filament to corroborate that
# the lock is backwards, and the shape to be elongated enough that its axis is meaningful.
# A real frozen mouse with a visible tail still recovers (its tail vote is strong and
# agrees); a tail-less compact toy is left to the manual head/tail override.
_STATIONARY_NEEDS_TAIL = True    # require anatomical-tail corroboration, never taper alone
_STATIONARY_TAIL_MIN = 0.30      # min |tail_signed| (raw, pre-damp) to corroborate a flip
# axis_strength below this == too round/compact for the taper to be trusted while still.
# Measured on synthetic masks: compact toy ~0.35, real elongated mouse ~0.74, so 0.45 sits
# safely between the two regimes.
_COMPACT_AXIS_STRENGTH = 0.45
# Body-bulk asymmetry confirmer ("hips are wider/heavier than the head"). An empirical +
# adversarial study found this cue is SIGN-REDUNDANT with the taper: it agrees with the
# end-shape vote almost everywhere, so it can only confirm (never overturn) it, and on a
# compact/ambiguous shape it carries the SAME wrong sign as the taper. It is therefore
# added as a tiny, gated additive vote that hardens an already-correct seed on clearly
# elongated bodies and stays silent (returns 0) on compact shapes and below a noise floor.
# It deliberately does NOT, and cannot, fix a backwards seed on a round/compact subject.
_ORIENT_MASS_W = 0.18                       # << taper (0.90) / tail (1.0); cannot flip a seed
_MASS_ASYM_MIN = 0.05                       # |mass fraction| below this -> no vote (noise floor)
_MASS_AXIS_STRENGTH_MIN = _COMPACT_AXIS_STRENGTH  # disable on round/compact silhouettes


def extract_skeleton(
    mask: np.ndarray,
    *,
    method: str = "pca",
    direction_hint: Optional[np.ndarray] = None,
) -> Optional[Skeleton]:
    """Return an eight-point skeleton for one binary mask.

    ``direction_hint`` is a tail-to-nose vector from a previous frame. It is
    only trusted when the current frame has weak tail evidence.
    """
    geom = analyze_mask(mask)
    if geom is None:
        return None
    # The public method switch is kept for compatibility with the LISBET-style
    # API. The live path uses PCA because it is the most stable on noisy masks.
    normalized_method = str(method or "pca").strip().lower()
    if normalized_method not in {"pca", "contour", "geodesic"}:
        normalized_method = "pca"
    return keypoints_pca(geom, direction_hint=direction_hint, method=normalized_method)


def analyze_mask(mask: np.ndarray) -> Optional[MaskGeom]:
    """Separate a mouse mask into a thick body core and a tail filament."""
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    full_binary = arr > 0
    if not bool(full_binary.any()):
        return None

    # Crop to the mask's bounding box (plus a small margin) so every heavy
    # OpenCV pass below (connected components, distance transform, up to four
    # morphological opens) runs on a tight ROI instead of the whole frame. A
    # mouse occupies a fraction of the frame, so this is the dominant speed-up
    # for live geometry. All work stays in crop space; the point-bearing
    # outputs are offset back to full-frame coordinates before returning, so
    # downstream keypoints are identical to the full-frame computation.
    ys0, xs0 = np.nonzero(full_binary)
    pad = 8
    y0 = max(0, int(ys0.min()) - pad)
    y1 = min(arr.shape[0], int(ys0.max()) + 1 + pad)
    x0 = max(0, int(xs0.min()) - pad)
    x1 = min(arr.shape[1], int(xs0.max()) + 1 + pad)
    offset = np.array([float(x0), float(y0)], dtype=np.float64)

    binary = full_binary[y0:y1, x0:x1].astype(np.uint8)

    binary = _largest_component(binary)
    if binary is None or not bool(binary.any()):
        return None
    binary = _close_small_holes(binary)

    ys, xs = np.nonzero(binary)
    if len(xs) < 12:
        return None

    mask_points = np.column_stack([xs, ys]).astype(np.float64)
    centroid = np.mean(mask_points, axis=0)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    body_radius = float(np.max(dist)) if dist.size else 1.0
    body_radius = max(1.0, body_radius)

    body_core = _opened_body_core(binary, body_radius)
    if body_core is None or int(np.count_nonzero(body_core)) < 8:
        body_core = binary.copy()

    body_ys, body_xs = np.nonzero(body_core)
    body_points = np.column_stack([body_xs, body_ys]).astype(np.float64)
    if len(body_points) < 8:
        body_points = mask_points
        body_core = binary.copy()

    tail_base, tail_tip, tail_confidence = _find_tail_attachment(
        binary,
        body_core,
        centroid,
        body_radius,
    )

    # Lift every point-bearing result from crop space to full-frame space.
    # body_points may alias mask_points (small-mask fallback), so offset once.
    body_points_shared = body_points is mask_points
    mask_points = mask_points + offset
    body_points = mask_points if body_points_shared else (body_points + offset)
    centroid = centroid + offset
    if tail_base is not None:
        tail_base = np.asarray(tail_base, dtype=np.float64) + offset
    if tail_tip is not None:
        tail_tip = np.asarray(tail_tip, dtype=np.float64) + offset

    return MaskGeom(
        mask=binary.astype(bool),
        body_core=body_core.astype(bool),
        mask_points=mask_points,
        body_points=body_points,
        centroid=centroid.astype(np.float64),
        body_radius=float(body_radius),
        tail_base=tail_base,
        tail_tip=tail_tip,
        tail_confidence=float(tail_confidence),
    )


def keypoints_pca(
    geom: MaskGeom,
    *,
    direction_hint: Optional[np.ndarray] = None,
    motion_hint: Optional[np.ndarray] = None,
    orientation: Optional[bool] = None,
    method: str = "pca",
) -> Optional[Skeleton]:
    """Build pose keypoints from body-core PCA and mask cross sections.

    ``orientation`` forces the head/tail decision (True = nose at the high-projection
    end of the body axis); the stateful wrapper passes it to apply a temporal lock.
    When None, the orientation is decided per-frame from the tail filament, motion,
    the previous direction, and a width fallback.
    """
    points = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None

    center = np.mean(points, axis=0)
    centered = points - center
    try:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float64)
    except Exception:
        axis = np.array([1.0, 0.0], dtype=np.float64)
        eigvals = np.array([0.0, 1.0], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis /= axis_norm

    body_proj = (points - center) @ axis
    low_proj = float(np.min(body_proj))
    high_proj = float(np.max(body_proj))
    body_length = max(1.0, high_proj - low_proj)
    # Elongation of the body core (computed early so the bulk-mass cue can be gated on
    # it: on a round/compact silhouette the mass-asymmetry sign is unreliable).
    axis_strength = _axis_strength(eigvals)

    # Per-frame shape evidence: > 0 favours the nose at the high-projection end.
    # ``taper_signed`` is the end-shape component alone (kept so the temporal lock can
    # trust the reliable silhouette cue even when the tail filament is gated off).
    evidence_signed, taper_signed, tail_signed = _orientation_shape_evidence(
        geom, center, axis, low_proj, high_proj, body_length, axis_strength
    )
    # Fold the previous direction and motion in as nudges so single-frame calls
    # still pick (and report a confidence for) a sensible orientation even when the
    # tail filament is missing this frame.
    ev_total = evidence_signed
    hint_dot = _orientation_hint_dot(axis, direction_hint)
    if hint_dot is not None:
        ev_total += _ORIENT_HINT_W * hint_dot
    motion_dot = _orientation_motion_dot(axis, motion_hint, body_length)
    if motion_dot is not None:
        ev_total += _ORIENT_MOTION_W * motion_dot

    if orientation is not None:
        nose_at_high = bool(orientation)
        strength = max(abs(evidence_signed), 0.5)   # a temporally-confirmed decision
    else:
        nose_at_high = ev_total >= 0.0
        strength = abs(ev_total)

    if nose_at_high:
        nose_proj, rear_proj = high_proj, low_proj
    else:
        nose_proj, rear_proj = low_proj, high_proj
    # Confidence reflects how decisively the combined cues picked an end.
    orientation_confidence = float(np.clip(0.30 + 0.55 * min(1.0, strength), 0.0, 1.0))

    nose_to_rear_sign = 1.0 if rear_proj >= nose_proj else -1.0
    tail_to_nose_vec = axis * (-nose_to_rear_sign)
    tail_to_nose_vec = _unit(tail_to_nose_vec, fallback=np.array([1.0, 0.0], dtype=np.float64))
    lateral = np.array([-tail_to_nose_vec[1], tail_to_nose_vec[0]], dtype=np.float64)

    nose = _end_point(geom.mask_points, center, axis, nose_proj, prefer_high=nose_proj > rear_proj)
    # Use the detected tail filament ONLY when it agrees with the chosen rear end.
    # After a temporal/motion-driven flip, a spurious filament (e.g. the tether
    # cable, or a tail curled forward) would otherwise drop the tail keypoint onto
    # the head; fall back to the geometric rear end in that case.
    use_detected_tail = False
    if geom.tail_base is not None:
        tail_proj = float((np.asarray(geom.tail_base, dtype=np.float64) - center) @ axis)
        tail_at_high = abs(tail_proj - high_proj) < abs(tail_proj - low_proj)
        use_detected_tail = (tail_at_high == (not nose_at_high))
    if use_detected_tail:
        tail_base = np.asarray(geom.tail_base, dtype=np.float64)
        tail_score = 0.70 + 0.25 * float(np.clip(geom.tail_confidence, 0.0, 1.0))
    else:
        tail_base = _end_point(geom.mask_points, center, axis, rear_proj, prefer_high=rear_proj > nose_proj)
        tail_score = 0.45

    neck_center = _section_center(geom.mask_points, center, axis, nose_proj, rear_proj, NECK_FRAC, body_length)
    ear_center = _section_center(geom.mask_points, center, axis, nose_proj, rear_proj, EAR_FRAC, body_length)
    # Hips are flank landmarks on the thick body core, not on the tail filament.
    # Using the full mask here lets a curved tail win the posterior cross-section
    # edge and produces anatomically impossible hip-on-tail overlays.
    hip_center = _section_center(geom.body_points, center, axis, nose_proj, rear_proj, HIP_FRAC, body_length)

    left_ear, right_ear, ear_width = _section_edges(
        geom.mask_points,
        center,
        axis,
        lateral,
        _fraction_to_projection(nose_proj, rear_proj, EAR_FRAC),
        body_length,
        fallback_center=ear_center,
    )
    left_hip, right_hip, hip_width = _section_edges(
        geom.body_points,
        center,
        axis,
        lateral,
        _fraction_to_projection(nose_proj, rear_proj, HIP_FRAC),
        body_length,
        fallback_center=hip_center,
        robust=True,
    )

    keypoints = np.vstack(
        [
            nose,
            left_ear,
            right_ear,
            neck_center,
            geom.centroid,
            left_hip,
            right_hip,
            tail_base,
        ]
    ).astype(np.float64)

    edge_score = float(np.clip(min(ear_width, hip_width) / max(1.0, geom.body_radius * 1.5), 0.35, 1.0))
    base_score = float(np.clip(0.45 + 0.35 * axis_strength, 0.45, 0.90))
    scores = np.array(
        [
            float(np.clip(0.50 + 0.45 * orientation_confidence, 0.0, 1.0)),
            edge_score,
            edge_score,
            base_score,
            1.0,
            edge_score,
            edge_score,
            float(np.clip(tail_score, 0.0, 1.0)),
        ],
        dtype=np.float64,
    )

    return Skeleton(
        keypoints=keypoints,
        scores=scores,
        orientation_confidence=float(np.clip(orientation_confidence, 0.0, 1.0)),
        method=method,
        axis=np.asarray(axis, dtype=np.float64).copy(),
        center=np.asarray(center, dtype=np.float64).copy(),
        body_length=float(body_length),
        evidence_signed=float(evidence_signed),
        taper_signed=float(taper_signed),
        tail_signed=float(tail_signed),
        axis_strength=float(axis_strength),
        nose_at_high=bool(nose_at_high),
    )


def repair_hip_keypoints_with_mask_geometry(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Move impossible hip keypoints off the tail and back onto body flanks.

    Pose networks can put one or both hips on the tail because the tail is still
    a valid foreground mask pixel. A hip is only accepted when it lies close to
    the thick body core and anterior to the tail base. Suspect hips are replaced
    with geometry-derived flank points from the same mask.
    """
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2).copy()
    sc = None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    if len(kp) < len(KP_ORDER):
        return kp, sc

    geom = analyze_mask(mask)
    if geom is None:
        return kp, sc
    geometry = keypoints_pca(geom, method="pca")
    if geometry is None or len(geometry.keypoints) < len(KP_ORDER):
        return kp, sc

    body_points = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
    if len(body_points) == 0:
        return kp, sc

    tail_base = np.asarray(geometry.keypoints[7], dtype=np.float64)
    nose = np.asarray(geometry.keypoints[0], dtype=np.float64)
    tail_to_nose = _unit(nose - tail_base, fallback=np.array([1.0, 0.0], dtype=np.float64))
    body_length = max(1.0, float(np.linalg.norm(nose - tail_base)))
    max_body_distance = max(3.0, float(geom.body_radius) * 0.45)
    min_anterior = max(2.0, body_length * 0.045)

    for index in (5, 6):
        point = kp[index]
        if _hip_point_is_suspect(
            point,
            body_points,
            tail_base,
            tail_to_nose,
            max_body_distance=max_body_distance,
            min_anterior=min_anterior,
        ):
            kp[index] = geometry.keypoints[index]
            if sc is not None and index < len(sc):
                replacement_score = float(geometry.scores[index]) if index < len(geometry.scores) else 0.55
                current = float(sc[index]) if np.isfinite(sc[index]) else replacement_score
                sc[index] = min(current, replacement_score, 0.70)
    return kp, sc


class MaskSkeletonExtractor:
    """Stateful live wrapper for mask-to-skeleton extraction."""

    def __init__(
        self,
        *,
        method: str = "pca",
        smooth: bool = True,
        alpha: float = 0.65,
        max_jump: float = 90.0,
        correct_body_core: bool = True,
        bodycore_gate: float = 0.15,
    ) -> None:
        self.method = str(method or "pca")
        self.smooth = bool(smooth)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_jump = float(max(1.0, max_jump))
        # Pull the nose/ears off the tail onto the tail-free body core (see
        # _correct_to_body_core). bodycore_gate is the disagreement threshold, in
        # body-lengths, beyond which the stock keypoint is replaced.
        self.correct_body_core = bool(correct_body_core)
        self.bodycore_gate = float(max(0.0, bodycore_gate))
        self._states: dict[object, dict[str, np.ndarray]] = {}
        self._pending: dict[object, int] = {}  # consecutive disagreeing-orientation frames
        self._pending_stationary: dict[object, int] = {}  # consecutive STRONG-disagree frames
        # Tracks whose orientation the user has manually asserted. While set, the
        # automatic stationary-recovery path is disabled for that track (only real
        # directed motion may then override the user's choice).
        self._manual: dict[object, bool] = {}
        self._manual_flip_pending: set[object] = set()  # flips requested before a track exists

    def reset(self) -> None:
        self._states.clear()
        self._pending.clear()
        self._pending_stationary.clear()
        self._manual.clear()
        self._manual_flip_pending.clear()

    def flip_orientation(self, track_id: object) -> bool:
        """Manually swap a track's head<->tail. Thread-unsafe; call from the worker.

        Inverts the locked tail->nose direction and drops the smoothing/left-right
        reference so the next frame is rebuilt cleanly in the new orientation. The
        track is marked manual so the automatic stationary-recovery path will not
        undo the user's correction; physical directed motion can still override it.
        Returns True when applied immediately, False when queued (track not seen yet).
        """
        track_id = int(track_id) if isinstance(track_id, (int, np.integer)) else track_id
        state = self._states.get(track_id)
        self._manual[track_id] = True
        self._pending[track_id] = 0
        self._pending_stationary[track_id] = 0
        if state is None or state.get("direction") is None:
            self._manual_flip_pending.add(track_id)
            return False
        direction = np.asarray(state["direction"], dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(direction)) or float(np.hypot(*direction)) < 1e-6:
            self._manual_flip_pending.add(track_id)
            return False
        state["direction"] = (-direction).copy()
        state["keypoints"] = None  # rebuild fresh next frame; do not smooth across the flip
        self._manual_flip_pending.discard(track_id)
        return True

    def estimate(
        self,
        mask: np.ndarray,
        *,
        track_id: object = "default",
        offset: tuple[float, float] | np.ndarray = (0.0, 0.0),
    ) -> Optional[Skeleton]:
        previous = self._states.get(track_id)
        if track_id in self._manual_flip_pending and previous is not None:
            # A flip requested before this track had a locked direction; apply it now.
            self.flip_orientation(track_id)
            previous = self._states.get(track_id)
        direction_hint = previous.get("direction") if previous is not None else None
        offset_arr = np.asarray(offset, dtype=np.float64).reshape(2)

        geom = analyze_mask(mask)
        if geom is None:
            return None
        # Motion is measured in full-frame coordinates (the per-mouse crop offset
        # shifts every frame), so the nose-leads-motion cue stays meaningful.
        centroid_full = np.asarray(geom.centroid, dtype=np.float64).reshape(2) + offset_arr
        motion_hint = None
        if previous is not None and previous.get("centroid") is not None:
            motion_hint = centroid_full - np.asarray(previous["centroid"], dtype=np.float64).reshape(2)

        skeleton = keypoints_pca(
            geom, direction_hint=direction_hint, motion_hint=motion_hint, method=self.method
        )
        if skeleton is None:
            return None

        # Temporal orientation lock: resist a single-frame head<->tail inversion;
        # only flip on sustained shape disagreement or clear directed motion.
        desired_high = self._resolve_orientation(track_id, skeleton, motion_hint)
        if skeleton.nose_at_high is not None and bool(desired_high) != bool(skeleton.nose_at_high):
            rebuilt = keypoints_pca(
                geom, motion_hint=motion_hint, orientation=bool(desired_high), method=self.method
            )
            if rebuilt is not None:
                skeleton = rebuilt

        keypoints = np.asarray(skeleton.keypoints, dtype=np.float64).reshape(-1, 2)
        if np.any(offset_arr):
            keypoints = keypoints + offset_arr.reshape(1, 2)
        if previous is not None:
            keypoints = self._lock_left_right(previous.get("keypoints"), keypoints)
            if self.smooth:
                keypoints = _smooth_keypoints(previous.get("keypoints"), keypoints, self.alpha, self.max_jump)
        if self.correct_body_core:
            keypoints = self._correct_to_body_core(keypoints, geom, offset_arr, previous)
        skeleton = Skeleton(
            keypoints=keypoints,
            scores=np.asarray(skeleton.scores, dtype=np.float64).reshape(-1),
            orientation_confidence=skeleton.orientation_confidence,
            method=skeleton.method,
        )

        # Stabilise the locked direction with a light EMA so axis wobble does not
        # drift the temporal reference; reset hard on a genuine flip.
        cur_dir = _tail_to_nose_direction(keypoints)
        cur_unit = _unit(cur_dir, fallback=np.array([0.0, 0.0], dtype=np.float64))
        prev_dir = previous.get("direction") if previous is not None else None
        if (
            prev_dir is not None
            and float(np.dot(cur_unit, _unit(prev_dir, fallback=cur_unit))) > 0.0
            and float(np.hypot(*cur_unit)) > 1e-6
        ):
            locked = _unit(0.6 * cur_unit + 0.4 * np.asarray(prev_dir, dtype=np.float64).reshape(2),
                           fallback=cur_unit)
        else:
            locked = cur_unit if float(np.hypot(*cur_unit)) > 1e-6 else (prev_dir if prev_dir is not None else cur_dir)
        self._states[track_id] = {
            "keypoints": np.asarray(keypoints, dtype=np.float64).reshape(-1, 2).copy(),
            "direction": np.asarray(locked, dtype=np.float64).reshape(2).copy(),
            "centroid": centroid_full.copy(),
        }
        return skeleton

    def _correct_to_body_core(
        self,
        keypoints: np.ndarray,
        geom: "MaskGeom",
        offset_arr: np.ndarray,
        previous: Optional[dict],
    ) -> np.ndarray:
        """Pull the nose and ears off the tail onto the tail-free body core.

        The PCA pose places the nose at the *whole-mask* axis extreme and the ears at
        *whole-mask* cross-section edges. When the tail curls forward over the body
        (grooming, tight turns) the tail wins those, snapping the nose onto the tail tip
        and an ear onto the tail. The body core (tail filament removed) is the reliable
        anatomical anchor -- this is exactly why the hips already use ``body_points``.

        The nose keeps pykaboo's own (temporally-locked) end and is only pulled back onto the
        body-core extreme when it has left the body core by more than ``self.bodycore_gate``
        body-lengths; the ears use the body-core flank edges, replaced only when the stock ear
        disagrees with that edge by the same margin. Well-placed keypoints are left untouched.
        """
        bp = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
        if len(bp) < 8 or geom.tail_base is None:
            return keypoints
        bp = bp + offset_arr.reshape(1, 2)        # body core -> same frame as keypoints
        kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2).copy()
        center = bp.mean(axis=0)
        try:
            evals, evecs = np.linalg.eigh(np.cov((bp - center).T))
            axis = evecs[:, int(np.argmax(evals))]
        except Exception:
            return keypoints
        axis = axis / (float(np.linalg.norm(axis)) or 1.0)
        proj = (bp - center) @ axis
        e_hi, e_lo = bp[int(np.argmax(proj))], bp[int(np.argmin(proj))]
        bl = float(np.linalg.norm(kp[0] - kp[7])) or 1.0
        gate = self.bodycore_gate
        prev_kp = None if previous is None else previous.get("keypoints")

        # Nose: keep pykaboo's OWN end decision (its multi-cue orientation is already temporally
        # locked). Only when the stock nose has LEFT the body core -- it jumped onto the curled tail
        # filament or a stray spur -- pull it back to the body-core on-axis extreme on the SAME end.
        # (Re-deriving the head end from tail_base is unreliable: a snout mis-read as a tail flips it,
        # which collapsed the pose on shapes where the detected filament sits on the snout side.)
        if float(np.min(np.linalg.norm(bp - kp[0], axis=1))) / bl > gate:
            kp[0] = e_hi if float((kp[0] - center) @ axis) >= 0.0 else e_lo

        # Ears: flank edges on the body core (same rationale as the hips).
        nose_at_high = float((kp[0] - center) @ axis) > 0
        hi, lo = float(proj.max()), float(proj.min())
        nose_proj, rear_proj = (hi, lo) if nose_at_high else (lo, hi)
        body_length = max(1.0, hi - lo)
        lateral = np.array([-axis[1], axis[0]], dtype=np.float64)
        target = _fraction_to_projection(nose_proj, rear_proj, EAR_FRAC)
        ear_l, ear_r, _ = _section_edges(bp, center, axis, lateral, target, body_length, fallback_center=center)
        # Match the two body-core edges to the stock L/R ears (min-cost assignment).
        if (np.linalg.norm(ear_l - kp[1]) + np.linalg.norm(ear_r - kp[2]) >
                np.linalg.norm(ear_r - kp[1]) + np.linalg.norm(ear_l - kp[2])):
            ear_l, ear_r = ear_r, ear_l
        if np.linalg.norm(kp[1] - ear_l) / bl > gate:
            kp[1] = ear_l
        if np.linalg.norm(kp[2] - ear_r) / bl > gate:
            kp[2] = ear_r
        # Temporal left/right lock so the corrected ears never swap labels.
        if prev_kp is not None and (
            np.linalg.norm(kp[1] - prev_kp[2]) + np.linalg.norm(kp[2] - prev_kp[1]) <
            np.linalg.norm(kp[1] - prev_kp[1]) + np.linalg.norm(kp[2] - prev_kp[2])
        ):
            kp[[1, 2]] = kp[[2, 1]]
        return kp

    def _resolve_orientation(
        self, track_id: object, skeleton: Skeleton, motion_hint: Optional[np.ndarray]
    ) -> bool:
        """Decide nose-at-high under a temporal lock + motion override + flip debounce."""
        state = self._states.get(track_id)
        if state is None or state.get("direction") is None or skeleton.axis is None:
            return bool(skeleton.nose_at_high)
        lock = np.asarray(state["direction"], dtype=np.float64).reshape(2)
        axis = np.asarray(skeleton.axis, dtype=np.float64).reshape(2)
        if not (np.all(np.isfinite(lock)) and np.all(np.isfinite(axis))) or float(np.hypot(*lock)) < 1e-6:
            return bool(skeleton.nose_at_high)
        prior_high = float(np.dot(lock, axis)) > 0.0
        body_length = max(1.0, float(skeleton.body_length))

        # Motion override: fast directed translation pins the nose to the leading end.
        if motion_hint is not None:
            mh = np.asarray(motion_hint, dtype=np.float64).reshape(-1)[:2]
            if np.all(np.isfinite(mh)):
                motion_along = float(np.dot(mh, axis))  # + = toward the high end
                if abs(motion_along) > _MOTION_OVERRIDE_FRAC * body_length:
                    self._pending[track_id] = 0
                    self._pending_stationary[track_id] = 0
                    return motion_along > 0.0

        # Debounced shape/tail flip against the locked orientation -- but only when the
        # animal shows evidence of actually REORIENTING: directed translation, or a
        # rotation of the body-axis line. A near-stationary mouse whose axis has not
        # turned cannot physically have swapped head<->tail; a "tail" appearing at the
        # locked nose end is then almost certainly a segmentation artifact (a snout
        # protrusion, or the real tail dropping out under partner occlusion). This is
        # the passive-investigated-mouse failure: no motion to override and the real
        # tail hidden between the bodies, so a spurious filament would otherwise flip
        # the orientation and freeze it backwards for the whole contact bout.
        evidence = float(skeleton.evidence_signed or 0.0)
        translating = False
        if motion_hint is not None:
            mh = np.asarray(motion_hint, dtype=np.float64).reshape(-1)[:2]
            if np.all(np.isfinite(mh)):
                translating = float(np.hypot(*mh)) > _FLIP_MIN_MOTION_FRAC * body_length
        lock_unit = _unit(lock, fallback=axis)
        axis_line_deg = float(np.degrees(np.arccos(
            float(np.clip(abs(float(np.dot(lock_unit, axis))), 0.0, 1.0)))))
        reorienting = translating or (axis_line_deg > _FLIP_MIN_AXIS_DEG)

        # Single debounced flip path. Now that candidate CL seeds orientation correctly
        # on the FIRST frame (full-silhouette pointiness + a strict long-thin tail test),
        # we no longer let a stationary "strong_shape" disagreement flip a still mouse.
        # A genuine head<->tail swap REQUIRES the animal to reorient: it must translate
        # by a fraction of a body length OR its body-axis LINE must rotate. A stationary
        # mouse with a steady axis cannot have physically swapped ends, so a "tail" or
        # filament appearing at the locked nose end (a tether cable, a snout protrusion,
        # or the real tail dropping out under partner occlusion) is rejected as an
        # artifact instead of inverting a correctly-seeded orientation.
        disagrees = (
            reorienting
            and ((evidence > 0.0) != prior_high)
            and abs(evidence) > _FLIP_MARGIN
        )
        count = self._pending.get(track_id, 0) + 1 if disagrees else 0
        self._pending[track_id] = count
        if count >= _FLIP_CONFIRM:
            self._pending[track_id] = 0
            self._pending_stationary[track_id] = 0
            return evidence > 0.0

        # Slow stationary recovery from a backwards seed (see _FLIP_CONFIRM_STATIONARY).
        # Skipped once the user has manually asserted this track's orientation. Crucially
        # this path must NOT be driven by the taper alone: taper is the very cue that is
        # systematically wrong for the failure case (a compact, round-headed motionless
        # object whose pointy end is the rear). So a stationary flip additionally requires
        # the anatomical TAIL filament to corroborate that the lock is backwards, and the
        # shape to be elongated enough for its axis to mean anything. A real frozen mouse
        # with a visible tail still recovers; a tail-less compact toy is deferred to the
        # manual head/tail override instead of being flipped to a confidently-wrong taper.
        if not self._manual.get(track_id, False):
            tail_signed = float(getattr(skeleton, "tail_signed", 0.0))
            axis_strength = float(getattr(skeleton, "axis_strength", 0.0))
            compact = axis_strength < _COMPACT_AXIS_STRENGTH
            tail_corroborates = (
                abs(tail_signed) >= _STATIONARY_TAIL_MIN
                and ((tail_signed > 0.0) != prior_high)
            )
            strong_disagree = (
                (not compact)
                and ((evidence > 0.0) != prior_high)
                and abs(evidence) > _FLIP_MARGIN_STRONG
                and (tail_corroborates or not _STATIONARY_NEEDS_TAIL)
            )
            s_count = self._pending_stationary.get(track_id, 0) + 1 if strong_disagree else 0
            self._pending_stationary[track_id] = s_count
            if s_count >= _FLIP_CONFIRM_STATIONARY:
                self._pending_stationary[track_id] = 0
                self._pending[track_id] = 0
                return evidence > 0.0
        else:
            self._pending_stationary[track_id] = 0
        return prior_high

    @staticmethod
    def _lock_left_right(previous: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current
        prev = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
        cur = np.asarray(current, dtype=np.float64).reshape(-1, 2).copy()
        if prev.shape != cur.shape or len(cur) < len(KP_ORDER):
            return cur
        for left_index, right_index in ((1, 2), (5, 6)):
            same = _finite_distance(cur[left_index], prev[left_index]) + _finite_distance(cur[right_index], prev[right_index])
            swapped = _finite_distance(cur[left_index], prev[right_index]) + _finite_distance(cur[right_index], prev[left_index])
            if swapped + 1e-6 < same:
                cur[[left_index, right_index]] = cur[[right_index, left_index]]
        return cur


def _largest_component(binary: np.ndarray) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return (labels == largest).astype(np.uint8)


def _close_small_holes(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def _opened_body_core(binary: np.ndarray, body_radius: float) -> Optional[np.ndarray]:
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return None
    radius_candidates = [
        int(round(body_radius * 0.55)),
        int(round(body_radius * 0.42)),
        int(round(body_radius * 0.32)),
        2,
    ]
    seen: set[int] = set()
    for radius in radius_candidates:
        radius = int(max(1, min(25, radius)))
        if radius in seen:
            continue
        seen.add(radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        opened = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        component = _largest_component(opened)
        if component is None:
            continue
        component_area = int(np.count_nonzero(component))
        if component_area >= max(8, int(area * 0.22)):
            return component.astype(np.uint8)
    return None


def _find_tail_attachment(
    binary: np.ndarray,
    body_core: np.ndarray,
    centroid: np.ndarray,
    body_radius: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    filament = np.logical_and(binary.astype(bool), ~body_core.astype(bool)).astype(np.uint8)
    if int(np.count_nonzero(filament)) < 3:
        return None, None, 0.0

    core_dilate_radius = max(1, int(round(min(5.0, max(1.0, body_radius * 0.20)))))
    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (core_dilate_radius * 2 + 1, core_dilate_radius * 2 + 1),
    )
    core_neighborhood = cv2.dilate(body_core.astype(np.uint8), core_kernel).astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filament, connectivity=8)
    if num_labels <= 1:
        return None, None, 0.0

    best_score = -float("inf")
    best_base = None
    best_tip = None
    best_length = 0.0
    min_area = max(3, int(0.002 * np.count_nonzero(binary)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        touching = np.logical_and(component, core_neighborhood)
        ys, xs = np.nonzero(component)
        if len(xs) == 0:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        if bool(touching.any()):
            tys, txs = np.nonzero(touching)
            base = np.array([float(np.mean(txs)), float(np.mean(tys))], dtype=np.float64)
            touch_bonus = 1.0
        else:
            distances = np.linalg.norm(pts - centroid.reshape(1, 2), axis=1)
            base = pts[int(np.argmin(distances))]
            touch_bonus = 0.35
        tip_distances = np.linalg.norm(pts - base.reshape(1, 2), axis=1)
        tip_index = int(np.argmax(tip_distances))
        tip = pts[tip_index]
        length = float(tip_distances[tip_index])
        score = touch_bonus * length + 0.03 * float(area)
        if score > best_score:
            best_score = score
            best_base = base
            best_tip = tip
            best_length = length

    if best_base is None or best_tip is None:
        return None, None, 0.0
    confidence = float(np.clip(best_length / max(1.0, body_radius * 1.8), 0.0, 1.0))
    return best_base, best_tip, confidence


def _orientation_hint_dot(axis: np.ndarray, direction_hint: Optional[np.ndarray]) -> Optional[float]:
    if direction_hint is None:
        return None
    hint = np.asarray(direction_hint, dtype=np.float64).reshape(-1)
    if hint.size < 2 or not np.all(np.isfinite(hint[:2])):
        return None
    hint_norm = float(np.linalg.norm(hint[:2]))
    if hint_norm <= 1e-6:
        return None
    return float(np.dot(axis, hint[:2] / hint_norm))


def _orientation_motion_dot(
    axis: np.ndarray, motion_hint: Optional[np.ndarray], body_length: float
) -> Optional[float]:
    """Signed motion vote: dot(axis, unit(motion)) weighted by speed/body_length.

    The nose leads forward locomotion, so when the centroid translates the body
    axis should point with the motion. Weak when slow, full weight near ``0.12``
    body lengths per frame. Returns None when there is no usable motion.
    """
    if motion_hint is None:
        return None
    mh = np.asarray(motion_hint, dtype=np.float64).reshape(-1)
    if mh.size < 2 or not np.all(np.isfinite(mh[:2])):
        return None
    speed = float(np.linalg.norm(mh[:2]))
    if speed <= 1e-6:
        return None
    speed_frac = float(np.clip(speed / max(1.0, 0.12 * float(body_length)), 0.0, 1.0))
    return float(np.dot(axis, mh[:2] / speed)) * speed_frac


def _end_cap_width(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    tip_proj: float,
    inward_sign: float,
    body_length: float,
) -> float:
    """Mean cross-section width of the silhouette over a cap reaching inward from one
    axis end. A pointy (nose) end stays narrow across the cap; a blunt (rump) end is
    wide right away. So a SMALLER value means a pointier end. The cap is binned so a
    long tapering snout is distinguished from a snub rump even when both tips are
    near-zero width at the very extreme pixel."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return 0.0
    lateral = np.array([-axis[1], axis[0]], dtype=np.float64)
    along = (pts - center.reshape(1, 2)) @ axis
    lat = (pts - center.reshape(1, 2)) @ lateral
    rel = (along - float(tip_proj)) * float(inward_sign)   # 0 at the tip, grows inward
    cap = max(2.0, 0.22 * float(body_length))
    widths = []
    nbins = 5
    for k in range(nbins):
        lo = cap * k / nbins
        hi = cap * (k + 1) / nbins
        m = (rel >= lo) & (rel < hi)
        if int(np.count_nonzero(m)) >= 2:
            widths.append(float(np.max(lat[m]) - np.min(lat[m])))
    if not widths:
        return 0.0
    return float(np.mean(widths))


def _overhang_stats(
    geom: MaskGeom,
    center: np.ndarray,
    axis: np.ndarray,
    core_extreme_proj: float,
    outward_sign: float,
) -> tuple[float, float, int]:
    """Measure the silhouette OVERHANG that sticks out past the opened body-core
    extreme on one axis end.

    ``core_extreme_proj`` is the body-core's projection extreme on this side and
    ``outward_sign`` (+1 toward high, -1 toward low) points away from the core. We
    select full-mask pixels whose axial projection lies beyond the core extreme on
    this side and report ``(axial_length L, mean_width w, pixel_count)``.

    A real TAIL is a long, thin overhang (large L, tiny w): the opening strips the
    thin tail off the thick core, so the tail lives entirely in the overhang. A
    pointy SNOUT is also stripped by the opening, but it is short relative to the
    body radius (and a tapering snout sits on the core's own narrowing end), which
    is how we tell the two apart downstream.
    """
    pts = np.asarray(geom.mask_points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0, 0.0, 0
    lateral = np.array([-axis[1], axis[0]], dtype=np.float64)
    along = (pts - center.reshape(1, 2)) @ axis
    rel = (along - float(core_extreme_proj)) * float(outward_sign)  # > 0 == beyond core
    sel = rel > 0.0
    count = int(np.count_nonzero(sel))
    if count < 2:
        return 0.0, 0.0, count
    L = float(np.max(rel[sel]))                 # axial reach of the overhang
    w = float(count) / max(1.0, L)              # mean cross-section width (pixels / length)
    return L, w, count


def _orientation_taper_vote(
    geom: MaskGeom,
    center: np.ndarray,
    axis: np.ndarray,
    body_length: float,
) -> float:
    """End-shape vote: > 0 favours the nose (the pointier end) at the high end.

    This is the silhouette method's primary anchor: the nose is the pointy, low-area
    end; the rump is the round, blunt end. It needs no tail filament, so it stays
    valid when the real tail is occluded between two animals in close contact.

    WHY this is measured on the FULL silhouette, not the opened core: the body
    opening strips a pointy snout off the core just like it strips a tail. If we
    cap-measure the OPENED core extremes, the snout is gone, the now-blunt nose end
    looks wide, and the sign INVERTS (nose mis-placed on the rear). So pointiness is
    read from ``geom.mask_points`` (the whole animal), which keeps the snout.
    """
    pts = np.asarray(geom.mask_points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return 0.0
    proj = (pts - center.reshape(1, 2)) @ axis
    lo = float(np.min(proj))
    hi = float(np.max(proj))
    span = max(1.0, hi - lo)

    # (1) Pointiness from the FULL silhouette: the narrower end cap is the nose.
    low_w = _end_cap_width(geom.mask_points, center, axis, lo, +1.0, span)
    high_w = _end_cap_width(geom.mask_points, center, axis, hi, -1.0, span)
    pointy_high = high_w < low_w
    mag = float(np.clip(abs(low_w - high_w) / max(1.0, low_w, high_w), 0.0, 1.0))

    # Body-core projection extremes (the opening removes both the tail and a snout).
    body = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
    if len(body) >= 4:
        core_proj = (body - center.reshape(1, 2)) @ axis
        core_lo_proj = float(np.min(core_proj))
        core_hi_proj = float(np.max(core_proj))
    else:
        core_lo_proj, core_hi_proj = lo, hi

    # (2) Strict tail per end: the overhang beyond the core extreme must be both LONG
    #     (vs the body radius) and THIN (high aspect L/w). A snout overhang is short
    #     or stubby, so it fails the aspect/length test.
    body_radius = max(1.0, float(geom.body_radius))
    lo_L, lo_w, _ = _overhang_stats(geom, center, axis, core_lo_proj, -1.0)
    hi_L, hi_w, _ = _overhang_stats(geom, center, axis, core_hi_proj, +1.0)
    lo_strict = (lo_L > 1.8 * body_radius) and (lo_L / max(1.0, lo_w) > 6.0)
    hi_strict = (hi_L > 1.8 * body_radius) and (hi_L / max(1.0, hi_w) > 6.0)

    # (3) Solid-core snout-wisp guard: if the CORE itself tapers toward an end, an
    #     overhang there is a stripped snout, not a tail. Cap-measure the core at its
    #     OWN extremes and compare. core_asym > 0 == core is pointier toward HIGH.
    core_span = max(1.0, core_hi_proj - core_lo_proj)
    core_lo_w = _end_cap_width(geom.body_points, center, axis, core_lo_proj, +1.0, core_span)
    core_hi_w = _end_cap_width(geom.body_points, center, axis, core_hi_proj, -1.0, core_span)
    core_asym = (core_lo_w - core_hi_w) / max(1.0, core_lo_w, core_hi_w)
    if core_asym > 0.35:
        hi_strict = False   # core tapers toward high => high overhang is a snout wisp
    if core_asym < -0.35:
        lo_strict = False   # core tapers toward low => low overhang is a snout wisp

    # (4) Decide. A single surviving strict tail pins the rear directly (a real tail
    #     is the strongest cue); otherwise fall back to full-silhouette pointiness.
    if lo_strict != hi_strict:
        nose_high = lo_strict        # tail at the low end => nose at the high end
    else:
        nose_high = pointy_high
    if mag <= 1e-3:
        return 0.0
    sign = 1.0 if nose_high else -1.0
    return float(sign * _ORIENT_TAPER_W * mag)


def _orientation_mass_vote(
    geom: MaskGeom,
    center: np.ndarray,
    axis: np.ndarray,
    low_proj: float,
    high_proj: float,
    axis_strength: float,
) -> float:
    """Body-bulk asymmetry vote: > 0 favours the NOSE at the high end.

    The rear half of a mouse carries more core mass (wider hips), so the LIGHTER half is
    the head and the nose sits at the lighter end. We split the opened body core at its
    median projection and compare pixel counts. Returns 0 (no vote) on compact shapes or
    when the asymmetry is below the noise floor -- exactly the regime where this cue's
    sign is unreliable. It is a weak confirmer of the taper, never an override (see
    _ORIENT_MASS_W).
    """
    if axis_strength < _MASS_AXIS_STRENGTH_MIN:
        return 0.0
    body = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
    if len(body) < 8:
        return 0.0
    proj = (body - center.reshape(1, 2)) @ axis
    # Split at the geometric MIDPOINT of the core extent (not the median of the points,
    # which would trivially balance the counts), then the heavier side is the rear.
    split = 0.5 * (float(low_proj) + float(high_proj))
    n_high = int(np.count_nonzero(proj > split))
    n_low = int(np.count_nonzero(proj < split))
    total = n_high + n_low
    if total <= 0:
        return 0.0
    # mass_frac > 0  <=>  low half heavier  <=>  rear at low  <=>  nose at high.
    mass_frac = float(n_low - n_high) / float(total)
    if abs(mass_frac) < _MASS_ASYM_MIN:
        return 0.0
    sign = 1.0 if mass_frac > 0.0 else -1.0
    return float(sign * _ORIENT_MASS_W * float(np.clip(abs(mass_frac), 0.0, 1.0)))


def _orientation_shape_evidence(
    geom: MaskGeom,
    center: np.ndarray,
    axis: np.ndarray,
    low_proj: float,
    high_proj: float,
    body_length: float,
    axis_strength: float = 0.0,
) -> tuple[float, float, float]:
    """Per-frame head/tail shape vote: > 0 favours the nose at the high end.

    Returns ``(total_vote, taper_vote, tail_signed)`` where ``tail_signed`` is the RAW
    anatomical-tail vote before the taper-disagreement damping (the lock uses it to
    require tail corroboration before flipping a stationary animal). The PRIMARY cue is
    the end shape (pointy =
    nose, blunt = rump). The anatomical tail filament and a weak width cue confirm it,
    BUT a filament that contradicts the end shape is damped: a pointy snout is often
    stripped off by the body opening and mis-detected as a filament, and when the real
    tail is occluded that snout-filament is the only one left, which is exactly what
    used to invert a passive mouse's nose and tail.
    """
    taper_vote = _orientation_taper_vote(geom, center, axis, body_length)

    tail_vote = 0.0
    tail_signed_raw = 0.0
    if geom.tail_base is not None:
        tail_proj = float((np.asarray(geom.tail_base, dtype=np.float64) - center) @ axis)
        d_low = abs(tail_proj - low_proj)
        d_high = abs(tail_proj - high_proj)
        sep = abs(d_low - d_high) / max(1.0, body_length)
        weight = (
            _ORIENT_TAIL_W
            * float(np.clip(0.35 + 0.65 * float(geom.tail_confidence), 0.0, 1.0))
            * float(np.clip(sep / 0.6, 0.25, 1.0))
        )
        # tail nearer the low end -> nose at high end -> positive vote
        tail_vote = (1.0 if d_low < d_high else -1.0) * weight
        tail_signed_raw = float(tail_vote)  # raw anatomical vote, before any damping
        # A filament at the POINTY end contradicts the shape -> it is the snout, not
        # the tail. Trust the silhouette and damp the filament rather than invert.
        if tail_vote * taper_vote < 0.0:
            tail_vote *= _TAIL_VS_SHAPE_DAMP

    low_width = _section_width(geom.mask_points, center, axis, low_proj, body_length)
    high_width = _section_width(geom.mask_points, center, axis, high_proj, body_length)
    denom = max(1.0, low_width, high_width)
    width_vote = (1.0 if low_width > high_width else -1.0) * _ORIENT_WIDTH_W * float(
        np.clip(abs(low_width - high_width) / denom, 0.0, 1.0)
    )
    # Weak, gated bulk-mass confirmer (sign-redundant with taper; see _ORIENT_MASS_W).
    mass_vote = _orientation_mass_vote(geom, center, axis, low_proj, high_proj, axis_strength)
    return (
        float(tail_vote + width_vote + taper_vote + mass_vote),
        float(taper_vote),
        float(tail_signed_raw),
    )


def _section_width(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    body_length: float,
) -> float:
    lateral = np.array([-axis[1], axis[0]], dtype=np.float64)
    selected = _section_points(points, center, axis, target_proj, body_length)
    if len(selected) < 2:
        return 0.0
    lat = (selected - center.reshape(1, 2)) @ lateral
    return float(np.max(lat) - np.min(lat))


def _axis_strength(eigvals: np.ndarray) -> float:
    vals = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    if vals.size < 2:
        return 0.0
    vals = np.sort(np.maximum(vals, 0.0))
    denom = float(vals[-1] + vals[-2])
    if denom <= 1e-9:
        return 0.0
    return float(np.clip((vals[-1] - vals[-2]) / denom, 0.0, 1.0))


def _fraction_to_projection(nose_proj: float, rear_proj: float, frac: float) -> float:
    return float(nose_proj + float(frac) * (rear_proj - nose_proj))


def _section_center(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    nose_proj: float,
    rear_proj: float,
    frac: float,
    body_length: float,
) -> np.ndarray:
    target = _fraction_to_projection(nose_proj, rear_proj, frac)
    selected = _section_points(points, center, axis, target, body_length)
    if len(selected) > 0:
        return np.mean(selected, axis=0).astype(np.float64)
    return center + axis * target


def _section_edges(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    lateral: np.ndarray,
    target_proj: float,
    body_length: float,
    *,
    fallback_center: np.ndarray,
    robust: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    selected = _section_points(points, center, axis, target_proj, body_length)
    if len(selected) < 2:
        offset = lateral * max(2.0, body_length * 0.035)
        return fallback_center + offset, fallback_center - offset, float(np.linalg.norm(offset) * 2.0)
    lat = (selected - center.reshape(1, 2)) @ lateral
    if robust and len(selected) >= 6:
        right_value, left_value = np.percentile(lat, [8.0, 92.0])
        left = selected[int(np.argmin(np.abs(lat - left_value)))]
        right = selected[int(np.argmin(np.abs(lat - right_value)))]
        width = float(max(0.0, left_value - right_value))
        return left.astype(np.float64), right.astype(np.float64), width
    left = selected[int(np.argmax(lat))]
    right = selected[int(np.argmin(lat))]
    return left.astype(np.float64), right.astype(np.float64), float(np.max(lat) - np.min(lat))


def _hip_point_is_suspect(
    point: np.ndarray,
    body_points: np.ndarray,
    tail_base: np.ndarray,
    tail_to_nose: np.ndarray,
    *,
    max_body_distance: float,
    min_anterior: float,
) -> bool:
    pt = np.asarray(point, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(pt)):
        return True
    body = np.asarray(body_points, dtype=np.float64).reshape(-1, 2)
    if len(body) == 0:
        return False
    nearest_body_distance = float(np.min(np.linalg.norm(body - pt.reshape(1, 2), axis=1)))
    if nearest_body_distance > float(max_body_distance):
        return True
    anterior = float(np.dot(pt - np.asarray(tail_base, dtype=np.float64).reshape(2), tail_to_nose))
    if anterior < float(min_anterior):
        return True
    return False


def _section_points(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    body_length: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    projs = (pts - center.reshape(1, 2)) @ axis
    base_window = max(2.0, float(body_length) * 0.045)
    for multiplier in (1.0, 1.8, 2.8, 4.0):
        mask = np.abs(projs - float(target_proj)) <= base_window * multiplier
        if int(np.count_nonzero(mask)) >= 3:
            return pts[mask]
    nearest = np.argsort(np.abs(projs - float(target_proj)))[: min(5, len(pts))]
    return pts[nearest]


def _end_point(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    *,
    prefer_high: bool,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return center + axis * target_proj
    projs = (pts - center.reshape(1, 2)) @ axis
    index = int(np.argmax(projs) if prefer_high else np.argmin(projs))
    return pts[index].astype(np.float64)


def _unit(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(2)
    return vec / norm


def _tail_to_nose_direction(keypoints: np.ndarray) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if len(kp) < len(KP_ORDER):
        return np.array([0.0, 0.0], dtype=np.float64)
    nose = kp[0]
    tail_base = kp[7]
    if not (np.all(np.isfinite(nose)) and np.all(np.isfinite(tail_base))):
        return np.array([0.0, 0.0], dtype=np.float64)
    return nose - tail_base


def _smooth_keypoints(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
    max_jump: float,
) -> np.ndarray:
    if previous is None:
        return current.copy()
    prev = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    if prev.shape != cur.shape:
        return cur.copy()
    out = cur.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for index in range(len(cur)):
        cx, cy = cur[index]
        px, py = prev[index]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            out[index] = (np.nan, np.nan)
            continue
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if float(np.hypot(cx - px, cy - py)) > float(max_jump):
            continue
        out[index] = (a * cx + (1.0 - a) * px, a * cy + (1.0 - a) * py)
    return out


def _finite_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.asarray(a, dtype=np.float64).reshape(2)
    pb = np.asarray(b, dtype=np.float64).reshape(2)
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        return 1e9
    return float(np.linalg.norm(pa - pb))
