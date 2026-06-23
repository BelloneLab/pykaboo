"""Orientation (nose<->tail) regression tests for the mask-geometry keypoint path.

These guard the live "mouse N nose and tail are inverted" failure mode and its fix:

* a clearly-shaped animal seeds the nose at the pointy (snout) end;
* a STATIONARY animal whose orientation locked backwards recovers automatically
  once the end-shape vote disagrees strongly and continuously (a real mis-seed),
  within ~_FLIP_CONFIRM_STATIONARY frames;
* a correctly-seeded STATIONARY animal never spuriously flips (the passive /
  investigated-mouse artifact the reorienting guard protects against);
* the manual user override (flip_orientation) swaps head<->tail and sticks.
"""
import numpy as np
import cv2

import mask_skeleton as ms


def _tadpole(nose_right: bool = True, width: int = 420, height: int = 200) -> np.ndarray:
    """A mouse-like silhouette: blunt body, a pointy snout, and a thin tail filament.

    The snout (pointy, low-area end) is the nose; the opposite end carries a thin
    tail filament. With ``nose_right`` the snout is on the +x side.
    """
    img = np.zeros((height, width), np.uint8)
    cy = height // 2
    cv2.ellipse(img, (210, cy), (70, 28), 0, 0, 360, 255, -1)           # body
    snout = np.array([[270, cy - 18], [270, cy + 18], [300, cy]], np.int32)
    cv2.fillConvexPoly(img, snout, 255)                                  # pointy nose at +x
    cv2.line(img, (140, cy), (70, cy), 255, 5)                           # tail filament at -x
    if not nose_right:
        img = img[:, ::-1].copy()
    return img


def _compact_toy(head_right: bool = True, width: int = 240, height: int = 200) -> np.ndarray:
    """A small, round-headed test object whose POINTY end is the REAR (taper inverts).

    This is the adversarial shape that breaks the 'pointy end == nose' heuristic: a
    blunt round head plus a short pointed rear, with a low elongation (compact) axis.
    """
    img = np.zeros((height, width), np.uint8)
    cy = height // 2
    cv2.circle(img, (150, cy), 46, 255, -1)                       # blunt round head
    rear = np.array([[120, cy - 30], [120, cy + 30], [70, cy]], np.int32)
    cv2.fillConvexPoly(img, rear, 255)                            # short pointy rear
    if not head_right:
        img = img[:, ::-1].copy()
    return img


def _ellipse_no_tail(width: int = 420, height: int = 200) -> np.ndarray:
    """An elongated body with NO tail filament (tail occluded / not segmented)."""
    img = np.zeros((height, width), np.uint8)
    cv2.ellipse(img, (210, height // 2), (110, 30), 0, 0, 360, 255, -1)
    return img


def _wide_rear_mouse(nose_right: bool = True, width: int = 460, height: int = 240) -> np.ndarray:
    """Elongated body with a clearly WIDE/heavy rear and a narrow head+snout + thin tail.

    The bulk-mass cue should confirm the rear (heavy half) here, where taper is also right.
    """
    img = np.zeros((height, width), np.uint8)
    cy = height // 2
    cv2.ellipse(img, (150, cy), (60, 52), 0, 0, 360, 255, -1)  # wide heavy rear
    cv2.ellipse(img, (250, cy), (55, 30), 0, 0, 360, 255, -1)  # torso
    cv2.ellipse(img, (330, cy), (30, 18), 0, 0, 360, 255, -1)  # small head
    cv2.fillConvexPoly(img, np.array([[358, cy - 9], [358, cy + 9], [388, cy]], np.int32), 255)
    cv2.line(img, (95, cy), (45, cy), 255, 6)                  # thin tail off the rear
    if not nose_right:
        img = img[:, ::-1].copy()
    return img


def _mass_vote(mask: np.ndarray) -> float:
    """Compute the body-bulk vote for a mask the way keypoints_pca does."""
    geom = ms.analyze_mask(mask)
    sk = ms.keypoints_pca(geom)
    proj = (np.asarray(geom.body_points).reshape(-1, 2) - sk.center) @ sk.axis
    return ms._orientation_mass_vote(
        geom, sk.center, sk.axis, float(proj.min()), float(proj.max()), sk.axis_strength
    )


def _make_skeleton(*, evidence, tail_signed, axis_strength, nose_at_high,
                   axis=(1.0, 0.0), body_length=100.0):
    """Minimal Skeleton carrying just the orientation internals _resolve_orientation reads."""
    kp = np.zeros((8, 2), dtype=np.float64)
    return ms.Skeleton(
        keypoints=kp,
        scores=np.full(8, 0.9),
        orientation_confidence=0.5,
        axis=np.asarray(axis, dtype=np.float64),
        center=np.zeros(2),
        body_length=float(body_length),
        evidence_signed=float(evidence),
        taper_signed=float(evidence),
        tail_signed=float(tail_signed),
        axis_strength=float(axis_strength),
        nose_at_high=bool(nose_at_high),
    )


def _nose_right(skeleton) -> bool:
    """True when the nose keypoint sits to the right of the tail keypoint."""
    return float(skeleton.keypoints[0][0]) > float(skeleton.keypoints[7][0])


def test_pointy_end_is_the_nose():
    geom = ms.analyze_mask(_tadpole(nose_right=True))
    assert geom is not None
    sk = ms.keypoints_pca(geom)
    assert sk is not None
    # The snout is on the +x side, so the nose keypoint must be there too.
    assert _nose_right(sk)
    # The end-shape vote should be decisively positive (above the strong margin).
    assert sk.evidence_signed > ms._FLIP_MARGIN_STRONG

    # Mirror image: nose must follow to the -x side.
    sk_left = ms.keypoints_pca(ms.analyze_mask(_tadpole(nose_right=False)))
    assert sk_left is not None
    assert not _nose_right(sk_left)


def test_stationary_backwards_seed_recovers():
    mask = _tadpole(nose_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    seed = ext.estimate(mask, track_id=1)
    assert seed is not None and _nose_right(seed)

    # Simulate a backwards seed: reverse the locked tail->nose direction.
    ext._states[1]["direction"] = -np.asarray(ext._states[1]["direction"], dtype=np.float64)

    recovered_at = None
    last = None
    # Allow a little headroom over the confirmation window.
    for i in range(ms._FLIP_CONFIRM_STATIONARY + 20):
        last = ext.estimate(mask, track_id=1)
        if recovered_at is None and _nose_right(last):
            recovered_at = i + 1
    assert last is not None and _nose_right(last)
    assert recovered_at is not None
    # Recovery is deliberately slow (time-discriminated from transient artifacts)
    # but must not need much more than the confirmation window.
    assert ms._FLIP_CONFIRM_STATIONARY <= recovered_at <= ms._FLIP_CONFIRM_STATIONARY + 5


def test_correctly_seeded_stationary_mouse_never_flips():
    """The passive / investigated-mouse guard: a still, correctly-oriented animal
    must keep its orientation indefinitely (no spurious stationary recovery)."""
    mask = _tadpole(nose_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    for _ in range(ms._FLIP_CONFIRM_STATIONARY * 3):
        sk = ext.estimate(mask, track_id=3)
        assert sk is not None
        assert _nose_right(sk)


def test_manual_flip_overrides_and_persists():
    mask = _tadpole(nose_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    for _ in range(5):
        sk = ext.estimate(mask, track_id=2)
    assert _nose_right(sk)

    assert ext.flip_orientation(2) is True

    # After a manual flip the nose moves to the other end and, because the user has
    # asserted orientation, automatic stationary recovery must not undo it.
    last = None
    for _ in range(ms._FLIP_CONFIRM_STATIONARY * 2):
        last = ext.estimate(mask, track_id=2)
        assert last is not None
        assert not _nose_right(last)  # stays flipped


def test_compact_toy_not_driven_wrong_from_correct_seed():
    """REGRESSION: a compact, round-headed motionless object whose taper points the
    WRONG way must NOT be auto-flipped to the pointy rear once it is oriented right.
    Before the tail/compact gate this flipped after _FLIP_CONFIRM_STATIONARY frames."""
    mask = _compact_toy(head_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    ext.estimate(mask, track_id=1)
    # Force the correct lock: nose toward +x (the round head).
    ext._states[1]["direction"] = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(ms._FLIP_CONFIRM_STATIONARY * 2 + 10):
        sk = ext.estimate(mask, track_id=1)
        assert sk is not None
        assert _nose_right(sk)  # nose stays on the round head; never driven to the rear


def test_forced_strong_wrong_evidence_cannot_flip_compact_stationary_lock():
    """Core regression assertion at the _resolve_orientation level: a strong, wrong-signed
    taper evidence with NO tail corroboration on a COMPACT shape must not overturn the
    lock. Pre-fix this returned the flipped orientation after _FLIP_CONFIRM_STATIONARY."""
    ext = ms.MaskSkeletonExtractor(smooth=False)
    ext._states[7] = {
        "direction": np.array([1.0, 0.0], dtype=np.float64),  # locked nose toward +x
        "keypoints": None,
        "centroid": np.zeros(2),
    }
    sk = _make_skeleton(evidence=-1.20, tail_signed=0.0, axis_strength=0.35, nose_at_high=False)
    for _ in range(ms._FLIP_CONFIRM_STATIONARY + 20):
        # prior_high = sign(lock . axis) = True; strong wrong evidence favours low end.
        assert ext._resolve_orientation(7, sk, None) is True  # never flips


def test_tail_corroborated_disagreement_arms_a_flip_while_compact_does_not():
    """The gate's two sides, asserted at the decision level. An ELONGATED shape with a
    confident tail vote that disagrees with the lock ARMS the stationary flip (it would
    recover via estimate(), as the tadpole test proves); the same evidence on a COMPACT
    shape, or with no tail, never arms it. _resolve_orientation is probed directly, so we
    check the *single-step* decision: armed flips show up once the counter saturates."""
    def decide(axis_strength, tail_signed):
        ext = ms.MaskSkeletonExtractor(smooth=False)
        ext._states[3] = {
            "direction": np.array([1.0, 0.0], dtype=np.float64),  # locked nose toward +x
            "keypoints": None,
            "centroid": np.zeros(2),
        }
        sk = _make_skeleton(evidence=-1.20, tail_signed=tail_signed,
                            axis_strength=axis_strength, nose_at_high=False)
        # First _FLIP_CONFIRM_STATIONARY calls must hold; the next reveals whether armed.
        held = all(ext._resolve_orientation(3, sk, None) is True
                   for _ in range(ms._FLIP_CONFIRM_STATIONARY - 1))
        return held, ext._resolve_orientation(3, sk, None)

    # elongated + corroborating tail => armed: holds through the window then flips.
    held, nxt = decide(axis_strength=0.75, tail_signed=-0.60)
    assert held and nxt is False
    # compact => never arms regardless of evidence strength.
    held, nxt = decide(axis_strength=0.35, tail_signed=-0.60)
    assert held and nxt is True


def test_tailless_elongated_backwards_seed_holds_until_motion():
    """Documents the accepted trade-off: an elongated mouse seeded backwards with NO
    tail filament does NOT auto-recover while perfectly still, but a motion hint (motion
    override) flips it immediately."""
    ext = ms.MaskSkeletonExtractor(smooth=False)
    ext._states[4] = {
        "direction": np.array([1.0, 0.0], dtype=np.float64),  # nose toward +x
        "keypoints": None,
        "centroid": np.zeros(2),
    }
    sk = _make_skeleton(evidence=-1.20, tail_signed=0.0, axis_strength=0.75, nose_at_high=False)
    for _ in range(ms._FLIP_CONFIRM_STATIONARY + 10):
        assert ext._resolve_orientation(4, sk, None) is True  # no tail -> holds
    # A decisive motion toward -x (the leading end) pins the nose there immediately.
    motion = np.array([-50.0, 0.0], dtype=np.float64)  # well above _MOTION_OVERRIDE_FRAC * body_len
    assert ext._resolve_orientation(4, sk, motion) is False


def test_manual_flip_on_compact_toy_sticks():
    """Manual override is independent of the new gates: it must still flip and hold."""
    mask = _compact_toy(head_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    for _ in range(4):
        before = ext.estimate(mask, track_id=2)
    before_right = _nose_right(before)
    assert ext.flip_orientation(2) is True
    for _ in range(ms._FLIP_CONFIRM_STATIONARY * 2):
        sk = ext.estimate(mask, track_id=2)
        assert _nose_right(sk) == (not before_right)  # flipped, and stays as the user set it


def test_mass_vote_gated_off_on_compact_shape():
    """The bulk cue must never inject a vote on a compact/round shape (where its sign is
    as unreliable as the taper it would amplify)."""
    assert _mass_vote(_compact_toy(head_right=True)) == 0.0
    assert _mass_vote(_compact_toy(head_right=False)) == 0.0


def test_mass_vote_below_floor_is_silent_on_tadpole():
    """A symmetric ellipse body has near-zero bulk asymmetry, so the cue stays silent and
    cannot perturb the already-correct taper-anchored seed."""
    assert _mass_vote(_tadpole(nose_right=True)) == 0.0
    assert _mass_vote(_tadpole(nose_right=False)) == 0.0


def test_mass_vote_sign_on_clear_bulk_asymmetry():
    """On a clearly wide-rear elongated body the cue confirms the nose at the lighter end,
    flips sign with the mask, and never exceeds its weight cap."""
    v_right = _mass_vote(_wide_rear_mouse(nose_right=True))
    v_left = _mass_vote(_wide_rear_mouse(nose_right=False))
    assert v_right > 0.0          # nose at the high (snout) end
    assert v_left < 0.0           # mirror flips the sign
    assert abs(v_right) <= ms._ORIENT_MASS_W + 1e-9
    assert abs(v_left) <= ms._ORIENT_MASS_W + 1e-9


def test_mass_vote_does_not_change_tadpole_seed():
    """Regression lock: adding the cue leaves the tadpole's anchored seed strong & correct."""
    sk = ms.keypoints_pca(ms.analyze_mask(_tadpole(nose_right=True)))
    assert _nose_right(sk)
    assert sk.evidence_signed > ms._FLIP_MARGIN_STRONG


def test_mass_weight_cannot_overturn_taper():
    """Subordination invariant: a maximally-wrong bulk vote cannot flip a taper-set seed."""
    taper = 0.90  # _ORIENT_TAPER_W, a confident snout taper favouring nose-at-high
    worst_wrong_mass = -ms._ORIENT_MASS_W
    assert taper + worst_wrong_mass > 0.0  # sign of total evidence is still nose-at-high


def test_flip_before_track_seen_is_queued_then_applied():
    mask = _tadpole(nose_right=True)
    ext = ms.MaskSkeletonExtractor(smooth=False)
    # Request a flip for a track that has not been processed yet.
    assert ext.flip_orientation(9) is False
    assert 9 in ext._manual_flip_pending
    # First frame seeds the lock; the queued flip applies on the following frame.
    ext.estimate(mask, track_id=9)
    after = ext.estimate(mask, track_id=9)
    assert after is not None
    assert not _nose_right(after)
    assert 9 not in ext._manual_flip_pending
