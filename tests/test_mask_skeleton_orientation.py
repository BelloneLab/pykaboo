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
