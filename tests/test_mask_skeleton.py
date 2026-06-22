import unittest

import numpy as np

try:
    import cv2
    from mask_skeleton import (
        KP_ORDER,
        MaskSkeletonExtractor,
        extract_skeleton,
        repair_hip_keypoints_with_mask_geometry,
    )
except Exception:
    cv2 = None
    KP_ORDER = ()
    MaskSkeletonExtractor = None
    extract_skeleton = None
    repair_hip_keypoints_with_mask_geometry = None


def _mouse_mask(tail: bool = True, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    mask = np.zeros((100, 140), dtype=np.uint8)
    cv2.ellipse(mask, (75 + shift_x, 50 + shift_y), (30, 16), 0, 0, 360, 1, -1)
    if tail:
        cv2.line(mask, (45 + shift_x, 50 + shift_y), (15 + shift_x, 56 + shift_y), 1, 3)
    return mask.astype(bool)


def _long_mouse(cx: int, tail_side: str, cy: int = 110) -> np.ndarray:
    """Horizontal body at ``cx`` with a clean tail filament on one side. The body is
    large so flipping the tail side moves the centroid only a few percent of a body
    length (i.e. effectively stationary)."""
    mask = np.zeros((220, 520), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (60, 30), 0, 0, 360, 1, -1)
    if tail_side == "left":
        cv2.line(mask, (cx - 60, cy), (cx - 120, cy + 8), 1, 3)
    else:
        cv2.line(mask, (cx + 60, cy), (cx + 120, cy + 8), 1, 3)
    return mask.astype(bool)


def _egg(cx: int = 210, cy: int = 110, narrow_sign: int = 1) -> np.ndarray:
    """A body 'egg': a wide blunt (rump) end and a narrower tapered (nose) end, with
    NO tail filament. ``narrow_sign=+1`` puts the narrow/nose end on the right. This is
    the silhouette-method case: the orientation must come from end SHAPE alone (the
    real tail is hidden, as for a passive mouse investigated at its rear)."""
    mask = np.zeros((240, 520), dtype=np.uint8)
    s = int(narrow_sign)
    x0, x1 = cx - 80, cx + 80
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        half_w = (40 * (1 - t) + 16 * t) if s > 0 else (16 * (1 - t) + 40 * t)
        cv2.line(mask, (x, int(cy - half_w)), (x, int(cy + half_w)), 1, 1)
    return mask.astype(bool)


def _egg_with_filament(cx: int = 210, cy: int = 110, narrow_sign: int = 1,
                       fil_sign: int = 1) -> np.ndarray:
    """An ``_egg`` plus a thin filament -- on the narrow end this mimics the pointy
    snout being stripped by the body opening and mis-detected as a tail."""
    mask = _egg(cx, cy, narrow_sign).astype(np.uint8)
    cv2.line(mask, (int(cx + fil_sign * 80), cy), (int(cx + fil_sign * 150), cy + 6), 1, 3)
    return mask.astype(bool)


def _pivot_mouse(cx: int, cy: int, ang_deg: float) -> np.ndarray:
    """Rigid body+tail rotated by ``ang_deg`` about its centroid (no translation).
    The tail sits at the -axis end, so the nose is at the +axis end."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (60, 28), ang_deg, 0, 360, 1, -1)
    th = np.deg2rad(ang_deg)
    ux, uy = float(np.cos(th)), float(np.sin(th))
    base = (int(round(cx - 60 * ux)), int(round(cy - 60 * uy)))
    tip = (int(round(cx - 120 * ux)), int(round(cy - 120 * uy)))
    cv2.line(mask, base, tip, 1, 3)
    return mask.astype(bool)


def _curved_tail_mask() -> np.ndarray:
    mask = np.zeros((160, 220), dtype=np.uint8)
    cv2.ellipse(mask, (115, 80), (42, 21), 0, 0, 360, 1, -1)
    tail_points = [(73, 80), (45, 90), (65, 102), (95, 99)]
    for start, end in zip(tail_points, tail_points[1:]):
        cv2.line(mask, start, end, 1, 5)
    return mask.astype(bool)


@unittest.skipUnless(cv2 is not None, "OpenCV is required for mask skeleton tests")
class MaskSkeletonTests(unittest.TestCase):
    def test_extract_skeleton_orients_nose_away_from_tail_filament(self):
        skeleton = extract_skeleton(_mouse_mask(tail=True))

        self.assertIsNotNone(skeleton)
        keypoints = skeleton.keypoints
        self.assertEqual(tuple(KP_ORDER), (
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "body",
            "left_hip",
            "right_hip",
            "tail_base",
        ))
        self.assertEqual(keypoints.shape, (8, 2))
        self.assertEqual(skeleton.scores.shape, (8,))
        self.assertGreater(skeleton.orientation_confidence, 0.75)
        self.assertGreater(keypoints[0, 0], keypoints[3, 0])
        self.assertGreater(keypoints[3, 0], keypoints[7, 0])
        self.assertLess(keypoints[7, 0], keypoints[4, 0])
        self.assertGreater(keypoints[7, 0], 35.0)
        self.assertLess(keypoints[7, 0], 55.0)

    def test_body_keypoint_matches_mask_centroid(self):
        mask = _mouse_mask(tail=True)
        skeleton = extract_skeleton(mask)

        ys, xs = np.nonzero(mask)
        centroid = np.array([float(np.mean(xs)), float(np.mean(ys))])
        np.testing.assert_allclose(skeleton.keypoints[4], centroid, atol=1.0)

    def test_keypoints_are_translation_invariant(self):
        """The bbox-crop fast path must yield the same keypoints (offset) as a
        full-frame computation: place the same mouse at two positions in a large
        frame and require the keypoints to differ only by the translation."""
        def mouse_at(cx: int, cy: int) -> np.ndarray:
            canvas = np.zeros((480, 640), dtype=np.uint8)
            cv2.ellipse(canvas, (cx, cy), (30, 16), 0, 0, 360, 1, -1)
            cv2.line(canvas, (cx - 30, cy), (cx - 60, cy + 6), 1, 3)
            return canvas.astype(bool)

        a = extract_skeleton(mouse_at(120, 110))
        b = extract_skeleton(mouse_at(420, 330))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        shift = np.array([300.0, 220.0])
        np.testing.assert_allclose(b.keypoints, a.keypoints + shift, atol=1.0)

    def test_stateful_extractor_uses_previous_direction_when_tail_is_missing(self):
        extractor = MaskSkeletonExtractor(smooth=False)

        first = extractor.estimate(_mouse_mask(tail=True), track_id=1)
        second = extractor.estimate(_mouse_mask(tail=False, shift_x=3), track_id=1)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreater(first.keypoints[0, 0], first.keypoints[7, 0])
        self.assertGreater(second.keypoints[0, 0], second.keypoints[7, 0])
        self.assertGreaterEqual(second.orientation_confidence, 0.40)

    def test_stationary_mouse_holds_lock_against_spurious_opposite_tail(self):
        """The passive-investigated-mouse failure: a stationary mouse must NOT flip
        head<->tail when a clean filament appears at the locked nose end (the real
        tail dropping out under occlusion, or a snout protrusion), because it has not
        translated and its body axis has not rotated. Without the reorientation gate
        the sustained disagreeing tail cue would overturn the lock and freeze the
        orientation backwards for the whole contact bout."""
        extractor = MaskSkeletonExtractor(smooth=False)
        established = None
        for _ in range(5):
            established = extractor.estimate(_long_mouse(260, "left"), track_id=11)
        self.assertIsNotNone(established)
        # tail on the left => nose at the high-x (right) end
        self.assertGreater(established.keypoints[0, 0], established.keypoints[7, 0])

        last = established
        for _ in range(6):
            last = extractor.estimate(_long_mouse(260, "right"), track_id=11)
        # lock holds: nose still on the right despite the disagreeing right-side tail
        self.assertGreater(last.keypoints[0, 0], last.keypoints[7, 0])

    def test_fast_translation_still_overrides_stale_orientation(self):
        """Regression guard: the reorientation gate must not block the motion override.
        A mouse that walks fast in -x with a trailing tail must orient its nose toward
        the motion, even if that overturns a stale lock."""
        extractor = MaskSkeletonExtractor(smooth=False)
        established = None
        for _ in range(5):
            established = extractor.estimate(_long_mouse(380, "left"), track_id=12)
        self.assertGreater(established.keypoints[0, 0], established.keypoints[7, 0])
        last = established
        for k in range(1, 6):
            # body marches left ~40 px/frame (> override frac); tail trails on the right
            last = extractor.estimate(_long_mouse(380 - 40 * k, "right"), track_id=12)
        # nose now leads the motion: nose at the low-x (left) end
        self.assertLess(last.keypoints[0, 0], last.keypoints[7, 0])

    def test_in_place_pivot_still_reorients(self):
        """Regression guard: a genuine in-place turn rotates the body-axis line, so the
        reorientation gate must still let the orientation follow it through ~180 deg."""
        extractor = MaskSkeletonExtractor(smooth=False)
        established = None
        for _ in range(5):
            established = extractor.estimate(_pivot_mouse(150, 150, 0.0), track_id=13)
        # tail at -axis (left) => nose at +x (right)
        self.assertGreater(established.keypoints[0, 0], established.keypoints[7, 0])
        last = established
        for ang in range(20, 181, 20):
            last = extractor.estimate(_pivot_mouse(150, 150, float(ang)), track_id=13)
        # after a full in-place 180 deg pivot the nose points the other way (left)
        self.assertLess(last.keypoints[0, 0], last.keypoints[7, 0])

    def test_orientation_from_end_shape_when_tail_absent(self):
        """Silhouette method: with no tail filament, the nose must land on the pointier
        (narrower) body-core end -- in BOTH orientations -- not be left to chance. This
        is the cue that fixes a passive mouse whose real tail is occluded in contact."""
        right = extract_skeleton(_egg(narrow_sign=1))
        left = extract_skeleton(_egg(narrow_sign=-1))
        self.assertIsNotNone(right)
        self.assertIsNotNone(left)
        # narrow end on the right -> nose on the right
        self.assertGreater(right.keypoints[0, 0], right.keypoints[7, 0])
        # narrow end on the left -> nose on the left
        self.assertLess(left.keypoints[0, 0], left.keypoints[7, 0])

    def test_snout_filament_does_not_invert_against_end_shape(self):
        """A filament on the NARROW (nose) end is the snout-stripped-as-tail artifact.
        The end shape must win: the nose stays on the narrow end instead of flipping
        onto the blunt rump."""
        extractor = MaskSkeletonExtractor(smooth=False)
        last = None
        for _ in range(6):
            last = extractor.estimate(_egg_with_filament(narrow_sign=1, fil_sign=1), track_id=41)
        self.assertGreater(last.keypoints[0, 0], last.keypoints[7, 0])  # nose on narrow (right)

    def test_end_shape_recovers_backwards_lock_once_moving(self):
        """Policy: a backwards-seeded STATIONARY mouse is deliberately NOT auto-corrected
        (that robustness is what stops a spurious tether-cable filament from inverting a
        correctly-seeded still mouse). The orientation recovers as soon as the mouse
        MOVES: directed translation re-engages the motion override / reorientation gate
        and the nose snaps to the leading, true-nose (narrow) end. White-box: seed
        correctly, corrupt the locked direction backwards, then TRANSLATE the egg."""
        extractor = MaskSkeletonExtractor(smooth=False)
        seeded = extractor.estimate(_egg(cx=160, narrow_sign=1), track_id=42)
        self.assertGreater(seeded.keypoints[0, 0], seeded.keypoints[7, 0])  # seeds correct
        # force a backwards lock (pretend tail->nose points left, nose at the blunt end)
        extractor._states[42]["direction"] = np.array([-1.0, 0.0])
        # The egg's narrow (true-nose) end is on the right, so march it RIGHT. A shift of
        # > 0.16*body_length per frame trips the motion override toward the leading end.
        last = seeded
        cx = 160
        for _ in range(6):
            cx += 40   # body_length ~160 px; 40 px/frame is well past 0.16*body_length
            last = extractor.estimate(_egg(cx=cx, narrow_sign=1), track_id=42)
        # once moving, the nose returns to the narrow (right / leading) end
        self.assertGreater(last.keypoints[0, 0], last.keypoints[7, 0])

    def test_geometry_hips_stay_anterior_to_tail_base(self):
        skeleton = extract_skeleton(_curved_tail_mask())

        self.assertIsNotNone(skeleton)
        nose = skeleton.keypoints[0]
        tail_base = skeleton.keypoints[7]
        tail_to_nose = nose - tail_base
        tail_to_nose = tail_to_nose / max(1e-9, float(np.linalg.norm(tail_to_nose)))
        for index in (5, 6):
            anterior = float(np.dot(skeleton.keypoints[index] - tail_base, tail_to_nose))
            self.assertGreater(anterior, 5.0)

    def test_repair_moves_pose_hips_off_tail_filament(self):
        mask = _curved_tail_mask()
        skeleton = extract_skeleton(mask)
        keypoints = skeleton.keypoints.copy()
        keypoints[5] = (60.0, 100.0)
        keypoints[6] = (55.0, 88.0)
        scores = np.ones(8, dtype=float)

        repaired, repaired_scores = repair_hip_keypoints_with_mask_geometry(keypoints, scores, mask)

        self.assertGreater(float(np.linalg.norm(repaired[5] - keypoints[5])), 20.0)
        self.assertGreater(float(np.linalg.norm(repaired[6] - keypoints[6])), 20.0)
        nose = repaired[0]
        tail_base = repaired[7]
        tail_to_nose = nose - tail_base
        tail_to_nose = tail_to_nose / max(1e-9, float(np.linalg.norm(tail_to_nose)))
        for index in (5, 6):
            anterior = float(np.dot(repaired[index] - tail_base, tail_to_nose))
            self.assertGreater(anterior, 5.0)
            self.assertLessEqual(repaired_scores[index], 0.70)


if __name__ == "__main__":
    unittest.main()
