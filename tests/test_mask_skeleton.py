import unittest

import numpy as np

try:
    import cv2
    from mask_skeleton import KP_ORDER, MaskSkeletonExtractor, extract_skeleton
except Exception:
    cv2 = None
    KP_ORDER = ()
    MaskSkeletonExtractor = None
    extract_skeleton = None


def _mouse_mask(tail: bool = True, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    mask = np.zeros((100, 140), dtype=np.uint8)
    cv2.ellipse(mask, (75 + shift_x, 50 + shift_y), (30, 16), 0, 0, 360, 1, -1)
    if tail:
        cv2.line(mask, (45 + shift_x, 50 + shift_y), (15 + shift_x, 56 + shift_y), 1, 3)
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
            "tail_tip",
        ))
        self.assertEqual(keypoints.shape, (8, 2))
        self.assertEqual(skeleton.scores.shape, (8,))
        self.assertGreater(skeleton.orientation_confidence, 0.75)
        self.assertGreater(keypoints[0, 0], keypoints[3, 0])
        self.assertGreater(keypoints[3, 0], keypoints[7, 0])
        self.assertLess(keypoints[7, 0], keypoints[4, 0])

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


if __name__ == "__main__":
    unittest.main()
