import unittest

import numpy as np

from live_tracking import LiveIdentityTracker, compute_body_center


class LiveTrackingTests(unittest.TestCase):
    def test_compute_body_center_uses_mask_centroid_then_bbox_fallback(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 6:8] = True
        center_from_mask = compute_body_center(mask, (0.0, 0.0, 9.0, 9.0))
        center_from_bbox = compute_body_center(None, (10.0, 20.0, 30.0, 40.0))

        self.assertAlmostEqual(center_from_mask[0], 6.5)
        self.assertAlmostEqual(center_from_mask[1], 2.5)
        self.assertEqual(center_from_bbox, (20.0, 30.0))

    def test_tracker_keeps_mouse_ids_stable_across_small_motion(self):
        tracker = LiveIdentityTracker(expected_mice=2)

        first = tracker.update(
            [
                {"center": (20.0, 20.0), "bbox": (10.0, 10.0, 30.0, 30.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (80.0, 20.0), "bbox": (70.0, 10.0, 90.0, 30.0), "confidence": 0.8, "class_id": 0, "mask": None},
            ]
        )
        second = tracker.update(
            [
                {"center": (24.0, 22.0), "bbox": (14.0, 12.0, 34.0, 32.0), "confidence": 0.92, "class_id": 0, "mask": None},
                {"center": (84.0, 18.0), "bbox": (74.0, 8.0, 94.0, 28.0), "confidence": 0.81, "class_id": 0, "mask": None},
            ]
        )

        self.assertEqual([mouse.mouse_id for mouse in first], [1, 2])
        self.assertEqual([mouse.mouse_id for mouse in second], [1, 2])
        self.assertLess(second[0].center[0], second[1].center[0])

    def _crossing_id1_x(self, use_motion_prediction):
        """Two mice on the same y line cross in x; return mouse-id-1's x per frame."""
        tracker = LiveIdentityTracker(expected_mice=2, use_motion_prediction=use_motion_prediction)
        id1_x = []
        for t in range(13):
            ax = 20.0 + 12.0 * t   # mouse A: starts left, moves right
            bx = 200.0 - 12.0 * t  # mouse B: starts right, moves left
            dets = [
                {"center": (ax, 50.0), "bbox": (ax - 20, 30.0, ax + 20, 70.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (bx, 50.0), "bbox": (bx - 20, 30.0, bx + 20, 70.0), "confidence": 0.9, "class_id": 0, "mask": None},
            ]
            states = tracker.update(dets)
            by_id = {s.mouse_id: s for s in states}
            if 1 in by_id:
                id1_x.append(by_id[1].center[0])
        return id1_x

    def test_motion_prediction_keeps_identity_through_a_crossing(self):
        # Mouse A (id 1) starts on the left and moves right; with motion prediction
        # mouse id 1 must follow A THROUGH the crossing, so its x rises monotonically.
        xs = self._crossing_id1_x(use_motion_prediction=True)
        self.assertGreater(len(xs), 10)
        self.assertGreater(xs[-1], xs[0] + 80.0)  # ended well to the right (followed A)
        # No backward jump at the crossing (a swap would make x drop).
        for prev, cur in zip(xs, xs[1:]):
            self.assertGreaterEqual(cur, prev - 1e-6)

    def test_without_prediction_identity_swaps_at_a_crossing(self):
        # Contrast: the legacy nearest-centroid path swaps IDs at the crossing, so
        # id 1's x is NOT monotonic (it bounces back). This proves the test bites.
        xs = self._crossing_id1_x(use_motion_prediction=False)
        self.assertTrue(any(cur < prev - 1.0 for prev, cur in zip(xs, xs[1:])))

    def test_distance_gate_blocks_a_teleport_steal(self):
        tracker = LiveIdentityTracker(expected_mice=2)
        tracker.update(
            [
                {"center": (100.0, 100.0), "bbox": (90.0, 90.0, 110.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (500.0, 100.0), "bbox": (490.0, 90.0, 510.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
            ]
        )
        # Track A moves a little; a spurious detection appears far away (a teleport).
        states = tracker.update(
            [
                {"center": (130.0, 100.0), "bbox": (120.0, 90.0, 140.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (900.0, 100.0), "bbox": (890.0, 90.0, 910.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
            ]
        )
        by_id = {s.mouse_id: s for s in states}
        # Track 1 keeps the nearby detection.
        self.assertIn(1, by_id)
        self.assertAlmostEqual(by_id[1].center[0], 130.0)
        # Track 2 must NOT have teleported onto the far detection.
        if 2 in by_id:
            self.assertLess(by_id[2].center[0], 600.0)
        # The far detection should exist under a fresh id, not id 2.
        far = [s for s in states if s.center[0] > 600.0]
        self.assertTrue(far and far[0].mouse_id != 2)

    def test_no_gate_allows_the_teleport_steal(self):
        tracker = LiveIdentityTracker(expected_mice=2, use_distance_gate=False)
        tracker.update(
            [
                {"center": (100.0, 100.0), "bbox": (90.0, 90.0, 110.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (500.0, 100.0), "bbox": (490.0, 90.0, 510.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
            ]
        )
        states = tracker.update(
            [
                {"center": (130.0, 100.0), "bbox": (120.0, 90.0, 140.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
                {"center": (900.0, 100.0), "bbox": (890.0, 90.0, 910.0, 110.0), "confidence": 0.9, "class_id": 0, "mask": None},
            ]
        )
        by_id = {s.mouse_id: s for s in states}
        # Without the gate, track 2 is force-matched onto the far detection (teleport).
        self.assertIn(2, by_id)
        self.assertGreater(by_id[2].center[0], 600.0)

    def test_model_class_assignment_uses_selected_class_order(self):
        tracker = LiveIdentityTracker(expected_mice=2)

        assigned = tracker.assign_by_model_class(
            [
                {"center": (15.0, 15.0), "bbox": (5.0, 5.0, 25.0, 25.0), "confidence": 0.8, "class_id": 7, "mask": None},
                {"center": (80.0, 18.0), "bbox": (70.0, 8.0, 90.0, 28.0), "confidence": 0.9, "class_id": 4, "mask": None},
            ],
            selected_class_ids=[4, 7],
        )

        self.assertEqual([mouse.mouse_id for mouse in assigned], [1, 2])
        self.assertEqual([mouse.class_id for mouse in assigned], [4, 7])


if __name__ == "__main__":
    unittest.main()
