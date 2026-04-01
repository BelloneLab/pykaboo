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
