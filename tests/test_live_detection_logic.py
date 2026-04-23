import unittest

import numpy as np

from live_detection_logic import LiveRuleEngine, format_roi_properties, occupied_roi_names, roi_geometry_properties
from live_detection_types import BehaviorROI, LiveDetectionResult, LiveTriggerRule, TrackedMouseState


class LiveDetectionLogicTests(unittest.TestCase):
    def test_roi_contains_point_for_rectangle_circle_and_polygon(self):
        rectangle = BehaviorROI(name="rect", roi_type="rectangle", data=[(10, 10, 30, 30)])
        circle = BehaviorROI(name="circle", roi_type="circle", data=[(50, 50, 10)])
        polygon = BehaviorROI(name="poly", roi_type="polygon", data=[(70, 70), (90, 70), (80, 90)])

        self.assertTrue(rectangle.contains_point(20, 20))
        self.assertFalse(rectangle.contains_point(5, 20))
        self.assertTrue(circle.contains_point(50, 55))
        self.assertFalse(circle.contains_point(65, 65))
        self.assertTrue(polygon.contains_point(80, 78))
        self.assertFalse(polygon.contains_point(95, 95))

    def test_occupied_roi_names_uses_tracked_mouse_centers(self):
        rois = {
            "open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)]),
            "closed_arm": BehaviorROI(name="closed_arm", roi_type="rectangle", data=[(50, 50, 75, 75)]),
        }
        result = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )

        self.assertEqual(occupied_roi_names(rois, result), {"open_arm"})
        self.assertEqual(occupied_roi_names(rois, None), set())

    def test_roi_geometry_properties_describe_supported_shapes(self):
        rectangle = BehaviorROI(name="rect", roi_type="rectangle", data=[(30, 10, 10, 25)])
        circle = BehaviorROI(name="circle", roi_type="circle", data=[(50, 60, 12)])
        polygon = BehaviorROI(name="poly", roi_type="polygon", data=[(0, 0), (10, 0), (5, 20)])

        rect_properties = roi_geometry_properties(rectangle)
        circle_properties = roi_geometry_properties(circle)
        polygon_properties = roi_geometry_properties(polygon)

        self.assertEqual(rect_properties["x"], 10.0)
        self.assertEqual(rect_properties["width"], 20.0)
        self.assertEqual(circle_properties["diameter"], 24.0)
        self.assertEqual(polygon_properties["point_count"], 3)
        self.assertIn("w=20", format_roi_properties(rectangle))
        self.assertIn("diameter=24", format_roi_properties(circle))
        self.assertIn("points=3", format_roi_properties(polygon))

    def test_level_rule_holds_output_while_mouse_is_in_roi(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-level",
                    rule_type="roi_occupancy",
                    output_id="DO1",
                    mode="level",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )
        outside = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(40.0, 40.0),
                    bbox=(30.0, 30.0, 50.0, 50.0),
                )
            ],
        )

        first_eval = engine.evaluate(inside, now_ms=1000)
        second_eval = engine.evaluate(outside, now_ms=2000)

        self.assertTrue(first_eval.output_states["DO1"])
        self.assertFalse(second_eval.output_states["DO1"])

    def test_gate_rule_holds_output_while_mouse_is_in_roi(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-gate",
                    rule_type="roi_occupancy",
                    output_id="DO1",
                    mode="gate",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )
        outside = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(40.0, 40.0),
                    bbox=(30.0, 30.0, 50.0, 50.0),
                )
            ],
        )

        self.assertTrue(engine.evaluate(inside, now_ms=1000).output_states["DO1"])
        self.assertFalse(engine.evaluate(outside, now_ms=2000).output_states["DO1"])

    def test_roi_rule_stays_active_through_missing_segmentation_until_actual_exit(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-gate",
                    rule_type="roi_occupancy",
                    output_id="DO1",
                    mode="gate",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )
        missing_segmentation = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[],
        )
        outside = LiveDetectionResult(
            frame_index=3,
            timestamp_s=3.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(40.0, 40.0),
                    bbox=(30.0, 30.0, 50.0, 50.0),
                )
            ],
        )

        self.assertTrue(engine.evaluate(inside, now_ms=1000).output_states["DO1"])
        self.assertTrue(engine.evaluate(missing_segmentation, now_ms=1500).output_states["DO1"])
        self.assertFalse(engine.evaluate(outside, now_ms=2000).output_states["DO1"])

    def test_pulse_rule_triggers_only_on_rising_edge(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-pulse",
                    rule_type="roi_occupancy",
                    output_id="DO2",
                    mode="pulse",
                    duration_ms=150,
                    pulse_count=1,
                    pulse_frequency_hz=1.0,
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )

        first_eval = engine.evaluate(inside, now_ms=1000)
        second_eval = engine.evaluate(inside, now_ms=1050)
        after_pulse = engine.evaluate(inside, now_ms=1200)

        self.assertEqual(first_eval.triggered_pulses, [("DO2", 150, 1, 1.0)])
        self.assertEqual(second_eval.triggered_pulses, [])
        self.assertTrue(second_eval.output_states["DO2"])
        self.assertFalse(after_pulse.output_states["DO2"])
        self.assertFalse(first_eval.level_output_states["DO2"])

    def test_pulse_train_rule_respects_count_and_frequency(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-pulse-train",
                    rule_type="roi_occupancy",
                    output_id="DO2",
                    mode="pulse",
                    duration_ms=100,
                    pulse_count=3,
                    pulse_frequency_hz=5.0,
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )

        first_eval = engine.evaluate(inside, now_ms=1000)
        first_gap = engine.evaluate(inside, now_ms=1150)
        second_pulse = engine.evaluate(inside, now_ms=1200)
        done = engine.evaluate(inside, now_ms=1500)

        self.assertEqual(first_eval.triggered_pulses, [("DO2", 100, 3, 5.0)])
        self.assertTrue(first_eval.output_states["DO2"])
        self.assertFalse(first_gap.output_states["DO2"])
        self.assertTrue(second_pulse.output_states["DO2"])
        self.assertFalse(done.output_states["DO2"])

    def test_exit_pulse_rule_triggers_on_falling_edge(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-exit-pulse",
                    rule_type="roi_occupancy",
                    output_id="DO2",
                    mode="pulse",
                    duration_ms=100,
                    pulse_count=1,
                    pulse_frequency_hz=1.0,
                    activation_pattern="exit",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )
        outside = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(40.0, 40.0),
                    bbox=(30.0, 30.0, 50.0, 50.0),
                )
            ],
        )

        entry_eval = engine.evaluate(inside, now_ms=1000)
        exit_eval = engine.evaluate(outside, now_ms=1100)

        self.assertEqual(entry_eval.triggered_pulses, [])
        self.assertEqual(exit_eval.triggered_pulses, [("DO2", 100, 1, 1.0)])
        self.assertTrue(exit_eval.output_states["DO2"])

    def test_continuous_pulse_rule_retriggers_after_inter_train_interval(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-continuous-pulse",
                    rule_type="roi_occupancy",
                    output_id="DO2",
                    mode="pulse",
                    duration_ms=100,
                    pulse_count=2,
                    pulse_frequency_hz=10.0,
                    inter_train_interval_ms=300,
                    activation_pattern="continuous",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )
        outside = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(40.0, 40.0),
                    bbox=(30.0, 30.0, 50.0, 50.0),
                )
            ],
        )
        missing_segmentation = LiveDetectionResult(
            frame_index=3,
            timestamp_s=3.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[],
        )

        first_train = engine.evaluate(inside, now_ms=1000)
        before_iti = engine.evaluate(inside, now_ms=1499)
        next_train = engine.evaluate(missing_segmentation, now_ms=1500)
        exit_eval = engine.evaluate(outside, now_ms=1700)
        reentry_train = engine.evaluate(inside, now_ms=1800)

        self.assertEqual(first_train.triggered_pulses, [("DO2", 100, 2, 10.0)])
        self.assertEqual(before_iti.triggered_pulses, [])
        self.assertEqual(next_train.triggered_pulses, [("DO2", 100, 2, 10.0)])
        self.assertEqual(exit_eval.triggered_pulses, [])
        self.assertEqual(reentry_train.triggered_pulses, [("DO2", 100, 2, 10.0)])

    def test_continuous_pulse_rule_retriggers_from_same_last_result_snapshot(self):
        engine = LiveRuleEngine()
        engine.set_rois({"open_arm": BehaviorROI(name="open_arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-continuous-pulse-same-result",
                    rule_type="roi_occupancy",
                    output_id="DO2",
                    mode="pulse",
                    duration_ms=100,
                    pulse_count=1,
                    pulse_frequency_hz=1.0,
                    inter_train_interval_ms=300,
                    activation_pattern="continuous",
                    mouse_id=1,
                    roi_name="open_arm",
                )
            ]
        )

        inside = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=100,
            height=100,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(10.0, 10.0),
                    bbox=(0.0, 0.0, 20.0, 20.0),
                )
            ],
        )

        first_train = engine.evaluate(inside, now_ms=1000)
        before_iti = engine.evaluate(inside, now_ms=1399)
        next_train = engine.evaluate(inside, now_ms=1400)

        self.assertEqual(first_train.triggered_pulses, [("DO2", 100, 1, 1.0)])
        self.assertEqual(before_iti.triggered_pulses, [])
        self.assertEqual(next_train.triggered_pulses, [("DO2", 100, 1, 1.0)])

    def test_gate_proximity_rule_holds_output_while_two_mice_are_close(self):
        engine = LiveRuleEngine()
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="prox",
                    rule_type="mouse_proximity",
                    output_id="DO3",
                    mode="gate",
                    mouse_id=1,
                    peer_mouse_id=2,
                    distance_px=20.0,
                )
            ]
        )

        close_result = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=120,
            height=120,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(mouse_id=1, class_id=0, confidence=0.9, center=(20.0, 20.0), bbox=(10.0, 10.0, 30.0, 30.0)),
                TrackedMouseState(mouse_id=2, class_id=1, confidence=0.9, center=(30.0, 25.0), bbox=(20.0, 15.0, 40.0, 35.0)),
            ],
        )

        far_result = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=120,
            height=120,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(mouse_id=1, class_id=0, confidence=0.9, center=(20.0, 20.0), bbox=(10.0, 10.0, 30.0, 30.0)),
                TrackedMouseState(mouse_id=2, class_id=1, confidence=0.9, center=(80.0, 85.0), bbox=(70.0, 75.0, 90.0, 95.0)),
            ],
        )
        missing_peer = LiveDetectionResult(
            frame_index=3,
            timestamp_s=3.0,
            width=120,
            height=120,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(mouse_id=1, class_id=0, confidence=0.9, center=(20.0, 20.0), bbox=(10.0, 10.0, 30.0, 30.0)),
            ],
        )

        self.assertTrue(engine.evaluate(close_result, now_ms=1000).output_states["DO3"])
        self.assertTrue(engine.evaluate(missing_peer, now_ms=1500).output_states["DO3"])
        self.assertFalse(engine.evaluate(far_result, now_ms=2000).output_states["DO3"])

    def test_mask_contact_rule_triggers_when_masks_touch(self):
        engine = LiveRuleEngine()
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="mask-touch",
                    rule_type="mask_contact",
                    output_id="DO4",
                    mode="pulse",
                    duration_ms=100,
                    pulse_count=1,
                    pulse_frequency_hz=1.0,
                    mouse_id=1,
                    peer_mouse_id=2,
                )
            ]
        )

        touch_a = np.zeros((6, 6), dtype=bool)
        touch_b = np.zeros((6, 6), dtype=bool)
        touch_a[2:4, 1:3] = True
        touch_b[2:4, 3:5] = True

        separate_b = np.zeros((6, 6), dtype=bool)
        separate_b[2:4, 4:6] = True

        touching = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=6,
            height=6,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(1.5, 2.5),
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    mask=touch_a,
                ),
                TrackedMouseState(
                    mouse_id=2,
                    class_id=0,
                    confidence=0.9,
                    center=(3.5, 2.5),
                    bbox=(3.0, 2.0, 5.0, 4.0),
                    mask=touch_b,
                ),
            ],
        )
        separated = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=6,
            height=6,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(1.5, 2.5),
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    mask=touch_a,
                ),
                TrackedMouseState(
                    mouse_id=2,
                    class_id=0,
                    confidence=0.9,
                    center=(4.5, 2.5),
                    bbox=(4.0, 2.0, 6.0, 4.0),
                    mask=separate_b,
                ),
            ],
        )

        entry_eval = engine.evaluate(touching, now_ms=1000)
        still_touching = engine.evaluate(touching, now_ms=1050)
        separated_eval = engine.evaluate(separated, now_ms=1100)
        reentry_eval = engine.evaluate(touching, now_ms=1200)

        self.assertEqual(entry_eval.triggered_pulses, [("DO4", 100, 1, 1.0)])
        self.assertEqual(still_touching.triggered_pulses, [])
        self.assertFalse(separated_eval.output_states["DO4"])
        self.assertEqual(reentry_eval.triggered_pulses, [("DO4", 100, 1, 1.0)])

    def test_mask_contact_rule_holds_previous_truth_when_masks_temporarily_missing(self):
        engine = LiveRuleEngine()
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="mask-touch-gate",
                    rule_type="mask_contact",
                    output_id="DO4",
                    mode="gate",
                    mouse_id=1,
                    peer_mouse_id=2,
                )
            ]
        )

        mask_a = np.zeros((6, 6), dtype=bool)
        mask_b = np.zeros((6, 6), dtype=bool)
        separate_mask_b = np.zeros((6, 6), dtype=bool)
        mask_a[2:4, 1:3] = True
        mask_b[2:4, 3:5] = True
        separate_mask_b[2:4, 4:6] = True

        touching = LiveDetectionResult(
            frame_index=1,
            timestamp_s=1.0,
            width=6,
            height=6,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(1.5, 2.5),
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    mask=mask_a,
                ),
                TrackedMouseState(
                    mouse_id=2,
                    class_id=0,
                    confidence=0.9,
                    center=(3.5, 2.5),
                    bbox=(3.0, 2.0, 5.0, 4.0),
                    mask=mask_b,
                ),
            ],
        )
        missing_masks = LiveDetectionResult(
            frame_index=2,
            timestamp_s=2.0,
            width=6,
            height=6,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(1.5, 2.5),
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    mask=None,
                ),
                TrackedMouseState(
                    mouse_id=2,
                    class_id=0,
                    confidence=0.9,
                    center=(3.5, 2.5),
                    bbox=(3.0, 2.0, 5.0, 4.0),
                    mask=None,
                ),
            ],
        )
        separated = LiveDetectionResult(
            frame_index=3,
            timestamp_s=3.0,
            width=6,
            height=6,
            inference_ms=5.0,
            tracked_mice=[
                TrackedMouseState(
                    mouse_id=1,
                    class_id=0,
                    confidence=0.9,
                    center=(1.5, 2.5),
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    mask=mask_a,
                ),
                TrackedMouseState(
                    mouse_id=2,
                    class_id=0,
                    confidence=0.9,
                    center=(4.5, 2.5),
                    bbox=(4.0, 2.0, 6.0, 4.0),
                    mask=separate_mask_b,
                ),
            ],
        )

        self.assertTrue(engine.evaluate(touching, now_ms=1000).output_states["DO4"])
        self.assertTrue(engine.evaluate(missing_masks, now_ms=1500).output_states["DO4"])
        self.assertFalse(engine.evaluate(separated, now_ms=2000).output_states["DO4"])


if __name__ == "__main__":
    unittest.main()
