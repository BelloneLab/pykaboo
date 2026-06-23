import unittest

import numpy as np

from live_detection_logic import (
    LiveRuleEngine,
    build_rule_label,
    format_roi_properties,
    occupied_roi_names,
    roi_geometry_properties,
)
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


class BehaviorClassRuleTests(unittest.TestCase):
    """The behavior_class rule type reads scene-level state pushed by the model worker."""

    def _rule(self, **kw):
        base = dict(
            rule_id="b1", rule_type="behavior_class", output_id="DO1",
            mode="gate", behavior_name="mounting",
        )
        base.update(kw)
        return LiveTriggerRule(**base)

    def test_roundtrip_serialization_keeps_behavior_name(self):
        rule = self._rule(behavior_name="anogenital")
        restored = LiveTriggerRule.from_dict(rule.to_dict())
        self.assertEqual(restored.rule_type, "behavior_class")
        self.assertEqual(restored.behavior_name, "anogenital")

    def test_roundtrip_keeps_subject_and_min_active(self):
        rule = self._rule(behavior_name="nose2anogenital", behavior_subject_id=2, min_active_ms=250)
        restored = LiveTriggerRule.from_dict(rule.to_dict())
        self.assertEqual(restored.behavior_subject_id, 2)
        self.assertEqual(restored.min_active_ms, 250)

    def test_subject_polarity_reads_per_track_actor(self):
        engine = LiveRuleEngine()
        rule_m1 = self._rule(rule_id="m1", behavior_name="nose2anogenital", behavior_subject_id=1)
        rule_m2 = self._rule(rule_id="m2", behavior_name="nose2anogenital", behavior_subject_id=2)
        rule_any = self._rule(rule_id="any", behavior_name="nose2anogenital", behavior_subject_id=0)
        engine.set_rules([rule_m1, rule_m2, rule_any])
        # Scene OR is True, but only mouse 1 is the actor performing the behavior.
        engine.set_behavior_state(
            {"nose2anogenital": True},
            {"nose2anogenital": 0.9},
            per_track={
                "1": {"binary": {"nose2anogenital": True}, "probs": {"nose2anogenital": 0.9}},
                "2": {"binary": {"nose2anogenital": False}, "probs": {"nose2anogenital": 0.1}},
            },
        )
        self.assertTrue(engine._rule_truth(rule_m1, {}))
        self.assertFalse(engine._rule_truth(rule_m2, {}))
        self.assertTrue(engine._rule_truth(rule_any, {}))  # subject 0 == scene level

    def test_subject_holds_when_per_track_missing(self):
        engine = LiveRuleEngine()
        rule = self._rule(behavior_name="nose2anogenital", behavior_subject_id=2)
        engine.set_rules([rule])
        engine.set_behavior_state(
            {"nose2anogenital": True}, {},
            per_track={"1": {"binary": {"nose2anogenital": True}}},
        )
        # subject 2 has no per-track decision -> hold (None), like a missing mouse
        self.assertIsNone(engine._rule_truth(rule, {}))

    def test_truth_holds_until_first_state_then_reads_scene(self):
        engine = LiveRuleEngine()
        rule = self._rule()
        engine.set_rules([rule])
        # before any model decision -> None (hold), so a gate output stays off
        self.assertIsNone(engine._rule_truth(rule, {}))
        self.assertFalse(engine.evaluate(None, now_ms=0).output_states["DO1"])

        engine.set_behavior_state({"mounting": True}, {"mounting": 0.91})
        self.assertTrue(engine._rule_truth(rule, {}))
        # an unknown class reads as False once state has been set at least once
        self.assertFalse(engine._rule_truth(self._rule(behavior_name="grooming"), {}))

        engine.set_behavior_state({"mounting": False}, {"mounting": 0.2})
        self.assertFalse(engine._rule_truth(rule, {}))

    def test_gate_output_tracks_behavior_state(self):
        engine = LiveRuleEngine()
        engine.set_rules([self._rule(mode="level")])
        engine.set_behavior_state({"mounting": True}, {"mounting": 0.9})
        self.assertTrue(engine.evaluate(None, now_ms=100).level_output_states["DO1"])
        engine.set_behavior_state({"mounting": False}, {"mounting": 0.1})
        self.assertFalse(engine.evaluate(None, now_ms=200).level_output_states["DO1"])

    def test_pulse_fires_once_on_behavior_onset(self):
        engine = LiveRuleEngine()
        engine.set_rules([self._rule(mode="pulse", activation_pattern="entry", duration_ms=100)])
        engine.set_behavior_state({"mounting": False}, {})
        self.assertEqual(len(engine.evaluate(None, now_ms=0).triggered_pulses), 0)
        engine.set_behavior_state({"mounting": True}, {})
        self.assertEqual(len(engine.evaluate(None, now_ms=50).triggered_pulses), 1)
        # still on -> no second pulse at entry
        self.assertEqual(len(engine.evaluate(None, now_ms=100).triggered_pulses), 0)

    def test_clear_runtime_state_forgets_behavior(self):
        engine = LiveRuleEngine()
        rule = self._rule()
        engine.set_rules([rule])
        engine.set_behavior_state({"mounting": True}, {})
        self.assertTrue(engine._rule_truth(rule, {}))
        engine.clear_runtime_state()
        self.assertIsNone(engine._rule_truth(rule, {}))


class MinActiveDurationTests(unittest.TestCase):
    """``min_active_ms`` gates a rule ON only after the condition holds long enough."""

    def _behavior_rule(self, **kw):
        base = dict(
            rule_id="d1", rule_type="behavior_class", output_id="DO1",
            mode="gate", behavior_name="nose2nose", min_active_ms=200,
        )
        base.update(kw)
        return LiveTriggerRule(**base)

    def test_gate_waits_for_min_duration_then_resets_on_drop(self):
        engine = LiveRuleEngine()
        engine.set_rules([self._behavior_rule()])
        engine.set_behavior_state({"nose2nose": True}, {})
        # Onset at t=1000: dwell starts, not yet 200 ms -> output stays low.
        self.assertFalse(engine.evaluate(None, now_ms=1000).output_states["DO1"])
        self.assertFalse(engine.evaluate(None, now_ms=1150).output_states["DO1"])
        # 200 ms elapsed -> fires.
        self.assertTrue(engine.evaluate(None, now_ms=1200).output_states["DO1"])
        # Behavior drops -> off and the dwell timer resets.
        engine.set_behavior_state({"nose2nose": False}, {})
        self.assertFalse(engine.evaluate(None, now_ms=1250).output_states["DO1"])
        # Re-onset must serve the full dwell again before firing.
        engine.set_behavior_state({"nose2nose": True}, {})
        self.assertFalse(engine.evaluate(None, now_ms=1300).output_states["DO1"])
        self.assertTrue(engine.evaluate(None, now_ms=1500).output_states["DO1"])

    def test_zero_min_active_fires_immediately(self):
        engine = LiveRuleEngine()
        engine.set_rules([self._behavior_rule(min_active_ms=0)])
        engine.set_behavior_state({"nose2nose": True}, {})
        self.assertTrue(engine.evaluate(None, now_ms=1000).output_states["DO1"])

    def test_min_active_applies_to_roi_dwell(self):
        engine = LiveRuleEngine()
        engine.set_rois({"arm": BehaviorROI(name="arm", roi_type="rectangle", data=[(0, 0, 25, 25)])})
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="roi-dwell", rule_type="roi_occupancy", output_id="DO1",
                    mode="gate", mouse_id=1, roi_name="arm", min_active_ms=500,
                )
            ]
        )
        inside = LiveDetectionResult(
            frame_index=1, timestamp_s=1.0, width=100, height=100, inference_ms=1.0,
            tracked_mice=[
                TrackedMouseState(mouse_id=1, class_id=0, confidence=0.9, center=(10.0, 10.0), bbox=(0.0, 0.0, 20.0, 20.0)),
            ],
        )
        self.assertFalse(engine.evaluate(inside, now_ms=1000).output_states["DO1"])
        self.assertFalse(engine.evaluate(inside, now_ms=1400).output_states["DO1"])
        self.assertTrue(engine.evaluate(inside, now_ms=1500).output_states["DO1"])


class RuleLabelTests(unittest.TestCase):
    def test_behavior_label_includes_subject_dwell_and_output_name(self):
        rule = LiveTriggerRule(
            rule_id="b", rule_type="behavior_class", output_id="DO1", mode="gate",
            behavior_name="nose2anogenital", behavior_subject_id=1, min_active_ms=200,
        )
        label = build_rule_label(rule, {"DO1": "Laser 473nm"})
        self.assertIn("nose2anogenital", label)
        self.assertIn("by M1", label)
        self.assertIn(">=200ms", label)
        self.assertIn("Laser 473nm (DO1)", label)

    def test_label_without_name_falls_back_to_do_id(self):
        rule = LiveTriggerRule(rule_id="b", rule_type="behavior_class", output_id="DO2",
                               mode="gate", behavior_name="mounting")
        label = build_rule_label(rule)
        self.assertIn("-> DO2", label)
        self.assertNotIn("by M", label)


if __name__ == "__main__":
    unittest.main()


class TtlTrailingHoldTests(unittest.TestCase):
    def _rule(self, **kw):
        return LiveTriggerRule(rule_id="r1", rule_type="roi_occupancy", output_id="DO1", **kw)

    def test_min_inactive_ms_holds_through_a_dropout(self):
        eng = LiveRuleEngine()
        rule = self._rule(min_active_ms=0, min_inactive_ms=200)
        self.assertTrue(eng._apply_min_active(rule, True, 0))     # qualifies ON
        self.assertTrue(eng._apply_min_active(rule, False, 100))  # dropout: held (100<200)
        self.assertTrue(eng._apply_min_active(rule, False, 250))  # still held (150<200)
        self.assertFalse(eng._apply_min_active(rule, False, 320))  # 220>=200 -> released

    def test_min_inactive_ms_re_arms_on_recovery(self):
        eng = LiveRuleEngine()
        rule = self._rule(min_active_ms=0, min_inactive_ms=200)
        self.assertTrue(eng._apply_min_active(rule, True, 0))
        self.assertTrue(eng._apply_min_active(rule, False, 50))   # brief dropout, held
        self.assertTrue(eng._apply_min_active(rule, True, 80))    # recovered -> stays ON, timer cleared
        self.assertTrue(eng._apply_min_active(rule, False, 250))  # new dropout starts fresh
        self.assertTrue(eng._apply_min_active(rule, False, 400))  # 150<200 still held
        self.assertFalse(eng._apply_min_active(rule, False, 470))  # 220>=200 -> released

    def test_default_zero_is_legacy_immediate_release(self):
        eng = LiveRuleEngine()
        rule = self._rule(min_active_ms=0, min_inactive_ms=0)
        self.assertTrue(eng._apply_min_active(rule, True, 0))
        self.assertFalse(eng._apply_min_active(rule, False, 1))   # legacy: immediate OFF

    def test_serialization_round_trips_min_inactive_ms(self):
        rule = self._rule(min_inactive_ms=350)
        back = LiveTriggerRule.from_dict(rule.to_dict())
        self.assertEqual(back.min_inactive_ms, 350)
