import unittest

from live_detection_logic import LiveRuleEngine
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

        self.assertEqual(first_eval.triggered_pulses, [("DO2", 150)])
        self.assertEqual(second_eval.triggered_pulses, [])
        self.assertTrue(second_eval.output_states["DO2"])
        self.assertFalse(after_pulse.output_states["DO2"])

    def test_proximity_rule_activates_when_two_mice_are_close(self):
        engine = LiveRuleEngine()
        engine.set_rules(
            [
                LiveTriggerRule(
                    rule_id="prox",
                    rule_type="mouse_proximity",
                    output_id="DO3",
                    mode="level",
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

        self.assertTrue(engine.evaluate(close_result, now_ms=1000).output_states["DO3"])
        self.assertFalse(engine.evaluate(far_result, now_ms=2000).output_states["DO3"])


if __name__ == "__main__":
    unittest.main()
