import math
import unittest

from user_flag_utils import project_user_flag_events


class UserFlagProjectionTests(unittest.TestCase):
    def test_projects_events_to_nearest_frames_and_pulse_window(self):
        projected = project_user_flag_events(
            [10.00, 10.05, 10.10, 10.15, 10.20],
            [
                {"timestamp_software": 10.06, "pulse_ms": 70, "label": "inj", "shortcut": "Space", "output_id": "DO8"},
                {"timestamp_software": 10.18, "pulse_ms": 20, "label": "stim", "shortcut": "S", "output_id": "DO2"},
            ],
        )

        self.assertEqual(projected["event"].tolist(), [0, 1, 0, 0, 1])
        self.assertEqual(projected["ttl"].tolist(), [0, 1, 1, 0, 1])
        self.assertEqual(projected["count"].tolist(), [0, 1, 1, 1, 2])
        self.assertTrue(math.isnan(projected["event_timestamp"][0]))
        self.assertAlmostEqual(projected["event_timestamp"][1], 10.06)
        self.assertAlmostEqual(projected["event_timestamp"][4], 10.18)
        self.assertEqual(projected["event_label"].tolist(), ["", "inj", "", "", "stim"])
        self.assertEqual(projected["event_shortcut"].tolist(), ["", "Space", "", "", "S"])
        self.assertEqual(projected["event_output"].tolist(), ["", "DO8", "", "", "DO2"])
        self.assertTrue(math.isnan(projected["event_pulse_ms"][0]))
        self.assertEqual(projected["event_pulse_ms"].tolist()[1], 70.0)
        self.assertEqual(projected["event_pulse_ms"].tolist()[4], 20.0)

    def test_returns_zeroed_columns_without_valid_frame_times(self):
        projected = project_user_flag_events(
            ["bad", None],
            [{"timestamp_software": 10.0, "pulse_ms": 50}],
        )

        self.assertEqual(projected["event"].tolist(), [0, 0])
        self.assertEqual(projected["ttl"].tolist(), [0, 0])
        self.assertEqual(projected["count"].tolist(), [0, 0])
        self.assertTrue(math.isnan(projected["event_timestamp"][0]))
        self.assertTrue(math.isnan(projected["event_timestamp"][1]))
        self.assertEqual(projected["event_label"].tolist(), ["", ""])
        self.assertEqual(projected["event_shortcut"].tolist(), ["", ""])
        self.assertEqual(projected["event_output"].tolist(), ["", ""])
        self.assertTrue(math.isnan(projected["event_pulse_ms"][0]))
        self.assertTrue(math.isnan(projected["event_pulse_ms"][1]))
