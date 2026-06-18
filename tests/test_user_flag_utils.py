import math
import unittest
from types import SimpleNamespace

import pandas as pd

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


class UserFlagTriggerTests(unittest.TestCase):
    def _make_window_shell(self, *, recording=True):
        from main_window_enhanced import MainWindow

        window = MainWindow.__new__(MainWindow)
        window.user_flag_configs = [
            {
                "flag_id": "flag-social",
                "label": "Social contact",
                "shortcut": "F",
                "output_id": "DO2",
                "pulse_ms": 80,
            }
        ]
        window.user_flag_events = []
        window.user_flag_preview_text = ""
        window.user_flag_preview_until_s = 0.0
        window.worker = SimpleNamespace(is_recording=bool(recording))
        window.is_arduino_connected = False
        window.arduino_worker = None
        window.is_camera_connected = True
        window.last_frame_size = (1600, 1333)
        window.live_output_mapping = {}
        window._focused_widget_blocks_space_record = lambda: False
        window.status_messages = []
        window._on_status_update = window.status_messages.append
        return window

    def test_trigger_user_flag_records_manual_event_while_recording(self):
        from main_window_enhanced import MainWindow

        window = self._make_window_shell(recording=True)
        MainWindow._trigger_user_flag(window, "flag-social")

        self.assertEqual(len(window.user_flag_events), 1)
        event = window.user_flag_events[0]
        self.assertEqual(event["flag_id"], "flag-social")
        self.assertEqual(event["label"], "Social contact")
        self.assertEqual(event["shortcut"], "F")
        self.assertEqual(event["output_id"], "DO2")
        self.assertEqual(event["pulse_ms"], 80)
        self.assertEqual(event["count"], 1)
        self.assertIn("marked", window.status_messages[-1])
        self.assertEqual(window.user_flag_preview_text, "FLAG: Social contact")

    def test_trigger_user_flag_preview_does_not_create_recording_event(self):
        from main_window_enhanced import MainWindow

        window = self._make_window_shell(recording=False)
        MainWindow._trigger_user_flag(window, "flag-social")

        self.assertEqual(window.user_flag_events, [])
        self.assertIn("previewed", window.status_messages[-1])
        self.assertEqual(window.user_flag_preview_text, "FLAG: Social contact")

    def test_recorded_user_flag_exports_to_nearest_frame_columns(self):
        from main_window_enhanced import MainWindow

        window = self._make_window_shell(recording=True)
        window.metadata = {"user_flags": window.user_flag_configs}
        window.user_flag_events = [
            {
                "timestamp_software": 10.061,
                "flag_id": "flag-social",
                "label": "Social contact",
                "shortcut": "F",
                "output_id": "DO2",
                "pulse_ms": 80,
                "count": 1,
            }
        ]
        frame_df = pd.DataFrame(
            {
                "frame_id": [1, 2, 3],
                "timestamp_software": [10.00, 10.05, 10.10],
            }
        )

        merged = MainWindow._merge_user_flag_events_into_frame_df(window, frame_df)

        self.assertEqual(merged["user_flag_event"].tolist(), [0, 1, 0])
        self.assertEqual(merged["user_flag_ttl"].tolist(), [0, 1, 1])
        self.assertEqual(merged["user_flag_count"].tolist(), [0, 1, 1])
        self.assertEqual(merged.loc[1, "user_flag_event_label"], "Social contact")
        self.assertEqual(merged.loc[1, "user_flag_event_shortcut"], "F")
        self.assertEqual(merged.loc[1, "user_flag_event_output"], "DO2")
        self.assertEqual(float(merged.loc[1, "user_flag_event_pulse_ms"]), 80.0)
