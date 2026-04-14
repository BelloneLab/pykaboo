import unittest

from recording_timing import (
    build_recording_timing_warnings,
    capture_duration_seconds,
    measured_capture_fps,
    percent_delta,
)


class RecordingTimingTests(unittest.TestCase):
    def test_capture_duration_clamps_negative_spans(self):
        self.assertEqual(capture_duration_seconds(10.0, 8.0), 0.0)
        self.assertEqual(capture_duration_seconds(10.0, 12.5), 2.5)
        self.assertIsNone(capture_duration_seconds(None, 12.5))

    def test_measured_capture_fps_uses_wallclock_duration(self):
        self.assertEqual(measured_capture_fps(900, 20.0), 45.0)
        self.assertIsNone(measured_capture_fps(0, 20.0))
        self.assertIsNone(measured_capture_fps(900, 0.0))

    def test_percent_delta_reports_signed_percent_difference(self):
        self.assertAlmostEqual(percent_delta(30.0, 45.0), 50.0)
        self.assertAlmostEqual(percent_delta(40.0, 30.0), -25.0)
        self.assertIsNone(percent_delta(0.0, 30.0))

    def test_build_recording_timing_warnings_flags_duration_and_fps_drift(self):
        warnings = build_recording_timing_warnings(
            requested_duration_s=30,
            capture_duration_s=20.0,
            encoded_fps=30.0,
            measured_fps=45.0,
            audio_duration_s=20.0,
        )
        self.assertTrue(any("capture duration" in warning.lower() for warning in warnings))
        self.assertTrue(any("encoded fps" in warning.lower() for warning in warnings))

    def test_build_recording_timing_warnings_ignores_small_deviations(self):
        warnings = build_recording_timing_warnings(
            requested_duration_s=30,
            capture_duration_s=30.1,
            encoded_fps=30.0,
            measured_fps=30.8,
            audio_duration_s=30.2,
            duration_tolerance_s=0.25,
            fps_tolerance_pct=5.0,
            audio_tolerance_s=0.25,
        )
        self.assertEqual(warnings, [])