"""Frame-count math that enforces exact recording durations.

These guard the conversion the acquisition thread uses to stop a recording at
an exact length (frames = round(fps * seconds)), so a file written at the same
fps plays back for exactly the requested duration regardless of GUI-thread lag.
"""
import unittest

from camera_worker import frames_for_duration


class FramesForDurationTests(unittest.TestCase):
    def test_exact_round_durations(self):
        self.assertEqual(frames_for_duration(30.0, 10.0), 300)
        self.assertEqual(frames_for_duration(60.0, 5.0), 300)
        self.assertEqual(frames_for_duration(25.0, 0.6), 15)  # 15.0 frames

    def test_file_playback_matches_request_within_one_frame(self):
        fps = 30.0
        seconds = 10.0
        n = frames_for_duration(fps, seconds)
        # File written at `fps` with `n` frames plays back for n/fps; this must
        # match the request to within a single frame interval.
        playback = n / fps
        self.assertLessEqual(abs(playback - seconds), 1.0 / fps)

    def test_unlimited_and_invalid_inputs_return_none(self):
        self.assertIsNone(frames_for_duration(30.0, None))
        self.assertIsNone(frames_for_duration(None, 10.0))
        self.assertIsNone(frames_for_duration(30.0, 0.0))
        self.assertIsNone(frames_for_duration(0.0, 10.0))
        self.assertIsNone(frames_for_duration(-30.0, 10.0))

    def test_minimum_one_frame(self):
        self.assertEqual(frames_for_duration(30.0, 0.001), 1)


if __name__ == "__main__":
    unittest.main()
