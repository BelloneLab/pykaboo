import unittest

import numpy as np

from audio_recorder import compress_waveform_for_display, level_to_dbfs


class AudioWaveformDisplayTests(unittest.TestCase):
    def test_level_to_dbfs_clamps_to_floor(self):
        self.assertEqual(level_to_dbfs(0.0, floor_db=-80.0), -80.0)
        self.assertAlmostEqual(level_to_dbfs(1.0, floor_db=-80.0), 0.0)

    def test_compress_waveform_returns_copy_when_already_small(self):
        waveform = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        compressed = compress_waveform_for_display(waveform, max_points=8)

        self.assertTrue(np.array_equal(compressed, waveform))
        self.assertIsNot(compressed, waveform)

    def test_compress_waveform_preserves_bucket_min_max(self):
        waveform = np.array(
            [0.1, 0.5, -0.2, 0.3, -0.9, 0.4, 0.2, -0.1], dtype=np.float32
        )

        compressed = compress_waveform_for_display(waveform, max_points=4)

        expected = np.array([-0.2, 0.5, -0.9, 0.4], dtype=np.float32)
        self.assertTrue(np.array_equal(compressed, expected))


if __name__ == "__main__":
    unittest.main()