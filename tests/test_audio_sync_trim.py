import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_recorder import UltrasoundRecorder, sf


@unittest.skipIf(sf is None, "soundfile is required for trim tests")
class AudioSyncTrimTests(unittest.TestCase):
    def test_trim_uses_encoded_video_duration_when_available(self):
        recorder = UltrasoundRecorder()
        recorder._samplerate = 1000
        recorder._channels = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "sync.wav"
            source = np.zeros((12_000, 1), dtype=np.float32)
            sf.write(str(wav_path), source, 1000, subtype="PCM_16", format="WAV")

            metadata = recorder._trim_to_video_span(
                str(wav_path),
                samples_written=12_000,
                audio_t0=None,
                recording_begin=100.0,
                video_t0=101.0,
                video_t1=111.0,
                target_duration_seconds=9.0,
            )

            info = sf.info(str(wav_path))
            self.assertEqual(info.frames, 9000)
            self.assertEqual(metadata["trim_target"], "encoded_video_duration")
            self.assertEqual(metadata["trim_leading_samples"], 1000)
            self.assertEqual(metadata["trim_trailing_samples"], 2000)
            self.assertAlmostEqual(metadata["duration_seconds"], 9.0)


if __name__ == "__main__":
    unittest.main()