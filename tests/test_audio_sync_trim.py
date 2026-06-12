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

    def test_pads_short_capture_to_exact_video_duration(self):
        recorder = UltrasoundRecorder()
        recorder._samplerate = 1000
        recorder._channels = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "short.wav"
            # Only 8 s of audio captured for a 9 s encoded video.
            source = np.zeros((9_000, 1), dtype=np.float32)
            sf.write(str(wav_path), source, 1000, subtype="PCM_16", format="WAV")

            metadata = recorder._trim_to_video_span(
                str(wav_path),
                samples_written=9_000,
                audio_t0=None,
                recording_begin=100.0,
                video_t0=101.0,
                video_t1=111.0,
                target_duration_seconds=9.0,
            )

            info = sf.info(str(wav_path))
            self.assertEqual(info.frames, 9000)
            self.assertEqual(metadata["trim_leading_samples"], 1000)
            self.assertEqual(metadata["pad_trailing_samples"], 1000)
            self.assertAlmostEqual(metadata["duration_seconds"], 9.0)
            self.assertTrue(metadata["trimmed"])

    def test_pad_only_when_no_leading_offset(self):
        recorder = UltrasoundRecorder()
        recorder._samplerate = 1000
        recorder._channels = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "padonly.wav"
            source = np.zeros((8_500, 1), dtype=np.float32)
            sf.write(str(wav_path), source, 1000, subtype="PCM_16", format="WAV")

            metadata = recorder._trim_to_video_span(
                str(wav_path),
                samples_written=8_500,
                audio_t0=None,
                recording_begin=100.0,
                video_t0=100.0,
                video_t1=None,
                target_duration_seconds=9.0,
            )

            info = sf.info(str(wav_path))
            self.assertEqual(info.frames, 9000)
            self.assertEqual(metadata["trim_leading_samples"], 0)
            self.assertEqual(metadata["pad_trailing_samples"], 500)
            self.assertAlmostEqual(metadata["duration_seconds"], 9.0)

    def test_leading_beyond_capture_keeps_file_untrimmed(self):
        recorder = UltrasoundRecorder()
        recorder._samplerate = 1000
        recorder._channels = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "lost.wav"
            source = np.zeros((2_000, 1), dtype=np.float32)
            sf.write(str(wav_path), source, 1000, subtype="PCM_16", format="WAV")

            # Video started 5 s after audio stopped: nothing aligned remains.
            metadata = recorder._trim_to_video_span(
                str(wav_path),
                samples_written=2_000,
                audio_t0=None,
                recording_begin=100.0,
                video_t0=105.0,
                video_t1=110.0,
                target_duration_seconds=5.0,
            )

            info = sf.info(str(wav_path))
            self.assertEqual(info.frames, 2000)  # untouched
            self.assertEqual(metadata["trim_skipped_reason"], "zero_length_trim")
            self.assertFalse(metadata["trimmed"])

    def test_first_block_wallclock_preferred_over_recording_begin(self):
        recorder = UltrasoundRecorder()
        recorder._samplerate = 1000
        recorder._channels = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "ref.wav"
            source = np.zeros((12_000, 1), dtype=np.float32)
            sf.write(str(wav_path), source, 1000, subtype="PCM_16", format="WAV")

            # recording_begin is half a second earlier than the first written
            # block; the leading trim must use the block stamp (0.5 s lead).
            metadata = recorder._trim_to_video_span(
                str(wav_path),
                samples_written=12_000,
                audio_t0=None,
                recording_begin=100.0,
                video_t0=101.0,
                video_t1=111.0,
                target_duration_seconds=9.0,
                first_block_wallclock=100.5,
            )

            info = sf.info(str(wav_path))
            self.assertEqual(info.frames, 9000)
            self.assertEqual(metadata["trim_leading_samples"], 500)
            self.assertAlmostEqual(metadata["duration_seconds"], 9.0)


if __name__ == "__main__":
    unittest.main()