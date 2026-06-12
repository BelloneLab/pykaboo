"""Stress tests for the UltrasoundPanel live spectrogram engine.

Feeds synthetic tone streams through _process_spectrogram_chunk at the
sample rates this lab actually uses (48 kHz USB mics, 250/384 kHz
Pettersson) with pathological chunk sizes, and asserts the rolling image
stays well-formed and the injected tone lands in the right kHz row.
"""
import unittest

import numpy as np

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None


def _make_panel():
    from audio_recorder import UltrasoundPanel

    return UltrasoundPanel()


@unittest.skipIf(QApplication is None, "PySide6 is required")
class SpectrogramEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _run_stream(self, panel, samplerate: int, tone_hz: float, chunk_sizes):
        panel.recorder._samplerate = samplerate
        panel._configure_spectrogram()
        phase = 0.0
        step = 2.0 * np.pi * tone_hz / samplerate
        for size in chunk_sizes:
            chunk = 0.5 * np.sin(phase + step * np.arange(size, dtype=np.float64))
            phase += step * size
            panel._process_spectrogram_chunk(chunk.astype(np.float32))

    def test_image_shape_and_tone_row_at_384k(self):
        panel = _make_panel()
        tone_hz = 70_000.0  # typical USV band
        # Mixed chunk sizes incl. tiny, prime, and oversized bursts.
        sizes = [7, 1024, 38400, 5003, 38400, 38400, 99991, 38400]
        self._run_stream(panel, 384_000, tone_hz, sizes)

        img = panel._spec_img
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (panel._spec_rows, panel._spec_cols))
        self.assertTrue(np.all(np.isfinite(img)))

        # The brightest recent column content must sit at the tone frequency.
        recent = img[:, -40:]
        peak_row = int(np.argmax(recent.max(axis=1)))
        khz_per_row = panel._spec_max_khz / panel._spec_rows
        peak_khz = (peak_row + 0.5) * khz_per_row
        self.assertAlmostEqual(peak_khz, tone_hz / 1000.0, delta=3.0)

    def test_low_rate_microphone_and_reconfigure(self):
        panel = _make_panel()
        self._run_stream(panel, 48_000, 8_000.0, [480, 4800, 4801, 4799, 9600])
        img = panel._spec_img
        self.assertEqual(img.shape, (panel._spec_rows, panel._spec_cols))
        self.assertLessEqual(panel._spec_max_khz, 24.0)

        # Switching to a Pettersson-style rate must reconfigure cleanly.
        panel.recorder._samplerate = 250_000
        panel._process_spectrogram_chunk(
            np.zeros(25_000, dtype=np.float32)
        )
        self.assertEqual(panel._spec_configured_rate, 250_000)
        self.assertTrue(np.all(np.isfinite(panel._spec_img)))

    def test_silence_stays_at_floor(self):
        panel = _make_panel()
        panel.recorder._samplerate = 192_000
        panel._configure_spectrogram()
        panel._process_spectrogram_chunk(np.zeros(192_00, dtype=np.float32))
        img = panel._spec_img
        self.assertTrue(np.all(img <= panel._SPEC_CEIL_DB))


if __name__ == "__main__":
    unittest.main()
