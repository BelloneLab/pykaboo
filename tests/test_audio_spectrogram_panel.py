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


def test_pettersson_wasapi_is_default_audio_device():
    from audio_recorder import AudioInputDevice, pick_default_device

    usb_camera = AudioInputDevice(
        index=1,
        name="Microphone (USB Live camera audio)",
        max_input_channels=2,
        default_samplerate=44_100.0,
        hostapi_index=0,
        hostapi_name="MME",
    )
    pettersson_mme = AudioInputDevice(
        index=3,
        name="Microphone (5- Pettersson M500-384kHz USB Ultrasound Microphone)",
        max_input_channels=1,
        default_samplerate=44_100.0,
        hostapi_index=0,
        hostapi_name="MME",
    )
    pettersson_wasapi = AudioInputDevice(
        index=23,
        name="Microphone (5- Pettersson M500-384kHz USB Ultrasound Microphone)",
        max_input_channels=1,
        default_samplerate=384_000.0,
        hostapi_index=2,
        hostapi_name="Windows WASAPI",
    )

    assert pick_default_device([usb_camera, pettersson_mme, pettersson_wasapi]) == pettersson_wasapi


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
        self.assertEqual(panel._spec_max_khz, 80.0)
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
        self.assertIn("80 kHz needs >= 160000 Hz", panel.preview_stats_label.text())

        # Switching to a Pettersson-style rate must reconfigure cleanly.
        panel.recorder._samplerate = 250_000
        panel._process_spectrogram_chunk(
            np.zeros(25_000, dtype=np.float32)
        )
        self.assertEqual(panel._spec_configured_rate, 250_000)
        self.assertEqual(panel._spec_max_khz, 80.0)
        self.assertTrue(np.all(np.isfinite(panel._spec_img)))

    def test_silence_stays_at_floor(self):
        panel = _make_panel()
        panel.recorder._samplerate = 192_000
        panel._configure_spectrogram()
        panel._process_spectrogram_chunk(np.zeros(192_00, dtype=np.float32))
        img = panel._spec_img
        self.assertTrue(np.all(img <= panel._SPEC_CEIL_DB))

    def test_panel_reselects_pettersson_before_monitoring(self):
        from audio_recorder import AudioInputDevice

        panel = _make_panel()
        usb_camera = AudioInputDevice(
            index=1,
            name="Microphone (USB Live camera audio)",
            max_input_channels=2,
            default_samplerate=44_100.0,
            hostapi_index=0,
            hostapi_name="MME",
        )
        pettersson = AudioInputDevice(
            index=23,
            name="Microphone (5- Pettersson M500-384kHz USB Ultrasound Microphone)",
            max_input_channels=1,
            default_samplerate=384_000.0,
            hostapi_index=2,
            hostapi_name="Windows WASAPI",
        )

        panel._devices = [pettersson, usb_camera]
        panel.device_combo.blockSignals(True)
        panel.device_combo.clear()
        panel.device_combo.addItem(pettersson.display, pettersson.index)
        panel.device_combo.addItem(usb_camera.display, usb_camera.index)
        panel.device_combo.setCurrentIndex(1)
        panel.device_combo.blockSignals(False)
        panel._apply_device(usb_camera)

        self.assertTrue(panel._prefer_pettersson_if_available())
        self.assertEqual(panel._current_device.index, 23)
        self.assertEqual(panel.device_combo.currentIndex(), 0)
        self.assertEqual(panel.sr_spin.value(), 384_000)


if __name__ == "__main__":
    unittest.main()
