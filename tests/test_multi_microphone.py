"""Tests for multi-microphone support in the ultrasound panel.

These exercise the new fan-out logic (per-mic WAV paths, armed-stream
enumeration, add/remove slots, focus switching) using synthetic devices, so
they never touch real hardware or open a PortAudio stream.
"""
import os
import unittest

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None

from audio_recorder import AudioInputDevice, sanitize_mic_label


PETTERSSON = AudioInputDevice(
    index=23,
    name="Microphone (Pettersson M500-384kHz USB Ultrasound Microphone)",
    max_input_channels=1,
    default_samplerate=384_000.0,
    hostapi_index=2,
    hostapi_name="Windows WASAPI",
)
PETTERSSON_B = AudioInputDevice(
    index=24,
    name="Microphone (2- Pettersson M500-384kHz USB Ultrasound Microphone)",
    max_input_channels=1,
    default_samplerate=384_000.0,
    hostapi_index=2,
    hostapi_name="Windows WASAPI",
)
USB_MIC = AudioInputDevice(
    index=1,
    name="Microphone (USB Live camera audio)",
    max_input_channels=2,
    default_samplerate=44_100.0,
    hostapi_index=0,
    hostapi_name="MME",
)


class SanitizeLabelTests(unittest.TestCase):
    def test_sanitize_strips_punctuation_and_truncates(self):
        self.assertEqual(
            sanitize_mic_label("Microphone (Pettersson M500)!", "mic2"),
            "Microphone_Pettersson_M500",
        )
        self.assertEqual(sanitize_mic_label("", "mic5"), "mic5")
        self.assertEqual(sanitize_mic_label("***", "mic9"), "mic9")
        self.assertLessEqual(len(sanitize_mic_label("x" * 200, "f")), 40)


@unittest.skipIf(QApplication is None, "PySide6 is required")
class MultiMicPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self, devices):
        from audio_recorder import UltrasoundPanel

        panel = UltrasoundPanel()
        # Keep the test hermetic: never write to the real QSettings and drop any
        # auto-restored slots so every test starts from just the primary mic.
        panel._save_mic_config = lambda: None
        for slot in list(panel._extra_slots):
            panel.remove_microphone(slot)
        panel.enable_check.setChecked(False)
        panel._devices = list(devices)
        return panel

    def test_stream_wav_path_primary_vs_extra(self):
        from audio_recorder import UltrasoundPanel

        self.assertEqual(
            UltrasoundPanel._stream_wav_path("C:/data/trial01.wav", ""),
            "C:/data/trial01.wav",
        )
        self.assertEqual(
            UltrasoundPanel._stream_wav_path("C:/data/trial01.wav", "Pettersson_2"),
            "C:/data/trial01__Pettersson_2.wav",
        )

    def test_add_and_remove_microphone(self):
        panel = self._panel([PETTERSSON, PETTERSSON_B, USB_MIC])
        primary_recorder = panel._primary_recorder

        slot_a = panel.add_microphone(PETTERSSON_B)
        slot_b = panel.add_microphone(USB_MIC)
        self.assertIsNotNone(slot_a)
        self.assertIsNotNone(slot_b)
        self.assertEqual(len(panel._extra_slots), 2)
        # Each mic owns an independent recorder.
        recorders = {id(primary_recorder), id(slot_a.recorder), id(slot_b.recorder)}
        self.assertEqual(len(recorders), 3)
        self.assertEqual(slot_b.current_device.index, USB_MIC.index)

        panel.remove_microphone(slot_a)
        self.assertEqual(len(panel._extra_slots), 1)
        self.assertIs(panel._extra_slots[0], slot_b)

    def test_armed_streams_unique_labels(self):
        panel = self._panel([PETTERSSON, PETTERSSON_B])
        panel._current_device = PETTERSSON  # arm the primary mic

        slot_a = panel.add_microphone(PETTERSSON_B)
        slot_b = panel.add_microphone(PETTERSSON_B)  # same device name on purpose

        streams = panel._armed_streams()
        labels = [label for label, _rec in streams]
        recorders = [rec for _label, rec in streams]
        # Primary first with an empty label; extras get unique suffixes.
        self.assertEqual(labels[0], "")
        self.assertIs(recorders[0], panel._primary_recorder)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3, f"labels must be unique: {labels}")
        self.assertIs(recorders[1], slot_a.recorder)
        self.assertIs(recorders[2], slot_b.recorder)

    def test_unchecked_rec_excludes_slot(self):
        panel = self._panel([PETTERSSON, PETTERSSON_B])
        panel._current_device = PETTERSSON
        slot = panel.add_microphone(PETTERSSON_B)
        slot.rec_check.setChecked(False)
        recorders = [rec for _label, rec in panel._armed_streams()]
        self.assertNotIn(slot.recorder, recorders)
        self.assertIn(panel._primary_recorder, recorders)

    def test_focus_switches_shared_recorder(self):
        panel = self._panel([PETTERSSON, PETTERSSON_B])
        self.assertIs(panel.recorder, panel._primary_recorder)
        slot = panel.add_microphone(PETTERSSON_B)

        slot.focus_radio.setChecked(True)
        self.assertIs(panel.recorder, slot.recorder)

        panel.primary_focus_radio.setChecked(True)
        self.assertIs(panel.recorder, panel._primary_recorder)

    def test_is_enabled_requires_master_and_armed_mic(self):
        panel = self._panel([PETTERSSON])
        panel._current_device = None
        panel.enable_check.setChecked(True)
        # Master on but nothing armed yet -> not enabled.
        self.assertFalse(panel.is_enabled())
        panel._current_device = PETTERSSON
        self.assertTrue(panel.is_enabled())
        panel.enable_check.setChecked(False)
        self.assertFalse(panel.is_enabled())


if __name__ == "__main__":
    unittest.main()
