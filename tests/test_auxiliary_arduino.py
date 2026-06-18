"""Tests for auxiliary (multi-board) Arduino support.

These exercise the pure data layers that matter for recordings, with no serial
hardware: the device-roster round-trip through settings, the per-frame
history -> prefixed-column CSV merge, pin normalization, and per-frame sampling
with an injected fake board. Qt-touching cases use the QApplication pattern from
test_audio_spectrogram_panel.py.
"""
import unittest

try:
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from auxiliary_arduino import (
    ArduinoDeviceManager,
    AuxiliaryArduinoWorker,
    sanitize_column_token,
)


class FakeSettings:
    """Minimal QSettings stand-in so tests never touch the registry."""

    def __init__(self):
        self._data = {}

    def value(self, key, default=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value

    def sync(self):
        pass


def test_sanitize_column_token():
    assert sanitize_column_token("lick L") == "lick_L"
    assert sanitize_column_token("beam-break!") == "beam_break"
    assert sanitize_column_token("  __x__ ") == "x"


def test_normalize_pins_dedupes_labels_and_pins():
    pins = AuxiliaryArduinoWorker.normalize_pins([
        {"pin": 2, "label": "a", "role": "Input"},
        {"pin": 2, "label": "b", "role": "Output"},   # duplicate pin -> dropped
        {"pin": 3, "label": "a", "role": "Output"},     # duplicate label -> a_2
        {"pin": 4, "label": "", "role": "input"},       # blank label -> pin4
    ])
    assert [p["pin"] for p in pins] == [2, 3, 4]
    assert [p["key"] for p in pins] == ["a", "a_2", "pin4"]
    assert pins[0]["role"] == "Input"
    assert pins[1]["role"] == "Output"


@unittest.skipIf(QApplication is None, "PySide6 is required")
class AuxiliaryArduinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_roster_round_trips_through_settings(self):
        settings = FakeSettings()
        mgr = ArduinoDeviceManager(settings)
        mgr.add_device(
            name="Beam Box",
            port="COM9",
            pins=[
                {"pin": 2, "label": "beam", "role": "Input"},
                {"pin": 7, "label": "laser", "role": "Output"},
            ],
        )
        mgr.save()

        reloaded = ArduinoDeviceManager(settings)
        reloaded.load()
        devices = reloaded.devices()
        self.assertEqual(len(devices), 1)
        dev = devices[0]
        self.assertEqual(dev.name, "Beam Box")
        self.assertEqual(dev.port_name, "COM9")
        snap = dev.snapshot()["pins"]
        self.assertEqual(
            snap,
            [
                {"pin": 2, "label": "beam", "role": "Input"},
                {"pin": 7, "label": "laser", "role": "Output"},
            ],
        )

    def test_used_ports_excludes_self(self):
        settings = FakeSettings()
        mgr = ArduinoDeviceManager(settings)
        a = mgr.add_device(name="A", port="COM3")
        mgr.add_device(name="B", port="COM5")
        self.assertEqual(mgr.used_ports(), {"COM3", "COM5"})
        self.assertEqual(mgr.used_ports(exclude_id=a.device_id), {"COM5"})

    @unittest.skipIf(pd is None, "pandas is required")
    def test_merge_into_frame_df_prefixes_and_frame_aligns(self):
        settings = FakeSettings()
        mgr = ArduinoDeviceManager(settings)
        worker = mgr.add_device(
            name="Beam Box",
            device_id="2",
            pins=[
                {"pin": 2, "label": "lickL", "role": "Input"},
                {"pin": 3, "label": "lickR", "role": "Input"},
            ],
        )
        # Frames 0 and 1 sampled; frame 2 has no auxiliary sample.
        worker.history = [
            {"frame_id": 0, "lickL": 0, "lickR": 1},
            {"frame_id": 1, "lickL": 1, "lickR": 0},
        ]

        df = pd.DataFrame({"frame_id": [0, 1, 2], "x": [10, 11, 12]})
        merged = mgr.merge_into_frame_df(df)

        self.assertIn("dev2_lickL_ttl", merged.columns)
        self.assertIn("dev2_lickR_ttl", merged.columns)
        self.assertEqual(list(merged["dev2_lickL_ttl"]), [0, 1, 0])
        self.assertEqual(list(merged["dev2_lickR_ttl"]), [1, 0, 0])
        # Original columns are untouched and row count is preserved.
        self.assertEqual(list(merged["x"]), [10, 11, 12])

    def test_sample_state_records_labelled_pins(self):
        settings = FakeSettings()
        mgr = ArduinoDeviceManager(settings)
        worker = mgr.add_device(
            name="Box",
            device_id="2",
            pins=[
                {"pin": 2, "label": "beam", "role": "Input"},
                {"pin": 7, "label": "laser", "role": "Output"},
            ],
        )
        # Inject a connected board + known live states (no serial involved).
        worker.board = object()
        worker.input_states[2] = True
        worker.output_shadow[7] = False
        worker.start_recording()
        worker.sample_state({"frame_id": 5})
        worker.sample_state({"frame_id": 6})
        history = worker.get_history()
        self.assertEqual(
            history,
            [
                {"frame_id": 5, "beam": 1, "laser": 0},
                {"frame_id": 6, "beam": 1, "laser": 0},
            ],
        )
        worker.board = None

    def test_sample_state_noop_without_board(self):
        settings = FakeSettings()
        mgr = ArduinoDeviceManager(settings)
        worker = mgr.add_device(name="Box", pins=[{"pin": 2, "label": "beam", "role": "Input"}])
        worker.start_recording()
        worker.sample_state({"frame_id": 1})  # board is None -> no row
        self.assertEqual(worker.get_history(), [])


if __name__ == "__main__":
    unittest.main()
