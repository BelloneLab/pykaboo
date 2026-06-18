"""Screenshot the ultrasound panel with multiple synthetic microphones added."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401
except Exception:
    pass

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    try:
        from branding import ensure_theme_assets
        ensure_theme_assets()
    except Exception:
        pass
    from app_theme import build_app_stylesheet
    app.setStyleSheet(build_app_stylesheet())

    from audio_recorder import AudioInputDevice, UltrasoundPanel

    devices = [
        AudioInputDevice(23, "Microphone (Pettersson M500-384kHz USB Ultrasound Microphone)",
                         1, 384_000.0, 2, "Windows WASAPI"),
        AudioInputDevice(24, "Microphone (2- Pettersson M500-384kHz USB Ultrasound Microphone)",
                         1, 384_000.0, 2, "Windows WASAPI"),
        AudioInputDevice(1, "Microphone (USB Live camera audio)", 2, 44_100.0, 0, "MME"),
    ]

    panel = UltrasoundPanel()
    panel._save_mic_config = lambda: None
    for slot in list(panel._extra_slots):
        panel.remove_microphone(slot)
    # Inject synthetic devices and rebuild the primary combo from them.
    panel._devices = list(devices)
    panel.device_combo.blockSignals(True)
    panel.device_combo.clear()
    for d in devices:
        panel.device_combo.addItem(("🎙  " if d.is_ultrasound else "    ") + d.display, d.index)
    panel.device_combo.setCurrentIndex(0)
    panel.device_combo.blockSignals(False)
    panel._apply_device(devices[0])

    slot_b = panel.add_microphone(devices[1])
    panel.add_microphone(devices[2])
    if slot_b is not None:
        slot_b.focus_radio.setChecked(True)  # focus the 2nd mic

    panel.resize(420, 900)
    panel.show()

    def snap():
        for _ in range(5):
            app.processEvents()
        out = REPO_ROOT / "dev_screenshots" / "multi_mic_panel.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.grab().save(str(out))
        print(f"[probe] saved {out}", flush=True)
        app.quit()

    QTimer.singleShot(800, snap)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
