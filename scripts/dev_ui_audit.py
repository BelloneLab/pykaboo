"""
UI audit harness: open every tool panel and capture screenshots.

Unlike dev_drive_app.py this never touches hardware — startup autoconnect is
suppressed so the audit can run on any machine. Used to iterate on theme,
panel layout, and visual polish.

Usage:
    python scripts/dev_ui_audit.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401  (DLL order: torch before PySpin)
except Exception:
    pass

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(REPO_ROOT / "dev_screenshots" / "ui_audit"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    # Block hardware autoconnect: this is a pure UI audit.
    window._startup_autoconnect_done = True
    window.resize(1760, 1020)
    window.show()

    def shot(name: str) -> None:
        for _ in range(4):
            app.processEvents()
        window.grab().save(str(outdir / name))
        print(f"[ui-audit] {name}", flush=True)

    steps: list[tuple[str, callable]] = []

    def add_step(delay_ms: int, fn) -> None:
        steps.append((delay_ms, fn))

    left_panels = [
        ("camera", "Camera Connection"),
        ("settings", "General Settings"),
        ("session", "Metadata and Planner"),
        ("file", "File Tools"),
    ]
    right_panels = [
        ("arduino", "Arduino Setup"),
        ("ttl", "TTL Monitor"),
        ("behavior", "Behavior Monitor"),
        ("live_detection", "Live Detection"),
        ("audio", "Ultrasound Audio"),
    ]

    sequence: list[callable] = []

    sequence.append(lambda: shot("00_startup.png"))

    for key, title in left_panels:
        def open_left(k=key, t=title):
            window._toggle_side_panel("left", k, t)
            shot(f"10_left_{k}.png")
        sequence.append(open_left)

    def close_left():
        window._hide_side_panel("left")
    sequence.append(close_left)

    for key, title in right_panels:
        def open_right(k=key, t=title):
            window._toggle_side_panel("right", k, t)
            shot(f"20_right_{k}.png")
        sequence.append(open_right)

    def close_right():
        window._hide_side_panel("right")
    sequence.append(close_right)

    def open_workspace_cards():
        window.btn_toggle_acquisition_panel.setChecked(True)
        window.btn_toggle_recording_panel.setChecked(True)
        shot("30_workspace_cards.png")
    sequence.append(open_workspace_cards)

    def open_audio_monitor():
        # Audio panel with monitor running + waveform preview enabled, so the
        # waveform aesthetic can be audited (uses whatever default mic exists).
        window._toggle_side_panel("right", "audio", "Ultrasound Audio")
        panel = window.audio_panel
        if panel is not None and panel.recorder.is_available:
            try:
                panel.preview_toggle.setChecked(True)
                panel.btn_monitor.setChecked(True)
            except Exception as exc:
                print(f"[ui-audit] audio monitor failed: {exc}", flush=True)
    sequence.append(open_audio_monitor)

    def shot_audio():
        shot("40_audio_spectrogram_live.png")
        panel = window.audio_panel
        if panel is not None:
            try:
                panel._set_preview_mode("waveform")
            except Exception:
                pass
    sequence.append(shot_audio)

    def shot_audio_wave():
        shot("41_audio_waveform_live.png")
        panel = window.audio_panel
        if panel is not None:
            try:
                panel.btn_monitor.setChecked(False)
            except Exception:
                pass
    sequence.append(shot_audio_wave)

    def drag_splitter():
        splitter = window.center_splitter
        if splitter is not None:
            total = sum(splitter.sizes())
            splitter.setSizes([int(total * 0.45), total - int(total * 0.45)])
        shot("50_splitter_resized.png")
    sequence.append(drag_splitter)

    def finish():
        print("[ui-audit] done", flush=True)
        window.close()
        QTimer.singleShot(300, app.quit)
    sequence.append(finish)

    delay = 1800
    for fn in sequence:
        QTimer.singleShot(delay, fn)
        delay += 900
    # Give the audio waveform time to fill before its screenshot.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
