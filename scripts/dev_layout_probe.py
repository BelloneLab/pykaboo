"""Probe: open both side panels at a non-fullscreen size and screenshot.

Used to reproduce/inspect the "content hidden when both tool panels are open
and the window is not maximized" layout bug.
"""
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
    outdir = Path(REPO_ROOT.parent / "dev_screenshots" / "layout_probe")
    outdir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    window._startup_autoconnect_done = True
    window.show()

    def shot(name: str) -> None:
        for _ in range(5):
            app.processEvents()
        window.grab().save(str(outdir / name))
        print(f"[probe] {name} @ {window.width()}x{window.height()}", flush=True)

    sizes = [(1280, 800), (1440, 900), (1100, 720)]
    seq = []

    for w, h in sizes:
        def do(w=w, h=h):
            window.resize(w, h)
            window._toggle_side_panel("left", "session", "Metadata and Planner")
            window._toggle_side_panel("right", "ttl", "TTL Monitor")
            window.btn_toggle_acquisition_panel.setChecked(True)
            window.btn_toggle_recording_panel.setChecked(True)
            shot(f"both_open_{w}x{h}.png")
        seq.append(do)

    def finish():
        window.close()
        QTimer.singleShot(200, app.quit)
    seq.append(finish)

    delay = 1400
    for fn in seq:
        QTimer.singleShot(delay, fn)
        delay += 1100
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
