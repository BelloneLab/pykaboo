"""Regression probe: the live preview must fill the workspace at launch.

Reproduces the bug where a `workspace_splitter_sizes` value saved while a
bottom panel was open (a small preview) was restored on every launch, leaving
the preview shrunken above a large empty controls strip. With both Acquisition
and Recording panels closed (the launch default), the preview should be
maximised regardless of the saved split.

Non-destructive: it saves the real QSettings value, injects a deliberately
small split, runs the launch sequence, then restores the original value.

Usage:
    python scripts/dev_split_probe.py
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

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

KEY = "workspace_splitter_sizes"
SMALL_SPLIT = "527,301"  # ~64% preview: the broken "image 1" launch state.


def main() -> int:
    app = QApplication(sys.argv)

    # Inject a small saved split before the window reads it, then restore.
    settings = QSettings("PyKaboo", "PyKaboo")
    original = settings.value(KEY, None)
    settings.setValue(KEY, SMALL_SPLIT)
    settings.sync()

    from main_window_enhanced import MainWindow

    window = MainWindow()
    window._startup_autoconnect_done = True  # pure UI check, no hardware
    window.show()

    result = {"ok": False}

    def check() -> None:
        spl = window.center_splitter
        sizes = spl.sizes() if spl is not None else [0, 0]
        total = sum(sizes) or 1
        preview_pct = sizes[0] * 100.0 / total
        panels_open = window._is_any_workspace_panel_open()
        print(f"[probe] injected={SMALL_SPLIT} -> launch sizes={sizes} "
              f"preview={preview_pct:.0f}% panels_open={panels_open}", flush=True)
        # Both panels start closed, so the preview must dominate the column.
        result["ok"] = (not panels_open) and preview_pct >= 85.0
        print("[probe] PASS" if result["ok"] else "[probe] FAIL", flush=True)

        out = REPO_ROOT / "dev_screenshots" / "split_probe.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(out))

        # Restore the operator's real setting.
        if original is None:
            settings.remove(KEY)
        else:
            settings.setValue(KEY, original)
        settings.sync()
        app.quit()

    QTimer.singleShot(1200, check)
    app.exec()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
