"""
Development harness: drive PyKaboo against real hardware.

Launches the full MainWindow, connects the primary camera plus auxiliary
streams, records a short synchronized session, and saves a screenshot at
every stage. Used to audit the GUI and verify CSV outputs end to end.

Usage:
    python scripts/dev_drive_app.py [--outdir DIR] [--record-seconds N]
"""
from __future__ import annotations

import argparse
import json
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

RESULTS: dict = {"steps": [], "errors": []}


def log(message: str) -> None:
    line = f"[harness +{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    RESULTS["steps"].append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(REPO_ROOT / "dev_screenshots"))
    parser.add_argument("--record-seconds", type=int, default=8)
    parser.add_argument("--skip-record", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session_dir = outdir / "session_output"
    session_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    window.resize(1720, 1020)
    window.show()

    def shot(name: str) -> None:
        app.processEvents()
        path = outdir / name
        window.grab().save(str(path))
        log(f"screenshot: {path.name}")

    state = {"aux_connected": 0}

    def step_startup():
        shot("01_startup.png")
        window._scan_cameras()
        cameras = []
        for index in range(window.combo_camera.count()):
            data = window.combo_camera.itemData(index)
            if data:
                cameras.append(data)
        RESULTS["cameras"] = cameras
        log(f"cameras discovered: {[c.get('label') for c in cameras]}")
        QTimer.singleShot(500, step_connect_primary)

    def step_connect_primary():
        if window.is_camera_connected:
            log("primary already connected (autoconnect)")
            QTimer.singleShot(2500, step_after_primary)
            return
        # Prefer the FLIR camera as primary; fall back to first entry.
        chosen = None
        for index in range(window.combo_camera.count()):
            data = window.combo_camera.itemData(index)
            if data and data.get("type") == "flir":
                chosen = index
                break
        if chosen is None and window.combo_camera.count() > 0:
            chosen = 0
        if chosen is None:
            RESULTS["errors"].append("no cameras found for primary")
            return finish()
        window.combo_camera.setCurrentIndex(chosen)
        log(f"connecting primary: {window.combo_camera.currentText()}")
        window._on_connect_clicked()
        QTimer.singleShot(3500, step_after_primary)

    def step_after_primary():
        log(f"primary connected={window.is_camera_connected}")
        shot("02_primary_connected.png")
        QTimer.singleShot(300, step_add_aux)

    def step_add_aux():
        window._on_add_camera_stream_clicked()
        if not window.aux_camera_tiles:
            RESULTS["errors"].append("aux tile was not created")
            return finish()
        tile = window.aux_camera_tiles[-1]
        tile.refresh_sources()
        if tile.combo_source.currentData() is None:
            log("no free camera for this aux tile; removing it")
            window._remove_camera_stream_tile(tile)
            return QTimer.singleShot(300, step_three_streams)
        log(f"aux{len(window.aux_camera_tiles)} connecting: {tile.combo_source.currentText()}")
        tile._on_connect_clicked()
        QTimer.singleShot(3000, step_check_aux)

    def step_check_aux():
        tile = window.aux_camera_tiles[-1]
        log(f"aux{len(window.aux_camera_tiles)} connected={tile.stream.is_connected}")
        state["aux_connected"] += 1 if tile.stream.is_connected else 0
        if len(window.aux_camera_tiles) < 2:
            return QTimer.singleShot(300, step_add_aux)
        QTimer.singleShot(300, step_three_streams)

    def step_three_streams():
        shot("03_multi_stream.png")
        if args.skip_record:
            return finish()
        QTimer.singleShot(300, step_record)

    def step_record():
        window.edit_save_folder.setText(str(session_dir))
        window.last_save_folder = str(session_dir)
        window.edit_filename.setText("multicam_check")
        for index in range(window.combo_encoder.count()):
            if "libx264" in window.combo_encoder.itemText(index):
                window.combo_encoder.setCurrentIndex(index)
                break
        log("starting recording...")
        window._on_record_clicked()
        QTimer.singleShot(1500, step_check_recording)

    def step_check_recording():
        recording = bool(window.worker.is_recording)
        aux_recording = [
            (tile.stream.display_name, tile.stream.is_recording)
            for tile in window.aux_camera_tiles
        ]
        log(f"recording primary={recording} aux={aux_recording}")
        RESULTS["recording_started"] = recording
        RESULTS["aux_recording"] = aux_recording
        if not recording:
            RESULTS["errors"].append("primary recording did not start")
            return finish()
        QTimer.singleShot(max(1, args.record_seconds - 2) * 1000, step_record_shot)

    def step_record_shot():
        shot("04_recording.png")
        QTimer.singleShot(2000, step_stop)

    def step_stop():
        log("stopping recording...")
        window._request_recording_stop("manual")
        QTimer.singleShot(6000, step_after_stop)

    def step_after_stop():
        shot("05_stopped.png")
        produced = sorted(str(p.relative_to(session_dir)) for p in session_dir.rglob("*") if p.is_file())
        RESULTS["produced_files"] = produced
        log(f"produced files: {produced}")
        finish()

    def finish():
        (outdir / "harness_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
        log("done — closing")
        QTimer.singleShot(400, window.close)
        QTimer.singleShot(1500, app.quit)

    QTimer.singleShot(2500, step_startup)
    code = app.exec()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
