"""Combined-load stress test: mask + pose geometry + behavior + recording at once.

Drives the real app end to end and samples preview FPS, inference FPS, end-to-end
lag, detection count and behavior status while everything runs together, then
reports whether the preview stayed fluid and the recording was produced cleanly.

This is the harness for the "the app should not lag with tracking mask + pose
geometry + behavior detection and recording all active" requirement. Run it on the
acquisition rig with the real camera for the true load:

    python scripts/dev_stress_pipeline.py --source flir --duration 30 \
        --keypoint-source mask_geometry --behavior --record

On a machine with no lab camera, use the built-in synthetic source to exercise the
capture -> process -> overlay -> record threading (no real mice, so detections are
empty, but the pipeline runs end to end):

    set PYKABOO_SHOW_SIMULATED_CAMERAS=1
    python scripts/dev_stress_pipeline.py --source simulated --duration 12 --record

Verdict thresholds are conservative defaults; tune with --min-preview-fps /
--max-lag-ms for your target (e.g. Full-HD 30 fps).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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

RESULTS: dict = {"steps": [], "errors": [], "samples": []}


def log(msg: str) -> None:
    print(f"[stress +{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    RESULTS["steps"].append(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["flir", "basler", "usb", "virtual", "simulated", "auto"], default="auto")
    parser.add_argument("--duration", type=int, default=20, help="seconds to hold combined load")
    parser.add_argument("--keypoint-source", default="mask_geometry",
                        choices=["mask_geometry", "yolo_pose"])
    parser.add_argument("--behavior", action="store_true", help="run live behavior detection")
    parser.add_argument("--no-track", dest="track", action="store_false", help="skip inference (camera+record only)")
    parser.add_argument("--record", action="store_true", help="record MP4 + CSV during the run")
    parser.add_argument("--warmup", type=int, default=50, help="seconds to wait for model warmup")
    parser.add_argument("--min-preview-fps", type=float, default=20.0)
    parser.add_argument("--max-lag-ms", type=float, default=120.0)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "dev_screenshots" / "stress"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session_dir = outdir / "session_output"
    session_dir.mkdir(parents=True, exist_ok=True)
    RESULTS["args"] = vars(args)

    if args.source == "simulated":
        os.environ["PYKABOO_SHOW_SIMULATED_CAMERAS"] = "1"

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    window._startup_autoconnect_done = True
    window.resize(1760, 1040)
    window.show()

    def fps_now() -> float:
        try:
            return float(window.label_fps.text().split(":")[-1].strip())
        except Exception:
            return -1.0

    def status() -> str:
        try:
            return window.live_detection_panel.label_status.text()
        except Exception:
            return ""

    def pick_camera() -> bool:
        window._scan_cameras()
        wanted = args.source
        combo = window.combo_camera
        candidates = []
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data and isinstance(data, dict):
                candidates.append((index, str(data.get("type", ""))))
        # The synthetic backend is listed in the combo as camera type "virtual".
        if wanted == "simulated":
            wanted = "virtual"
        order = ([wanted] if wanted != "auto" else ["flir", "basler", "usb", "virtual"])
        for want in order:
            for index, ctype in candidates:
                if ctype == want:
                    combo.setCurrentIndex(index)
                    window._on_connect_clicked()
                    log(f"connecting camera type={ctype}")
                    return True
        RESULTS["errors"].append(f"no camera of type(s) {order} found; candidates={candidates}")
        return False

    def start():
        if not pick_camera():
            return finish()
        QTimer.singleShot(3500, after_connect)

    def after_connect():
        log(f"connected={window.is_camera_connected} preview_fps={fps_now():.1f}")
        if not window.is_camera_connected:
            RESULTS["errors"].append("camera failed to connect")
            return finish()
        try:
            RESULTS["acquire_resolution"] = [int(window.worker.width), int(window.worker.height)]
        except Exception:
            pass
        if not args.track:
            return begin_record_phase()
        # Select keypoint source (mask geometry is the geometry-only fast path).
        try:
            combo_ks = getattr(window.live_detection_panel, "combo_keypoint_source", None)
            if combo_ks is not None:
                idx = combo_ks.findData(args.keypoint_source)
                if idx >= 0:
                    combo_ks.setCurrentIndex(idx)
                    log(f"keypoint source = {args.keypoint_source}")
        except Exception as exc:
            log(f"keypoint source select failed: {exc}")
        log("enabling tracking (loading models, warmup)...")
        try:
            window._set_tracking_button_checked(True)
            window._on_tracking_mode_toggled(True)
        except Exception as exc:
            RESULTS["errors"].append(f"tracking enable failed: {exc}")
            return finish()
        QTimer.singleShot(max(1, args.warmup) * 1000, enable_behavior)

    def enable_behavior():
        log(f"inference status: {status()!r}")
        if args.behavior:
            try:
                window._on_run_behavior_toggled(True)
                log("behavior detection enabled")
            except Exception as exc:
                log(f"behavior enable failed: {exc}")
        QTimer.singleShot(3000, begin_record_phase)

    def begin_record_phase():
        if args.record:
            try:
                window.edit_save_folder.setText(str(session_dir))
                window.last_save_folder = str(session_dir)
                window.edit_filename.setText("stress")
                window._on_record_clicked()
                log("recording started")
            except Exception as exc:
                RESULTS["errors"].append(f"record start failed: {exc}")
        RESULTS["sample_start"] = time.time()
        sample_loop()

    def sample_loop():
        elapsed = time.time() - RESULTS["sample_start"]
        result = window.live_detection_last_result
        mice = len(getattr(result, "tracked_mice", []) or []) if result else 0
        infer_ms = float(getattr(result, "inference_ms", 0.0) or 0.0) if result else 0.0
        lag_ms = float(getattr(result, "end_to_end_ms", 0.0) or 0.0) if result else 0.0
        RESULTS["samples"].append({
            "t": round(elapsed, 2), "preview_fps": fps_now(), "mice": mice,
            "inference_ms": round(infer_ms, 2), "lag_ms": round(lag_ms, 2),
        })
        if elapsed < args.duration:
            QTimer.singleShot(500, sample_loop)
        else:
            stop_phase()

    def stop_phase():
        if args.record:
            log("stopping recording...")
            try:
                window._request_recording_stop("manual")
            except Exception as exc:
                RESULTS["errors"].append(f"record stop failed: {exc}")
            QTimer.singleShot(6000, finish)
        else:
            finish()

    def summarize():
        samples = RESULTS["samples"]
        fps_vals = [s["preview_fps"] for s in samples if s["preview_fps"] > 0]
        lag_vals = [s["lag_ms"] for s in samples if s["lag_ms"] > 0]
        infer_vals = [s["inference_ms"] for s in samples if s["inference_ms"] > 0]
        mice_vals = [s["mice"] for s in samples]
        summary: dict = {
            "preview_fps_mean": round(statistics.mean(fps_vals), 1) if fps_vals else -1,
            "preview_fps_min": round(min(fps_vals), 1) if fps_vals else -1,
            "inference_fps_mean": round(1000.0 / statistics.mean(infer_vals), 1) if infer_vals else -1,
            "lag_ms_mean": round(statistics.mean(lag_vals), 1) if lag_vals else -1,
            "lag_ms_p95": round(sorted(lag_vals)[int(0.95 * (len(lag_vals) - 1))], 1) if lag_vals else -1,
            "mice_max": max(mice_vals) if mice_vals else 0,
        }
        produced = sorted(str(p.relative_to(session_dir)) for p in session_dir.rglob("*") if p.is_file())
        summary["produced_files"] = produced
        mp4s = [p for p in session_dir.rglob("*.mp4")]
        if mp4s:
            summary["mp4_size_mb"] = round(max(p.stat().st_size for p in mp4s) / 1e6, 2)
        fluid = (summary["preview_fps_min"] >= args.min_preview_fps if fps_vals else False)
        lag_ok = (summary["lag_ms_p95"] <= args.max_lag_ms) if lag_vals else (not args.track)
        summary["verdict_fluid_preview"] = bool(fluid)
        summary["verdict_lag_ok"] = bool(lag_ok)
        summary["verdict_pass"] = bool(fluid and lag_ok and not RESULTS["errors"])
        RESULTS["summary"] = summary
        return summary

    def finish():
        s = summarize()
        (outdir / "stress_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
        log("==== SUMMARY ====")
        for k, v in s.items():
            log(f"  {k}: {v}")
        if RESULTS["errors"]:
            log(f"errors: {RESULTS['errors']}")
        QTimer.singleShot(400, window.close)
        QTimer.singleShot(1500, app.quit)

    QTimer.singleShot(1500, start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
