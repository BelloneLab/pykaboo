"""Drive PyKaboo with live mask+pose tracking on the real FLIR feed.

Autoconnects the primary camera (Full HD 60 from saved settings), turns on
tracking mode (mask + pose), waits for the models to warm up, then captures a
series of overlay screenshots plus a short Full HD recording. Reports achieved
preview FPS, inference FPS, and the recorded video resolution/fps so we can
verify the >30 fps Full HD target end to end.

Usage:
    python scripts/dev_drive_tracking.py [--outdir DIR] [--record-seconds N]
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


def log(msg: str) -> None:
    line = f"[track +{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RESULTS["steps"].append(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(REPO_ROOT / "dev_screenshots" / "tracking"))
    parser.add_argument("--record-seconds", type=int, default=6)
    parser.add_argument("--shots", type=int, default=6)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session_dir = outdir / "session_output"
    session_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    window.resize(1760, 1040)
    window.show()

    def shot(name: str) -> None:
        app.processEvents()
        path = outdir / name
        window.grab().save(str(path))
        log(f"screenshot: {path.name}")

    def shot_overlay(name: str) -> None:
        """Render the latest inference result on its full-res frame for a clean,
        zoomed look at mask + skeleton quality."""
        try:
            import numpy as np
            from PySide6.QtGui import QImage
            result = window.live_detection_last_result
            if result is None:
                return
            packet = window.live_inference_frame_cache.get(int(result.frame_index))
            if packet is None and window.live_inference_frame_cache:
                packet = window.live_inference_frame_cache[max(window.live_inference_frame_cache)]
            if packet is None:
                return
            frame = np.asarray(packet.frame)
            rgb = window._decorate_live_frame(frame, include_recording_hud=False, overlay_result_override=result)
            rgb = np.ascontiguousarray(rgb)
            h, w = rgb.shape[:2]
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            img.save(str(outdir / name))
            log(f"overlay-render: {name}")
        except Exception as exc:
            log(f"overlay-render failed: {exc}")

    def fps_now() -> float:
        try:
            text = window.label_fps.text()  # "FPS: 59.8"
            return float(text.split(":")[-1].strip())
        except Exception:
            return -1.0

    def inference_status() -> str:
        try:
            return window.live_detection_panel.label_status.text()
        except Exception:
            return ""

    state = {"shot_index": 0}

    def start():
        log("waiting for autoconnect...")
        QTimer.singleShot(4000, ensure_connected)

    def ensure_connected():
        if not window.is_camera_connected:
            # Force-connect the FLIR if autoconnect has not fired yet.
            window._scan_cameras()
            for index in range(window.combo_camera.count()):
                data = window.combo_camera.itemData(index)
                if data and data.get("type") == "flir":
                    window.combo_camera.setCurrentIndex(index)
                    window._on_connect_clicked()
                    break
        QTimer.singleShot(3000, after_connect)

    def after_connect():
        log(f"connected={window.is_camera_connected} fps={fps_now():.1f}")
        try:
            w = window.worker
            RESULTS["acquire_resolution"] = [int(w.width), int(w.height)]
            RESULTS["image_format"] = window.combo_image_format.currentText()
        except Exception:
            pass
        shot("01_fullhd_preview.png")
        log(f"acquisition res={RESULTS.get('acquire_resolution')} fmt={RESULTS.get('image_format')}")
        QTimer.singleShot(500, enable_tracking)

    def enable_tracking():
        cfg = window.live_detection_panel.detection_config()
        log(f"seg='{Path(cfg.get('checkpoint_path','')).name}' pose='{Path(cfg.get('pose_checkpoint_path','')).name}'")
        if not cfg.get("checkpoint_path"):
            RESULTS["errors"].append("no seg checkpoint configured")
            return finish()
        log("enabling tracking mode (loading models)...")
        window._set_tracking_button_checked(True)
        window._on_tracking_mode_toggled(True)
        # RF-DETR optimize_for_inference + pose load + cudnn warmup take ~40 s.
        QTimer.singleShot(50000, wait_for_results)

    def wait_for_results():
        log(f"inference status: {inference_status()!r}")
        RESULTS["inference_status"] = inference_status()
        QTimer.singleShot(1500, capture_loop)

    def capture_loop():
        idx = state["shot_index"]
        result = window.live_detection_last_result
        mice = len(getattr(result, "tracked_mice", []) or []) if result else 0
        lag = float(getattr(result, "end_to_end_ms", 0.0) or 0.0) if result else 0.0
        infer_fps = (1000.0 / lag) if lag > 0 else 0.0
        log(f"shot {idx}: mice={mice} preview_fps={fps_now():.1f} infer~{infer_fps:.1f}fps status={inference_status()!r}")
        shot(f"overlay_{idx:02d}.png")
        shot_overlay(f"render_{idx:02d}.png")
        RESULTS.setdefault("overlay_samples", []).append(
            {"mice": mice, "preview_fps": fps_now(), "inference_fps": infer_fps}
        )
        state["shot_index"] += 1
        if state["shot_index"] < args.shots:
            QTimer.singleShot(900, capture_loop)
        else:
            QTimer.singleShot(500, start_recording)

    def start_recording():
        window.edit_save_folder.setText(str(session_dir))
        window.last_save_folder = str(session_dir)
        window.edit_filename.setText("fullhd_track")
        # Arm audio + waveform so we can verify A/V equal-duration sync and grab
        # a screenshot of the live waveform.
        try:
            if window.audio_panel is not None:
                window.audio_panel.enable_check.setChecked(True)
                if hasattr(window.audio_panel, "waveform_preview_toggle"):
                    window.audio_panel.waveform_preview_toggle.setChecked(True)
                RESULTS["audio_enabled"] = bool(window.audio_panel.is_enabled())
                log(f"audio enabled={RESULTS['audio_enabled']}")
        except Exception as exc:
            log(f"audio enable failed: {exc}")
        log("starting Full HD recording with live overlay...")
        window._on_record_clicked()
        QTimer.singleShot(max(1, args.record_seconds) * 1000, stop_recording)

    def stop_recording():
        shot("rec_overlay.png")
        log("stopping recording...")
        window._request_recording_stop("manual")
        QTimer.singleShot(6000, after_stop)

    def after_stop():
        produced = sorted(str(p.relative_to(session_dir)) for p in session_dir.rglob("*") if p.is_file())
        RESULTS["produced_files"] = produced
        log(f"produced: {produced}")
        # Compare MP4 and WAV durations for A/V equal-duration sync.
        try:
            import soundfile as _sf
            mp4 = next((p for p in session_dir.rglob("*.mp4") if "track" in p.name), None)
            wav = next((p for p in session_dir.rglob("*.wav")), None)
            if mp4 is not None:
                import subprocess
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(mp4)],
                    capture_output=True, text=True,
                )
                RESULTS["mp4_duration_s"] = float(out.stdout.strip() or 0.0)
            if wav is not None and wav.is_file():
                info = _sf.info(str(wav))
                RESULTS["wav_duration_s"] = float(info.frames) / float(info.samplerate)
            if "mp4_duration_s" in RESULTS and "wav_duration_s" in RESULTS:
                RESULTS["av_delta_ms"] = (RESULTS["wav_duration_s"] - RESULTS["mp4_duration_s"]) * 1000.0
                log(f"A/V sync: mp4={RESULTS['mp4_duration_s']:.3f}s wav={RESULTS['wav_duration_s']:.3f}s "
                    f"delta={RESULTS['av_delta_ms']:.1f}ms")
        except Exception as exc:
            log(f"A/V duration check failed: {exc}")
        finish()

    def finish():
        (outdir / "tracking_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
        log("done")
        QTimer.singleShot(400, window.close)
        QTimer.singleShot(1500, app.quit)

    QTimer.singleShot(2500, start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
