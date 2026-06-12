"""
End-to-end audio/video sync check without the GUI.

Connects a USB camera through CameraWorker, records ~6 s of video with
libx264 while the UltrasoundRecorder captures the default microphone, then
finalizes the WAV against the encoded video duration and verifies (via
ffprobe) that both files have exactly the same duration.

Usage:
    python scripts/dev_avsync_check.py [--outdir DIR] [--seconds N] [--cam-index I]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401  (DLL order: torch before PySpin)
except Exception:
    pass

from PySide6.QtCore import QCoreApplication, QTimer

RESULTS: dict = {"ok": False, "errors": []}


def log(message: str) -> None:
    line = f"[avsync +{time.strftime('%H:%M:%S')}] {message}"
    sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace") + b"\n")
    sys.stdout.flush()


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    return float(out.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(REPO_ROOT / "dev_screenshots" / "avsync_check"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--cam-index", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_path = outdir / "avsync_check"

    app = QCoreApplication(sys.argv)

    from camera_worker import CameraWorker
    from audio_recorder import UltrasoundRecorder, enumerate_input_devices, pick_default_device

    worker = CameraWorker()
    worker.set_encoder("libx264")
    worker.set_preview_enabled(False)

    recorder = UltrasoundRecorder()
    state = {"first_frame": None, "last_frame": None, "frames": 0}

    def on_frame_recorded(metadata: dict) -> None:
        ts = metadata.get("timestamp_software")
        if ts is None:
            return
        if state["first_frame"] is None:
            state["first_frame"] = float(ts)
            recorder.mark_video_started(float(ts))
        state["last_frame"] = float(ts)
        state["frames"] += 1

    worker.frame_recorded.connect(on_frame_recorded)
    worker.error_occurred.connect(lambda msg: RESULTS["errors"].append(str(msg)))
    worker.status_update.connect(lambda msg: log(f"worker: {msg}"))
    recorder.error_occurred.connect(lambda msg: RESULTS["errors"].append(f"audio: {msg}"))
    recorder.status_changed.connect(lambda msg: log(f"audio: {msg}"))

    def fail(message: str) -> None:
        RESULTS["errors"].append(message)
        log(f"FAIL: {message}")
        finish()

    def step_connect() -> None:
        from camera_backends import discover_usb_cameras

        cams = discover_usb_cameras()
        log(f"usb cameras: {[c['label'] for c in cams]}")
        RESULTS["usb_cameras"] = [c["label"] for c in cams]
        target = None
        for cam in cams:
            if int(cam.get("index", -1)) == args.cam_index:
                target = cam
                break
        if target is None and cams:
            target = cams[0]
        if target is None:
            return fail("no USB camera available")
        if not worker.connect_camera(target):
            return fail("camera connect failed")
        worker.start()
        log(f"connected {target['label']}")
        QTimer.singleShot(1500, step_audio)

    def step_audio() -> None:
        devices = enumerate_input_devices()
        device = pick_default_device(devices)
        if device is None:
            return fail("no audio input device")
        if not recorder.open_stream(device):
            return fail("audio stream open failed")
        log(f"audio stream open on {device.name}")
        QTimer.singleShot(800, step_record)

    def step_record() -> None:
        if not recorder.begin_recording(str(base_path) + ".wav"):
            return fail("audio begin_recording failed")
        if not worker.start_recording(str(base_path)):
            return fail("video start_recording failed")
        log("recording...")
        QTimer.singleShot(int(args.seconds * 1000), step_stop)

    def step_stop() -> None:
        worker.stop_recording()
        QTimer.singleShot(1200, step_finalize)

    def step_finalize() -> None:
        frames = int(getattr(worker, "frame_counter", 0) or 0)
        fps = float(getattr(worker, "last_recording_output_fps", 0.0) or 0.0)
        encoded_duration = frames / fps if fps > 0 else None
        RESULTS["frames"] = frames
        RESULTS["encoded_fps"] = fps
        RESULTS["encoded_duration_s"] = encoded_duration
        if state["last_frame"] is not None:
            recorder.mark_video_stopped(state["last_frame"])
        metadata = recorder.finalize(target_duration_seconds=encoded_duration) or {}
        RESULTS["audio_metadata"] = {
            k: v for k, v in metadata.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        recorder.close_stream()
        QTimer.singleShot(300, step_verify)

    def step_verify() -> None:
        mp4 = Path(str(base_path) + ".mp4")
        wav = Path(str(base_path) + ".wav")
        try:
            video_s = ffprobe_duration(mp4)
            audio_s = ffprobe_duration(wav)
        except Exception as exc:
            return fail(f"ffprobe failed: {exc}")
        RESULTS["mp4_duration_s"] = video_s
        RESULTS["wav_duration_s"] = audio_s
        RESULTS["av_delta_ms"] = (audio_s - video_s) * 1000.0
        RESULTS["ok"] = abs(audio_s - video_s) <= 0.02 and not RESULTS["errors"]
        log(
            f"mp4 {video_s:.3f}s | wav {audio_s:.3f}s | "
            f"delta {RESULTS['av_delta_ms']:+.1f} ms | ok={RESULTS['ok']}"
        )
        finish()

    def finish() -> None:
        try:
            worker.stop()
        except Exception:
            pass
        try:
            recorder.close_stream()
        except Exception:
            pass
        (outdir / "avsync_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
        QTimer.singleShot(400, app.quit)

    QTimer.singleShot(200, step_connect)
    app.exec()
    return 0 if RESULTS.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
