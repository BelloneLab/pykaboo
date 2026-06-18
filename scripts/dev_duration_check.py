"""
Hardware test: verify a duration-limited recording produces exact-length files.

Connects the primary camera plus any free auxiliary cameras, sets a Limited
recording duration, then presses Record and lets the new acquisition-thread
frame-count cap stop it (no manual stop). Afterwards it probes every produced
mp4 (container duration + frame count + fps) and the wav, and prints a
requested-vs-actual table so we can confirm the files match the set time.

Usage:
    python scripts/dev_duration_check.py [--seconds 10] [--outdir DIR] [--encoder libx264]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401  (DLL order: torch before PySpin)
except Exception:
    pass

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def _ffprobe(path: Path) -> dict:
    """Return container duration, stream nb_frames, and avg fps via ffprobe."""
    info: dict = {"file": path.name}
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "format=duration:stream=nb_frames,avg_frame_rate,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(out.stdout or "{}")
        fmt = data.get("format", {})
        stream = (data.get("streams") or [{}])[0]
        info["container_duration_s"] = float(fmt.get("duration", 0.0) or 0.0)
        info["frames"] = int(stream.get("nb_frames", 0) or 0)
        avg = stream.get("avg_frame_rate", "0/1")
        num, _, den = avg.partition("/")
        info["avg_fps"] = (float(num) / float(den)) if den and float(den) != 0 else 0.0
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _wav_duration(path: Path) -> dict:
    info: dict = {"file": path.name}
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as handle:
            info["duration_s"] = len(handle) / float(handle.samplerate)
            info["samplerate"] = handle.samplerate
    except Exception as exc:
        info["error"] = str(exc)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--outdir", default=str(REPO_ROOT.parent / "dev_screenshots" / "duration_check"))
    parser.add_argument("--encoder", default="", help="substring of encoder to force, e.g. libx264")
    parser.add_argument("--max-aux", type=int, default=2)
    parser.add_argument(
        "--aux-usb-index", type=int, default=None,
        help="Connect exactly one aux stream bound to this USB camera index (e.g. 1).",
    )
    args = parser.parse_args()

    import shutil

    outdir = Path(args.outdir)
    session_dir = outdir / "session"
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)  # wipe stale nested runs
    session_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    from main_window_enhanced import MainWindow

    window = MainWindow()
    window.resize(1600, 980)
    window.show()
    results: dict = {"requested_seconds": args.seconds, "steps": [], "files": []}

    def log(msg: str) -> None:
        print(f"[dur-check] {msg}", flush=True)
        results["steps"].append(msg)

    def connect_primary():
        window._scan_cameras()
        if not window.is_camera_connected:
            chosen = None
            for i in range(window.combo_camera.count()):
                data = window.combo_camera.itemData(i)
                if data and data.get("type") == "flir":
                    chosen = i
                    break
            if chosen is None and window.combo_camera.count() > 0:
                chosen = 0
            if chosen is None:
                log("ERROR: no cameras found")
                return finish()
            window.combo_camera.setCurrentIndex(chosen)
            log(f"connecting primary: {window.combo_camera.currentText()}")
            window._on_connect_clicked()
        QTimer.singleShot(3500, add_aux)

    def _usb_index_of(camera_info) -> Optional[int]:
        if not camera_info:
            return None
        for key in ("index", "video_index"):
            val = camera_info.get(key)
            if val is not None and str(val).strip() != "":
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return None

    def add_aux():
        # Targeted mode: connect exactly one aux stream on a specific USB index.
        if args.aux_usb_index is not None:
            if window.aux_camera_tiles:
                return QTimer.singleShot(500, configure_audio)
            window._on_add_camera_stream_clicked()
            if not window.aux_camera_tiles:
                return QTimer.singleShot(500, configure_audio)
            tile = window.aux_camera_tiles[-1]
            tile.refresh_sources()
            target = None
            for i in range(tile.combo_source.count()):
                info = tile.combo_source.itemData(i)
                if info and str(info.get("type")) == "usb" and _usb_index_of(info) == args.aux_usb_index:
                    target = i
                    break
            if target is None:
                log(f"USB index {args.aux_usb_index} not available; aux sources: "
                    f"{[tile.combo_source.itemText(i) for i in range(tile.combo_source.count())]}")
                window._remove_camera_stream_tile(tile)
                return QTimer.singleShot(500, configure_audio)
            tile.combo_source.setCurrentIndex(target)
            log(f"connecting aux (USB index {args.aux_usb_index}): {tile.combo_source.currentText()}")
            tile._on_connect_clicked()
            return QTimer.singleShot(3500, configure_audio)

        added = len(window.aux_camera_tiles)
        if added >= args.max_aux:
            return QTimer.singleShot(500, configure_audio)
        window._on_add_camera_stream_clicked()
        if not window.aux_camera_tiles:
            return QTimer.singleShot(500, configure_audio)
        tile = window.aux_camera_tiles[-1]
        tile.refresh_sources()
        if tile.combo_source.currentData() is None:
            log("no more free cameras")
            window._remove_camera_stream_tile(tile)
            return QTimer.singleShot(500, configure_audio)
        log(f"connecting aux: {tile.combo_source.currentText()}")
        tile._on_connect_clicked()
        QTimer.singleShot(3500, add_aux)

    def configure_audio():
        """Select the Pettersson ultrasound mic (384 kHz) and arm synced WAV."""
        panel = getattr(window, "audio_panel", None)
        if panel is None:
            log("no audio panel; skipping audio")
            return QTimer.singleShot(300, configure_and_record)
        try:
            panel.refresh_devices()
            devices = list(getattr(panel, "_devices", []) or [])
            chosen = None
            for dev in devices:
                disp = str(getattr(dev, "display", "")).lower()
                if getattr(dev, "is_pettersson", False) or "pettersson" in disp or "m500" in disp:
                    chosen = dev
                    break
            if chosen is None:
                log(f"Pettersson mic NOT found. devices: {[getattr(d,'display','?') for d in devices]}")
            else:
                panel._select_device(chosen)
                panel.sr_spin.setValue(384000)
                panel.ch_spin.setValue(1)
                panel.enable_check.setChecked(True)
                results["audio_device"] = str(getattr(chosen, "display", ""))
                results["audio_samplerate_requested"] = 384000
                log(f"audio device = {results['audio_device']} @ 384000 Hz; sync armed")
        except Exception as exc:
            log(f"audio config error: {exc}")
        QTimer.singleShot(1500, configure_and_record)

    def configure_and_record():
        window.edit_save_folder.setText(str(session_dir))
        window.last_save_folder = str(session_dir)
        window.edit_filename.setText("dur_check")
        if args.encoder:
            for i in range(window.combo_encoder.count()):
                if args.encoder in window.combo_encoder.itemText(i):
                    window.combo_encoder.setCurrentIndex(i)
                    break
        # Set Limited duration = args.seconds, decomposed into HH:MM:SS because
        # the seconds spinbox is capped at 59 (setting 300 there clamps to 59).
        window.check_unlimited.setCurrentText("Limited")
        total = int(args.seconds)
        window.spin_hours.setValue(total // 3600)
        window.spin_minutes.setValue((total % 3600) // 60)
        window.spin_seconds.setValue(total % 60)
        aux_live = [t.stream.display_name for t in window.aux_camera_tiles if t.stream.is_connected]
        audio_armed = bool(getattr(window, "audio_panel", None) and window.audio_panel.enable_check.isChecked())
        log(f"primary connected={window.is_camera_connected}; aux live={aux_live}; audio_armed={audio_armed}")
        log(f"requested duration = {args.seconds}s; pressing Record (no manual stop)")
        results["record_started_perf"] = time.perf_counter()
        window._on_record_clicked()
        QTimer.singleShot(800, poll_stop)

    def poll_stop():
        if window.worker is not None and window.worker.is_recording:
            QTimer.singleShot(250, poll_stop)
            return
        elapsed = time.perf_counter() - results.get("record_started_perf", time.perf_counter())
        results["wallclock_record_to_stop_s"] = round(elapsed, 3)
        log(f"recording stopped by cap after ~{elapsed:.2f}s wall-clock; finalizing...")
        QTimer.singleShot(5000, measure)

    def measure():
        files = sorted(p for p in session_dir.rglob("*") if p.is_file())
        log(f"produced: {[p.name for p in files]}")
        for path in files:
            if path.suffix.lower() == ".mp4":
                results["files"].append(_ffprobe(path))
            elif path.suffix.lower() == ".wav":
                results["files"].append(_wav_duration(path))
        finish()

    def finish():
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "duration_check_results.json").write_text(json.dumps(results, indent=2, default=str))
        print("\n===== DURATION CHECK =====", flush=True)
        print(f"requested: {args.seconds}.000 s", flush=True)
        for f in results["files"]:
            if "container_duration_s" in f:
                d = f["container_duration_s"]
                delta = d - args.seconds
                print(
                    f"  MP4 {f['file']:42} dur={d:7.3f}s  delta={delta:+.3f}s  "
                    f"frames={f.get('frames')}  fps={f.get('avg_fps'):.3f}",
                    flush=True,
                )
            elif "duration_s" in f:
                d = f["duration_s"]
                print(f"  WAV {f['file']:42} dur={d:7.3f}s  delta={d-args.seconds:+.3f}s", flush=True)
            else:
                print(f"  ??? {f.get('file')}: {f.get('error')}", flush=True)
        durations = []
        for f in results["files"]:
            d = f.get("container_duration_s", f.get("duration_s"))
            if isinstance(d, (int, float)) and d > 0:
                durations.append(d)
        if durations:
            spread_ms = (max(durations) - min(durations)) * 1000.0
            print(f"duration spread across all files: {spread_ms:.1f} ms "
                  f"(min={min(durations):.3f}s max={max(durations):.3f}s)", flush=True)
            results["duration_spread_ms"] = spread_ms
        print(f"wall-clock record->stop: {results.get('wallclock_record_to_stop_s')} s", flush=True)
        print("==========================", flush=True)
        QTimer.singleShot(400, window.close)
        QTimer.singleShot(1200, app.quit)

    QTimer.singleShot(2500, connect_primary)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
