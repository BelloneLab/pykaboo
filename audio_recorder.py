"""Ultrasonic audio recorder synchronised with video acquisition.

Captures audio from a PortAudio input device (via the `sounddevice` library)
into a background thread, writes it as a 16-bit PCM WAV file whose basename
matches the video, and exposes a dockable Qt panel that lets the user pick
the device (auto-selecting the Pettersson M500 when present), watch a rolling
waveform/level meter, and confirm the mic is connected before recording.

Sync strategy (wall-clock trimming)
-----------------------------------
The PortAudio input stream is started *before* video recording begins. The
first sample-block callback records ``t_audio_start`` (``time.time()``). Once
the camera worker has flipped ``is_recording = True`` it exposes
``recording_started_at`` (also ``time.time()``). After the recording stops
the main window reports the video start timestamp and (optionally) the stop
timestamp to :meth:`UltrasoundRecorder.finalize`, which trims leading /
trailing samples so that sample 0 of the saved WAV corresponds exactly to the
first accepted video frame and the last sample lines up with the last frame
accepted by the encoder. This keeps audio↔video drift well under one frame
interval even for very long recordings.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

try:  # PortAudio wrapper
    import sounddevice as sd  # type: ignore
    _SOUNDDEVICE_IMPORT_ERROR: Optional[str] = None
except Exception as _sd_exc:  # pragma: no cover - optional dependency
    sd = None  # type: ignore
    _SOUNDDEVICE_IMPORT_ERROR = str(_sd_exc)

try:  # libsndfile wrapper (required for >4 GB files, nicer API than stdlib wave)
    import soundfile as sf  # type: ignore
    _SOUNDFILE_IMPORT_ERROR: Optional[str] = None
except Exception as _sf_exc:  # pragma: no cover - optional dependency
    sf = None  # type: ignore
    _SOUNDFILE_IMPORT_ERROR = str(_sf_exc)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioInputDevice:
    """Lightweight description of one PortAudio input device."""

    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi_index: int
    hostapi_name: str

    @property
    def display(self) -> str:
        sr = int(round(self.default_samplerate))
        return f"[{self.index}] {self.name}  ·  {sr} Hz  ·  {self.hostapi_name}"

    @property
    def is_ultrasound(self) -> bool:
        lowered = self.name.lower()
        if "pettersson" in lowered or "m500" in lowered or "ultrasound" in lowered:
            return True
        return self.default_samplerate >= 96_000.0


def enumerate_input_devices() -> List[AudioInputDevice]:
    if sd is None:
        return []
    try:
        raw_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return []

    out: List[AudioInputDevice] = []
    for index, device in enumerate(raw_devices):
        try:
            max_in = int(device.get("max_input_channels", 0) or 0)
        except Exception:
            max_in = 0
        if max_in <= 0:
            continue
        hostapi_index = int(device.get("hostapi", 0) or 0)
        hostapi_name = "?"
        if 0 <= hostapi_index < len(hostapis):
            hostapi_name = str(hostapis[hostapi_index].get("name", "?"))
        out.append(
            AudioInputDevice(
                index=index,
                name=str(device.get("name", f"Device {index}")),
                max_input_channels=max_in,
                default_samplerate=float(device.get("default_samplerate", 0.0) or 0.0),
                hostapi_index=hostapi_index,
                hostapi_name=hostapi_name,
            )
        )
    # Prefer ultrasound devices first, then higher-sample-rate devices.
    out.sort(
        key=lambda d: (
            0 if d.is_ultrasound else 1,
            -d.default_samplerate,
            d.name.lower(),
        )
    )
    return out


def pick_default_device(devices: List[AudioInputDevice]) -> Optional[AudioInputDevice]:
    for dev in devices:
        if "pettersson" in dev.name.lower() or "m500" in dev.name.lower():
            return dev
    for dev in devices:
        if dev.default_samplerate >= 192_000.0:
            return dev
    return devices[0] if devices else None


def level_to_dbfs(value: float, floor_db: float = -72.0) -> float:
    """Return a dBFS reading capped to *floor_db* for UI display."""
    amplitude = abs(float(value))
    if amplitude <= 1e-12:
        return floor_db
    return max(floor_db, 20.0 * float(np.log10(amplitude)))


def compress_waveform_for_display(waveform: np.ndarray, max_points: int) -> np.ndarray:
    """Reduce a waveform to a display-friendly min/max envelope."""
    data = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if data.size <= max_points:
        return data.copy()

    bucket_count = max(1, int(max_points) // 2)
    bucket_edges = np.linspace(0, data.size, bucket_count + 1, dtype=np.int32)
    envelope = np.empty(bucket_count * 2, dtype=np.float32)
    write_index = 0

    for bucket_index in range(bucket_count):
        start = int(bucket_edges[bucket_index])
        end = int(bucket_edges[bucket_index + 1])
        if end <= start:
            continue
        chunk = data[start:end]
        envelope[write_index] = float(np.min(chunk))
        envelope[write_index + 1] = float(np.max(chunk))
        write_index += 2

    if write_index == 0:
        return data[:max_points].copy()
    return envelope[:write_index]


# ---------------------------------------------------------------------------
# Recorder backend
# ---------------------------------------------------------------------------


class UltrasoundRecorder(QObject):
    """Threaded PortAudio capture + disk writer.

    Lifecycle:
      * :meth:`open_stream`  — start the PortAudio stream (used for preview).
      * :meth:`close_stream` — stop the stream (end preview).
      * :meth:`begin_recording` — open the target WAV and start writing
        incoming callback blocks to disk; call this just *before* the video
        recording starts.
      * :meth:`finalize`    — stop writing, trim the file so that it lines
        up with the video start/stop timestamps, and close the WAV.

    ``preview_ready`` is emitted at ~30 Hz with the most recent down-sampled
    chunk so a pyqtgraph plot can draw it. ``level_ready`` carries RMS & peak.
    ``status_changed`` emits a human-readable string. ``error_occurred`` is
    emitted on any failure.
    """

    preview_ready = Signal(object)           # np.ndarray float32 (mono)
    level_ready = Signal(float, float)       # rms, peak  (both in [0, 1])
    status_changed = Signal(str)
    error_occurred = Signal(str)
    recording_finalized = Signal(str, dict)  # wav_path, metadata

    # Preview a fixed time slice so the live view stays useful at high sample rates.
    _PREVIEW_WINDOW_SECONDS = 0.02
    _PREVIEW_MIN_SAMPLES = 2048
    _PREVIEW_RATE_HZ = 10.0

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lock = threading.RLock()
        self._stream = None  # type: ignore[assignment]
        self._device: Optional[AudioInputDevice] = None
        self._samplerate: int = 0
        self._channels: int = 1

        self._preview_buffer = np.zeros(self._PREVIEW_MIN_SAMPLES, dtype=np.float32)
        self._preview_write_index = 0
        self._preview_last_emit = 0.0
        self._preview_interval = 1.0 / self._PREVIEW_RATE_HZ
        self._waveform_preview_enabled = False

        # Writer state
        self._writer = None  # soundfile.SoundFile or None
        self._writer_lock = threading.RLock()
        self._writer_path: Optional[str] = None
        self._writing = False
        self._samples_written = 0
        self._audio_start_wallclock: Optional[float] = None   # first-callback time.time()
        self._recording_begin_wallclock: Optional[float] = None  # time.time() at begin_recording()
        self._video_start_wallclock: Optional[float] = None
        self._video_stop_wallclock: Optional[float] = None
        self._pending_samples: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4096)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop_event = threading.Event()
        self._first_callback_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return sd is not None

    @property
    def availability_error(self) -> Optional[str]:
        if sd is None:
            return _SOUNDDEVICE_IMPORT_ERROR or "sounddevice not installed"
        return None

    @property
    def is_streaming(self) -> bool:
        return self._stream is not None

    @property
    def is_recording(self) -> bool:
        return self._writing

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def preview_window_seconds(self) -> float:
        samplerate = max(1, self._samplerate or 48_000)
        return self._preview_buffer.shape[0] / float(samplerate)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def device(self) -> Optional[AudioInputDevice]:
        return self._device

    def set_waveform_preview_enabled(self, enabled: bool) -> None:
        self._waveform_preview_enabled = bool(enabled)

    def open_stream(
        self,
        device: AudioInputDevice,
        samplerate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> bool:
        """(Re)open the PortAudio input stream for live preview."""
        if sd is None:
            self.error_occurred.emit(
                f"sounddevice unavailable: {_SOUNDDEVICE_IMPORT_ERROR}"
            )
            return False

        with self._lock:
            self.close_stream()

            sr = int(samplerate or round(device.default_samplerate) or 48_000)
            ch = int(channels or min(device.max_input_channels, 1) or 1)

            try:
                stream = sd.InputStream(
                    device=device.index,
                    samplerate=sr,
                    channels=ch,
                    dtype="float32",
                    blocksize=0,         # let PortAudio choose
                    latency="low",
                    callback=self._audio_callback,
                )
                stream.start()
            except Exception as exc:
                self.error_occurred.emit(f"Failed to open {device.name}: {exc}")
                return False

            self._stream = stream
            self._device = device
            self._samplerate = sr
            self._channels = ch
            self._reset_preview_buffer(sr)
            self._audio_start_wallclock = None
            self._first_callback_event.clear()
            self.status_changed.emit(
                f"Listening: {device.name} @ {sr} Hz · {ch} ch"
            )
            return True

    def close_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            # Make sure any in-flight recording is torn down too.
            if self._writing:
                try:
                    self._abort_writer()
                except Exception:
                    pass
            self._reset_preview_buffer(self._samplerate or 48_000)
            self._audio_start_wallclock = None
            self._first_callback_event.clear()
            self.status_changed.emit("Idle")

    def wait_until_ready(self, timeout: float = 0.75) -> bool:
        """Return True once the PortAudio callback has delivered at least one block."""
        if self._audio_start_wallclock is not None:
            return True
        return self._first_callback_event.wait(max(0.0, float(timeout)))

    def begin_recording(self, wav_path: str) -> bool:
        """Start writing callback samples to *wav_path* (a .wav path)."""
        if sf is None:
            self.error_occurred.emit(
                f"soundfile unavailable: {_SOUNDFILE_IMPORT_ERROR}"
            )
            return False
        with self._lock:
            if self._stream is None:
                self.error_occurred.emit(
                    "Audio stream is not open — cannot start synchronised recording"
                )
                return False
            if self._writing:
                return True  # already recording

            wav_path = str(wav_path)
            try:
                Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
                writer = sf.SoundFile(
                    wav_path,
                    mode="w",
                    samplerate=self._samplerate,
                    channels=self._channels,
                    subtype="PCM_16",
                    format="WAV",
                )
            except Exception as exc:
                self.error_occurred.emit(f"Failed to open WAV {wav_path}: {exc}")
                return False

            with self._writer_lock:
                self._writer = writer
                self._writer_path = wav_path
                self._samples_written = 0
                self._video_start_wallclock = None
                self._video_stop_wallclock = None
                while not self._pending_samples.empty():
                    try:
                        self._pending_samples.get_nowait()
                    except queue.Empty:
                        break
                self._writer_stop_event.clear()
                # Stamp the recording start before _writing=True so that
                # _trim_to_video_span uses a reference aligned with the
                # actual capture, not the preview stream start.
                self._recording_begin_wallclock = time.time()
                self._writing = True

            self._writer_thread = threading.Thread(
                target=self._writer_loop, name="UltrasoundWriter", daemon=True
            )
            self._writer_thread.start()
            self.status_changed.emit(f"Recording audio → {Path(wav_path).name}")
            return True

    def mark_video_started(self, wallclock: Optional[float] = None) -> None:
        """Record the wall-clock time of the first video frame/accept moment."""
        ts = float(wallclock) if wallclock is not None else time.time()
        with self._writer_lock:
            self._video_start_wallclock = ts

    def mark_video_stopped(self, wallclock: Optional[float] = None) -> None:
        ts = float(wallclock) if wallclock is not None else time.time()
        with self._writer_lock:
            self._video_stop_wallclock = ts

    def finalize(self, target_duration_seconds: Optional[float] = None) -> Optional[Dict[str, object]]:
        """Stop writing and trim the WAV to match the saved video duration."""
        with self._lock:
            if not self._writing:
                return None
            self._writer_stop_event.set()
            self._writing = False

        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self._writer_thread = None

        with self._writer_lock:
            writer = self._writer
            path = self._writer_path
            samples_written = int(self._samples_written)
            audio_t0 = self._audio_start_wallclock
            recording_begin = self._recording_begin_wallclock
            video_t0 = self._video_start_wallclock
            video_t1 = self._video_stop_wallclock
            self._writer = None
            self._writer_path = None
            self._samples_written = 0
            self._recording_begin_wallclock = None

        if writer is not None:
            try:
                writer.flush()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass

        if path is None:
            return None

        metadata = self._trim_to_video_span(
            path,
            samples_written,
            audio_t0,
            recording_begin,
            video_t0,
            video_t1,
            target_duration_seconds=target_duration_seconds,
        )
        self.status_changed.emit(
            f"Audio saved → {Path(path).name}  ({metadata.get('duration_seconds', 0.0):.2f}s)"
        )
        self.recording_finalized.emit(path, metadata)
        return metadata

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback — runs in a dedicated PortAudio thread."""
        # Preview path: always run regardless of recording state.
        try:
            if status:
                # XRun / InputOverflow shows up here. We forward once as info.
                self.status_changed.emit(f"PortAudio: {status}")
            if self._audio_start_wallclock is None:
                # Estimate first-sample wall-clock: now() minus the duration of
                # samples already buffered in this callback.
                sr = self._samplerate or 1
                self._audio_start_wallclock = time.time() - (frames / float(sr))
            self._first_callback_event.set()

            block = np.asarray(indata, dtype=np.float32)
            if block.ndim == 2 and block.shape[1] > 1:
                mono = block.mean(axis=1)
            elif block.ndim == 2:
                mono = block[:, 0]
            else:
                mono = block

            if not self._writing:
                self._update_preview(mono)

            if self._writing:
                # Copy before forwarding — PortAudio reuses the buffer.
                try:
                    self._pending_samples.put_nowait(block.copy())
                except queue.Full:
                    # Drop block rather than block the audio thread — signal it.
                    self.error_occurred.emit(
                        "Audio writer queue overflowed — samples dropped"
                    )
        except Exception:
            # Never raise from the PortAudio callback.
            traceback.print_exc()

    def _reset_preview_buffer(self, samplerate: int) -> None:
        sample_count = max(
            self._PREVIEW_MIN_SAMPLES,
            int(round(max(1, int(samplerate)) * self._PREVIEW_WINDOW_SECONDS)),
        )
        self._preview_buffer = np.zeros(sample_count, dtype=np.float32)
        self._preview_write_index = 0
        self._preview_last_emit = 0.0

    def _update_preview(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        # Fill the rolling preview buffer.
        buf = self._preview_buffer
        n = buf.shape[0]
        if mono.size >= n:
            buf[:] = mono[-n:]
            self._preview_write_index = 0
        else:
            idx = self._preview_write_index
            end = idx + mono.size
            if end <= n:
                buf[idx:end] = mono
            else:
                first = n - idx
                buf[idx:] = mono[:first]
                buf[: mono.size - first] = mono[first:]
            self._preview_write_index = (idx + mono.size) % n

        now = time.time()
        if now - self._preview_last_emit < self._preview_interval:
            return
        self._preview_last_emit = now

        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
        if self._waveform_preview_enabled:
            idx = self._preview_write_index
            if idx == 0:
                wave = buf.copy()
            else:
                wave = np.concatenate((buf[idx:], buf[:idx]))
            self.preview_ready.emit(wave)
        self.level_ready.emit(min(rms, 1.0), min(peak, 1.0))

    def _writer_loop(self) -> None:
        """Drain the pending-samples queue onto the SoundFile writer."""
        while not self._writer_stop_event.is_set() or not self._pending_samples.empty():
            try:
                block = self._pending_samples.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._writer_lock:
                    writer = self._writer
                    if writer is None:
                        continue
                    writer.write(block)
                    self._samples_written += int(block.shape[0])
            except Exception as exc:
                self.error_occurred.emit(f"WAV write failed: {exc}")
                break

    def _abort_writer(self) -> None:
        """Emergency teardown (called when the stream closes mid-recording)."""
        self._writer_stop_event.set()
        self._writing = False
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._writer_thread = None
        with self._writer_lock:
            writer = self._writer
            self._writer = None
            self._writer_path = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def _trim_to_video_span(
        self,
        wav_path: str,
        samples_written: int,
        audio_t0: Optional[float],
        recording_begin: Optional[float],
        video_t0: Optional[float],
        video_t1: Optional[float],
        target_duration_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        """Rewrite *wav_path* so that sample 0 == first video frame.

        ``recording_begin`` is the wall-clock time stamped the moment
        ``begin_recording()`` set ``_writing = True``.  It is always used as
        the audio-capture reference for the leading-trim calculation, because
        ``audio_t0`` (the first PortAudio callback) may have fired long before
        recording started (during the live preview), which would otherwise
        over-estimate the leading offset and steal time from the end of the WAV.
        ``audio_t0`` is kept in metadata for diagnostics. When available,
        ``target_duration_seconds`` should be the encoded MP4 duration
        (recorded frames / output fps); that is the most accurate trim target.
        """
        sr = self._samplerate or 1
        ch = self._channels or 1
        actual_samples = int(samples_written)
        if sf is not None:
            try:
                info = sf.info(wav_path)
                actual_samples = int(getattr(info, "frames", actual_samples) or actual_samples)
            except Exception:
                actual_samples = int(samples_written)

        meta: Dict[str, object] = {
            "samplerate": sr,
            "channels": ch,
            "samples_captured": int(actual_samples),
            "audio_start_wallclock": audio_t0,
            "recording_begin_wallclock": recording_begin,
            "video_start_wallclock": video_t0,
            "video_stop_wallclock": video_t1,
            "video_duration_seconds": None,
            "trim_target": "",
            "trim_leading_samples": 0,
            "trim_trailing_samples": 0,
            "duration_seconds": actual_samples / float(sr),
            "trimmed": False,
            "trim_skipped_reason": "",
        }

        if actual_samples <= 0:
            meta["trim_skipped_reason"] = "empty_capture"
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.error_occurred.emit(
                "Ultrasound capture produced no samples; empty WAV removed."
            )
            return meta

        # Use recording_begin (stamped at begin_recording()) as the audio
        # reference so leading trim is relative to when samples were actually
        # being written, not the earlier preview-stream callback.
        ref_t0 = recording_begin if recording_begin is not None else audio_t0
        if sf is None or ref_t0 is None or video_t0 is None:
            return meta

        leading = int(round((video_t0 - ref_t0) * sr))
        if leading < 0:
            leading = 0
        if leading >= actual_samples:
            leading = actual_samples

        target_duration_value = None
        if target_duration_seconds is not None:
            try:
                target_duration_value = float(target_duration_seconds)
            except (TypeError, ValueError):
                target_duration_value = None
            if target_duration_value is not None and target_duration_value <= 0.0:
                target_duration_value = None

        if target_duration_value is not None:
            video_duration_samples = int(round(target_duration_value * sr))
            meta["video_duration_seconds"] = float(target_duration_value)
            meta["trim_target"] = "encoded_video_duration"
        elif video_t1 is not None:
            video_duration_seconds = max(0.0, float(video_t1) - float(video_t0))
            video_duration_samples = int(round(video_duration_seconds * sr))
            meta["video_duration_seconds"] = float(video_duration_seconds)
            meta["trim_target"] = "captured_video_span"
        else:
            video_duration_samples = actual_samples - leading
            meta["video_duration_seconds"] = video_duration_samples / float(sr)
            meta["trim_target"] = "audio_capture"
        video_duration_samples = max(0, video_duration_samples)

        keep = min(video_duration_samples, actual_samples - leading)
        trailing = actual_samples - leading - keep

        if leading == 0 and trailing == 0:
            return meta

        if keep <= 0:
            meta["trim_skipped_reason"] = "zero_length_trim"
            meta["trim_leading_samples"] = int(leading)
            meta["trim_trailing_samples"] = int(max(0, actual_samples - leading))
            meta["duration_seconds"] = actual_samples / float(sr)
            self.status_changed.emit(
                "Audio kept untrimmed because sync trim would have removed all samples."
            )
            return meta

        try:
            # Read the full file, slice, and rewrite. Libsndfile streams on disk
            # so memory usage stays O(chunk) if we use a block iterator.
            tmp_path = wav_path + ".trim.wav"
            with sf.SoundFile(wav_path, mode="r") as src:
                subtype = src.subtype
                fmt = src.format
                with sf.SoundFile(
                    tmp_path,
                    mode="w",
                    samplerate=sr,
                    channels=ch,
                    subtype=subtype,
                    format=fmt,
                ) as dst:
                    remaining = keep
                    src.seek(leading)
                    block_size = max(sr, 1) * 2  # ~2 s blocks
                    while remaining > 0:
                        chunk = src.read(
                            min(block_size, remaining),
                            dtype="float32",
                            always_2d=True,
                        )
                        if chunk.shape[0] == 0:
                            break
                        dst.write(chunk)
                        remaining -= chunk.shape[0]
            os.replace(tmp_path, wav_path)
            meta["trim_leading_samples"] = int(leading)
            meta["trim_trailing_samples"] = int(trailing)
            meta["duration_seconds"] = keep / float(sr)
            meta["trimmed"] = True
        except Exception as exc:
            self.error_occurred.emit(f"WAV trim failed (kept untrimmed file): {exc}")

        return meta


# ---------------------------------------------------------------------------
# Dock panel
# ---------------------------------------------------------------------------


class UltrasoundPanel(QWidget):
    """Tool panel hosting device selection, live waveform + enable toggle."""

    enable_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("UltrasoundPanel")

        self.recorder = UltrasoundRecorder(self)
        self.recorder.preview_ready.connect(self._on_preview_ready)
        self.recorder.level_ready.connect(self._on_level_ready)
        self.recorder.status_changed.connect(self._on_status_changed)
        self.recorder.error_occurred.connect(self._on_error)
        self.recorder.recording_finalized.connect(self._on_recording_finalized)

        self._devices: List[AudioInputDevice] = []
        self._current_device: Optional[AudioInputDevice] = None
        self._peak_hold = 0.0
        self._peak_hold_decay = 0.92  # per preview tick
        self._last_rms = 0.0
        self._last_peak = 0.0
        self._preview_display_gain = 1.0
        self.waveform_preview_visible = False
        self._preview_target_peak = 0.6
        self._preview_max_display_gain = 48.0
        self._preview_min_peak = 5e-5
        self._waveform_max_points = 1400
        # Throttle the pyqtgraph repaint aggressively. The waveform is only a
        # diagnostic view, so a very low refresh rate is preferable to dropped frames.
        self._waveform_last_paint_monotonic = 0.0
        self._waveform_min_paint_interval_s = 1.0 / 8.0
        self._waveform_last_xrange_ms: Optional[float] = None
        self._waveform_stats_last_update_monotonic = 0.0
        self._waveform_stats_min_interval_s = 0.5

        self._build_ui()
        self.refresh_devices()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Device picker ---------------------------------------------------
        device_group = QGroupBox("Ultrasound Microphone")
        dg_layout = QVBoxLayout(device_group)
        dg_layout.setContentsMargins(10, 14, 10, 10)
        dg_layout.setSpacing(8)

        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.currentIndexChanged.connect(self._on_device_index_changed)
        dg_layout.addWidget(self.device_combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_refresh = QPushButton("Refresh devices")
        self.btn_refresh.setObjectName("secondary")
        self.btn_refresh.clicked.connect(self.refresh_devices)
        btn_row.addWidget(self.btn_refresh)

        self.btn_monitor = QPushButton("Start monitor")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.toggled.connect(self._on_monitor_toggled)
        btn_row.addWidget(self.btn_monitor)
        dg_layout.addLayout(btn_row)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setSpacing(6)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8_000, 500_000)
        self.sr_spin.setSingleStep(1_000)
        self.sr_spin.setSuffix(" Hz")
        self.sr_spin.setValue(384_000)
        form.addRow("Sample rate", self.sr_spin)

        self.ch_spin = QSpinBox()
        self.ch_spin.setRange(1, 8)
        self.ch_spin.setValue(1)
        form.addRow("Channels", self.ch_spin)
        dg_layout.addLayout(form)

        self.status_label = QLabel("No input device detected.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        dg_layout.addWidget(self.status_label)

        root.addWidget(device_group)

        # Sync enable toggle ---------------------------------------------
        sync_group = QGroupBox("Synchronised Recording")
        sync_layout = QVBoxLayout(sync_group)
        sync_layout.setContentsMargins(10, 14, 10, 10)
        sync_layout.setSpacing(8)

        self.enable_check = QCheckBox(
            "Record ultrasound WAV synchronised with each video"
        )
        self.enable_check.setToolTip(
            "When enabled, pressing Record starts an audio stream on the selected\n"
            "device before the video recording begins. The WAV is saved next to\n"
            "the .mp4 using the same base filename and trimmed to match the video\n"
            "start/stop timestamps."
        )
        self.enable_check.toggled.connect(self._on_enable_toggled)
        sync_layout.addWidget(self.enable_check)

        self.sync_status_label = QLabel("")
        self.sync_status_label.setWordWrap(True)
        self.sync_status_label.setStyleSheet("color: #6fe06e; font-size: 11px;")
        sync_layout.addWidget(self.sync_status_label)

        root.addWidget(sync_group)

        # Waveform + level meter -----------------------------------------
        waveform_group = QGroupBox("Live Waveform")
        wf_layout = QVBoxLayout(waveform_group)
        wf_layout.setContentsMargins(8, 14, 8, 10)
        wf_layout.setSpacing(6)

        self.waveform_preview_toggle = QCheckBox(
            "Show live waveform preview (recommended off while recording)"
        )
        self.waveform_preview_toggle.setChecked(False)
        self.waveform_preview_toggle.toggled.connect(self._on_waveform_preview_toggled)
        wf_layout.addWidget(self.waveform_preview_toggle)

        self.waveform_content = QWidget()
        waveform_content_layout = QVBoxLayout(self.waveform_content)
        waveform_content_layout.setContentsMargins(0, 0, 0, 0)
        waveform_content_layout.setSpacing(6)

        self.plot_widget = pg.PlotWidget(background="#0b1120")
        self.plot_widget.setFrameShape(QFrame.NoFrame)
        self.plot_widget.setMinimumHeight(154)
        self.plot_widget.setYRange(-1.05, 1.05, padding=0)
        self.plot_widget.setXRange(-20.0, 0.0, padding=0)
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.12)
        self.plot_widget.setMouseEnabled(False, False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setStyleSheet(
            "border: 1px solid #1f2937; border-radius: 8px; background: #0b1120;"
        )
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setContentsMargins(8, 8, 8, 8)
        view_box = plot_item.getViewBox()
        view_box.setDefaultPadding(0.0)
        view_box.enableAutoRange(x=False, y=False)
        self._waveform_fill_curve = self.plot_widget.plot(
            pen=None,
            fillLevel=0.0,
            brush=pg.mkBrush(51, 200, 255, 44),
        )
        self._waveform_curve = self.plot_widget.plot(
            pen=pg.mkPen("#6be3ff", width=1.8)
        )
        self._zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen("#45475a", width=1, style=Qt.DashLine)
        )
        self.plot_widget.addItem(self._zero_line)
        waveform_content_layout.addWidget(self.plot_widget)

        self.waveform_stats_label = QLabel(
            "Waveform preview hidden to reduce frame drops"
        )
        self.waveform_stats_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        waveform_content_layout.addWidget(self.waveform_stats_label)

        meter_row = QHBoxLayout()
        meter_row.setSpacing(8)

        meter_col = QVBoxLayout()
        meter_col.setSpacing(3)
        meter_col.addWidget(self._meter_caption("RMS"))
        self.rms_bar = self._make_meter_bar("#6fe06e")
        meter_col.addWidget(self.rms_bar)
        meter_row.addLayout(meter_col)

        peak_col = QVBoxLayout()
        peak_col.setSpacing(3)
        peak_col.addWidget(self._meter_caption("Peak"))
        self.peak_bar = self._make_meter_bar("#f38ba8")
        peak_col.addWidget(self.peak_bar)
        meter_row.addLayout(peak_col)

        waveform_content_layout.addLayout(meter_row)
        wf_layout.addWidget(self.waveform_content)

        root.addWidget(waveform_group, 1)
        self._set_waveform_preview_visible(False)

        root.addStretch()

    def _meter_caption(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
        return lbl

    def _meter_level_to_value(self, value: float) -> int:
        floor_db = -72.0
        db_value = level_to_dbfs(value, floor_db=floor_db)
        proportion = (db_value - floor_db) / abs(floor_db)
        return int(round(max(0.0, min(1.0, proportion)) * 1000.0))

    def _update_waveform_stats(self, force: bool = False) -> None:
        if not force:
            now_monotonic = time.monotonic()
            if (
                now_monotonic - self._waveform_stats_last_update_monotonic
                < self._waveform_stats_min_interval_s
            ):
                return
            self._waveform_stats_last_update_monotonic = now_monotonic
        else:
            self._waveform_stats_last_update_monotonic = time.monotonic()
        window_ms = self.recorder.preview_window_seconds * 1000.0
        rms_db = level_to_dbfs(self._last_rms)
        peak_db = level_to_dbfs(self._last_peak)
        self.waveform_stats_label.setText(
            f"{window_ms:.1f} ms window | display x{self._preview_display_gain:.1f} | "
            f"RMS {rms_db:.1f} dBFS | Peak {peak_db:.1f} dBFS"
        )

    def _reset_live_waveform_view(self) -> None:
        self._preview_display_gain = 1.0
        self._peak_hold = 0.0
        self._last_rms = 0.0
        self._last_peak = 0.0
        self._waveform_fill_curve.setData([], [])
        self._waveform_curve.setData([], [])
        self._waveform_last_xrange_ms = None
        self._waveform_last_paint_monotonic = 0.0
        self._waveform_stats_last_update_monotonic = 0.0
        self.rms_bar.setValue(0)
        self.peak_bar.setValue(0)
        self.waveform_stats_label.setText(
            "Adaptive preview | start monitor to inspect input"
            if self.waveform_preview_visible
            else "Waveform preview hidden to reduce frame drops"
        )

    def _set_waveform_preview_visible(self, visible: bool) -> None:
        self.waveform_preview_visible = bool(visible)
        self.waveform_content.setVisible(self.waveform_preview_visible)
        self.recorder.set_waveform_preview_enabled(self.waveform_preview_visible)
        if not self.waveform_preview_visible:
            self._reset_live_waveform_view()
        else:
            self._update_waveform_stats(force=True)

    def _make_meter_bar(self, chunk_color: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(14)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bar.setStyleSheet(
            "QProgressBar { background: #1e1e2e; border-radius: 3px; }"
            f"QProgressBar::chunk {{ background: {chunk_color}; border-radius: 3px; }}"
        )
        return bar

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    @Slot()
    def refresh_devices(self) -> None:
        if not self.recorder.is_available:
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            self.device_combo.addItem("sounddevice not installed", None)
            self.device_combo.setEnabled(False)
            self.device_combo.blockSignals(False)
            self.btn_monitor.setEnabled(False)
            self.enable_check.setEnabled(False)
            err = self.recorder.availability_error or ""
            self.status_label.setText(
                f"Audio capture unavailable — install sounddevice+soundfile.\n{err}"
            )
            self.status_label.setStyleSheet("color: #f38ba8; font-size: 11px;")
            return

        self._devices = enumerate_input_devices()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        if not self._devices:
            self.device_combo.addItem("No input devices found", None)
            self.device_combo.setEnabled(False)
            self.btn_monitor.setEnabled(False)
            self.enable_check.setEnabled(False)
            self.status_label.setText("No microphone detected — plug in the Pettersson.")
            self.status_label.setStyleSheet("color: #f38ba8; font-size: 11px;")
        else:
            self.device_combo.setEnabled(True)
            self.btn_monitor.setEnabled(True)
            self.enable_check.setEnabled(True)
            for dev in self._devices:
                label = ("🎙  " if dev.is_ultrasound else "    ") + dev.display
                self.device_combo.addItem(label, dev.index)
            default = pick_default_device(self._devices)
            if default is not None:
                idx = self._devices.index(default)
                self.device_combo.setCurrentIndex(idx)
                self._apply_device(default)
        self.device_combo.blockSignals(False)

    @Slot(int)
    def _on_device_index_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._devices):
            return
        device = self._devices[index]
        self._apply_device(device)
        if self.btn_monitor.isChecked():
            # Restart preview on the new device.
            self.recorder.close_stream()
            self.recorder.open_stream(
                device, samplerate=self.sr_spin.value(), channels=self.ch_spin.value()
            )

    def _apply_device(self, device: AudioInputDevice) -> None:
        self._current_device = device
        sr_default = int(round(device.default_samplerate)) or 48_000
        self.sr_spin.blockSignals(True)
        self.sr_spin.setValue(sr_default)
        self.sr_spin.blockSignals(False)
        self.ch_spin.blockSignals(True)
        self.ch_spin.setMaximum(max(1, device.max_input_channels))
        self.ch_spin.setValue(min(self.ch_spin.value(), max(1, device.max_input_channels)))
        self.ch_spin.blockSignals(False)
        tag = "Ultrasound ready" if device.is_ultrasound else "Standard mic detected"
        self.status_label.setText(
            f"{tag}: {device.name}  ·  default {sr_default} Hz  ·  {device.hostapi_name}"
        )
        color = "#6fe06e" if device.is_ultrasound else "#e0bd6e"
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    @Slot(bool)
    def _on_monitor_toggled(self, checked: bool) -> None:
        if checked:
            self._start_monitor_stream()
        else:
            self.recorder.close_stream()
            self.btn_monitor.setText("Start monitor")
            self._reset_live_waveform_view()

    @Slot(bool)
    def _on_enable_toggled(self, checked: bool) -> None:
        """Auto-start the audio stream when the user enables sync recording."""
        self.enable_toggled.emit(checked)
        if checked and not self.recorder.is_streaming:
            self._start_monitor_stream()

    @Slot(bool)
    def _on_waveform_preview_toggled(self, checked: bool) -> None:
        self._set_waveform_preview_visible(checked)

    def _start_monitor_stream(self) -> bool:
        """Open the PortAudio stream on the current device (non-recording path)."""
        device = self._current_device
        if device is None:
            self.btn_monitor.blockSignals(True)
            self.btn_monitor.setChecked(False)
            self.btn_monitor.blockSignals(False)
            return False
        ok = self.recorder.open_stream(
            device, samplerate=self.sr_spin.value(), channels=self.ch_spin.value()
        )
        if not ok:
            self.btn_monitor.blockSignals(True)
            self.btn_monitor.setChecked(False)
            self.btn_monitor.blockSignals(False)
            return False
        self.btn_monitor.blockSignals(True)
        self.btn_monitor.setChecked(True)
        self.btn_monitor.setText("Stop monitor")
        self.btn_monitor.blockSignals(False)
        return True

    # ------------------------------------------------------------------
    # Recorder signals
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_preview_ready(self, waveform: np.ndarray) -> None:
        if not self.waveform_preview_visible:
            return
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            self._reset_live_waveform_view()
            return

        centered = samples - float(np.mean(samples, dtype=np.float64))
        raw_peak = float(np.max(np.abs(centered))) if centered.size else 0.0
        if raw_peak >= self._preview_min_peak:
            target_gain = min(
                self._preview_max_display_gain,
                max(1.0, self._preview_target_peak / raw_peak),
            )
        else:
            target_gain = 1.0

        alpha = 0.35 if target_gain < self._preview_display_gain else 0.16
        self._preview_display_gain += (target_gain - self._preview_display_gain) * alpha
        display_wave = np.clip(centered * self._preview_display_gain, -1.0, 1.0)
        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._waveform_last_paint_monotonic
            < self._waveform_min_paint_interval_s
        ):
            return
        self._waveform_last_paint_monotonic = now_monotonic

        display_wave = compress_waveform_for_display(
            display_wave, self._waveform_max_points
        )

        window_ms = self.recorder.preview_window_seconds * 1000.0
        x_axis = np.linspace(-window_ms, 0.0, display_wave.size, dtype=np.float32)
        if self._waveform_last_xrange_ms != window_ms:
            self.plot_widget.setXRange(-window_ms, 0.0, padding=0)
            self._waveform_last_xrange_ms = window_ms
        self._waveform_fill_curve.setData(x_axis, display_wave)
        self._waveform_curve.setData(x_axis, display_wave)
        self._update_waveform_stats()

    @Slot(float, float)
    def _on_level_ready(self, rms: float, peak: float) -> None:
        if not self.waveform_preview_visible:
            return
        self._last_rms = float(rms)
        self._last_peak = float(peak)
        self._peak_hold = max(peak, self._peak_hold * self._peak_hold_decay)
        self.rms_bar.setValue(self._meter_level_to_value(rms))
        self.peak_bar.setValue(self._meter_level_to_value(self._peak_hold))
        self._update_waveform_stats()

    @Slot(str)
    def _on_status_changed(self, text: str) -> None:
        self.sync_status_label.setText(text)

    @Slot(str)
    def _on_error(self, text: str) -> None:
        self.sync_status_label.setText(f"⚠ {text}")
        self.sync_status_label.setStyleSheet("color: #f38ba8; font-size: 11px;")
        # Restore "ok" style after a short delay so repeated errors stay visible.
        QTimer.singleShot(
            4000,
            lambda: self.sync_status_label.setStyleSheet(
                "color: #6fe06e; font-size: 11px;"
            ),
        )

    @Slot(str, dict)
    def _on_recording_finalized(self, path: str, metadata: dict) -> None:
        dur = float(metadata.get("duration_seconds", 0.0) or 0.0)
        leading_ms = 1000.0 * float(metadata.get("trim_leading_samples", 0) or 0) / max(1, self.recorder.samplerate)
        self.sync_status_label.setText(
            f"Saved {Path(path).name}  ·  {dur:.2f}s  ·  aligned (-{leading_ms:.1f} ms lead)"
        )

    # ------------------------------------------------------------------
    # Integration API used by MainWindow
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return (
            self.enable_check.isChecked()
            and self.recorder.is_available
            and sf is not None
        )

    def prepare_for_recording(self, wav_path: str) -> bool:
        """Start writing to *wav_path*.  **Must not block the GUI thread.**

        The PortAudio stream must already be running (via the monitor toggle
        or :meth:`ensure_stream_running`).  If it is not, this method returns
        ``False`` immediately — it never opens a stream on the hot path so
        that Arduino TTL ↔ video latency stays at zero.
        """
        if not self.is_enabled():
            return False
        if not self.recorder.is_streaming:
            self._on_error(
                "Turn on the audio monitor before recording "
                "(stream must already be running)."
            )
            return False
        return self.recorder.begin_recording(wav_path)

    def notify_video_started(self, wallclock: Optional[float] = None) -> None:
        self.recorder.mark_video_started(wallclock)

    def notify_video_stopped(self, wallclock: Optional[float] = None) -> None:
        self.recorder.mark_video_stopped(wallclock)

    def finalize_recording(self, target_duration_seconds: Optional[float] = None) -> Optional[Dict[str, object]]:
        return self.recorder.finalize(target_duration_seconds=target_duration_seconds)

    def shutdown(self) -> None:
        try:
            self.recorder.close_stream()
        except Exception:
            pass
