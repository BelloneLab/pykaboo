from __future__ import annotations

from typing import List, Optional


def capture_duration_seconds(
    start_wallclock: Optional[float],
    stop_wallclock: Optional[float],
) -> Optional[float]:
    if start_wallclock is None or stop_wallclock is None:
        return None
    return max(0.0, float(stop_wallclock) - float(start_wallclock))


def measured_capture_fps(
    recorded_frames: int,
    capture_duration_s: Optional[float],
) -> Optional[float]:
    if capture_duration_s is None or float(capture_duration_s) <= 0.0:
        return None
    if int(recorded_frames) <= 0:
        return None
    return float(recorded_frames) / float(capture_duration_s)


def percent_delta(
    reference_value: Optional[float],
    measured_value: Optional[float],
) -> Optional[float]:
    if reference_value is None or measured_value is None:
        return None
    reference = float(reference_value)
    if abs(reference) <= 1e-9:
        return None
    return ((float(measured_value) - reference) / reference) * 100.0


def build_recording_timing_warnings(
    requested_duration_s: Optional[int] = None,
    capture_duration_s: Optional[float] = None,
    encoded_fps: Optional[float] = None,
    measured_fps: Optional[float] = None,
    audio_duration_s: Optional[float] = None,
    duration_tolerance_s: float = 0.25,
    fps_tolerance_pct: float = 5.0,
    audio_tolerance_s: float = 0.25,
) -> List[str]:
    warnings: List[str] = []

    if requested_duration_s is not None and requested_duration_s > 0 and capture_duration_s is not None:
        duration_delta_s = float(capture_duration_s) - float(requested_duration_s)
        if abs(duration_delta_s) > float(duration_tolerance_s):
            warnings.append(
                "Measured capture duration deviates from the requested limit by "
                f"{duration_delta_s:+.3f} s."
            )

    fps_delta_pct = percent_delta(encoded_fps, measured_fps)
    if fps_delta_pct is not None and abs(float(fps_delta_pct)) > float(fps_tolerance_pct):
        warnings.append(
            "Encoded FPS differs from measured capture FPS by "
            f"{fps_delta_pct:+.2f}% — MP4 playback duration may not match wall-clock capture."
        )

    if capture_duration_s is not None and audio_duration_s is not None:
        audio_delta_s = float(audio_duration_s) - float(capture_duration_s)
        if abs(audio_delta_s) > float(audio_tolerance_s):
            warnings.append(
                "Saved audio duration differs from measured captured video span by "
                f"{audio_delta_s:+.3f} s."
            )

    return warnings