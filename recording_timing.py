"""
Pure timing math for recording self-audits.

After a recording stops, the app cross-checks three independent measures of
"how long was this" — wall-clock capture span, frames divided by encoded FPS,
and saved audio length — and warns when they disagree (a sign of dropped
frames or an FPS mismatch between capture and the MP4). These functions are
deliberately side-effect free so they can be unit-tested in isolation
(see tests/test_recording_timing.py).
"""
from __future__ import annotations

from typing import List, Optional


def capture_duration_seconds(
    start_wallclock: Optional[float],
    stop_wallclock: Optional[float],
) -> Optional[float]:
    """Wall-clock seconds between recording start and stop.

    Returns None if either timestamp is missing, and clamps to 0 so a
    out-of-order pair never yields a negative duration.
    """
    if start_wallclock is None or stop_wallclock is None:
        return None
    return max(0.0, float(stop_wallclock) - float(start_wallclock))


def measured_capture_fps(
    recorded_frames: int,
    capture_duration_s: Optional[float],
) -> Optional[float]:
    """Actual average FPS = frames captured / wall-clock capture duration.

    Returns None when the duration or frame count is non-positive (nothing
    meaningful to divide).
    """
    if capture_duration_s is None or float(capture_duration_s) <= 0.0:
        return None
    if int(recorded_frames) <= 0:
        return None
    return float(recorded_frames) / float(capture_duration_s)


def encoded_video_duration_seconds(
    recorded_frames: int,
    encoded_fps: Optional[float],
) -> Optional[float]:
    """Playback length the MP4 will report = frames / the FPS it was encoded at.

    This is what a player shows, which can differ from the real capture span
    when the encoder FPS does not match the achieved capture rate.
    """
    if encoded_fps is None:
        return None
    fps = float(encoded_fps)
    if fps <= 0.0:
        return None
    if int(recorded_frames) <= 0:
        return None
    return float(recorded_frames) / fps


def percent_delta(
    reference_value: Optional[float],
    measured_value: Optional[float],
) -> Optional[float]:
    """Signed percentage difference of measured vs reference ((m-r)/r*100).

    Returns None when either value is missing or the reference is ~0 (the
    percentage would be undefined / explosive).
    """
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
    encoded_video_duration_s: Optional[float] = None,
    duration_tolerance_s: float = 0.25,
    fps_tolerance_pct: float = 5.0,
    audio_tolerance_s: float = 0.25,
) -> List[str]:
    """Compare the timing measures and return human-readable warning lines.

    Emits a warning when (a) the captured duration drifts from the requested
    limit, (b) encoded FPS differs from measured capture FPS beyond tolerance,
    or (c) saved audio length diverges from the video span. An empty list means
    every measure agreed within tolerance. Each tolerance is tunable so callers
    can tighten or relax the audit.
    """
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

    audio_reference_s = encoded_video_duration_s
    audio_reference_label = "encoded video duration"
    if audio_reference_s is None:
        audio_reference_s = capture_duration_s
        audio_reference_label = "measured captured video span"

    if audio_reference_s is not None and audio_duration_s is not None:
        audio_delta_s = float(audio_duration_s) - float(audio_reference_s)
        if abs(audio_delta_s) > float(audio_tolerance_s):
            warnings.append(
                f"Saved audio duration differs from {audio_reference_label} by "
                f"{audio_delta_s:+.3f} s."
            )

    return warnings