"""Time-resolved wavelet features for behavior segmentation.

Static and rolling features describe the *level* of a signal. Many behaviors are
distinguished instead by their *rhythm*: wrestling and struggling are oscillatory,
freezing is flat, tail rattling is a fast tremor. A continuous wavelet transform
(Morlet) gives, for each frame, the signal's energy at several behavioral
timescales while preserving temporal resolution, so it slots straight into the
framewise TCN as extra input channels.

The transform is applied per contiguous identity time series (never across video
or identity boundaries) on a curated set of dynamic source signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt

    _HAS_PYWT = True
except Exception:  # pragma: no cover - graceful fallback
    _HAS_PYWT = False

# Source signals to decompose, in priority order. Missing ones are skipped.
DEFAULT_WAVELET_SIGNALS = [
    "speed",
    "area_velocity",
    "angular_velocity",
    "pose_nose_speed",
    "pose_body_speed",
    "pose_tail_speed",
    "pose_body_length",
    "pose_heading_angvel",
    "pp_body_body",
    "pp_nose_nose",
    "pp_approach_speed",
]

# Behavioral timescales in seconds (fast tremor -> slow bout envelope).
DEFAULT_PERIODS_SECONDS = [0.12, 0.25, 0.5, 1.0, 2.0, 4.0]


def periods_to_scales(periods_seconds: list[float], frame_rate: float) -> np.ndarray:
    """Convert target oscillation periods to Morlet CWT scales."""

    central = pywt.central_frequency("morl") if _HAS_PYWT else 0.8125
    return np.asarray(
        [central * frame_rate * float(p) for p in periods_seconds], dtype=np.float64
    )


def _cwt_power(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Return ``[num_scales, T]`` log1p wavelet power for a 1-D signal."""

    x = np.nan_to_num(signal.astype(np.float64), nan=0.0)
    x = x - x.mean()
    if not _HAS_PYWT:
        # Fallback: band-pass energy via difference-of-gaussian envelopes.
        out = np.zeros((len(scales), len(x)))
        for i, s in enumerate(scales):
            k = max(int(round(s)), 1)
            kernel = np.ones(k) / k
            smooth = np.convolve(x, kernel, mode="same")
            out[i] = np.log1p(np.abs(x - smooth))
        return out
    coeffs, _ = pywt.cwt(x, scales, "morl")
    return np.log1p(np.abs(coeffs))


def add_wavelet_features(
    track_df: pd.DataFrame,
    signal_cols: list[str],
    frame_rate: float,
    periods_seconds: list[float] | None = None,
) -> pd.DataFrame:
    """Append wavelet-power columns for ``signal_cols`` to one identity table.

    ``track_df`` must be a single contiguous identity series sorted by frame.
    Adds ``<signal>_cwt_p{i}`` columns (one per scale).
    """

    periods_seconds = periods_seconds or DEFAULT_PERIODS_SECONDS
    available = [c for c in signal_cols if c in track_df.columns]
    if not available:
        return track_df
    scales = periods_to_scales(periods_seconds, frame_rate)
    new_cols: dict[str, np.ndarray] = {}
    for col in available:
        power = _cwt_power(track_df[col].to_numpy(), scales)
        for i in range(power.shape[0]):
            new_cols[f"{col}_cwt_p{i}"] = power[i].astype(np.float32)
    if new_cols:
        track_df = pd.concat(
            [track_df, pd.DataFrame(new_cols, index=track_df.index)], axis=1
        )
    return track_df


def wavelet_column_names(
    signal_cols: list[str], periods_seconds: list[float] | None = None
) -> list[str]:
    periods_seconds = periods_seconds or DEFAULT_PERIODS_SECONDS
    names: list[str] = []
    for col in signal_cols:
        for i in range(len(periods_seconds)):
            names.append(f"{col}_cwt_p{i}")
    return names
