import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metadata_normalization import (
    infer_timestamp_tick_scale,
    normalize_metadata_csv_file,
    normalize_recording_timestamps,
)


def _spinnaker_like_frame_df(n_frames: int = 30) -> pd.DataFrame:
    """Mimic the real exported metadata: raw epoch + raw nanosecond ticks."""
    frame_interval_s = 1.0 / 30.0
    software_origin = 1_762_000_000.0
    tick_origin = 14_967_653_718_296.0
    ticks_per_second = 1e9 / 30.0 * 30.0  # 1 GHz camera clock

    rows = []
    for index in range(n_frames):
        elapsed = index * frame_interval_s
        rows.append(
            {
                "frame_id": index,
                "timestamp_software": software_origin + elapsed,
                "timestamp_camera": tick_origin + elapsed * ticks_per_second,
                "timestamp_ticks": tick_origin + elapsed * ticks_per_second,
                "camera_frame_id": 188_693 + index,
                "user_flag_event_timestamp_software": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.loc[10, "user_flag_event_timestamp_software"] = software_origin + 10 * frame_interval_s
    return df


def test_software_timestamps_start_at_zero():
    normalized = normalize_recording_timestamps(_spinnaker_like_frame_df())
    assert normalized["timestamp_software"].iloc[0] == 0.0
    assert normalized["timestamp_software"].iloc[-1] == pytest.approx(29 / 30.0, abs=1e-6)


def test_camera_tick_columns_start_at_zero_in_seconds():
    normalized = normalize_recording_timestamps(_spinnaker_like_frame_df())
    for column in ("timestamp_camera", "timestamp_ticks"):
        assert normalized[column].iloc[0] == 0.0
        # Tick columns must be converted to seconds, matching software elapsed time.
        assert normalized[column].iloc[-1] == pytest.approx(29 / 30.0, rel=1e-3)


def test_camera_and_ticks_share_one_scale():
    normalized = normalize_recording_timestamps(_spinnaker_like_frame_df())
    np.testing.assert_allclose(
        normalized["timestamp_camera"].to_numpy(),
        normalized["timestamp_ticks"].to_numpy(),
        atol=1e-5,
    )


def test_camera_frame_id_rebased_to_zero():
    normalized = normalize_recording_timestamps(_spinnaker_like_frame_df())
    assert normalized["camera_frame_id"].iloc[0] == 0
    assert normalized["camera_frame_id"].iloc[-1] == 29


def test_event_timestamps_share_software_origin():
    normalized = normalize_recording_timestamps(_spinnaker_like_frame_df())
    event_values = normalized["user_flag_event_timestamp_software"].dropna()
    assert len(event_values) == 1
    assert float(event_values.iloc[0]) == pytest.approx(10 / 30.0, abs=1e-6)


def test_ttl_states_like_export_normalizes_both_clocks():
    """The *_ttl_states.csv layout: small tick counts + epoch software stamps."""
    df = pd.DataFrame(
        {
            "frame_id": [1, 2, 3, 4],
            "timestamp_camera": [120001, 121001, 122001, 123001],
            "timestamp_software": [1711361400.000, 1711361400.033, 1711361400.067, 1711361400.100],
            "gate": [1, 1, 0, 0],
        }
    )
    normalized = normalize_recording_timestamps(df)
    assert normalized["timestamp_software"].iloc[0] == 0.0
    assert normalized["timestamp_camera"].iloc[0] == 0.0
    assert normalized["timestamp_camera"].iloc[-1] == pytest.approx(0.1, rel=0.05)
    # Signal columns must be untouched.
    assert normalized["gate"].tolist() == [1, 1, 0, 0]


def test_empty_and_none_frames_pass_through():
    assert normalize_recording_timestamps(None) is None
    empty = pd.DataFrame()
    assert normalize_recording_timestamps(empty) is empty


def test_tick_scale_inference_prefers_software_ratio():
    ticks = pd.Series(np.arange(10) * 33_333_333.0)
    software = pd.Series(np.arange(10) / 30.0)
    scale = infer_timestamp_tick_scale(ticks, software)
    assert scale == pytest.approx(1e9, rel=1e-3)


def test_tick_scale_inference_magnitude_fallback():
    # 30 fps with a 1 GHz (nanosecond) camera clock.
    ticks_ns = pd.Series(np.arange(10) * 33_333_333.0)
    assert infer_timestamp_tick_scale(ticks_ns) == pytest.approx(1e9)
    # 30 fps with a microsecond clock.
    ticks_us = pd.Series(np.arange(10) * 33_333.0)
    assert infer_timestamp_tick_scale(ticks_us) == pytest.approx(1e6)
    # 30 fps with a millisecond clock.
    ticks_ms = pd.Series(np.arange(10) * 33.0)
    assert infer_timestamp_tick_scale(ticks_ms) == pytest.approx(1e3)


def test_normalize_metadata_csv_file_roundtrip(tmp_path):
    csv_path = tmp_path / "aux_metadata.csv"
    df = _spinnaker_like_frame_df()
    df["raw_mean"] = 12.0
    df.to_csv(csv_path, index=False)

    assert normalize_metadata_csv_file(csv_path) is True

    rewritten = pd.read_csv(csv_path)
    assert rewritten["timestamp_software"].iloc[0] == 0.0
    assert rewritten["timestamp_camera"].iloc[0] == 0.0
    assert "raw_mean" not in rewritten.columns


def test_normalize_metadata_csv_file_missing_file(tmp_path):
    assert normalize_metadata_csv_file(tmp_path / "missing.csv") is False
