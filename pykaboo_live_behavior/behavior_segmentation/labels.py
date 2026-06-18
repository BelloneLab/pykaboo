"""Manual behavior labels: loading, frame alignment, and label maps.

Supports two on-disk formats:

- interval labels: one row per behavior bout with start/end frames
- framewise labels: one row per labeled frame

Both are converted to a framewise integer (multiclass) or multi-hot (multilabel)
target array per (video_id, subject_id, object_id) track. The background label is
assigned to frames with no manual annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LabelsConfig
from .storage import read_json, write_json


class LabelAlignmentError(RuntimeError):
    """Raised when manual labels cannot be aligned to mask frames."""


INTERVAL_COLUMNS = [
    "video_id",
    "subject_id",
    "object_id",
    "start_frame",
    "end_frame",
    "behavior",
]
FRAMEWISE_COLUMNS = ["video_id", "frame_idx", "subject_id", "object_id", "behavior"]
WIDE_METADATA_COLUMNS = {
    "time",
    "frame",
    "frame_idx",
    "frame_index",
    "video_id",
    "subject_id",
    "object_id",
    "label_source",
    "annotator",
}

ATTACK_BEHAVIOR_ALIASES = {
    "attack": "attack",
    "attack_biting": "attack_biting",
    "attack_wrestling": "attack_wrestling",
}


@dataclass
class LabelMap:
    """Stable mapping between behavior names and integer ids."""

    names: list[str]
    background_label: str = "background"

    @property
    def name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names)}

    @property
    def id_to_name(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.names)}

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def background_id(self) -> int:
        return self.name_to_id.get(self.background_label, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"names": self.names, "background_label": self.background_label}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelMap":
        return cls(
            names=list(payload["names"]),
            background_label=payload.get("background_label", "background"),
        )

    def save(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        return cls.from_dict(read_json(path))


def normalize_object_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_video_id_for_matching(value: Any) -> str:
    text = normalize_object_id(value).lower()
    return "".join(ch for ch in text if ch.isalnum())


def video_ids_match(label_video_id: Any, target_video_id: Any) -> bool:
    label_norm = normalize_video_id_for_matching(label_video_id)
    target_norm = normalize_video_id_for_matching(target_video_id)
    if not label_norm or not target_norm:
        return False
    return (
        label_norm == target_norm
        or label_norm.startswith(target_norm)
        or target_norm.startswith(label_norm)
    )


def select_label_rows_for_video(
    labels_df: pd.DataFrame, video_id: str
) -> pd.DataFrame:
    exact = labels_df[labels_df["video_id"].astype(str) == str(video_id)]
    if not exact.empty:
        return exact
    unique_ids = [
        str(value)
        for value in labels_df["video_id"].dropna().astype(str).unique()
        if str(value).strip()
    ]
    matching_ids = [value for value in unique_ids if video_ids_match(value, video_id)]
    if len(matching_ids) == 1:
        return labels_df[labels_df["video_id"].astype(str) == matching_ids[0]]
    if len(unique_ids) == 1:
        return labels_df
    return exact


def infer_video_id_from_label_path(path: Path) -> str:
    stem = path.stem.strip()
    for marker in ("_Raw data", " Raw data", "-Raw data", "_Analysis", "-Analysis"):
        marker_pos = stem.find(marker)
        if marker_pos > 0:
            return stem[:marker_pos].strip()
    return stem


def wide_behavior_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        lower = str(column).strip().lower()
        if lower in WIDE_METADATA_COLUMNS:
            continue
        if lower.startswith("result") or lower.startswith("unnamed"):
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        present = values.dropna()
        if present.empty:
            continue
        unique = set(float(value) for value in present.unique())
        if unique.issubset({0.0, 1.0}) and float(values.fillna(0).sum()) > 0:
            columns.append(str(column))
    return columns


def canonical_label_column_name(column: Any) -> str:
    """Normalize known behavior-name aliases without touching metadata columns."""

    text = str(column).strip()
    lower = text.lower()
    if lower in WIDE_METADATA_COLUMNS:
        return text
    normalized = lower.replace(" ", "_")
    return ATTACK_BEHAVIOR_ALIASES.get(normalized, text)


def canonicalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate behavior columns caused by capitalization differences."""

    out = pd.DataFrame(index=df.index)
    for column in df.columns:
        target = canonical_label_column_name(column)
        if target not in out.columns:
            out[target] = df[column]
            continue
        left = pd.to_numeric(out[target], errors="coerce")
        right = pd.to_numeric(df[column], errors="coerce")
        if left.notna().any() or right.notna().any():
            out[target] = pd.concat([left, right], axis=1).max(axis=1).fillna(0)
        else:
            out[target] = out[target].where(out[target].notna(), df[column])
    return out


def read_label_csv(path: str | Path) -> pd.DataFrame:
    """Read label CSVs with either comma or semicolon delimiters.

    Some BORIS-style exports use ``;`` as the field separator and decimal commas
    in the ``time`` column. Auto-detect that dialect so wide behavior columns are
    visible before alignment.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    delimiter = ";" if header.count(";") > header.count(",") else ","
    decimal = "," if delimiter == ";" else "."
    df = pd.read_csv(path, sep=delimiter, decimal=decimal, encoding="utf-8-sig")
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    return canonicalize_label_columns(df)


def looks_like_wide_behavior_dataframe(df: pd.DataFrame) -> bool:
    lower_columns = {str(column).strip().lower() for column in df.columns}
    has_time_or_frame = bool(
        # "frames" (1-based) is the free-interaction model's convention.
        lower_columns & {"time", "frame", "frames", "frame_idx", "frame_index"}
    )
    return has_time_or_frame and bool(wide_behavior_columns(df))


def infer_frame_rate_from_time_column(times: pd.Series) -> float:
    """Infer annotation FPS from a wide CSV time column."""

    values = pd.to_numeric(times, errors="coerce").dropna().to_numpy(dtype=float)
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    if diffs.size:
        return float(1.0 / np.median(diffs))
    return 25.0


def frame_indices_from_wide_dataframe(
    df: pd.DataFrame, frame_rate: float | None = None
) -> np.ndarray:
    # Case-insensitive frame-column lookup. "frames" (1-based, free-interaction
    # model) is converted to 0-based to match the COCO frame indices.
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    for name in ("frame_idx", "frame_index", "frame", "frames"):
        if name in lower_to_actual:
            values = pd.to_numeric(df[lower_to_actual[name]], errors="coerce")
            if values.notna().all():
                idx = values.round().astype(int).to_numpy()
                if name == "frames" and idx.size and idx.min() >= 1:
                    idx = idx - 1  # 1-based -> 0-based
                return idx
    if "time" in df.columns:
        values = pd.to_numeric(df["time"], errors="coerce")
        fps = (
            float(frame_rate)
            if frame_rate
            else infer_frame_rate_from_time_column(values)
        )
        frames = np.arange(len(df), dtype=np.int64)
        valid = values.notna().to_numpy()
        frames[valid] = np.round(values.to_numpy(dtype=float)[valid] * fps).astype(
            np.int64
        )
        return frames
    return np.arange(len(df), dtype=np.int64)


def wide_behavior_dataframe_to_framewise(
    df: pd.DataFrame, path: Path, frame_rate: float | None = None
) -> pd.DataFrame:
    video_id = (
        str(df["video_id"].iloc[0])
        if "video_id" in df.columns and not df["video_id"].dropna().empty
        else infer_video_id_from_label_path(path)
    )
    subject_id = (
        df["subject_id"].map(normalize_object_id)
        if "subject_id" in df.columns
        else pd.Series([""] * len(df))
    )
    object_id = (
        df["object_id"].map(normalize_object_id)
        if "object_id" in df.columns
        else pd.Series([""] * len(df))
    )
    frame_indices = frame_indices_from_wide_dataframe(df, frame_rate)
    rows: list[dict[str, Any]] = []
    for behavior in wide_behavior_columns(df):
        active = pd.to_numeric(df[behavior], errors="coerce").fillna(0).to_numpy() > 0
        active_positions = np.flatnonzero(active)
        for pos in active_positions:
            rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": int(frame_indices[pos]),
                    "subject_id": subject_id.iloc[pos],
                    "object_id": object_id.iloc[pos],
                    "behavior": behavior,
                    "label_source": "wide_framewise_csv",
                    "annotator": "",
                }
            )
    if not rows:
        return pd.DataFrame(columns=FRAMEWISE_COLUMNS)
    return pd.DataFrame(rows)


def wide_behavior_dataframe_to_intervals(
    df: pd.DataFrame, path: Path, frame_rate: float | None = None
) -> pd.DataFrame:
    framewise = wide_behavior_dataframe_to_framewise(df, path, frame_rate)
    if framewise.empty:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    rows: list[dict[str, Any]] = []
    for key, group in framewise.groupby(
        ["video_id", "subject_id", "object_id", "behavior"], dropna=False, sort=False
    ):
        video_id, subject_id, object_id, behavior = key
        frames = sorted(int(frame) for frame in group["frame_idx"].unique())
        if not frames:
            continue
        start = frames[0]
        previous = frames[0]
        for frame in frames[1:] + [None]:
            if frame is not None and frame == previous + 1:
                previous = frame
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "subject_id": normalize_object_id(subject_id),
                    "object_id": normalize_object_id(object_id),
                    "start_frame": int(start),
                    "end_frame": int(previous),
                    "behavior": behavior,
                    "label_source": "wide_framewise_csv",
                    "annotator": "",
                }
            )
            if frame is not None:
                start = frame
                previous = frame
    return pd.DataFrame(rows)


def load_labels_dataframe(
    path: str | Path | list[str] | list[Path],
    label_format: str,
    frame_rate: float | None = None,
) -> pd.DataFrame:
    """Load an interval or framewise label CSV with validation of columns."""

    if isinstance(path, (list, tuple)):
        frames = [
            load_labels_dataframe(item, label_format, frame_rate=frame_rate)
            for item in path
            if str(item)
        ]
        if not frames:
            raise LabelAlignmentError("No labels files configured.")
        return pd.concat(frames, ignore_index=True)

    path = Path(path)
    if not path.exists():
        raise LabelAlignmentError(f"Labels file not found: {path}")
    df = read_label_csv(path)
    required = INTERVAL_COLUMNS if label_format == "interval" else FRAMEWISE_COLUMNS
    missing = [
        col
        for col in required
        if col not in df.columns and col not in {"object_id", "subject_id"}
    ]
    if missing:
        if looks_like_wide_behavior_dataframe(df):
            df = (
                wide_behavior_dataframe_to_intervals(df, path, frame_rate)
                if label_format == "interval"
                else wide_behavior_dataframe_to_framewise(df, path, frame_rate)
            )
            missing = [
                col
                for col in required
                if col not in df.columns and col not in {"object_id", "subject_id"}
            ]
        if missing:
            detected = wide_behavior_columns(df)
            detail = (
                f" Detected wide behavior columns: {detected}"
                if detected
                else ""
            )
            raise LabelAlignmentError(
                f"Labels file {path} missing required columns: {missing}."
                f"{detail}"
            )
    if "subject_id" not in df.columns:
        df["subject_id"] = ""
    if "object_id" not in df.columns:
        df["object_id"] = ""
    df["subject_id"] = df["subject_id"].map(lambda v: normalize_object_id(v) or "")
    df["object_id"] = df["object_id"].map(normalize_object_id)
    return df


def build_label_map(
    labels_df: pd.DataFrame, labels_config: LabelsConfig
) -> LabelMap:
    """Build a stable label map placing background first, then sorted behaviors."""

    behaviors = sorted(
        {
            str(name)
            for name in labels_df["behavior"].dropna().unique()
            if str(name) != labels_config.background_label
        }
    )
    if labels_config.behavior_priority:
        ordered = [b for b in labels_config.behavior_priority if b in behaviors]
        ordered += [b for b in behaviors if b not in ordered]
        behaviors = ordered
    names = [labels_config.background_label] + behaviors
    return LabelMap(names=names, background_label=labels_config.background_label)


def track_key(video_id: str, subject_id: str, object_id: str) -> tuple[str, str, str]:
    return (str(video_id), str(subject_id), normalize_object_id(object_id))


def align_interval_labels(
    labels_df: pd.DataFrame,
    video_id: str,
    frame_indices: list[int],
    label_map: LabelMap,
    labels_config: LabelsConfig,
    end_frame_inclusive: bool = True,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Convert interval labels into per-track framewise target arrays."""

    return _align(
        labels_df,
        video_id,
        frame_indices,
        label_map,
        labels_config,
        end_frame_inclusive,
        framewise=False,
    )


def align_framewise_labels(
    labels_df: pd.DataFrame,
    video_id: str,
    frame_indices: list[int],
    label_map: LabelMap,
    labels_config: LabelsConfig,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Convert framewise labels into per-track framewise target arrays."""

    return _align(
        labels_df,
        video_id,
        frame_indices,
        label_map,
        labels_config,
        True,
        framewise=True,
    )


def _align(
    labels_df: pd.DataFrame,
    video_id: str,
    frame_indices: list[int],
    label_map: LabelMap,
    labels_config: LabelsConfig,
    end_frame_inclusive: bool,
    framewise: bool,
) -> dict[tuple[str, str, str], np.ndarray]:
    frame_to_pos = {frame: pos for pos, frame in enumerate(frame_indices)}
    num_frames = len(frame_indices)
    multilabel = labels_config.mode == "multilabel"
    name_to_id = label_map.name_to_id

    video_rows = select_label_rows_for_video(labels_df, video_id)
    tracks: dict[tuple[str, str, str], np.ndarray] = {}

    def get_target(key: tuple[str, str, str]) -> np.ndarray:
        if key not in tracks:
            if multilabel:
                arr = np.zeros((label_map.num_classes, num_frames), dtype=np.int64)
                arr[label_map.background_id, :] = 1
            else:
                arr = np.full(num_frames, label_map.background_id, dtype=np.int64)
            tracks[key] = arr
        return tracks[key]

    priority_rank = {
        name: rank for rank, name in enumerate(labels_config.behavior_priority)
    }

    if framewise:
        iterator = video_rows.sort_values("frame_idx").itertuples(index=False)
        for row in iterator:
            behavior = str(row.behavior)
            if behavior not in name_to_id:
                continue
            frame = int(row.frame_idx)
            if frame not in frame_to_pos:
                continue
            pos = frame_to_pos[frame]
            key = track_key(video_id, row.subject_id, row.object_id)
            target = get_target(key)
            class_id = name_to_id[behavior]
            if multilabel:
                target[class_id, pos] = 1
                target[label_map.background_id, pos] = 0
            else:
                target[pos] = class_id
        return tracks

    # Interval mode: sort so higher priority behaviors are applied last (win).
    def sort_rank(behavior: str) -> int:
        return priority_rank.get(behavior, len(priority_rank))

    rows = list(video_rows.itertuples(index=False))
    rows.sort(key=lambda r: sort_rank(str(r.behavior)))
    for row in rows:
        behavior = str(row.behavior)
        if behavior not in name_to_id:
            continue
        start = int(row.start_frame)
        end = int(row.end_frame)
        if not end_frame_inclusive:
            end -= 1
        key = track_key(video_id, row.subject_id, row.object_id)
        target = get_target(key)
        class_id = name_to_id[behavior]
        for frame in range(start, end + 1):
            if frame not in frame_to_pos:
                continue
            pos = frame_to_pos[frame]
            if multilabel:
                target[class_id, pos] = 1
                target[label_map.background_id, pos] = 0
            else:
                target[pos] = class_id
    return tracks


def labels_to_intervals(
    framewise: np.ndarray,
    frame_indices: list[int],
    label_map: LabelMap,
    min_bout_frames: int = 1,
) -> list[dict[str, Any]]:
    """Convert a multiclass framewise label array back to interval rows."""

    intervals: list[dict[str, Any]] = []
    if len(framewise) == 0:
        return intervals
    current = int(framewise[0])
    start_pos = 0
    for pos in range(1, len(framewise) + 1):
        changed = pos == len(framewise) or int(framewise[pos]) != current
        if changed:
            length = pos - start_pos
            if current != label_map.background_id and length >= min_bout_frames:
                intervals.append(
                    {
                        "behavior": label_map.id_to_name[current],
                        "start_frame": frame_indices[start_pos],
                        "end_frame": frame_indices[pos - 1],
                        "num_frames": length,
                    }
                )
            if pos < len(framewise):
                current = int(framewise[pos])
                start_pos = pos
    return intervals


def coverage_summary(
    tracks: dict[tuple[str, str, str], np.ndarray],
    label_map: LabelMap,
) -> pd.DataFrame:
    """Per (video, subject, object, behavior) frame counts for QA."""

    rows: list[dict[str, Any]] = []
    for (video_id, subject_id, object_id), target in tracks.items():
        if target.ndim == 2:
            counts = target.sum(axis=1)
            for class_id, count in enumerate(counts):
                if count == 0:
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "behavior": label_map.id_to_name[class_id],
                        "num_frames": int(count),
                    }
                )
        else:
            ids, counts = np.unique(target, return_counts=True)
            for class_id, count in zip(ids, counts):
                rows.append(
                    {
                        "video_id": video_id,
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "behavior": label_map.id_to_name[int(class_id)],
                        "num_frames": int(count),
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class AlignedLabels:
    """Container for aligned labels across videos plus the label map."""

    label_map: LabelMap
    tracks: dict[tuple[str, str, str], np.ndarray] = field(default_factory=dict)

    def coverage(self) -> pd.DataFrame:
        return coverage_summary(self.tracks, self.label_map)
