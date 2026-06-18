"""Conversion of model predictions into export tables, plots, and diagnostics.

This module defines the :class:`TrackPrediction` container produced by inference
and the functions that turn predictions into the framewise, bout, and embedding
tables specified by the project, plus matplotlib summary plots and data quality
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import LabelMap, labels_to_intervals
from .metrics import extract_bouts
from .storage import write_table


@dataclass
class TrackPrediction:
    """Per-frame predictions for a single (video, subject, object) track."""

    video_id: str
    subject_id: str
    object_id: str
    frame_indices: np.ndarray
    labels: np.ndarray  # [T] class ids or [num_classes, T] binary labels
    probabilities: np.ndarray  # [num_classes, T]
    embeddings: np.ndarray  # [embedding_dim, T]
    # optional raw contact geometry the model saw, per frame ([T] arrays keyed by
    # channel, e.g. "pp_nose_tail"/"pp_nose_nose"), used by the contact-gate rule.
    geometry: dict[str, np.ndarray] | None = None

    @property
    def confidence(self) -> np.ndarray:
        return self.probabilities.max(axis=0)


def dominant_label_sequence(
    labels: np.ndarray,
    probabilities: np.ndarray,
    background_id: int,
) -> np.ndarray:
    """Return one display label per frame from class ids or binary labels."""

    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int64)
    probs = np.asarray(probabilities)
    out = np.full(labels.shape[1], background_id, dtype=np.int64)
    for frame_pos in range(labels.shape[1]):
        active = np.flatnonzero(labels[:, frame_pos] > 0)
        active = active[active != background_id]
        if active.size:
            best = active[np.argmax(probs[active, frame_pos])]
            out[frame_pos] = int(best)
    return out


def active_behavior_names(
    labels: np.ndarray,
    probabilities: np.ndarray,
    frame_pos: int,
    label_map: LabelMap,
) -> list[str]:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return [label_map.id_to_name[int(labels[frame_pos])]]
    active = np.flatnonzero(labels[:, frame_pos] > 0)
    active = active[active != label_map.background_id]
    if active.size == 0:
        return [label_map.background_label]
    order = active[np.argsort(probabilities[active, frame_pos])[::-1]]
    return [label_map.id_to_name[int(class_id)] for class_id in order]


def framewise_predictions_dataframe(
    predictions: list[TrackPrediction], label_map: LabelMap
) -> pd.DataFrame:
    """Build the framewise prediction table with one probability column per class."""

    rows: list[dict[str, Any]] = []
    prob_cols = [f"prob_{name}" for name in label_map.names]
    for pred in predictions:
        confidence = pred.confidence
        display_labels = dominant_label_sequence(
            pred.labels, pred.probabilities, label_map.background_id
        )
        for i, frame in enumerate(pred.frame_indices):
            active_names = active_behavior_names(
                pred.labels, pred.probabilities, i, label_map
            )
            row: dict[str, Any] = {
                "video_id": pred.video_id,
                "frame_idx": int(frame),
                "subject_id": pred.subject_id,
                "object_id": pred.object_id,
                "predicted_behavior": label_map.id_to_name[int(display_labels[i])],
                "predicted_behaviors": ";".join(active_names),
                "confidence": float(confidence[i]),
            }
            if np.asarray(pred.labels).ndim == 2:
                for class_id, name in label_map.id_to_name.items():
                    row[f"active_{name}"] = int(pred.labels[class_id, i])
            for class_id, col in enumerate(prob_cols):
                row[col] = float(pred.probabilities[class_id, i])
            rows.append(row)
    return pd.DataFrame(rows)


def bouts_dataframe(
    predictions: list[TrackPrediction], label_map: LabelMap, frame_rate: float
) -> pd.DataFrame:
    """Build the behavior-bout table from postprocessed framewise labels."""

    rows: list[dict[str, Any]] = []
    for pred in predictions:
        confidence = pred.confidence
        labels = np.asarray(pred.labels)
        if labels.ndim == 2:
            for class_id in range(labels.shape[0]):
                if class_id == label_map.background_id:
                    continue
                for bout in extract_bouts(labels[class_id], background_id=0):
                    start_frame = int(pred.frame_indices[bout.start])
                    end_frame = int(pred.frame_indices[bout.end])
                    seg_conf = pred.probabilities[class_id, bout.start : bout.end + 1]
                    rows.append(
                        {
                            "video_id": pred.video_id,
                            "subject_id": pred.subject_id,
                            "object_id": pred.object_id,
                            "behavior": label_map.id_to_name[class_id],
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "start_time_sec": start_frame / frame_rate,
                            "end_time_sec": end_frame / frame_rate,
                            "mean_confidence": float(seg_conf.mean()),
                            "max_confidence": float(seg_conf.max()),
                        }
                    )
            continue
        for bout in extract_bouts(labels, label_map.background_id):
            start_frame = int(pred.frame_indices[bout.start])
            end_frame = int(pred.frame_indices[bout.end])
            seg_conf = confidence[bout.start : bout.end + 1]
            rows.append(
                {
                    "video_id": pred.video_id,
                    "subject_id": pred.subject_id,
                    "object_id": pred.object_id,
                    "behavior": label_map.id_to_name[bout.behavior],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_sec": start_frame / frame_rate,
                    "end_time_sec": end_frame / frame_rate,
                    "mean_confidence": float(seg_conf.mean()),
                    "max_confidence": float(seg_conf.max()),
                }
            )
    return pd.DataFrame(rows)


def embeddings_dataframe(
    predictions: list[TrackPrediction], label_map: LabelMap
) -> pd.DataFrame:
    """Build the per-frame embedding table for clustering."""

    frames: list[pd.DataFrame] = []
    for pred in predictions:
        embed_dim = pred.embeddings.shape[0]
        confidence = pred.confidence
        display_labels = dominant_label_sequence(
            pred.labels, pred.probabilities, label_map.background_id
        )
        base = pd.DataFrame(
            {
                "video_id": pred.video_id,
                "frame_idx": pred.frame_indices.astype(int),
                "subject_id": pred.subject_id,
                "object_id": pred.object_id,
                "predicted_behavior": [
                    label_map.id_to_name[int(c)] for c in display_labels
                ],
                "confidence": confidence,
            }
        )
        embed_cols = pd.DataFrame(
            pred.embeddings.T,
            columns=[f"embedding_{i:03d}" for i in range(embed_dim)],
        )
        frames.append(pd.concat([base, embed_cols], axis=1))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def corrections_to_interval_csv(
    corrections: list[dict[str, Any]], path: str | Path
) -> Path:
    """Write GUI correction intervals to the training-compatible interval schema."""

    columns = [
        "video_id",
        "subject_id",
        "object_id",
        "start_frame",
        "end_frame",
        "behavior",
        "label_source",
        "annotator",
    ]
    df = pd.DataFrame(corrections)
    for col in columns:
        if col not in df.columns:
            df[col] = "" if col not in {"start_frame", "end_frame"} else 0
    df = df[columns]
    return write_table(df, path)


def labels_array_to_corrections(
    labels: np.ndarray,
    frame_indices: np.ndarray,
    label_map: LabelMap,
    video_id: str,
    subject_id: str,
    object_id: str,
    annotator: str = "gui",
    min_bout_frames: int = 1,
) -> list[dict[str, Any]]:
    """Convert a framewise label array to interval correction rows."""

    labels = np.asarray(labels)
    if labels.ndim == 2:
        rows: list[dict[str, Any]] = []
        frames = list(frame_indices.astype(int))
        for class_id in range(labels.shape[0]):
            if class_id == label_map.background_id:
                continue
            intervals = labels_to_intervals(
                labels[class_id], frames, LabelMap(["background", label_map.id_to_name[class_id]]), min_bout_frames
            )
            for interval in intervals:
                rows.append(
                    {
                        "video_id": video_id,
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "start_frame": interval["start_frame"],
                        "end_frame": interval["end_frame"],
                        "behavior": label_map.id_to_name[class_id],
                        "label_source": "corrected",
                        "annotator": annotator,
                    }
                )
        return rows

    intervals = labels_to_intervals(
        labels, list(frame_indices.astype(int)), label_map, min_bout_frames
    )
    rows: list[dict[str, Any]] = []
    for interval in intervals:
        rows.append(
            {
                "video_id": video_id,
                "subject_id": subject_id,
                "object_id": object_id,
                "start_frame": interval["start_frame"],
                "end_frame": interval["end_frame"],
                "behavior": interval["behavior"],
                "label_source": "corrected",
                "annotator": annotator,
            }
        )
    return rows


def write_summary_plots(
    confusion: np.ndarray,
    class_names: list[str],
    per_class: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Render a confusion matrix heatmap and per-class F1 bar chart."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 6))
    normalized = confusion / np.clip(confusion.sum(axis=1, keepdims=True), 1, None)
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    paths.append(cm_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_class["behavior"], per_class["f1"], color="#35c2ff")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    f1_path = output_dir / "per_class_f1.png"
    fig.savefig(f1_path, dpi=130)
    plt.close(fig)
    paths.append(f1_path)
    return paths


def write_feature_diagnostics(
    identity_df: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Write basic data-quality diagnostics tables and plots from features."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    masks_per_frame = (
        identity_df.assign(present=1 - identity_df.get("missing_mask_flag", 0))
        .groupby(["video_id", "frame_idx"])["present"]
        .sum()
        .reset_index(name="num_masks")
    )
    outputs["masks_per_frame"] = write_table(
        masks_per_frame, output_dir / "masks_per_frame.csv"
    )

    missing = (
        identity_df.groupby(["video_id", "subject_id"])["missing_mask_flag"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "missing_frames", "count": "total_frames"})
    )
    outputs["missing_summary"] = write_table(
        missing, output_dir / "missing_mask_summary.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for subject, group in identity_df.groupby("subject_id"):
        ax.plot(group["frame_idx"], group.get("area", 0), label=str(subject), lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("area")
    ax.set_title("Identity area over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    area_path = output_dir / "area_over_time.png"
    fig.savefig(area_path, dpi=120)
    plt.close(fig)
    outputs["area_plot"] = area_path
    return outputs
