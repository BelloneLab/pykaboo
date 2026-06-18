"""Load DeepLabCut-style multi-animal pose tracking CSVs.

The LISBET tracker exports an 8-keypoint skeleton per mouse
(``nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail``) as a
DeepLabCut multi-index CSV with header rows ``scorer / individuals / bodyparts /
coords``. Each (animal, bodypart) has ``x``, ``y`` and ``likelihood`` columns.

This module turns that file into a tidy long table keyed by
``(video_id, frame_idx, identity)`` so it can be merged with the mask-derived
feature tables. Identity strings are mapped to the COCO ``track_id`` space
(``mouse1`` -> ``"1"``, ``mouse2`` -> ``"2"``) so pose, masks and labels all share
one identity vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYPOINT_ORDER = [
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "body",
    "left_hip",
    "right_hip",
    "tail",
]

# Default mapping from DLC animal names to COCO track_id identity strings.
DEFAULT_ANIMAL_TO_IDENTITY = {"mouse1": "1", "mouse2": "2"}


@dataclass
class PoseData:
    """Per-frame keypoints for every animal in one video.

    ``coords`` has shape ``[num_frames, num_animals, num_keypoints, 3]`` where the
    last axis is ``(x, y, likelihood)``. ``identities`` lines up with axis 1.
    """

    video_id: str
    frame_indices: np.ndarray  # [F]
    identities: list[str]  # length A, COCO identity strings
    keypoint_names: list[str]
    coords: np.ndarray  # [F, A, K, 3]
    source_path: str | None = None

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def identity_index(self, identity: str) -> int | None:
        try:
            return self.identities.index(str(identity))
        except ValueError:
            return None


def infer_pose_video_id(path: Path) -> str:
    stem = path.stem
    for marker in ("_tracking", "_pose", "_dlc", "_DLC"):
        pos = stem.find(marker)
        if pos > 0:
            return stem[:pos]
    return stem


def load_pose_csv(
    path: str | Path,
    video_id: str | None = None,
    animal_to_identity: dict[str, str] | None = None,
    min_likelihood: float = 0.0,
) -> PoseData:
    """Parse a DeepLabCut multi-animal CSV into a :class:`PoseData`.

    Points with ``likelihood < min_likelihood`` are set to NaN so downstream
    feature code can interpolate or flag them. The raw likelihood is preserved.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pose CSV not found: {path}")
    animal_to_identity = animal_to_identity or DEFAULT_ANIMAL_TO_IDENTITY
    video_id = video_id or infer_pose_video_id(path)

    # header rows: 0=scorer, 1=individuals, 2=bodyparts, 3=coords; col 0 = frame.
    df = pd.read_csv(path, header=[1, 2, 3], index_col=0)
    df = df.sort_index()
    frame_indices = df.index.to_numpy().astype(np.int64)

    animals = list(dict.fromkeys(df.columns.get_level_values(0)))
    bodyparts = list(dict.fromkeys(df.columns.get_level_values(1)))
    # Keep canonical order where possible.
    keypoint_names = [b for b in KEYPOINT_ORDER if b in bodyparts]
    keypoint_names += [b for b in bodyparts if b not in keypoint_names]

    identities = [animal_to_identity.get(a, a) for a in animals]
    num_frames = len(frame_indices)
    coords = np.full((num_frames, len(animals), len(keypoint_names), 3), np.nan)

    for ai, animal in enumerate(animals):
        for ki, kp in enumerate(keypoint_names):
            for ci, coord in enumerate(("x", "y", "likelihood")):
                key = (animal, kp, coord)
                if key in df.columns:
                    coords[:, ai, ki, ci] = pd.to_numeric(
                        df[key], errors="coerce"
                    ).to_numpy()

    if min_likelihood > 0:
        lik = coords[:, :, :, 2]
        bad = lik < min_likelihood
        xy = coords[:, :, :, :2]
        xy[bad] = np.nan
        coords[:, :, :, :2] = xy

    return PoseData(
        video_id=video_id,
        frame_indices=frame_indices,
        identities=identities,
        keypoint_names=keypoint_names,
        coords=coords,
        source_path=str(path),
    )


def find_pose_csv_for_coco(coco_path: str | Path) -> Path | None:
    """Best-effort discovery of the pose CSV that accompanies a COCO masks file.

    LISBET writes ``<stem>_masks_coco.json`` next to ``<stem>_tracking.csv``.
    """

    coco_path = Path(coco_path)
    stem = coco_path.stem
    base = stem
    for marker in ("_masks_coco", "_masks", "_coco"):
        if base.endswith(marker):
            base = base[: -len(marker)]
            break
    candidates = [
        coco_path.with_name(f"{base}_tracking.csv"),
        coco_path.with_name(f"{base}_pose.csv"),
        coco_path.with_name(f"{base}.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def pose_quality_summary(pose: PoseData) -> dict[str, Any]:
    """Quick diagnostics: mean likelihood and missing fraction per identity."""

    out: dict[str, Any] = {"video_id": pose.video_id, "num_frames": pose.num_frames}
    lik = pose.coords[:, :, :, 2]
    xy = pose.coords[:, :, :, :2]
    for ai, identity in enumerate(pose.identities):
        out[f"mean_likelihood[{identity}]"] = float(np.nanmean(lik[:, ai]))
        out[f"missing_xy_frac[{identity}]"] = float(
            np.isnan(xy[:, ai]).any(axis=-1).mean()
        )
    return out
