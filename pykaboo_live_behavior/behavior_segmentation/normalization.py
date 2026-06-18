"""Feature normalization fitted on the training split only.

A :class:`FeatureNormalizer` stores per-channel mean and standard deviation and
applies a z-score transform. Statistics are JSON serializable so they travel
inside model checkpoints, guaranteeing inference uses the same scaling as
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeatureNormalizer:
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    video_median: dict[str, np.ndarray] | None = None
    video_iqr: dict[str, np.ndarray] | None = None
    video_mean: dict[str, np.ndarray] | None = None
    video_std: dict[str, np.ndarray] | None = None

    @classmethod
    def fit(
        cls, matrices: list[np.ndarray], feature_names: list[str]
    ) -> "FeatureNormalizer":
        """Fit mean/std over a list of ``[T, C]`` matrices stacked along time."""

        if not matrices:
            raise ValueError("Cannot fit normalizer on empty data.")
        stacked = np.concatenate(matrices, axis=0).astype(np.float64)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(feature_names=list(feature_names), mean=mean, std=std)

    @classmethod
    def fit_with_video_robust(
        cls,
        tracks,
        feature_names: list[str],
    ) -> "FeatureNormalizer":
        """Fit global z-score after per-video robust median/IQR scaling."""

        by_video: dict[str, list[np.ndarray]] = {}
        for track in tracks:
            by_video.setdefault(str(track.video_id), []).append(track.features)
        video_median: dict[str, np.ndarray] = {}
        video_iqr: dict[str, np.ndarray] = {}
        scaled: list[np.ndarray] = []
        for video_id, matrices in by_video.items():
            stacked = np.concatenate(matrices, axis=0).astype(np.float64)
            median = np.nanmedian(stacked, axis=0)
            q25 = np.nanpercentile(stacked, 25, axis=0)
            q75 = np.nanpercentile(stacked, 75, axis=0)
            iqr = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
            video_median[video_id] = median
            video_iqr[video_id] = iqr
            scaled.extend([(matrix.astype(np.float64) - median) / iqr for matrix in matrices])
        base = cls.fit(scaled, feature_names)
        base.video_median = video_median
        base.video_iqr = video_iqr
        return base

    @classmethod
    def fit_with_video_coral(
        cls,
        tracks,
        feature_names: list[str],
    ) -> "FeatureNormalizer":
        """Fit a diagonal CORAL-style normalizer from per-video mean/std stats.

        Each training video is first centered and scaled by its own distribution,
        then one global z-score is fitted. At inference, unseen videos get their
        own test-time stats before the checkpoint's global transform is applied.
        """

        by_video: dict[str, list[np.ndarray]] = {}
        for track in tracks:
            by_video.setdefault(str(track.video_id), []).append(track.features)
        video_mean: dict[str, np.ndarray] = {}
        video_std: dict[str, np.ndarray] = {}
        scaled: list[np.ndarray] = []
        for video_id, matrices in by_video.items():
            stacked = np.concatenate(matrices, axis=0).astype(np.float64)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            video_mean[video_id] = mean
            video_std[video_id] = std
            scaled.extend(
                [(matrix.astype(np.float64) - mean) / std for matrix in matrices]
            )
        base = cls.fit(scaled, feature_names)
        base.video_mean = video_mean
        base.video_std = video_std
        return base

    def transform(self, matrix: np.ndarray, video_id: str | None = None) -> np.ndarray:
        x = matrix.astype(np.float64)
        if (
            video_id is not None
            and self.video_median is not None
            and self.video_iqr is not None
        ):
            key = str(video_id)
            if key in self.video_median:
                median = self.video_median[key]
                iqr = self.video_iqr[key]
            else:
                median = np.nanmedian(x, axis=0)
                q25 = np.nanpercentile(x, 25, axis=0)
                q75 = np.nanpercentile(x, 75, axis=0)
                iqr = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
            x = (x - median) / iqr
        if (
            video_id is not None
            and self.video_mean is not None
            and self.video_std is not None
        ):
            key = str(video_id)
            if key in self.video_mean:
                mean = self.video_mean[key]
                std = self.video_std[key]
            else:
                mean = np.nanmean(x, axis=0)
                std = np.nanstd(x, axis=0)
                std = np.where(std < 1e-6, 1.0, std)
            x = (x - mean) / std
        out = (x - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def with_video_stats_from_tracks(self, tracks) -> "FeatureNormalizer":
        """Return a copy enriched with robust stats for any unseen track videos."""

        if (
            self.video_median is None
            and self.video_iqr is None
            and self.video_mean is None
            and self.video_std is None
        ):
            return FeatureNormalizer(
                feature_names=list(self.feature_names),
                mean=self.mean.copy(),
                std=self.std.copy(),
            )
        video_median = (
            {key: value.copy() for key, value in self.video_median.items()}
            if self.video_median is not None
            else {}
        )
        video_iqr = (
            {key: value.copy() for key, value in self.video_iqr.items()}
            if self.video_iqr is not None
            else {}
        )
        video_mean = (
            {key: value.copy() for key, value in self.video_mean.items()}
            if self.video_mean is not None
            else {}
        )
        video_std = (
            {key: value.copy() for key, value in self.video_std.items()}
            if self.video_std is not None
            else {}
        )
        by_video: dict[str, list[np.ndarray]] = {}
        for track in tracks:
            video_id = str(track.video_id)
            needs_robust = self.video_median is not None and video_id not in video_median
            needs_coral = self.video_mean is not None and video_id not in video_mean
            if needs_robust or needs_coral:
                by_video.setdefault(video_id, []).append(track.features)
        for video_id, matrices in by_video.items():
            stacked = np.concatenate(matrices, axis=0).astype(np.float64)
            if self.video_median is not None and video_id not in video_median:
                median = np.nanmedian(stacked, axis=0)
                q25 = np.nanpercentile(stacked, 25, axis=0)
                q75 = np.nanpercentile(stacked, 75, axis=0)
                iqr = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
                video_median[video_id] = median
                video_iqr[video_id] = iqr
            if self.video_mean is not None and video_id not in video_mean:
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                std = np.where(std < 1e-6, 1.0, std)
                video_mean[video_id] = mean
                video_std[video_id] = std
        return FeatureNormalizer(
            feature_names=list(self.feature_names),
            mean=self.mean.copy(),
            std=self.std.copy(),
            video_median=video_median or None,
            video_iqr=video_iqr or None,
            video_mean=video_mean or None,
            video_std=video_std or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "video_median": {
                key: value.tolist() for key, value in (self.video_median or {}).items()
            },
            "video_iqr": {
                key: value.tolist() for key, value in (self.video_iqr or {}).items()
            },
            "video_mean": {
                key: value.tolist() for key, value in (self.video_mean or {}).items()
            },
            "video_std": {
                key: value.tolist() for key, value in (self.video_std or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureNormalizer":
        return cls(
            feature_names=list(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            video_median={
                str(key): np.asarray(value, dtype=np.float64)
                for key, value in payload.get("video_median", {}).items()
            }
            or None,
            video_iqr={
                str(key): np.asarray(value, dtype=np.float64)
                for key, value in payload.get("video_iqr", {}).items()
            }
            or None,
            video_mean={
                str(key): np.asarray(value, dtype=np.float64)
                for key, value in payload.get("video_mean", {}).items()
            }
            or None,
            video_std={
                str(key): np.asarray(value, dtype=np.float64)
                for key, value in payload.get("video_std", {}).items()
            }
            or None,
        )
