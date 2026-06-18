"""Online feature extractor: reproduce the EXACT offline 432-feature surface, live.

Design principle: maximize parity by reusing the *same* offline feature functions
(vendored ``behavior_segmentation``) rather than re-deriving them. Each frame we keep
a rolling buffer of per-mouse {bbox, score, mask, keypoints} and rebuild the in-memory
``CocoVideo`` + ``PoseData`` for the buffer, then call the offline pipeline on it:

    extract_video_features (mask geometry + motion + nearest + pair)
    extract_pose_feature_tables (egocentric pose + dyadic pose)
    add_wavelet_features (Morlet CWT channels)
    build_inference_tracks (subject + obj_ partner + pair, both directed tracks)

Two parity-safe optimizations keep this real-time:

  1. Mask-geometry CACHE. The expensive per-frame work (cv2 moments/contours/hull at
     ~8 ms/frame) is memoized by (frame_idx, identity) so it runs once per frame even
     though the buffer is re-processed every frame.
  2. Rolling features DISABLED. The lean 432 columns contain ZERO rolling ``*_w*``
     stats, so we set ``rolling_windows_seconds=[]`` to skip ~2200 columns of pandas
     rolling work. The surviving 432 columns are byte-identical (verified by
     replay_parity_test.py).

Known online/offline gaps (documented, measured by the parity test):
  - Wavelet right-edge: the newest frames lack future support; the look-ahead L
    (handled in live_engine) recovers the fast scales.
  - ``area_norm`` divides by the whole-video median area offline; online we use the
    buffer median (small early drift).
  - ``pair_contact_flag`` uses a whole-video median distance threshold offline; online
    it uses the buffer median.
These only affect a handful of channels and are quantified in the parity report.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from behavior_segmentation import features as _feat_mod
from behavior_segmentation.coco_masks import CocoVideo, MaskRecord
from behavior_segmentation.config import load_config
from behavior_segmentation.dataset import build_feature_column_list, build_inference_tracks
from behavior_segmentation.features import extract_video_features
from behavior_segmentation.pose import KEYPOINT_ORDER, PoseData
from behavior_segmentation.pose_features import extract_pose_feature_tables
from behavior_segmentation.social_pipeline import _merge_on_meta
from behavior_segmentation.wavelet_features import DEFAULT_WAVELET_SIGNALS, add_wavelet_features

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml")
POSE_MIN_LIKELIHOOD = 0.3  # matches social_pipeline.load_pose_csv(min_likelihood=0.3)

# Pair signals temporarily merged onto each subject for the CWT (then dropped, cwt kept).
_PAIR_SIGNAL_MAP = ("pp_body_body", "pp_nose_nose", "pp_approach_speed")


# --------------------------------------------------------------------------- #
# Mask-geometry cache (parity-safe: same numbers, computed once per frame)
# --------------------------------------------------------------------------- #

_ORIG_MASK_SHAPE_FEATURES = _feat_mod.mask_shape_features
_SHAPE_CACHE: dict[tuple, dict] = {}


def _cached_mask_shape_features(mask, record, frame_width, frame_height):
    key = (record.frame_idx, record.identity)
    cached = _SHAPE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    out = _ORIG_MASK_SHAPE_FEATURES(mask, record, frame_width, frame_height)
    _SHAPE_CACHE[key] = out
    return dict(out)


# Install once. extract_identity_features looks up the module global at call time.
_feat_mod.mask_shape_features = _cached_mask_shape_features

# mask_iou(prev, cur) is the SECOND big per-frame cost (cProfile: ~240 ms over a
# 480-frame buffer, dominated by H*W boolean reductions). The IoU between a frame's
# mask and its predecessor never changes once computed, so memoize it. We key on the
# array object identities (the pykaboo boolean masks persist in the rolling buffer,
# and ``np.asarray(bool_arr, dtype=bool)`` returns the same object, so ids are stable
# for a frame's lifetime) and verify with ``is`` to be safe against id reuse.
_ORIG_MASK_IOU = _feat_mod.mask_iou
_IOU_CACHE: dict[tuple[int, int], tuple] = {}


def _cached_mask_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return _ORIG_MASK_IOU(mask_a, mask_b)
    key = (id(mask_a), id(mask_b))
    ent = _IOU_CACHE.get(key)
    if ent is not None and ent[0] is mask_a and ent[1] is mask_b:
        return ent[2]
    val = _ORIG_MASK_IOU(mask_a, mask_b)
    _IOU_CACHE[key] = (mask_a, mask_b, val)
    return val


_feat_mod.mask_iou = _cached_mask_iou


@dataclass
class LiveMaskRecord(MaskRecord):
    """A MaskRecord that returns an already-decoded boolean mask (no RLE/pycocotools)."""

    mask_array: np.ndarray | None = None

    def decode_mask(self):  # noqa: D401
        return self.mask_array


# --------------------------------------------------------------------------- #
# Per-frame input record
# --------------------------------------------------------------------------- #

@dataclass
class FrameRecord:
    frame_idx: int
    timestamp_s: float
    width: int
    height: int
    # per-identity payloads keyed by "1"/"2"
    mice: dict  # {"1": {present,bbox_xywh,score,mask,keypoints,keypoint_scores}, ...}


# --------------------------------------------------------------------------- #
# Online feature extractor
# --------------------------------------------------------------------------- #

class OnlineFeatureExtractor:
    """Rolling-buffer reproduction of build_social_features + track assembly."""

    def __init__(
        self,
        feature_names: list[str],
        frame_rate: float,
        window_frames: int,
        identities: tuple[str, str] = ("1", "2"),
        history_pad: int = 64,
        min_window: int | None = None,
        config_path: str = CONFIG_PATH,
    ):
        self.feature_names = list(feature_names)
        self.frame_rate = float(frame_rate)
        self.win = int(window_frames)
        self.identities = tuple(identities)
        self.buffer_len = self.win + int(history_pad)
        self.min_window = int(min_window) if min_window is not None else self.win

        # Load the SAME config the training cache used, then force the trained fps
        # (do NOT auto-detect from timestamps; the model expects 30-fps semantics) and
        # disable rolling (parity-safe speedup; lean set has no rolling columns).
        cfg = load_config(config_path)
        cfg = cfg.model_copy(deep=True)
        cfg.data.frame_rate = self.frame_rate
        try:
            cfg.features.rolling_windows_seconds = []
        except Exception:
            pass
        self.config = cfg
        self.egocentric = bool(getattr(cfg.features, "egocentric", True))

        self._buf: deque[FrameRecord] = deque(maxlen=self.buffer_len)
        self._full_names_cache: list[str] | None = None

    # ------------------------------------------------------------------ #
    def push(self, frame: FrameRecord) -> None:
        self._buf.append(frame)
        self._prune_cache()

    def ready(self) -> bool:
        return len(self._buf) >= self.min_window

    def _prune_cache(self) -> None:
        if not self._buf:
            return
        live_frames = {f.frame_idx for f in self._buf}
        stale = [k for k in _SHAPE_CACHE if k[0] not in live_frames]
        for k in stale:
            _SHAPE_CACHE.pop(k, None)
        # Drop IoU cache entries whose masks have left the buffer (id no longer live).
        live_ids: set[int] = set()
        for f in self._buf:
            for m in f.mice.values():
                arr = m.get("mask")
                if arr is not None:
                    live_ids.add(id(arr))
        stale_iou = [k for k in _IOU_CACHE if k[0] not in live_ids or k[1] not in live_ids]
        for k in stale_iou:
            _IOU_CACHE.pop(k, None)

    # ------------------------------------------------------------------ #
    def _build_coco_pose(self):
        frames = list(self._buf)
        frame_indices = [f.frame_idx for f in frames]
        W = frames[-1].width
        H = frames[-1].height
        records: list[MaskRecord] = []
        A = len(self.identities)
        K = len(KEYPOINT_ORDER)
        coords = np.full((len(frames), A, K, 3), np.nan, dtype=np.float64)
        id_to_a = {ident: a for a, ident in enumerate(self.identities)}

        for fi, fr in enumerate(frames):
            for ident in self.identities:
                m = fr.mice.get(ident)
                if m is None or not m.get("present", False):
                    continue
                bbox = m["bbox_xywh"]
                mask = m.get("mask")
                records.append(
                    LiveMaskRecord(
                        frame_idx=fr.frame_idx,
                        identity=ident,
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        score=float(m.get("score", 1.0)),
                        height=H,
                        width=W,
                        segmentation=None,
                        category_id=m.get("category_id"),
                        mask_array=(np.asarray(mask, dtype=bool) if mask is not None else None),
                    )
                )
                kps = m.get("keypoints")
                if kps is not None:
                    kps = np.asarray(kps, dtype=np.float64)  # (K,2)
                    scores = m.get("keypoint_scores")
                    scores = (np.asarray(scores, dtype=np.float64)
                              if scores is not None else np.ones(K))
                    a = id_to_a[ident]
                    coords[fi, a, :, 0] = kps[:, 0]
                    coords[fi, a, :, 1] = kps[:, 1]
                    coords[fi, a, :, 2] = scores
                    # replicate load_pose_csv(min_likelihood=0.3): low-conf -> NaN x/y
                    low = scores < POSE_MIN_LIKELIHOOD
                    coords[fi, a, low, 0] = np.nan
                    coords[fi, a, low, 1] = np.nan

        timestamps = {f.frame_idx: float(f.timestamp_s) for f in frames}
        video = CocoVideo(
            video_id="live",
            width=W,
            height=H,
            frame_indices=list(frame_indices),
            records=records,
            frame_timestamps=timestamps,
        )
        pose = PoseData(
            video_id="live",
            frame_indices=np.asarray(frame_indices),
            identities=list(self.identities),
            keypoint_names=list(KEYPOINT_ORDER),
            coords=coords,
        )
        return video, pose, W, H

    # ------------------------------------------------------------------ #
    def _social_features(self, video: CocoVideo, pose: PoseData, W: int, H: int):
        """Mirror of build_social_features (no fps override), in memory."""
        identity_df, pair_df = extract_video_features(video, self.config)
        pose_id, pose_pair = extract_pose_feature_tables(
            pose, W, H, self.config.data.frame_rate, egocentric=self.egocentric
        )
        meta = ["video_id", "frame_idx", "subject_id", "object_id"]
        identity_df = _merge_on_meta(identity_df, pose_id, meta)
        pair_df = _merge_on_meta(pair_df, pose_pair, meta)

        # wavelets: attach pair signals per subject, run CWT, drop temp signals
        signals = list(DEFAULT_WAVELET_SIGNALS)
        present = [c for c in signals if c in identity_df.columns]
        parts: list[pd.DataFrame] = []
        for subject, grp in identity_df.groupby("subject_id", sort=False):
            grp = grp.sort_values("frame_idx").copy()
            if not pair_df.empty:
                pj = pair_df[pair_df["subject_id"].astype(str) == str(subject)]
                if not pj.empty:
                    pj = pj.sort_values("frame_idx")
                    keep = [c for c in _PAIR_SIGNAL_MAP if c in pj.columns]
                    grp = grp.merge(pj[["frame_idx", *keep]], on="frame_idx", how="left")
            sigs = present + [c for c in _PAIR_SIGNAL_MAP if c in grp.columns]
            grp = add_wavelet_features(grp, sigs, self.config.data.frame_rate)
            drop = [c for c in _PAIR_SIGNAL_MAP if c in grp.columns]
            grp = grp.drop(columns=drop, errors="ignore")
            parts.append(grp)
        identity_df = pd.concat(parts, ignore_index=True)

        identity_df = identity_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not pair_df.empty:
            pair_df = pair_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return identity_df, pair_df

    # ------------------------------------------------------------------ #
    def compute_windows(self):
        """Return {subject_id: raw_features[win_eff, 432]} for the current buffer.

        Channels are in the model's exact ``feature_names`` order (missing -> zeros).
        NOT normalized (the engine normalizes). Returns None during warmup.
        """
        if not self.ready():
            return None
        video, pose, W, H = self._build_coco_pose()
        identity_df, pair_df = self._social_features(video, pose, W, H)

        full_names = build_feature_column_list(identity_df, pair_df, is_pair=True)
        self._full_names_cache = full_names
        tracks = build_inference_tracks(identity_df, pair_df, full_names, is_pair=True)

        out: dict[str, np.ndarray] = {}
        for t in tracks:
            idx = {n: i for i, n in enumerate(t.feature_names)}
            n_frames = t.features.shape[0]
            cols = []
            for c in self.feature_names:
                j = idx.get(c)
                cols.append(t.features[:, j] if j is not None
                            else np.zeros(n_frames, np.float32))
            mat = np.stack(cols, axis=1).astype(np.float32)  # [T, 432]
            if mat.shape[0] > self.win:
                mat = mat[-self.win:]
            out[str(t.subject_id)] = mat
        return out
