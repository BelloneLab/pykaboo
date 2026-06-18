"""Pose-keypoint feature extraction for single-mouse posture and social geometry.

These features are the behaviorally informative signals that mask-shape features
alone cannot capture: head/anogenital orientation, body elongation (rearing vs
freezing vs stretched), and the dyadic geometry of two interacting mice
(nose-to-nose facing for attack, nose-to-tail for anogenital sniffing, approach
velocity, relative heading). All quantities are normalized by frame size so they
transfer across videos.

Output tables share the ``(video_id, frame_idx, subject_id, object_id)`` key with
the mask feature tables so they can be merged column-wise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pose import PoseData

EPS = 1e-6


def _kp(pose: PoseData, ai: int, name: str) -> np.ndarray:
    """Return ``[F, 2]`` xy for one keypoint of one animal (NaN where missing)."""

    ki = pose.keypoint_names.index(name)
    return pose.coords[:, ai, ki, :2]


def _interp_xy(xy: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Linear-interpolate short NaN gaps in an ``[F, 2]`` track, ffill/bfill edges."""

    out = xy.copy()
    for c in range(out.shape[1]):
        s = pd.Series(out[:, c])
        s = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
        s = s.ffill().bfill()
        out[:, c] = s.to_numpy()
    return out


def _angle(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[:, 1], v[:, 0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.hypot(v[:, 0], v[:, 1]) + EPS
    return v / n[:, None]


def _norm(v: np.ndarray) -> np.ndarray:
    return np.hypot(v[:, 0], v[:, 1])


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""

    return (a + np.pi) % (2 * np.pi) - np.pi


def _safe_keypoint(pose: PoseData, ai: int, name: str) -> np.ndarray:
    if name in pose.keypoint_names:
        return _interp_xy(_kp(pose, ai, name))
    return np.full((pose.num_frames, 2), np.nan, dtype=np.float64)


def _all_keypoints(pose: PoseData, ai: int) -> np.ndarray:
    """Return interpolated xy keypoints for one animal as ``[F, K, 2]``."""

    points = [_safe_keypoint(pose, ai, name) for name in pose.keypoint_names]
    return np.stack(points, axis=1) if points else np.zeros((pose.num_frames, 0, 2))


def _reference_frame(pose: PoseData, ai: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-centered origin, heading and body length for egocentric features."""

    points = _all_keypoints(pose, ai)
    if "body" in pose.keypoint_names:
        center = points[:, pose.keypoint_names.index("body")]
    elif points.shape[1]:
        center = np.nanmean(points, axis=1)
    else:
        center = np.zeros((pose.num_frames, 2), dtype=np.float64)
    nose = _safe_keypoint(pose, ai, "nose")
    neck = _safe_keypoint(pose, ai, "neck")
    tail = _safe_keypoint(pose, ai, "tail")
    heading_vec = nose - neck
    bad_heading = _norm(heading_vec) < EPS
    if np.any(bad_heading):
        heading_vec[bad_heading] = nose[bad_heading] - tail[bad_heading]
    heading = _angle(heading_vec)
    scale = _norm(nose - tail)
    finite = np.isfinite(scale) & (scale > EPS)
    fallback = np.nanmedian(scale[finite]) if finite.any() else 1.0
    scale = np.where(finite, scale, fallback)
    scale = np.clip(scale, EPS, None)
    return center, heading, scale


def _rotate_points(
    points: np.ndarray,
    center: np.ndarray,
    heading: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Transform ``[F, K, 2]`` points into the subject's body-normalized frame."""

    rel = points - center[:, None, :]
    cos = np.cos(-heading)[:, None]
    sin = np.sin(-heading)[:, None]
    out = np.empty_like(rel, dtype=np.float64)
    out[..., 0] = rel[..., 0] * cos - rel[..., 1] * sin
    out[..., 1] = rel[..., 0] * sin + rel[..., 1] * cos
    return out / scale[:, None, None]


def _rotate_vectors(vectors: np.ndarray, heading: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the subject frame and normalize by body length."""

    cos = np.cos(-heading)
    sin = np.sin(-heading)
    out = np.empty_like(vectors, dtype=np.float64)
    out[:, 0] = vectors[:, 0] * cos - vectors[:, 1] * sin
    out[:, 1] = vectors[:, 0] * sin + vectors[:, 1] * cos
    return out / scale[:, None]


def egocentric_transform(
    pose: PoseData,
    subject_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all animals' keypoints in one subject's egocentric frame.

    Output coordinates are ``[F, A, K, 2]``. The origin is the subject body
    keypoint, the x-axis faces from neck to nose, and units are subject body
    lengths. Constant translation, rotation and scale of the full scene therefore
    leave the coordinates unchanged up to floating-point tolerance.
    """

    center, heading, scale = _reference_frame(pose, subject_index)
    transformed = []
    for ai in range(len(pose.identities)):
        transformed.append(_rotate_points(_all_keypoints(pose, ai), center, heading, scale))
    coords = np.stack(transformed, axis=1) if transformed else np.zeros((pose.num_frames, 0, 0, 2))
    return coords, heading, scale


def single_pose_features(
    pose: PoseData,
    ai: int,
    diag: float,
    frame_rate: float,
    egocentric: bool = True,
) -> pd.DataFrame:
    """Per-frame posture and kinematics features for one animal.

    Distances are divided by the frame diagonal so they are scale-free.
    """

    if egocentric:
        return single_pose_features_egocentric(pose, ai, frame_rate)

    nose = _interp_xy(_kp(pose, ai, "nose"))
    neck = _interp_xy(_kp(pose, ai, "neck"))
    body = _interp_xy(_kp(pose, ai, "body"))
    tail = _interp_xy(_kp(pose, ai, "tail"))
    lear = _interp_xy(_kp(pose, ai, "left_ear"))
    rear = _interp_xy(_kp(pose, ai, "right_ear"))
    lhip = _interp_xy(_kp(pose, ai, "left_hip"))
    rhip = _interp_xy(_kp(pose, ai, "right_hip"))
    # NaN-safe mean likelihood (frames with all keypoints missing -> 0).
    lik_raw = pose.coords[:, ai, :, 2]
    valid = ~np.isnan(lik_raw)
    denom = valid.sum(axis=1)
    lik = np.where(denom > 0, np.nansum(np.where(valid, lik_raw, 0.0), axis=1) / np.maximum(denom, 1), 0.0)

    # Body axis = neck -> tail (heading points from tail toward nose).
    spine = nose - tail
    heading = _angle(nose - neck)  # head direction
    body_len = _norm(spine) / diag
    nose_neck = _norm(nose - neck) / diag
    neck_body = _norm(neck - body) / diag
    body_tail = _norm(body - tail) / diag
    ear_span = _norm(lear - rear) / diag
    hip_span = _norm(lhip - rhip) / diag

    # Elongation / curl: how stretched the spine is vs sum of segments.
    seg_sum = (nose_neck + neck_body + body_tail) + EPS
    straightness = (body_len / seg_sum).clip(0, 2)

    # Spine curvature: angle between head segment and tail segment at body.
    head_seg = neck - body
    tail_seg = body - tail
    curvature = _wrap(_angle(head_seg) - _angle(tail_seg))

    df = pd.DataFrame(
        {
            "pose_body_length": body_len,
            "pose_nose_neck": nose_neck,
            "pose_neck_body": neck_body,
            "pose_body_tail": body_tail,
            "pose_ear_span": ear_span,
            "pose_hip_span": hip_span,
            "pose_straightness": straightness,
            "pose_curvature": curvature,
            "pose_heading_sin": np.sin(heading),
            "pose_heading_cos": np.cos(heading),
            "pose_mean_likelihood": lik,
        }
    )

    # Kinematics of nose, body and tail (speed / acceleration), scale-free.
    for name, pts in (("nose", nose), ("body", body), ("tail", tail)):
        d = np.zeros_like(pts)
        d[1:] = np.diff(pts, axis=0)
        speed = _norm(d) / diag * frame_rate
        accel = np.zeros_like(speed)
        accel[1:] = np.diff(speed) * frame_rate
        df[f"pose_{name}_speed"] = speed
        df[f"pose_{name}_accel"] = accel

    # Angular velocity of heading.
    hd = np.unwrap(heading)
    ang = np.zeros_like(hd)
    ang[1:] = np.diff(hd) * frame_rate
    df["pose_heading_angvel"] = ang
    df["pose_body_length_vel"] = np.concatenate(
        [[0.0], np.diff(body_len)]
    ) * frame_rate
    return df


def single_pose_features_egocentric(
    pose: PoseData,
    ai: int,
    frame_rate: float,
) -> pd.DataFrame:
    """Per-animal features in the animal's own rotation and scale frame."""

    coords, heading, scale = egocentric_transform(pose, ai)
    own = coords[:, ai]
    idx = {name: i for i, name in enumerate(pose.keypoint_names)}

    def point(name: str) -> np.ndarray:
        if name in idx:
            return own[:, idx[name]]
        return np.zeros((pose.num_frames, 2), dtype=np.float64)

    nose = point("nose")
    neck = point("neck")
    body = point("body")
    tail = point("tail")
    lear = point("left_ear")
    rear = point("right_ear")
    lhip = point("left_hip")
    rhip = point("right_hip")

    lik_raw = pose.coords[:, ai, :, 2]
    valid = ~np.isnan(lik_raw)
    denom = valid.sum(axis=1)
    lik = np.where(
        denom > 0,
        np.nansum(np.where(valid, lik_raw, 0.0), axis=1) / np.maximum(denom, 1),
        0.0,
    )

    body_len = _norm(nose - tail)
    nose_neck = _norm(nose - neck)
    neck_body = _norm(neck - body)
    body_tail = _norm(body - tail)
    ear_span = _norm(lear - rear)
    hip_span = _norm(lhip - rhip)
    seg_sum = (nose_neck + neck_body + body_tail) + EPS
    straightness = (body_len / seg_sum).clip(0, 2)
    curvature = _wrap(_angle(neck - body) - _angle(body - tail))

    df = pd.DataFrame(
        {
            "pose_body_length": body_len,
            "pose_nose_neck": nose_neck,
            "pose_neck_body": neck_body,
            "pose_body_tail": body_tail,
            "pose_ear_span": ear_span,
            "pose_hip_span": hip_span,
            "pose_straightness": straightness,
            "pose_curvature": curvature,
            "pose_mean_likelihood": lik,
        }
    )

    for name in pose.keypoint_names:
        pts = point(name)
        safe = name.replace(" ", "_")
        df[f"pose_{safe}_ego_x"] = pts[:, 0]
        df[f"pose_{safe}_ego_y"] = pts[:, 1]

    for name, pts in (("nose", nose), ("body", body), ("tail", tail)):
        d = np.zeros_like(pts)
        d[1:] = np.diff(pts, axis=0)
        speed = _norm(d) * frame_rate
        accel = np.zeros_like(speed)
        accel[1:] = np.diff(speed) * frame_rate
        df[f"pose_{name}_speed"] = speed
        df[f"pose_{name}_accel"] = accel

    center, _, _ = _reference_frame(pose, ai)
    center_delta = np.zeros_like(center)
    center_delta[1:] = np.diff(center, axis=0)
    center_vel = _rotate_vectors(center_delta, heading, scale) * frame_rate
    center_speed = _norm(center_vel)
    center_accel = np.zeros_like(center_speed)
    center_accel[1:] = np.diff(center_speed) * frame_rate
    df["pose_ego_forward_velocity"] = center_vel[:, 0]
    df["pose_ego_lateral_velocity"] = center_vel[:, 1]
    df["pose_ego_speed"] = center_speed
    df["pose_ego_accel"] = center_accel

    hd = np.unwrap(heading)
    ang = np.zeros_like(hd)
    ang[1:] = np.diff(hd) * frame_rate
    scale_vel = np.zeros_like(scale)
    scale_vel[1:] = np.diff(scale) / np.clip(scale[:-1], EPS, None) * frame_rate
    df["pose_heading_angvel"] = ang
    df["pose_body_length_vel"] = scale_vel
    return df


def pair_pose_features(
    pose: PoseData,
    si: int,
    oi: int,
    diag: float,
    frame_rate: float,
    egocentric: bool = True,
) -> pd.DataFrame:
    """Directed social-geometry features: subject ``si`` relative to object ``oi``.

    These capture the canonical dyadic motifs of rodent aggression assays.
    """

    if egocentric:
        return pair_pose_features_egocentric(pose, si, oi, frame_rate)

    s_nose = _interp_xy(_kp(pose, si, "nose"))
    s_neck = _interp_xy(_kp(pose, si, "neck"))
    s_body = _interp_xy(_kp(pose, si, "body"))
    s_tail = _interp_xy(_kp(pose, si, "tail"))
    o_nose = _interp_xy(_kp(pose, oi, "nose"))
    o_body = _interp_xy(_kp(pose, oi, "body"))
    o_tail = _interp_xy(_kp(pose, oi, "tail"))

    s_head = _angle(s_nose - s_neck)

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _norm(a - b) / diag

    nose_nose = dist(s_nose, o_nose)
    nose_tail = dist(s_nose, o_tail)  # subject sniffing object anogenital
    nose_body = dist(s_nose, o_body)
    body_body = dist(s_body, o_body)
    tail_nose = dist(s_tail, o_nose)  # object sniffing subject

    # Facing: angle between subject heading and the vector subject->object body.
    to_obj = o_body - s_body
    facing = _wrap(_angle(to_obj) - s_head)
    # Approach speed: rate of decrease of body-body distance.
    approach = np.zeros_like(body_body)
    approach[1:] = -np.diff(body_body) * frame_rate

    # Relative heading of the two animals (aligned chase vs face-off).
    o_head = _angle(o_nose - o_body)
    rel_heading = _wrap(s_head - o_head)

    df = pd.DataFrame(
        {
            "pp_nose_nose": nose_nose,
            "pp_nose_tail": nose_tail,
            "pp_nose_body": nose_body,
            "pp_body_body": body_body,
            "pp_tail_nose": tail_nose,
            "pp_min_internose": np.minimum(nose_nose, tail_nose),
            "pp_facing_cos": np.cos(facing),
            "pp_facing_sin": np.sin(facing),
            "pp_approach_speed": approach,
            "pp_rel_heading_cos": np.cos(rel_heading),
            "pp_rel_heading_sin": np.sin(rel_heading),
        }
    )
    # Contact: any keypoint pair very close (proxy for physical interaction).
    contact = (np.minimum.reduce([nose_nose, nose_tail, nose_body, body_body]) < 0.06)
    df["pp_contact_flag"] = contact.astype(np.float32)
    df["pp_body_body_vel"] = np.concatenate([[0.0], np.diff(body_body)]) * frame_rate
    return df


def pair_pose_features_egocentric(
    pose: PoseData,
    si: int,
    oi: int,
    frame_rate: float,
) -> pd.DataFrame:
    """Directed pair features in the subject animal's egocentric frame."""

    coords, _heading, _scale = egocentric_transform(pose, si)
    subject = coords[:, si]
    obj = coords[:, oi]
    idx = {name: i for i, name in enumerate(pose.keypoint_names)}

    def spoint(name: str) -> np.ndarray:
        if name in idx:
            return subject[:, idx[name]]
        return np.zeros((pose.num_frames, 2), dtype=np.float64)

    def opoint(name: str) -> np.ndarray:
        if name in idx:
            return obj[:, idx[name]]
        return np.zeros((pose.num_frames, 2), dtype=np.float64)

    s_nose = spoint("nose")
    s_body = spoint("body")
    s_tail = spoint("tail")
    o_nose = opoint("nose")
    o_body = opoint("body")
    o_tail = opoint("tail")

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _norm(a - b)

    nose_nose = dist(s_nose, o_nose)
    nose_tail = dist(s_nose, o_tail)
    nose_body = dist(s_nose, o_body)
    body_body = dist(s_body, o_body)
    tail_nose = dist(s_tail, o_nose)

    to_obj = o_body - s_body
    facing = _angle(to_obj)
    o_head = _angle(o_nose - o_body)
    rel_heading = _wrap(-o_head)
    approach = np.zeros_like(body_body)
    approach[1:] = -np.diff(body_body) * frame_rate
    rel = o_body - s_body
    rel_vel = np.zeros_like(rel)
    rel_vel[1:] = np.diff(rel, axis=0) * frame_rate

    df = pd.DataFrame(
        {
            "pp_nose_nose": nose_nose,
            "pp_nose_tail": nose_tail,
            "pp_nose_body": nose_body,
            "pp_body_body": body_body,
            "pp_tail_nose": tail_nose,
            "pp_min_internose": np.minimum(nose_nose, tail_nose),
            "pp_facing_cos": np.cos(facing),
            "pp_facing_sin": np.sin(facing),
            "pp_approach_speed": approach,
            "pp_rel_heading_cos": np.cos(rel_heading),
            "pp_rel_heading_sin": np.sin(rel_heading),
            "pp_relative_velocity_forward": rel_vel[:, 0],
            "pp_relative_velocity_lateral": rel_vel[:, 1],
            "pp_relative_speed": _norm(rel_vel),
        }
    )
    for name in pose.keypoint_names:
        pts = opoint(name)
        safe = name.replace(" ", "_")
        df[f"pp_obj_{safe}_ego_x"] = pts[:, 0]
        df[f"pp_obj_{safe}_ego_y"] = pts[:, 1]
    contact = (np.minimum.reduce([nose_nose, nose_tail, nose_body, body_body]) < 0.5)
    df["pp_contact_flag"] = contact.astype(np.float32)
    df["pp_body_body_vel"] = np.concatenate([[0.0], np.diff(body_body)]) * frame_rate
    return df


def extract_pose_feature_tables(
    pose: PoseData,
    frame_width: int,
    frame_height: int,
    frame_rate: float,
    egocentric: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-identity and ordered-pair pose feature tables for one video."""

    diag = float(np.hypot(frame_width, frame_height)) or 1.0
    identity_frames: list[pd.DataFrame] = []
    for ai, identity in enumerate(pose.identities):
        feats = single_pose_features(pose, ai, diag, frame_rate, egocentric=egocentric)
        feats.insert(0, "video_id", pose.video_id)
        feats.insert(1, "frame_idx", pose.frame_indices)
        feats.insert(2, "subject_id", str(identity))
        feats.insert(3, "object_id", "")
        identity_frames.append(feats)
    identity_df = (
        pd.concat(identity_frames, ignore_index=True)
        if identity_frames
        else pd.DataFrame()
    )

    pair_rows: list[pd.DataFrame] = []
    for si, s_id in enumerate(pose.identities):
        for oi, o_id in enumerate(pose.identities):
            if si == oi:
                continue
            feats = pair_pose_features(pose, si, oi, diag, frame_rate, egocentric=egocentric)
            feats.insert(0, "video_id", pose.video_id)
            feats.insert(1, "frame_idx", pose.frame_indices)
            feats.insert(2, "subject_id", str(s_id))
            feats.insert(3, "object_id", str(o_id))
            pair_rows.append(feats)
    pair_df = (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    )

    for frame in (identity_df, pair_df):
        if not frame.empty:
            num_cols = frame.select_dtypes(include=["float64"]).columns
            frame[num_cols] = frame[num_cols].astype(np.float32)
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            frame.fillna(0.0, inplace=True)
    return identity_df, pair_df
