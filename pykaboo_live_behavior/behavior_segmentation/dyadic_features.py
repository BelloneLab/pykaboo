"""Egocentric + rich relational (dyadic) pose features for cross-video transfer.

The cross-video failure is mostly an INVARIANCE problem: features in cage
coordinates encode where/how the animal sits in the arena and the camera zoom, none
of which transfer across sessions. This module re-expresses everything in each
animal's OWN egocentric frame (translate to body center, rotate to heading, scale by
body length), so the representation is invariant to global translation, rotation, and
scale. On top of that it computes the relational geometry that actually discriminates
aggression: where the partner sits in the subject's frame, body-axis alignment
(parallel vs facing vs perpendicular), anogenital (nose-to-tail) contact, an on-top
proxy, approach velocity, and burst motion energy (attacks are fast).

Output: a tidy per-frame, ordered-pair feature table keyed by
``(video_id, frame_idx, subject_id, object_id)`` so it merges with the pairwise
feature tables. Every feature here is invariant by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pose import PoseData

EPS = 1e-6


def _interp(xy: np.ndarray, max_gap: int = 10) -> np.ndarray:
    out = xy.copy()
    for c in range(out.shape[1]):
        s = pd.Series(out[:, c]).interpolate("linear", limit=max_gap, limit_area="inside")
        out[:, c] = s.ffill().bfill().to_numpy()
    return out


def _kp(pose: PoseData, ai: int, name: str) -> np.ndarray:
    ki = pose.keypoint_names.index(name)
    return _interp(pose.coords[:, ai, ki, :2])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rotate(v: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate [F,2] vectors by angle -phi (into the egocentric frame)."""
    c, s = np.cos(phi), np.sin(phi)
    x = c * v[:, 0] + s * v[:, 1]
    y = -s * v[:, 0] + c * v[:, 1]
    return np.stack([x, y], axis=1)


def _ego_frame(pose: PoseData, ai: int):
    nose = _kp(pose, ai, "nose")
    neck = _kp(pose, ai, "neck")
    body = _kp(pose, ai, "body")
    tail = _kp(pose, ai, "tail")
    center = body
    phi = np.arctan2((nose - neck)[:, 1], (nose - neck)[:, 0])
    L = np.hypot(*(nose - tail).T) + EPS
    return center, phi, L, dict(nose=nose, neck=neck, body=body, tail=tail)


def _to_ego(p: np.ndarray, center: np.ndarray, phi: np.ndarray, L: np.ndarray) -> np.ndarray:
    return _rotate(p - center, phi) / L[:, None]


def dyadic_pair_features(pose: PoseData, si: int, oi: int, frame_rate: float) -> pd.DataFrame:
    cs, phis, Ls, S = _ego_frame(pose, si)
    co, phio, Lo, O = _ego_frame(pose, oi)

    # partner keypoints expressed in the SUBJECT's egocentric frame (invariant config)
    o_nose_e = _to_ego(O["nose"], cs, phis, Ls)
    o_body_e = _to_ego(O["body"], cs, phis, Ls)
    o_tail_e = _to_ego(O["tail"], cs, phis, Ls)

    def d(a, b):  # distance normalized by subject body length
        return np.hypot(*(a - b).T) / Ls

    nose_nose = d(S["nose"], O["nose"])
    s_nose_o_tail = d(S["nose"], O["tail"])     # subject sniffs partner anogenital
    o_nose_s_tail = d(O["nose"], S["tail"])     # partner sniffs subject
    body_body = d(S["body"], O["body"])
    nose_body = d(S["nose"], O["body"])
    min_contact = np.minimum.reduce([nose_nose, s_nose_o_tail, o_nose_s_tail, nose_body])

    # body-axis alignment: 0 = parallel/same, pi = anti-parallel; |.|~pi/2 = perpendicular
    rel_axis = _wrap(phis - phio)
    # facing: subject heading vs vector to partner
    to_o = O["body"] - S["body"]
    facing = _wrap(np.arctan2(to_o[:, 1], to_o[:, 0]) - phis)

    # on-top proxy: who is "ahead/above" along subject axis (partner body x in ego frame)
    on_top = o_body_e[:, 0]                      # >0 partner is in front of subject

    # approach velocity (rate of body-body closing) and relative speed
    approach = np.zeros_like(body_body)
    approach[1:] = -np.diff(body_body) * frame_rate

    # burst motion energy (fast motion = attack); world speed / L, summed over keypoints
    def speed(pts):
        v = np.zeros_like(pts); v[1:] = np.diff(pts, axis=0)
        return np.hypot(*v.T) * frame_rate
    s_motion = sum(speed(S[k]) for k in S) / (Ls)
    o_motion = sum(speed(O[k]) for k in O) / (Ls)
    rel_motion = np.abs(s_motion - o_motion)
    s_accel = np.zeros_like(s_motion); s_accel[1:] = np.diff(s_motion) * frame_rate

    df = pd.DataFrame({
        "dy_o_nose_x": o_nose_e[:, 0], "dy_o_nose_y": o_nose_e[:, 1],
        "dy_o_body_x": o_body_e[:, 0], "dy_o_body_y": o_body_e[:, 1],
        "dy_o_tail_x": o_tail_e[:, 0], "dy_o_tail_y": o_tail_e[:, 1],
        "dy_nose_nose": nose_nose, "dy_nose_anogenital": s_nose_o_tail,
        "dy_partner_anogenital": o_nose_s_tail, "dy_body_body": body_body,
        "dy_min_contact": min_contact,
        "dy_axis_align_cos": np.cos(rel_axis), "dy_axis_align_sin": np.sin(rel_axis),
        "dy_facing_cos": np.cos(facing), "dy_facing_sin": np.sin(facing),
        "dy_on_top": on_top, "dy_approach_speed": approach,
        "dy_subject_motion": s_motion, "dy_partner_motion": o_motion,
        "dy_relative_motion": rel_motion, "dy_subject_accel": s_accel,
        "dy_contact_flag": (min_contact < 0.5).astype(np.float32),
    })
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def extract_dyadic_features(pose: PoseData, frame_rate: float) -> pd.DataFrame:
    """Egocentric+relational features for every ordered pair, merge-ready."""
    rows = []
    for si, s_id in enumerate(pose.identities):
        for oi, o_id in enumerate(pose.identities):
            if si == oi:
                continue
            feats = dyadic_pair_features(pose, si, oi, frame_rate)
            feats.insert(0, "video_id", pose.video_id)
            feats.insert(1, "frame_idx", pose.frame_indices)
            feats.insert(2, "subject_id", str(s_id))
            feats.insert(3, "object_id", str(o_id))
            rows.append(feats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
