"""Stream-2 contact / depth-order mask features (the hybrid-plan thesis core).

These are the mask-derived features that the 8 keypoints physically CANNOT
produce, computed directly from the segmentation rasters + the (reliable) neck and
tail keypoints. They target exactly the three behaviors the data audit flagged as
mask-dependent (anogenital, mounting, rearing) and the mounting<->nose-to-body 55%
aliasing.

Per directed track (subject A = self, partner B = the other mouse), per frame:

  mc_contact_dist   contour-to-contour min distance A->B  (/body-length; 0 if touching)
  mc_contact_arc    fraction of A's boundary within eps of B  (how much they touch)
  mc_contact_s_on_B contact location projected on B's neck->tail axis, s in [0,1]
                    (0=head third, ~1=anogenital third) -- THE anogenital/nose-to-body
                    discriminator
  mc_contact_s_on_A same projection on A's own axis (is A contacting with its nose?)
  mc_nose_to_Bmask  A's nose -> nearest point on B's mask boundary (/body-length);
                    robust where the body keypoint is unreliable (89% of frames)
  mc_overlap_frac   |A intersect B| / |A|  (occlusion fraction)
  mc_ccount         connected-component count of A (occlusion fragments the silhouette)
  mc_area_ratio     area_A / running-median(area_A)  (the occluded animal shrinks)
  mc_depth_order    area_ratio_A - area_ratio_B, gated by overlap: >0 => A intact on
                    top => mounting evidence. Pose cannot produce this.

The anchor for body length uses nose<->tail (both >0.91 likelihood), NOT the broken
body keypoint. Distances are normalized by mean body length in pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .coco_masks import CocoVideo, MaskRecord, load_coco_videos
from .config import load_config
from .pose import KEYPOINT_ORDER, load_pose_csv

# NOTE: overlap_frac and depth_order were DROPPED after measurement. The RF-DETR
# masks are mutually-exclusive instance masks (mask_A intersect mask_B ~ 0 even
# during mounting), so overlap-based depth ordering is unrecoverable. The occlusion
# cue that DOES survive is the occluded animal's silhouette collapsing/vanishing,
# captured by mc_area_ratio (area / running-median) and mc_ccount (component count
# -> 0 when the mask disappears). See experiments/free_data_audit.py findings.
_PERFRAME = [
    "mc_contact_dist", "mc_contact_arc", "mc_contact_s_on_B", "mc_contact_s_on_A",
    "mc_nose_to_Bmask", "mc_ccount",
]
FEATURE_NAMES = _PERFRAME + ["mc_area_ratio", "mc_mask_valid"]
_KP = {n: i for i, n in enumerate(KEYPOINT_ORDER)}
_MEDIAN_WIN = 31  # frames (~1 s at 30 fps) for the area running median


def _best_by_frame_identity(video: CocoVideo) -> dict[tuple[int, str], MaskRecord]:
    best: dict[tuple[int, str], MaskRecord] = {}
    for fi, recs in video.records_by_frame().items():
        for r in recs:
            k = (fi, r.identity)
            if k not in best or r.score > best[k].score:
                best[k] = r
    return best


def _crop_union(mA, mB, margin=8):
    """Crop both masks to the union bounding box (+margin) for cheap raster ops."""
    union = mA | mB
    ys = np.where(union.any(axis=1))[0]
    xs = np.where(union.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = max(0, ys[0] - margin), min(union.shape[0], ys[-1] + margin + 1)
    x0, x1 = max(0, xs[0] - margin), min(union.shape[1], xs[-1] + margin + 1)
    return (mA[y0:y1, x0:x1], mB[y0:y1, x0:x1], y0, x0)


def _boundary(m: np.ndarray) -> np.ndarray:
    """Boolean boundary pixels of a binary mask (1px erosion difference)."""
    er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return m & (er == 0)


def _proj_s(pts_xy: np.ndarray, neck: np.ndarray, tail: np.ndarray) -> float:
    """Mean projection of points onto the neck->tail axis, clamped to [0,1]."""
    axis = tail - neck
    L2 = float(axis @ axis)
    if L2 < 1e-6 or pts_xy.shape[0] == 0:
        return np.nan
    s = ((pts_xy - neck) @ axis) / L2
    return float(np.clip(s, 0.0, 1.0).mean())


def _frame_pair_features(mA, mB, kpA, kpB, Lbar_px, eps_px):
    """Directed A->B contact/occlusion features for one frame (cropped rasters).

    kpA/kpB are [K,2] pixel coords already shifted into the crop frame.
    Returns (contact_dist, contact_arc, s_on_B, s_on_A, nose_to_Bmask,
             overlap_frac, ccount_A, area_A).
    """
    areaA = float(mA.sum())
    if areaA == 0 or mB.sum() == 0:
        return (np.nan,) * 6 + (np.nan, areaA)
    # distance transform: distance from every pixel to the nearest B pixel
    dtB = cv2.distanceTransform((~mB).astype(np.uint8), cv2.DIST_L2, 3)
    bndA = _boundary(mA)
    dt_on_bndA = dtB[bndA]
    contact_dist = float(dt_on_bndA.min()) if dt_on_bndA.size else np.nan
    # contact set = A pixels within eps of B (touch zone)
    contact = mA & (dtB <= eps_px)
    contact_arc = (float((contact & bndA).sum()) / float(bndA.sum())
                   if bndA.sum() else 0.0)
    cy, cx = np.where(contact)
    if cy.size:
        pts = np.stack([cx, cy], axis=1).astype(np.float32)  # (x,y)
        s_on_B = _proj_s(pts, kpB[_KP["neck"]], kpB[_KP["tail"]])
        s_on_A = _proj_s(pts, kpA[_KP["neck"]], kpA[_KP["tail"]])
    else:
        s_on_B = s_on_A = np.nan
    # nose_A -> nearest B mask pixel (guard NaN/low-likelihood nose)
    nx, ny = kpA[_KP["nose"]]
    if (np.isfinite(nx) and np.isfinite(ny)
            and 0 <= int(ny) < dtB.shape[0] and 0 <= int(nx) < dtB.shape[1]):
        nose_to_B = float(dtB[int(ny), int(nx)])
    else:
        nose_to_B = np.nan
    overlap = float((mA & mB).sum())
    overlap_frac = overlap / areaA
    ccount = cv2.connectedComponents(mA.astype(np.uint8))[0] - 1
    L = max(Lbar_px, 1e-6)
    return (contact_dist / L, contact_arc, s_on_B, s_on_A, nose_to_B / L,
            overlap_frac, float(ccount), areaA)


def _contact_tracks_from_loaded(video, pose, ids: list[str] | None = None,
                                log: Callable[[str], None] | None = None
                                ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Core compute on already-loaded ``video`` (CocoVideo) + ``pose`` (PoseData).

    Dataset-agnostic: works for the free-interaction and aggression models alike
    (both are 2-animal, 8-keypoint, identity "1"/"2"). Returns
    {subject_id -> [T, F] directed contact features}, FEATURE_NAMES.
    """
    best = _best_by_frame_identity(video)
    frames = video.frame_indices
    if ids is None:
        ids = [i for i in ("1", "2") if i in pose.identities] or list(pose.identities[:2])
    if len(ids) < 2:
        return {}, FEATURE_NAMES
    a_id, b_id = ids[0], ids[1]
    pid = {ident: pose.identities.index(ident) for ident in ids}
    T = len(frames)
    pairs = [(a_id, b_id), (b_id, a_id)]

    raw = {(a, b): {n: np.full(T, np.nan) for n in _PERFRAME} for a, b in pairs}
    areas = {a_id: np.full(T, np.nan), b_id: np.full(T, np.nan)}

    for ti, fi in enumerate(frames):
        rA = best.get((fi, a_id)); rB = best.get((fi, b_id))
        kp = pose.coords[fi] if fi < pose.coords.shape[0] else None
        if rA is None or rB is None or kp is None:
            continue
        mA = rA.decode_mask(); mB = rB.decode_mask()
        if mA is None or mB is None:
            continue
        cr = _crop_union(mA, mB)
        if cr is None:
            continue
        cmA, cmB, y0, x0 = cr
        kpA = kp[pid[a_id], :, :2].copy(); kpA[:, 0] -= x0; kpA[:, 1] -= y0
        kpB = kp[pid[b_id], :, :2].copy(); kpB[:, 0] -= x0; kpB[:, 1] -= y0

        def blen(kpc):
            return float(np.linalg.norm(kpc[_KP["nose"]] - kpc[_KP["tail"]]))
        Lbar = 0.5 * (blen(kpA) + blen(kpB))
        if not np.isfinite(Lbar) or Lbar < 1.0:
            Lbar = 40.0  # fallback when nose/tail are NaN (rare)
        eps = max(3.0, 0.05 * Lbar)
        for (a, b), (cma, cmb, ka, kb) in {
            (a_id, b_id): (cmA, cmB, kpA, kpB),
            (b_id, a_id): (cmB, cmA, kpB, kpA),
        }.items():
            f = _frame_pair_features(cma, cmb, ka, kb, Lbar, eps)
            (cd, arc, sB, sA, n2B, _ofrac, cc, area) = f
            d = raw[(a, b)]
            d["mc_contact_dist"][ti] = cd
            d["mc_contact_arc"][ti] = arc
            d["mc_contact_s_on_B"][ti] = sB
            d["mc_contact_s_on_A"][ti] = sA
            d["mc_nose_to_Bmask"][ti] = n2B
            d["mc_ccount"][ti] = cc
            areas[a][ti] = area
        if log and ti % 2000 == 0:
            log(f"  {video.video_id} contact frame {ti}/{T}")

    def running_median_ratio(area: np.ndarray) -> np.ndarray:
        out = np.full_like(area, np.nan)
        half = _MEDIAN_WIN // 2
        for i in range(len(area)):
            lo, hi = max(0, i - half), min(len(area), i + half + 1)
            w = area[lo:hi]
            w = w[~np.isnan(w)]
            if w.size and not np.isnan(area[i]):
                med = np.median(w)
                out[i] = area[i] / med if med > 0 else np.nan
        return out

    ratio = {a: running_median_ratio(areas[a]) for a in ids}
    out: dict[str, np.ndarray] = {}
    for a, b in pairs:
        d = raw[(a, b)]
        valid = (~np.isnan(d["mc_contact_dist"])).astype(np.float32)
        cols = [d[n] for n in _PERFRAME] + [ratio[a], valid]
        mat = np.stack(cols, axis=1).astype(np.float32)
        out[a] = np.nan_to_num(mat, nan=0.0)
    return out, FEATURE_NAMES


def build_contact_tracks(coco_path: str | Path, pose_path: str | Path,
                         config_path: str = "configs/default.yaml",
                         log: Callable[[str], None] | None = None
                         ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Path-based wrapper: load COCO + pose, then compute contact tracks."""
    cfg = load_config(config_path)
    video = next(iter(load_coco_videos(coco_path, cfg.data).values()))
    pose = load_pose_csv(pose_path)
    return _contact_tracks_from_loaded(video, pose, log=log)


def contact_pair_dataframe(video, pose, video_id: str | None = None):
    """Per-(subject, object) contact features as a pair_df-compatible DataFrame
    keyed by (video_id, frame_idx, subject_id, object_id), for merging into the
    main social pipeline. Returns an empty frame if fewer than 2 animals."""
    import pandas as pd

    tracks, names = _contact_tracks_from_loaded(video, pose)
    if not tracks:
        return pd.DataFrame()
    frames = list(video.frame_indices)
    vid = video_id or video.video_id
    rows = []
    for subject, mat in tracks.items():
        partner = next((x for x in tracks if x != subject), "")
        df = pd.DataFrame(mat[: len(frames)], columns=names)
        df.insert(0, "object_id", partner)
        df.insert(0, "subject_id", subject)
        df.insert(0, "frame_idx", frames[: len(df)])
        df.insert(0, "video_id", vid)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cache builder (CPU, safe to run while a GPU job trains)
# --------------------------------------------------------------------------- #

def build_or_load_contact_cache(cache_dir: str | Path = "outputs/free_maskcontact",
                                rebuild: bool = False,
                                log: Callable[[str], None] | None = None
                                ) -> dict[str, dict[str, np.ndarray]]:
    from .free_social import discover_videos
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, dict[str, np.ndarray]] = {}
    for entry in discover_videos():
        npz = cache_dir / f"{entry.video_id}.npz"
        if npz.exists() and not rebuild:
            z = np.load(npz)
            by_video[entry.video_id] = {"1": z["id1"], "2": z["id2"]}
            continue
        if entry.pose_csv is None:
            if log:
                log(f"skip {entry.video_id}: no pose")
            continue
        if log:
            log(f"building contact features for {entry.video_id}")
        tracks, _ = build_contact_tracks(entry.coco, entry.pose_csv, log=log)
        np.savez_compressed(npz, id1=tracks["1"], id2=tracks["2"])
        by_video[entry.video_id] = tracks
    return by_video
