"""Offline analyzer: run the rule-based detector on a recorded pykaboo session and
report the behavior-label distribution + timeline, so the rules can be tuned against
real data (and against what the overlay video showed).

Reads the DLC tracking CSV (bodycenter + kp1..kp8 per mouse) and optionally the COCO
masks JSON. Builds per-frame FrameRecords and runs RuleBasedSocialDetector.

Usage:
  python analyze_recording.py <tracking_dlc.csv> [--masks <masks_coco.json>] [--params k=v ...]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from rule_based_social import LABELS, RuleBasedSocialDetector, RuleParams


def load_coco_index(path):
    """frame_idx -> {track_str: [polygon, ...]} plus (W, H). Polygons decoded lazily."""
    import json
    import re
    j = json.load(open(path))
    W = int(j["images"][0]["width"]); H = int(j["images"][0]["height"])
    id_to_frame = {}
    for im in j["images"]:
        m = re.search(r"frame[_:]?(\d+)", str(im.get("file_name", "")))
        id_to_frame[im["id"]] = int(m.group(1)) if m else int(im["id"])
    idx = {}
    for an in j["annotations"]:
        fi = id_to_frame.get(an["image_id"])
        tid = str(an.get("track_id") or an.get("category_id") or "")
        seg = an.get("segmentation")
        if fi is None or not seg:
            continue
        idx.setdefault(fi, {}).setdefault(tid, []).append(seg)
    return idx, W, H


def _decode_polys(seg_list, W, H):
    import cv2
    mask = np.zeros((H, W), np.uint8)
    for seg in seg_list:
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

# pykaboo DLC export: kp1..kp8 are the 8 keypoints in KP_ORDER
# (nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail).
KP_COLS = [f"kp{i}" for i in range(1, 9)]


def load_tracking(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
    # columns: (scorer, individuals, bodyparts, coords)
    # keep only real animals (those that actually have kp1 columns), not the time col
    individuals = sorted({
        c[1] for c in df.columns
        if c[1] and not str(c[1]).startswith("Unnamed")
        and any(cc[1] == c[1] and cc[2] == "kp1" for cc in df.columns)
    })
    n = len(df)
    times = df[df.columns[0]].to_numpy().astype(float) if df.columns[0][3] == "time" else np.arange(n) / 30.0
    # find the 'time' column robustly
    time_col = None
    for c in df.columns:
        if str(c[3]).lower() == "time" or str(c[2]).lower() == "time":
            time_col = c
            break
    if time_col is not None:
        times = df[time_col].to_numpy().astype(float)
    out = {}
    for ind in individuals:
        kps = np.full((n, 8, 3), np.nan)
        for ki, name in enumerate(KP_COLS):
            for j, coord in enumerate(("x", "y", "likelihood")):
                col = [c for c in df.columns if c[1] == ind and c[2] == name and c[3] == coord]
                if col:
                    kps[:, ki, j] = df[col[0]].to_numpy().astype(float)
        out[ind] = kps
    return times, out, individuals


def build_frames(times, tracks, ind_map):
    n = len(times)
    frames = []
    for i in range(n):
        mice = {}
        for ind, sid in ind_map.items():
            kp = tracks[ind][i]            # [8,3]
            xy = kp[:, :2]
            sc = kp[:, 2]
            present = np.isfinite(xy).any()
            mice[sid] = {
                "present": bool(present),
                "bbox_xywh": None,
                "score": 0.95,
                "mask": None,
                "keypoints": xy,
                "keypoint_scores": sc,
            }
        frames.append(SimpleNamespace(frame_idx=i, timestamp_s=float(times[i]),
                                      width=0, height=0, mice=mice))
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="param overrides k=v")
    ap.add_argument("--timeline", action="store_true")
    args = ap.parse_args(argv)

    times, tracks, individuals = load_tracking(args.csv)
    ind_map = {}
    for ind in individuals:
        digits = "".join(ch for ch in ind if ch.isdigit())
        ind_map[ind] = digits if digits in ("1", "2") else ("1" if not ind_map else "2")
    print(f"individuals={individuals} -> {ind_map}; frames={len(times)}")

    p = RuleParams()
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        setattr(p, k, type(cur)(v))
        print(f"  param {k} = {getattr(p, k)}")

    det = RuleBasedSocialDetector(identities=("1", "2"), params=p)

    coco_idx = None
    if args.masks:
        coco_idx, mW, mH = load_coco_index(args.masks)
        print(f"masks: {len(coco_idx)} frames with polygons ({mW}x{mH})")

    # report body length / tolerances on a mid frame
    frames = build_frames(times, tracks, ind_map)
    counts = Counter()
    per_track_counts = {"1": Counter(), "2": Counter()}
    bodylens = []
    timeline = []
    n_contact = 0
    for fr in frames:
        # body length estimate (mouse1 nose-tail)
        kp1 = fr.mice["1"]["keypoints"]
        if kp1 is not None and np.all(np.isfinite(kp1[[0, 7]])):
            bodylens.append(float(np.hypot(*(kp1[0] - kp1[7]))))
        if coco_idx is not None:
            polys = coco_idx.get(fr.frame_idx, {})
            for sid in ("1", "2"):
                seg = polys.get(sid)
                fr.mice[sid]["mask"] = _decode_polys(seg, mW, mH) if seg else None
        st = det.process(fr)
        if st is None:
            continue
        act = [k for k in LABELS if st.active.get(k)]
        counts.update(act if act else ["none"])
        for sid in ("1", "2"):
            per_track_counts[sid].update([st.per_track[sid].get("top", "none")])  # the actual chip
        if args.timeline:
            timeline.append((st.frame_idx, act or ["none"]))

    if bodylens:
        print(f"body length (mouse1 nose-tail): median={np.median(bodylens):.1f}px "
              f"min={np.min(bodylens):.0f} max={np.max(bodylens):.0f}")
        bl = float(np.median(bodylens))
        ct = float(np.clip(0.30 * bl, 5.0, 1.5 * bl)) if p.close_tol == 0 else p.close_tol
        print(f"derived close_tol={ct:.1f}  side_tol={p.side_tol}  follow_tol={p.follow_tol}  "
              f"stat={p.stationary_threshold}  move~{max(0.8*p.stationary_threshold,0.012*bl):.2f} "
              f"fast~{max(1.6*p.stationary_threshold,0.030*bl):.2f}")

    # keypoint sanity: nose should lead, tail should trail along the body axis
    f0 = frames[len(frames) // 2]
    kp = f0.mice["1"]["keypoints"]
    if kp is not None:
        print(f"sanity (mid frame) nose={kp[0].round(0)} body={kp[4].round(0)} tail={kp[7].round(0)}")

    total = sum(counts.values())
    print(f"\nscene label distribution ({total} active-label hits over scored frames):")
    for name, c in counts.most_common():
        print(f"  {name:<26} {c:5d}  {100*c/total:5.1f}%")
    for sid in ("1", "2"):
        tot = sum(per_track_counts[sid].values())
        print(f"\nper-mouse {sid} chip (argmax) distribution:")
        for name, c in per_track_counts[sid].most_common(8):
            print(f"  {name:<26} {c:5d}  {100*c/max(tot,1):5.1f}%")

    if args.timeline:
        print("\ntimeline (frame: active):")
        prev = None
        for fi, act in timeline:
            key = tuple(act)
            if key != prev:
                print(f"  f{fi:4d} t={fi/30.0:6.2f}s  {', '.join(act)}")
                prev = key


if __name__ == "__main__":
    main()
