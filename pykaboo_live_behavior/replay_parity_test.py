"""Parity gate: replay a recorded video frame-by-frame through the live engine and
compare to the EXACT offline pipeline. Run this before wiring anything to hardware.

It measures the two things that decide whether the live system is trustworthy:

  1. FEATURE parity: the live extractor's 432-vector for a decision frame vs the
     offline build_free_inference_tracks 432-vector for the same frame. Reports per
     column MAE and the worst columns (expected offenders: the slow wavelet scales at
     the right edge, area_norm, pair_contact_flag).
  2. PREDICTION parity: live windowed model probabilities (scene MAX over the two
     mice) vs offline overlap-averaged probabilities, per class MAE, swept over the
     look-ahead L. This is the number that matters for triggering.

Usage:
  python3 replay_parity_test.py --coco <masks_coco.json> --pose <tracking.csv> \
      [--ckpt checkpoints/free_embtcn_attention_optimized.pt] [--max-frames 1200] \
      [--lookaheads 0 4 8 16 32] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from behavior_segmentation.coco_masks import load_coco_videos
from behavior_segmentation.config import load_config
from behavior_segmentation.free_infer import build_free_inference_tracks
from behavior_segmentation.pose import load_pose_csv
from live_features import CONFIG_PATH, FrameRecord, OnlineFeatureExtractor
from model_runtime import LiveModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "free_embtcn_attention_optimized.pt")
DEFAULT_COCO = "/home/andry/tracking_project/free_interaction_model/dataset/training/225_adu_masks_coco.json"
DEFAULT_POSE = "/home/andry/tracking_project/free_interaction_model/dataset/training/225_adu_tracking.csv"


def offline_track_probs(model: LiveModel, feats_TxD: np.ndarray) -> np.ndarray:
    """Offline overlap-averaged probabilities for one track [T, D] -> [K, T]."""
    feats = model.normalize(feats_TxD)
    T = feats.shape[0]
    win, stride = model.win, model.eval_stride
    prob_sum = np.zeros((model.num_classes, T), dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)
    starts = list(range(0, max(T - win, 0) + 1, max(stride, 1)))
    if not starts or starts[-1] + win < T:
        starts.append(max(T - win, 0))
    for s in starts:
        e = min(s + win, T)
        p = model.forward_window(feats[s:e])  # [K, e-s]
        prob_sum[:, s:e] += p
        counts[s:e] += 1.0
    counts = np.clip(counts, 1.0, None)
    return prob_sum / counts


def build_frame_records(coco_path, pose_path, frame_rate):
    """Decode the recorded video into a list of FrameRecord (plays the role of pykaboo)."""
    cfg = load_config(CONFIG_PATH)
    videos = load_coco_videos(coco_path, cfg.data)
    video = max(videos.values(), key=lambda v: v.num_frames)
    pose = load_pose_csv(pose_path, video_id=video.video_id, min_likelihood=0.0)
    # NOTE: min_likelihood=0.0 here; the extractor re-applies the 0.3 threshold to
    # match social_pipeline. We feed raw coords + scores so the masking matches.
    recs_by_frame = video.records_by_frame()
    id_index = {ident: pose.identity_index(ident) for ident in ("1", "2")}
    ts = {fi: t for fi, t in zip(video.frame_indices, video.timestamps)}

    frames = []
    for pos, fi in enumerate(video.frame_indices):
        mice = {}
        for rec in recs_by_frame.get(fi, []):
            ident = str(rec.identity)
            if ident not in ("1", "2"):
                continue
            mask = rec.decode_mask()
            kps = ksc = None
            ai = id_index.get(ident)
            if ai is not None and pos < pose.coords.shape[0]:
                c = pose.coords[pos, ai]  # [K,3]
                kps = c[:, :2].copy()
                ksc = c[:, 2].copy()
            mice[ident] = {
                "present": True,
                "bbox_xywh": tuple(float(b) for b in rec.bbox),
                "score": float(rec.score),
                "mask": (np.asarray(mask, bool) if mask is not None else None),
                "keypoints": kps,
                "keypoint_scores": ksc,
                "category_id": rec.category_id,
            }
        for ident in ("1", "2"):
            mice.setdefault(ident, {"present": False, "bbox_xywh": (0.0, 0.0, 0.0, 0.0),
                                    "score": 0.0, "mask": None, "keypoints": None,
                                    "keypoint_scores": None, "category_id": None})
        frames.append(FrameRecord(frame_idx=int(fi), timestamp_s=float(ts.get(fi, fi / frame_rate)),
                                  width=video.width, height=video.height, mice=mice))
    return frames, video.video_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--coco", default=DEFAULT_COCO)
    ap.add_argument("--pose", default=DEFAULT_POSE)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--lookaheads", type=int, nargs="*", default=[0, 4, 8, 16, 32])
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    print(f"loading model {args.ckpt}", flush=True)
    model = LiveModel(args.ckpt, device=args.device)
    print(f"  {model}", flush=True)

    print(f"decoding recorded video {os.path.basename(args.coco)} ...", flush=True)
    frames, vid = build_frame_records(args.coco, args.pose, model.frame_rate)
    if args.max_frames:
        frames = frames[: args.max_frames]
    T = len(frames)
    print(f"  {vid}: {T} frames", flush=True)

    # ---- OFFLINE reference (whole clip) ----
    print("running OFFLINE pipeline (build_free_inference_tracks + overlap-avg) ...", flush=True)
    cfg = load_config(CONFIG_PATH)
    cfg = cfg.model_copy(deep=True)
    cfg.data.frame_rate = model.frame_rate
    feats, off_tracks = build_free_inference_tracks(args.coco, cfg, model.feature_names, pose_path=args.pose)
    off_by_sid = {str(t.subject_id): t for t in off_tracks}
    off_feat = {sid: t.features[:T] for sid, t in off_by_sid.items()}          # [T,432] selected
    off_prob = {sid: offline_track_probs(model, t.features)[:, :T] for sid, t in off_by_sid.items()}
    off_scene = np.zeros((model.num_classes, T))
    for sid in off_prob:
        off_scene = np.maximum(off_scene, off_prob[sid])

    # ---- LIVE streaming, once per look-ahead ----
    behaviors = [model.labels[k] for k in range(model.num_classes)]
    print("\n=== PREDICTION parity (scene MAX prob, live vs offline) ===", flush=True)
    print(f"{'L (frames/s)':>14} | " + " ".join(f"{b[:9]:>9}" for b in behaviors[1:]) + " |  mean", flush=True)

    best_feat_report = None
    for L in args.lookaheads:
        ext = OnlineFeatureExtractor(model.feature_names, model.frame_rate, model.win,
                                     identities=("1", "2"))
        live_prob_rows = {}   # frame_idx -> scene prob [K]
        live_feat_rows = {sid: {} for sid in ("1", "2")}  # frame_idx -> [432]
        for fr in frames:
            ext.push(fr)
            w = ext.compute_windows()
            if w is None:
                continue
            sids = [s for s in ("1", "2") if s in w]
            Teff = min(w[s].shape[0] for s in sids)
            col = max(Teff - 1 - L, 0)
            batch = np.stack([model.normalize(w[s])[-Teff:] for s in sids], 0)
            probs = model.forward_window(batch)  # [B,K,Teff]
            scene = np.zeros(model.num_classes)
            for bi, s in enumerate(sids):
                scene = np.maximum(scene, probs[bi, :, col])
                live_feat_rows[s][fr.frame_idx - L if len(ext._buf) > L else fr.frame_idx] = \
                    w[s][col]
            dframe = ext._buf[-1 - L] if len(ext._buf) > L else ext._buf[-1]
            live_prob_rows[dframe.frame_idx] = scene

        # align on common frame indices
        common = sorted(set(live_prob_rows) & set(range(T)))
        if not common:
            print(f"  L={L}: no overlap", flush=True)
            continue
        lp = np.stack([live_prob_rows[f] for f in common], 1)  # [K, M]
        op = off_scene[:, common]
        mae = np.abs(lp - op).mean(axis=1)  # [K]
        row = " ".join(f"{mae[k]:9.3f}" for k in range(1, model.num_classes))
        print(f"{L:>6}/{L/model.frame_rate:>6.2f}s | {row} | {mae[1:].mean():.3f}", flush=True)

        # feature parity at this L (use sid "1")
        if "1" in live_feat_rows and live_feat_rows["1"]:
            fcommon = sorted(set(live_feat_rows["1"]) & set(range(T)))
            lf = np.stack([live_feat_rows["1"][f] for f in fcommon], 0)  # [M,432]
            of = off_feat["1"][fcommon]
            col_mae = np.abs(lf - of).mean(axis=0)  # [432]
            best_feat_report = (L, col_mae)

    # ---- feature parity detail (last L) ----
    if best_feat_report is not None:
        L, col_mae = best_feat_report
        order = np.argsort(col_mae)[::-1]
        print(f"\n=== FEATURE parity (subject '1', L={L}) ===", flush=True)
        print(f"overall mean |live-offline| per column = {col_mae.mean():.5f}", flush=True)
        print(f"columns with MAE < 1e-3: {(col_mae < 1e-3).sum()}/{len(col_mae)}", flush=True)
        print("top-15 worst columns (expected: slow wavelet p4/p5, area_norm, contact):", flush=True)
        for j in order[:15]:
            print(f"  {model.feature_names[j]:<34} MAE={col_mae[j]:.4f}", flush=True)

    print("\nInterpretation: feature MAE near 0 for non-wavelet columns confirms the "
          "online extractor reproduces the trained surface. Pick the smallest L whose "
          "prediction MAE is acceptable for your trigger (see PLAN.md latency budget).",
          flush=True)


if __name__ == "__main__":
    main()
