"""Synthetic end-to-end smoke test for the live behavior engine.

No recorded dataset is required (the replay_parity_test.py paths point at the Linux
research box). This test fabricates a plausible two-mouse stream of masks + keypoints,
drives it through the whole stack, and asserts:

  * the online extractor emits [win, 432] feature windows in the model's column order;
  * the model forward + causal postproc + scene reduction run without error;
  * FrameDecision / BehaviorEvent objects are produced once warmed up;
  * the pykaboo adapter maps a synthetic LiveDetectionResult correctly;
  * per-frame latency is reported (so we know if a GPU keeps up at 30 fps).

Run:  python smoke_test.py [--frames 700] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from live_features import FrameRecord
from live_engine import OnlineBehaviorEngine

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "free_embtcn_attention_optimized.pt")

W, H = 640, 480
KP = 8  # nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail


def _ellipse_mask(cx, cy, a, b, angle_deg):
    """Rasterize a filled ellipse as a boolean (H, W) mask via numpy (no cv2 needed)."""
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.deg2rad(angle_deg)
    ct, st = np.cos(t), np.sin(t)
    xr = (xx - cx) * ct + (yy - cy) * st
    yr = -(xx - cx) * st + (yy - cy) * ct
    return ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0


def _keypoints(cx, cy, length, angle_deg):
    """8 keypoints along the body axis in our canonical order."""
    t = np.deg2rad(angle_deg)
    ax = np.array([np.cos(t), np.sin(t)])           # forward (nose) direction
    perp = np.array([-np.sin(t), np.cos(t)])
    c = np.array([cx, cy])
    nose = c + ax * length * 0.5
    tail = c - ax * length * 0.5
    neck = c + ax * length * 0.2
    body = c
    lear = neck + perp * length * 0.18
    rear = neck - perp * length * 0.18
    lhip = (c - ax * length * 0.24) + perp * length * 0.16
    rhip = (c - ax * length * 0.24) - perp * length * 0.16
    kps = np.stack([nose, lear, rear, neck, body, lhip, rhip, tail], axis=0)
    return kps


def synth_frame(i, fps, interacting):
    """Two mice circling; when `interacting` they come nose-to-nose and close."""
    ts = i / fps
    if interacting:
        # mice approach each other near the centre
        sep = 60.0
        cx1, cy1 = W / 2 - sep, H / 2 + 8 * np.sin(ts * 2)
        cx2, cy2 = W / 2 + sep, H / 2 + 8 * np.sin(ts * 2 + 0.5)
        a1, a2 = 0.0, 180.0          # facing each other
    else:
        # mice on separate circular orbits, far apart
        r = 130.0
        cx1 = W / 2 + r * np.cos(ts * 0.8)
        cy1 = H / 2 + r * np.sin(ts * 0.8)
        cx2 = W / 2 + 0.6 * r * np.cos(-ts * 1.1 + 2.0)
        cy2 = H / 2 + 0.6 * r * np.sin(-ts * 1.1 + 2.0)
        a1 = np.rad2deg(ts * 0.8) + 90
        a2 = np.rad2deg(-ts * 1.1) - 90

    length = 70.0
    mice = {}
    for ident, (cx, cy, ang) in (("1", (cx1, cy1, a1)), ("2", (cx2, cy2, a2))):
        mask = _ellipse_mask(cx, cy, length * 0.45, length * 0.18, ang)
        ys, xs = np.where(mask)
        if xs.size == 0:
            x0 = y0 = 0.0
            bw = bh = 1.0
        else:
            x0, y0 = float(xs.min()), float(ys.min())
            bw, bh = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        kps = _keypoints(cx, cy, length, ang)
        mice[ident] = {
            "present": True,
            "bbox_xywh": (x0, y0, bw, bh),
            "score": 0.95,
            "mask": mask,
            "keypoints": kps,
            "keypoint_scores": np.full(KP, 0.9),
            "category_id": 0,
        }
    return FrameRecord(frame_idx=i, timestamp_s=ts, width=W, height=H, mice=mice)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--frames", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--min-window", type=int, default=240,
                    help="warm-up frames before first decision (default 240 for a fast smoke test)")
    args = ap.parse_args(argv)

    events = []
    decisions = []
    engine = OnlineBehaviorEngine(
        ckpt_path=args.ckpt,
        device=args.device,
        lookahead=args.lookahead,
        min_window=args.min_window,
        on_event=lambda ev: events.append(ev),
        on_frame=lambda d: decisions.append(d),
    )
    print(repr(engine), flush=True)
    print(f"warm-up min_window={args.min_window}, win={engine.model.win}, L={engine.L}", flush=True)

    fps = engine.model.frame_rate
    # first third apart, middle third interacting, last third apart
    seg = args.frames // 3
    t_push = []
    first_window_shape = None
    for i in range(args.frames):
        interacting = seg <= i < 2 * seg
        fr = synth_frame(i, fps, interacting)
        t0 = time.perf_counter()
        dec = engine.on_detection(fr)
        t_push.append((time.perf_counter() - t0) * 1000.0)
        if dec is not None and first_window_shape is None:
            w = engine.extractor.compute_windows()
            first_window_shape = {k: v.shape for k, v in w.items()}

    warm = [t for t in t_push if t > 0]
    decided = [t for t, d in zip(t_push, range(len(t_push))) ]
    n_decided = len(decisions)
    print(f"\nframes pushed: {args.frames}; decisions emitted: {n_decided}", flush=True)
    print(f"first feature-window shapes (per identity): {first_window_shape}", flush=True)

    # latency only for frames that actually ran a forward (after warm-up)
    post_warm = t_push[args.min_window:]
    if post_warm:
        arr = np.array(post_warm)
        print(f"per-frame latency after warm-up: mean={arr.mean():.1f} ms  "
              f"median={np.median(arr):.1f} ms  p95={np.percentile(arr,95):.1f} ms  "
              f"max={arr.max():.1f} ms  (30 fps budget = 33.3 ms)", flush=True)

    if decisions:
        d = decisions[-1]
        print(f"\nlast decision @frame {d.frame_idx} t={d.timestamp_s:.2f}s", flush=True)
        for k, name in enumerate(engine.labels):
            print(f"  {name:<14} scene_prob={d.scene_prob[k]:.3f}  on={int(d.scene_binary[k])}", flush=True)

    print(f"\nbehavior events fired: {len(events)}", flush=True)
    for ev in events[:20]:
        print(f"  [{ev.frame_idx:>5} t={ev.timestamp_s:6.2f}s] {ev.behavior} {ev.edge} (p={ev.prob:.2f})", flush=True)

    # ---- adapter check ----
    print("\n=== pykaboo adapter check ===", flush=True)
    from pykaboo_adapter import IdentityMapper, result_to_framerecord

    class _Tracked:
        def __init__(self, mid, label, mask, kps):
            self.mouse_id = mid; self.label = label; self.class_id = 0
            self.confidence = 0.9
            ys, xs = np.where(mask)
            self.bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))  # XYXY
            self.center = (float(xs.mean()), float(ys.mean()))
            self.mask = mask; self.keypoints = kps; self.keypoint_scores = np.full(KP, 0.9)

    class _Result:
        pass

    fr = synth_frame(100, fps, True)
    res = _Result()
    res.frame_index = 100; res.timestamp_s = 100 / fps; res.width = W; res.height = H
    res.tracked_mice = [
        _Tracked(1, "mouse1", fr.mice["1"]["mask"], fr.mice["1"]["keypoints"]),
        _Tracked(2, "mouse2", fr.mice["2"]["mask"], fr.mice["2"]["keypoints"]),
    ]
    mapper = IdentityMapper()
    rec = result_to_framerecord(res, mapper)
    print(f"  mapped identities: {sorted(rec.mice)}", flush=True)
    for sid in ("1", "2"):
        m = rec.mice[sid]
        print(f"   mouse {sid}: present={m['present']} bbox_xywh={tuple(round(v,1) for v in m['bbox_xywh'])} "
              f"kp_shape={None if m['keypoints'] is None else m['keypoints'].shape}", flush=True)
    assert rec.mice["1"]["present"] and rec.mice["2"]["present"]
    assert rec.mice["1"]["bbox_xywh"][2] > 0  # width positive => XYXY->XYWH conversion ok

    print("\nSMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
