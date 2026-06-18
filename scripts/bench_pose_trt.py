"""Quick benchmark: YOLO pose .engine vs .pt (per-frame predict latency)."""
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path

import numpy as np


def bench(model, img, iters, **kw):
    for _ in range(10):
        model.predict(img, verbose=False, **kw)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(img, verbose=False, **kw)
        s.append(time.perf_counter() - t0)
    return statistics.fmean(s) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    img = np.random.randint(0, 256, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    pt = Path(args.pose)
    eng = pt.with_suffix(".engine")

    pt_model = YOLO(str(pt))
    pt_ms = bench(pt_model, img, args.iters, imgsz=args.imgsz, half=True, device=0)
    print(f"torch .pt   : {pt_ms:6.2f} ms ({1000.0/pt_ms:5.1f} fps)", flush=True)

    if eng.is_file():
        eng_model = YOLO(str(eng), task="pose")
        eng_ms = bench(eng_model, img, args.iters, imgsz=args.imgsz)
        print(f"TensorRT eng: {eng_ms:6.2f} ms ({1000.0/eng_ms:5.1f} fps) | {pt_ms/eng_ms:.2f}x vs .pt", flush=True)
    else:
        print(f"no engine at {eng}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
