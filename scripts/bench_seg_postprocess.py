"""Measure where the seg pipeline time goes: engine forward vs PostProcess.

PostProcess upsamples `num_select` candidate masks to the full target size every
frame. With num_select=200 and a 1600x1182 output that is 200 big interpolations,
even though only a couple of detections survive the threshold. This shows the cost
and how much a smaller num_select saves.
"""
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch


def timeit(fn, iters, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append(time.perf_counter() - t0)
    return statistics.fmean(s) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-h", type=int, default=1182)
    ap.add_argument("--out-w", type=int, default=1600)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()

    from PySide6.QtCore import QCoreApplication
    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceWorker

    w = LiveInferenceWorker()
    w.status_changed.connect(lambda m: None)
    model = w._load_model("rfdetr-seg-medium", str(Path(a.checkpoint)), acceleration_mode="max_gpu_trt")
    ctx = model.model
    eng = ctx.inference_model
    pp = ctx.postprocess
    res = int(model._optimized_resolution)
    dev = ctx.device
    print(f"engine res={res}, postprocess.num_select={getattr(pp, 'num_select', '?')}", flush=True)

    x = torch.randn(1, 3, res, res, device=dev, dtype=model._optimized_dtype)
    with torch.no_grad():
        out = eng(x)
    preds = {"pred_boxes": out[0], "pred_logits": out[1]}
    if len(out) >= 3:
        preds["pred_masks"] = out[2]
    target = torch.tensor([[a.out_h, a.out_w]], device=dev, dtype=torch.int64)

    fwd_ms = timeit(lambda: eng(x), a.iters)
    print(f"engine forward                : {fwd_ms:6.2f} ms", flush=True)

    for ns in (getattr(pp, "num_select", 200), 100, 50, 20, 10):
        pp.num_select = int(ns)
        with torch.no_grad():
            pp_ms = timeit(lambda: pp(preds, target_sizes=target), a.iters)
        print(f"postprocess @ num_select={ns:<4d}: {pp_ms:6.2f} ms  (-> {fwd_ms + pp_ms:6.2f} ms seg+post)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
