"""Build (and validate) a TensorRT .engine from an RF-DETR seg checkpoint.

Pipeline:
  1. Load the checkpoint exactly as the live app does (same variant + resolution).
  2. Export rfdetr's deploy-mode graph to ONNX at the model's native resolution.
  3. Build an FP16 TensorRT engine from that ONNX via the TensorRT Python API
     (no trtexec / NVIDIA zip required).
  4. Save the engine + a JSON metadata sidecar next to the checkpoint.
  5. Validate: compare engine outputs and final post-processed detections against
     the fp32 PyTorch deploy model on the same input.

Engines are hardware/driver/TensorRT-version specific. Build on the target GPU.

Example:
  python scripts/build_rfdetr_engine.py \
      --checkpoint "D:/Models/Segmentation_models/rfdetr_v3_weights_20260427/medium/checkpoint_best_total.pth"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from rfdetr_trt import (
    RFDETRTensorRTModule,
    engine_path_for_checkpoint,
    write_engine_metadata,
)


def _log(msg: str) -> None:
    print(f"[build_rfdetr_engine] {msg}", flush=True)


def load_model(checkpoint: str, model_key: str):
    """Load the rfdetr model the same way LiveInferenceWorker does.

    Uses acceleration_mode='compatibility' so model.model.inference_model is the
    fp32 deploy graph, which we use as the numerical reference for validation.
    """
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceWorker

    worker = LiveInferenceWorker()
    worker.status_changed.connect(lambda m: _log(f"status: {m}"))
    worker.error_occurred.connect(lambda m: _log(f"error: {m}"))
    model = worker._load_model(model_key, checkpoint, acceleration_mode="compatibility")
    return model


def export_onnx(model, onnx_dir: Path, resolution: int) -> Path:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Exporting ONNX at {resolution}x{resolution} -> {onnx_dir}")
    model.export(output_dir=str(onnx_dir), shape=(resolution, resolution), batch_size=1, verbose=False)
    onnx_path = onnx_dir / "inference_model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Expected ONNX at {onnx_path} but it was not created")
    return onnx_path


def _build_engine_subprocess(
    onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gb: float, verbose: bool, opt_level: int
) -> int:
    """Build the engine in a torch-free subprocess and return its exit code.

    Isolation matters: aggressive TensorRT optimization levels can hard-crash the
    interpreter (Myelin FP16 compilation) on some builds. Running the build in a
    child process lets us detect the crash via the exit code and fall back to a
    safer optimization level instead of taking the whole tool down with it.
    """
    import subprocess

    script = Path(__file__).with_name("trt_build_from_onnx.py")
    cmd = [
        sys.executable,
        str(script),
        "--onnx", str(onnx_path),
        "--engine", str(engine_path),
        "--workspace-gb", str(workspace_gb),
        "--opt-level", str(opt_level),
    ]
    if not fp16:
        cmd.append("--fp32")
    if verbose:
        cmd.append("--verbose")
    label = "fp16" if fp16 else "fp32"
    _log(f"Building engine ({label}, opt_level={opt_level}) in subprocess...")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    _log(f"  subprocess exit={proc.returncode} ({time.perf_counter() - t0:.1f}s)")
    return proc.returncode


def build_engine(
    onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gb: float, verbose: bool, opt_level: int = -1
) -> None:
    """Build the engine, trying safe optimization levels and FP32 as fallbacks.

    For FP16 we prefer opt_level 2 (fast runtime) and fall back to 1; both avoid the
    Myelin FP16 builder crash seen at the default level 5 on TensorRT 10.13. As a last
    resort we build an FP32 (TF32) engine so the user always gets a working engine.
    """
    if engine_path.exists():
        engine_path.unlink()

    if opt_level >= 0:
        ladder = [opt_level]
    elif fp16:
        ladder = [2, 1]
    else:
        ladder = [5]

    for level in ladder:
        rc = _build_engine_subprocess(
            onnx_path, engine_path, fp16=fp16, workspace_gb=workspace_gb, verbose=verbose, opt_level=level
        )
        if rc == 0 and engine_path.is_file():
            _log(f"Engine built ({engine_path.stat().st_size / 1e6:.1f} MB) at opt_level={level} -> {engine_path}")
            return
        _log(f"  opt_level={level} build failed (exit {rc}); trying next fallback")

    if fp16:
        _log("FP16 build failed at all optimization levels; falling back to an FP32 engine")
        rc = _build_engine_subprocess(
            onnx_path, engine_path, fp16=False, workspace_gb=workspace_gb, verbose=verbose, opt_level=5
        )
        if rc == 0 and engine_path.is_file():
            _log(f"FP32 fallback engine built ({engine_path.stat().st_size / 1e6:.1f} MB) -> {engine_path}")
            return

    raise RuntimeError("TensorRT engine build failed (all fallbacks exhausted)")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def _make_input(resolution: int, device, seed: int = 0) -> torch.Tensor:
    """A realistic normalized input: random uint8 image -> /255 -> ImageNet norm."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _frame_input(video: str, frame_idx: int, resolution: int, device):
    """Load one video frame and normalize it exactly like the live worker does."""
    import cv2

    cap = cv2.VideoCapture(video)
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    x = torch.nn.functional.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(1e-9)


def validate(model, engine_path: Path, resolution: int, threshold: float = 0.3, video: str = "", frame_idx: int = 0) -> bool:
    device = model.model.device
    ref_module = model.model.inference_model
    if ref_module is None:
        _log("WARNING: no fp32 reference inference_model available; skipping numeric validation")
        return True

    eng = RFDETRTensorRTModule(engine_path, device=str(device))
    _log(f"Engine I/O: input {eng.input_shape} {eng.input_dtype}; outputs {eng.output_shapes}")

    if video:
        _log(f"Validation input: frame {frame_idx} of {video}")
        x = _frame_input(video, frame_idx, resolution, device)
    else:
        _log("Validation input: random noise (background-query boxes are arbitrary; rely on labels/masks + a --video frame)")
        x = _make_input(resolution, device)

    with torch.no_grad():
        ref_out = ref_module(x.to(dtype=torch.float32))
        eng_out = eng(x)

    # Raw-tensor similarity. Background-query boxes are chaotic, so dets cosine is
    # reported but not gated; classification logits and mask logits are robust and
    # are the meaningful precision checks.
    names = ["dets", "labels", "masks"][: len(ref_out)]
    ok = True
    gates = {"labels": 0.99, "masks": 0.90}
    for name, r, e in zip(names, ref_out, eng_out):
        r = r.float()
        e = e.float()
        if r.shape != e.shape:
            _log(f"  [{name}] SHAPE MISMATCH ref={tuple(r.shape)} eng={tuple(e.shape)}")
            ok = False
            continue
        cos = _cosine(r, e)
        gate = gates.get(name)
        flag = ""
        if gate is not None and cos <= gate:
            flag = "  <-- FAIL"
            ok = False
        _log(f"  [{name}] cos={cos:.5f} max_abs_diff={float((r - e).abs().max()):.4g} shape={tuple(r.shape)}{flag}")

    # End-to-end: compare post-processed detections (this is what actually matters).
    postprocess = model.model.postprocess
    target_sizes = torch.tensor([[resolution, resolution]], device=device, dtype=torch.int64)

    def detect(out):
        preds = {"pred_boxes": out[0], "pred_logits": out[1]}
        if len(out) >= 3:
            preds["pred_masks"] = out[2]
        res = postprocess(preds, target_sizes=target_sizes)[0]
        keep = res["scores"] > threshold
        masks = res["masks"][keep] if "masks" in res else None
        return res["boxes"][keep], res["scores"][keep], res["labels"][keep], masks

    rb, rs, rl, rm = detect(ref_out)
    eb, es, el, em = detect(eng_out)
    _log(f"  detections > {threshold}: ref={len(rs)} engine={len(es)}")

    if video:
        if len(rs) == 0:
            _log("  WARNING: reference found no detections on this frame; try a different --frame")
        else:
            iou = _box_iou(rb, eb) if len(es) else torch.zeros((len(rs), 0), device=device)
            for i in range(len(rs)):
                if iou.shape[1] == 0:
                    _log(f"  ref det {i} (score {float(rs[i]):.3f}) has NO engine match")
                    ok = False
                    continue
                best = int(iou[i].argmax())
                best_iou = float(iou[i, best])
                score_gap = abs(float(rs[i]) - float(es[best]))
                mask_iou = float("nan")
                if rm is not None and em is not None and len(em) > best:
                    mr = rm[i].bool().flatten()
                    me = em[best].bool().flatten()
                    inter = (mr & me).sum().float()
                    union = (mr | me).sum().float().clamp_min(1)
                    mask_iou = float(inter / union)
                tag = "" if (best_iou > 0.85 and score_gap < 0.1) else "  <-- WEAK MATCH"
                _log(
                    f"  ref det {i}: box_iou={best_iou:.3f} score_gap={score_gap:.3f} mask_iou={mask_iou:.3f}{tag}"
                )
                if best_iou <= 0.7:
                    ok = False

    _log(f"VALIDATION {'PASSED' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to the .pth seg checkpoint")
    ap.add_argument("--model-key", default="rfdetr-seg-medium", help="rfdetr model key (default: rfdetr-seg-medium)")
    ap.add_argument("--output", default=None, help="Output .engine path (default: sibling of the checkpoint)")
    ap.add_argument("--resolution", type=int, default=0, help="Override export resolution (default: model native)")
    ap.add_argument("--workspace-gb", type=float, default=4.0, help="TensorRT workspace memory pool (GB)")
    ap.add_argument("--opt-level", type=int, default=-1, help="Force builder opt level (default: auto fallback ladder)")
    ap.add_argument("--fp32", action="store_true", help="Build an FP32 engine instead of FP16")
    ap.add_argument("--no-validate", action="store_true", help="Skip numeric validation against PyTorch")
    ap.add_argument("--validate-only", action="store_true", help="Only validate an existing engine (no export/build)")
    ap.add_argument("--video", default="", help="Validate detections on a real frame from this video")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to use from --video")
    ap.add_argument("--keep-onnx", action="store_true", help="Keep the intermediate ONNX file")
    ap.add_argument("--verbose", action="store_true", help="Verbose TensorRT logging")
    args = ap.parse_args()

    checkpoint = str(Path(args.checkpoint))
    if not Path(checkpoint).is_file():
        _log(f"ERROR: checkpoint not found: {checkpoint}")
        return 2

    if not torch.cuda.is_available():
        _log("ERROR: CUDA is not available; cannot build/run a TensorRT engine")
        return 2

    engine_path = Path(args.output) if args.output else engine_path_for_checkpoint(checkpoint)

    _log(f"Loading model ({args.model_key}) from {checkpoint}")
    model = load_model(checkpoint, args.model_key)
    resolution = int(args.resolution) if args.resolution > 0 else int(model.model.resolution)
    _log(f"Native resolution: {model.model.resolution}; building at {resolution}")

    if args.validate_only:
        if not engine_path.is_file():
            _log(f"ERROR: engine not found for --validate-only: {engine_path}")
            return 2
        ok = validate(model, engine_path, resolution, video=args.video, frame_idx=args.frame)
        _log("DONE")
        return 0 if ok else 1

    onnx_workdir = Path(tempfile.mkdtemp(prefix="rfdetr_onnx_"))
    try:
        onnx_path = export_onnx(model, onnx_workdir, resolution)
        build_engine(
            onnx_path,
            engine_path,
            fp16=not args.fp32,
            workspace_gb=args.workspace_gb,
            verbose=args.verbose,
            opt_level=args.opt_level,
        )
        if args.keep_onnx:
            kept = engine_path.with_suffix(".onnx")
            kept.write_bytes(onnx_path.read_bytes())
            _log(f"Kept ONNX -> {kept}")
    finally:
        if not args.keep_onnx:
            import shutil

            shutil.rmtree(onnx_workdir, ignore_errors=True)

    # Probe the engine for metadata, write sidecar.
    eng = RFDETRTensorRTModule(engine_path, device=str(model.model.device))
    write_engine_metadata(
        engine_path,
        checkpoint_path=checkpoint,
        model_key=args.model_key,
        resolution=resolution,
        batch_size=eng.batch_size,
        fp16=not args.fp32,
        output_names=list(eng.ordered_output_names),
        output_shapes=eng.output_shapes,
    )
    del eng

    ok = True
    if not args.no_validate:
        ok = validate(model, engine_path, resolution, video=args.video, frame_idx=args.frame)

    _log("DONE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
