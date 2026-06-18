"""One-shot probe: what TensorRT/ONNX/rfdetr export tooling exists in this env."""
import importlib.metadata as md
import sys


def ver(name):
    try:
        return md.version(name)
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({type(exc).__name__})"


print("python", sys.executable, flush=True)
for p in [
    "torch", "torchvision", "rfdetr", "onnx", "onnxruntime",
    "onnxruntime-gpu", "tensorrt", "torch-tensorrt", "torch_tensorrt",
    "pycuda", "cuda-python", "supervision",
]:
    print(f"{p:18s} {ver(p)}", flush=True)

print("-" * 50, flush=True)
try:
    import torch
    print("torch", torch.__version__, "| cuda build", torch.version.cuda, flush=True)
    print("cuda available", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), flush=True)
except Exception as exc:  # noqa: BLE001
    print("torch import FAILED:", repr(exc), flush=True)

print("-" * 50, flush=True)
try:
    import rfdetr
    print("rfdetr file", getattr(rfdetr, "__file__", "?"), flush=True)
    seg = getattr(rfdetr, "RFDETRSegMedium", None)
    print("RFDETRSegMedium present:", seg is not None, flush=True)
    if seg is not None:
        import inspect
        exp = getattr(seg, "export", None)
        print("has .export:", exp is not None, flush=True)
        if exp is not None:
            try:
                print("export sig:", inspect.signature(exp), flush=True)
            except (TypeError, ValueError) as exc:
                print("export sig unavailable:", exc, flush=True)
        opt = getattr(seg, "optimize_for_inference", None)
        print("has .optimize_for_inference:", opt is not None, flush=True)
except Exception as exc:  # noqa: BLE001
    print("rfdetr import FAILED:", repr(exc), flush=True)

print("-" * 50, flush=True)
try:
    import tensorrt as trt
    print("tensorrt", trt.__version__, flush=True)
except Exception as exc:  # noqa: BLE001
    print("tensorrt import FAILED:", repr(exc), flush=True)
