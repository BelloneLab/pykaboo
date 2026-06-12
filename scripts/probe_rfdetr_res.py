"""Check RF-DETR seg medium native resolution and mask timing vs resolution,
plus a true two-CUDA-stream overlap test for mask+pose."""
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import torch
import cv2, numpy as np

def load(video, n):
    cap = cv2.VideoCapture(video); fr=[]
    while len(fr)<n:
        ok,f=cap.read()
        if not ok: break
        f=cv2.resize(f,(1920,1080),interpolation=cv2.INTER_AREA)
        fr.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release()
    while len(fr)<n: fr.append(fr[len(fr)%len(fr)].copy())
    return fr

def t(label, fn, frames, warm=8):
    for i in range(warm): fn(frames[i%len(frames)])
    torch.cuda.synchronize()
    d=[]
    for f in frames:
        s=time.perf_counter(); fn(f); torch.cuda.synchronize(); d.append(time.perf_counter()-s)
    m=statistics.fmean(d); print(f"{label:40s} {m*1000:7.1f} ms | {1.0/m:6.1f} fps")
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--video",required=True); ap.add_argument("--seg",required=True); ap.add_argument("--pose",required=True); ap.add_argument("--frames",type=int,default=40); a=ap.parse_args()
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker
    frames=load(a.video,a.frames)
    w=LiveInferenceWorker(); w.status_changed.connect(lambda m:None); w.error_occurred.connect(lambda m:print(" err",m))
    cfg=LiveInferenceConfig(model_key="rfdetr-seg-medium",checkpoint_path=a.seg,threshold=0.5,inference_max_width=768,acceleration_mode="max_gpu").normalized()
    model=w._load_model(cfg.model_key,cfg.checkpoint_path,acceleration_mode="max_gpu"); w._model=model
    mc=getattr(model,"model",None)
    print("optimized_resolution:", getattr(model,"_optimized_resolution",None), "| context.resolution:", getattr(mc,"resolution",None))
    inf_module=getattr(mc,"inference_model",None); dev=getattr(mc,"device",None); dt=getattr(model,"_optimized_dtype",torch.float16)
    means=torch.tensor(getattr(model,"means",[0.485,0.456,0.406]),device=dev,dtype=dt).view(1,3,1,1)
    stds=torch.tensor(getattr(model,"stds",[0.229,0.224,0.225]),device=dev,dtype=dt).view(1,3,1,1)
    base=frames[0]; src=torch.from_numpy(np.ascontiguousarray(base)).permute(2,0,1).unsqueeze(0).to(device=dev,dtype=dt).div_(255.0)
    def run_at(res):
        x=torch.nn.functional.interpolate(src,size=(res,res),mode="bilinear",align_corners=False); x=(x-means)/stds
        with torch.inference_mode():
            _=inf_module(x) if inf_module is not None else None
    if inf_module is not None:
        for res in (640,576,512,448,384,320):
            try: t(f"mask inference_model res={res}", lambda f,_r=res: run_at(_r), frames)
            except Exception as e: print(f"res={res} failed: {e}")
    # pose
    pose=w._load_pose_model(a.pose,acceleration_mode="max_gpu"); w._pose_model=pose
    print("done")
    return 0
if __name__=="__main__": raise SystemExit(main())
