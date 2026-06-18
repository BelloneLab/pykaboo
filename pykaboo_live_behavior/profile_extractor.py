"""Profile where compute_windows() spends its time, to target the real-time fix."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from smoke_test import synth_frame, W, H
from live_features import OnlineFeatureExtractor
from model_runtime import LiveModel
from behavior_segmentation.dataset import build_feature_column_list, build_inference_tracks

HERE = os.path.dirname(os.path.abspath(__file__))
ckpt = os.path.join(HERE, "checkpoints", "free_embtcn_attention_optimized.pt")
model = LiveModel(ckpt, device="cuda")
ext = OnlineFeatureExtractor(model.feature_names, model.frame_rate, model.win, identities=("1", "2"))

# fill a full 480-frame buffer
N = 480
for i in range(N):
    ext.push(synth_frame(i, model.frame_rate, 240 <= i < 360))
print(f"buffer filled: {len(ext._buf)} frames")

def timeit(fn, n=5):
    fn()  # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return np.median(ts)

# stage timings
def stage_build():
    return ext._build_coco_pose()
video, pose, Wd, Hd = ext._build_coco_pose()

from behavior_segmentation.features import extract_video_features
from behavior_segmentation.pose_features import extract_pose_feature_tables

t_build = timeit(stage_build)
t_vid = timeit(lambda: extract_video_features(video, ext.config))
identity_df, pair_df = extract_video_features(video, ext.config)
t_pose = timeit(lambda: extract_pose_feature_tables(pose, Wd, Hd, ext.config.data.frame_rate, egocentric=ext.egocentric))
t_social = timeit(lambda: ext._social_features(video, pose, Wd, Hd))
idf, pdf = ext._social_features(video, pose, Wd, Hd)
full_names = build_feature_column_list(idf, pdf, is_pair=True)
t_tracks = timeit(lambda: build_inference_tracks(idf, pdf, full_names, is_pair=True))
t_cw = timeit(lambda: ext.compute_windows())

w = ext.compute_windows()
sids = list(w)
batch = np.stack([model.normalize(w[s]) for s in sids], 0)
t_fwd = timeit(lambda: model.forward_window(batch))

print(f"\n--- stage medians (480-frame buffer) ---")
print(f"build_coco_pose         : {t_build:8.1f} ms")
print(f"extract_video_features  : {t_vid:8.1f} ms")
print(f"extract_pose_features   : {t_pose:8.1f} ms")
print(f"_social_features (total): {t_social:8.1f} ms")
print(f"build_inference_tracks  : {t_tracks:8.1f} ms")
print(f"compute_windows (total) : {t_cw:8.1f} ms")
print(f"model.forward (2 tracks): {t_fwd:8.1f} ms")
