# Bundled default models

The Live Detection panel defaults to these files (resolved by
[`default_models.py`](../default_models.py)).

| File | What it is | Committed? |
|---|---|---|
| `checkpoint_best_total.engine` | RF-DETR seg medium, TensorRT FP16 (432 px) | yes |
| `checkpoint_best_total.engine.json` | engine build provenance | yes |
| `poseModel_largebest.engine` | YOLO pose, TensorRT FP16 | yes |
| `checkpoint_best_total.pth` | RF-DETR seg weights (portable) | no — [Release asset](https://github.com/BelloneLab/pykaboo/releases/tag/models-v1) |
| `poseModel_largebest.pt` | YOLO pose weights (portable) | no — [Release asset](https://github.com/BelloneLab/pykaboo/releases/tag/models-v1) |

The `.pth`/`.pt` weights exceed GitHub's 100 MB file limit, so they are published as
**Release assets** under [`models-v1`](https://github.com/BelloneLab/pykaboo/releases/tag/models-v1).
Fetch them into this folder with:

```
python scripts/download_models.py
```

## Important: engines are machine-specific

A `.engine` only loads on the **same GPU + TensorRT version** it was built on
(here: RTX 3060, TensorRT 10.13, CUDA 12.6). On a matching machine the app runs
the engines directly via the engine-only path (no `.pth`/`.pt` needed).

On any **other** machine the engines will fail to load. To use the models there:

1. Download the portable weights into this folder:
   ```
   python scripts/download_models.py
   ```
   (or grab them manually from the
   [models-v1 release](https://github.com/BelloneLab/pykaboo/releases/tag/models-v1)).
2. Rebuild the engines for that machine:
   ```
   python scripts/build_rfdetr_engine.py --checkpoint pykaboo/models/checkpoint_best_total.pth
   python scripts/build_yolo_pose_engine.py --pose pykaboo/models/poseModel_largebest.pt
   ```

With the `.pth`/`.pt` present, the app also runs on CPU / non-TensorRT GPUs (it
falls back to the torch path automatically).
