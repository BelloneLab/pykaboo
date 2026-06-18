"""Bundled default models shipped with PyKaboo.

The app ships a default RF-DETR segmentation checkpoint and a YOLO pose checkpoint
(plus prebuilt TensorRT engines) under ``pykaboo/models/``. These helpers resolve
those paths so a fresh install points the live-detection panel at working models
without the user having to browse for them.

The ``.engine`` files are hardware- and TensorRT-version specific: they are loaded
only in the "Max GPU (TensorRT)" GPU mode and the app falls back to the portable
``.pt`` / ``.pth`` weights on any other machine (or rebuild them with
``scripts/build_rfdetr_engine.py`` / ``scripts/build_yolo_pose_engine.py``).
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Prefer the portable weights when present (they work on any machine and let the
# engine be rebuilt); otherwise fall back to the shipped .engine. The repo ships
# only the .engine files (the .pth/.pt exceed GitHub's 100 MB file limit), so on a
# fresh clone these resolve to the engines and run via the engine-only path.
DEFAULT_SEG_CHECKPOINT = MODELS_DIR / "checkpoint_best_total.pth"
DEFAULT_SEG_ENGINE = MODELS_DIR / "checkpoint_best_total.engine"
DEFAULT_POSE_CHECKPOINT = MODELS_DIR / "poseModel_largebest.pt"
DEFAULT_POSE_ENGINE = MODELS_DIR / "poseModel_largebest.engine"
DEFAULT_MODEL_KEY = "rfdetr-seg-medium"


def _first_existing(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def default_seg_checkpoint() -> str:
    """Bundled RF-DETR seg path: the .pth if present, else the .engine, else ""."""
    return _first_existing(DEFAULT_SEG_CHECKPOINT, DEFAULT_SEG_ENGINE)


def default_pose_checkpoint() -> str:
    """Bundled YOLO pose path: the .pt if present, else the .engine, else ""."""
    return _first_existing(DEFAULT_POSE_CHECKPOINT, DEFAULT_POSE_ENGINE)
