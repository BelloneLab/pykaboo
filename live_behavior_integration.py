"""Import shim that exposes the live behavior engine to the pykaboo app.

The behavior detection stack (EmbTCN-Attention temporal model + offline-parity feature
extractor) lives in the ``pykaboo_live_behavior`` sub-package and pulls in torch +
the vendored ``behavior_segmentation`` code. We isolate that here so the main window
can import behavior support behind a single ``BEHAVIOR_AVAILABLE`` flag and degrade
gracefully (the rest of live detection / TTL keeps working) if torch or the model
package is missing on a given machine.

The default checkpoint is the free-interaction model shipped with the package.
"""

from __future__ import annotations

import os

_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pykaboo_live_behavior")
DEFAULT_BEHAVIOR_CHECKPOINT = os.path.join(
    _PKG_DIR, "checkpoints", "free_embtcn_attention_optimized.pt"
)

BEHAVIOR_AVAILABLE = False
BEHAVIOR_IMPORT_ERROR: str | None = None

LiveBehaviorWorker = None  # type: ignore
BehaviorFrameState = None  # type: ignore
read_checkpoint_labels = None  # type: ignore

# Behavior class names for the fast rule-based detector (torch-free; always known).
RULE_BEHAVIOR_LABELS: list[str] = [
    "nose2nose", "sidebyside", "sidereside", "nose2anogenital", "nose2body",
    "oriented_toward", "following", "chasing", "approach",
    "withdrawal_from_partner", "escape", "withdrawal_after_contact", "fighting",
]
# Default ML class names (the shipped free-interaction checkpoint).
ML_BEHAVIOR_LABELS: list[str] = [
    "nose-to-nose", "nose-to-body", "anogenital", "passive", "rearing", "mounting",
]

try:
    import sys

    # On Windows torch must be imported BEFORE camera_backends/PySpin (vendor SDK
    # DLLs otherwise shadow a dependency of torch's shm.dll -> WinError 127). main.py
    # already does this at startup; we repeat it defensively so importing the behavior
    # stack also works in standalone / alternate import orders. Cache-hit no-op once
    # torch is loaded. The behavior feature stack pulls torch via behavior_segmentation.
    try:
        from torch_runtime import import_torch as _import_torch

        _import_torch(required=False)
    except Exception:
        pass

    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    from pykaboo_behavior_worker import (  # noqa: E402
        BehaviorFrameState,
        LiveBehaviorWorker,
        read_checkpoint_labels,
    )

    BEHAVIOR_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on optional torch install
    BEHAVIOR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def default_checkpoint_exists() -> bool:
    return os.path.isfile(DEFAULT_BEHAVIOR_CHECKPOINT)
