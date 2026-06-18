"""Adapter: pykaboo live detections -> OnlineBehaviorEngine -> closed-loop TTL.

This is the only file that knows about pykaboo's data shapes. It maps pykaboo's
``LiveDetectionResult`` / ``TrackedMouseState`` onto the engine's per-frame input
contract, handling every offline-parity gotcha found in research/D_pykaboo_api.md:

  * keypoints: pykaboo mask geometry's 8th point is ``tail_base``; the behavior
    engine treats index 7 as the rear/tail anchor. The ORDER is identical, so we
    index by position when keypoint_source is mask_geometry. If you use yolo_pose
    with a different order, pass ``kp_reorder``.
  * bbox: pykaboo is XYXY; the engine wants COCO XYWH. Converted here.
  * identities: pykaboo uses integer ``mouse_id``; the engine wants stable string
    ids "1"/"2". ``IdentityMapper`` assigns them stably across the session.
  * geometry: we pass pykaboo's boolean mask through; the feature code recomputes
    center/area/shape with cv2 exactly as offline (do NOT trust precomputed center).
  * resolution: the model uses frame-size-normalized features, so any consistent
    resolution works AS LONG AS you pass the true per-frame (width, height). No
    rescaling needed. Set pykaboo ``expected_mouse_count = 2``.

Nothing here imports pykaboo (so it loads anywhere); pykaboo objects are duck-typed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from live_features import FrameRecord


class IdentityMapper:
    """Stable mapping from pykaboo mouse_id/label -> "1"/"2" across the session."""

    def __init__(self, prefer_label: bool = True):
        self.prefer_label = prefer_label
        self._map: dict = {}
        self._next = 1

    def _key(self, tracked) -> object:
        label = getattr(tracked, "label", "") or ""
        if self.prefer_label and label:
            # "mouse1"/"mouse2" -> "1"/"2" directly if they end in a digit
            digits = "".join(ch for ch in label if ch.isdigit())
            if digits:
                return f"label:{digits}"
            return f"label:{label}"
        return f"id:{getattr(tracked, 'mouse_id', id(tracked))}"

    def get(self, tracked) -> str:
        # Prefer a direct integer mouse_id of 1/2 so the engine's subject ids match
        # str(mouse_id) deterministically -- this lets the preview overlay map each
        # displayed mouse to its per-track behavior subtitle with no shared state.
        mid = getattr(tracked, "mouse_id", None)
        if isinstance(mid, int) and str(mid) in ("1", "2"):
            return str(mid)
        k = self._key(tracked)
        if k.startswith("label:") and k.split(":", 1)[1] in ("1", "2"):
            return k.split(":", 1)[1]
        if k not in self._map:
            if self._next > 2:
                # more than two mice seen; assign by insertion order, clamp warning
                self._map[k] = str(self._next)
            else:
                self._map[k] = str(self._next)
            self._next += 1
        return self._map[k]


def result_to_framerecord(
    result,
    mapper: IdentityMapper,
    identities: tuple[str, str] = ("1", "2"),
    kp_reorder: list[int] | None = None,
) -> FrameRecord:
    """Convert a pykaboo LiveDetectionResult into a FrameRecord for the engine."""

    width = int(getattr(result, "width", 0))
    height = int(getattr(result, "height", 0))
    mice: dict = {}

    for tracked in getattr(result, "tracked_mice", []) or []:
        ident = mapper.get(tracked)
        if ident not in identities:
            continue  # ignore extra detections beyond the dyad
        bbox = getattr(tracked, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)

        mask = getattr(tracked, "mask", None)
        mask_arr = None if mask is None else np.asarray(mask, dtype=bool)

        kps = getattr(tracked, "keypoints", None)
        kp_scores = getattr(tracked, "keypoint_scores", None)
        if kps is not None:
            kps = np.asarray(kps, dtype=np.float64)
            if kp_reorder is not None:
                kps = kps[kp_reorder]
            if kp_scores is not None:
                kp_scores = np.asarray(kp_scores, dtype=np.float64)
                if kp_reorder is not None:
                    kp_scores = kp_scores[kp_reorder]

        mice[ident] = {
            "present": True,
            "bbox_xywh": bbox_xywh,
            "score": float(getattr(tracked, "confidence", 1.0)),
            "mask": mask_arr,
            "keypoints": kps,
            "keypoint_scores": kp_scores,
            "category_id": getattr(tracked, "class_id", None),
        }

    for ident in identities:
        mice.setdefault(ident, {"present": False, "bbox_xywh": (0.0, 0.0, 0.0, 0.0),
                                "score": 0.0, "mask": None, "keypoints": None,
                                "keypoint_scores": None, "category_id": None})

    return FrameRecord(
        frame_idx=int(getattr(result, "frame_index", 0)),
        timestamp_s=float(getattr(result, "timestamp_s", 0.0)),
        width=width,
        height=height,
        mice=mice,
    )


# --------------------------------------------------------------------------- #
# Example closed-loop wiring (pykaboo Qt signal -> engine -> Arduino TTL)
# --------------------------------------------------------------------------- #

class PykabooClosedLoop:
    """Glue object: connect to LiveInferenceWorker.result_ready, fire TTL on events.

    Example (inside pykaboo, on the GUI thread):

        from pykaboo_adapter import PykabooClosedLoop
        loop = PykabooClosedLoop(
            ckpt_path="checkpoints/free_embtcn_attention_optimized.pt",
            arduino_worker=app.arduino_worker,            # ArduinoOutputWorker
            behavior_to_output={"mounting": "DO1", "fighting": "DO2"},
            pulse_ms=200, lookahead=8, device="cuda",
        )
        worker.result_ready.connect(loop.on_result)       # LiveInferenceWorker

    Remember: set pykaboo expected_mouse_count = 2 and keypoint_source = "mask_geometry".
    """

    def __init__(
        self,
        ckpt_path: str,
        arduino_worker=None,
        behavior_to_output: dict | None = None,
        pulse_ms: int = 200,
        lookahead: int = 8,
        device: str | None = None,
        min_bout_frames: int | None = None,
        smooth_win: int | None = None,
        merge_gap_frames: int | None = None,
        on_frame=None,
        verbose: bool = True,
    ):
        from live_engine import OnlineBehaviorEngine  # local import keeps module light

        self.arduino_worker = arduino_worker
        self.behavior_to_output = behavior_to_output or {}
        self.pulse_ms = int(pulse_ms)
        self.verbose = verbose
        self.mapper = IdentityMapper()
        triggers = list(self.behavior_to_output.keys()) or None
        self.engine = OnlineBehaviorEngine(
            ckpt_path=ckpt_path,
            device=device,
            lookahead=lookahead,
            trigger_behaviors=triggers,
            on_event=self._on_event,
            on_frame=on_frame,
            smooth_win=smooth_win,
            min_bout_frames=min_bout_frames,
            merge_gap_frames=merge_gap_frames,
        )

    def on_result(self, result) -> None:
        """Slot for LiveInferenceWorker.result_ready (one LiveDetectionResult)."""
        frame = result_to_framerecord(result, self.mapper)
        self.engine.on_detection(frame)

    def _on_event(self, ev) -> None:
        if self.verbose:
            print(f"[{ev.frame_idx:>7d} t={ev.timestamp_s:7.2f}s] "
                  f"{ev.behavior} {ev.edge} (p={ev.prob:.2f})", flush=True)
        if ev.edge != "onset":
            return
        out = self.behavior_to_output.get(ev.behavior)
        if out and self.arduino_worker is not None:
            # pykaboo ArduinoOutputWorker.start_live_output_pulse(output_id, duration_ms)
            self.arduino_worker.start_live_output_pulse(out, self.pulse_ms)
