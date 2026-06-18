# D. Pykaboo Live Detection API and Closed-Loop Integration Seam

Research target: `BelloneLab/pykaboo` (https://github.com/BelloneLab/pykaboo), the live
acquisition + detection + TTL-trigger Windows app that will feed our EmbTCN-Attention
behavior model in real time for closed-loop optogenetic triggering.

Status: **repo is PUBLIC and reachable**. It has a real, documented live per-frame
detection pipeline (RF-DETR-Seg via TensorRT + YOLO/mask-geometry pose) and a real
closed-loop ROI/proximity/contact -> Arduino-TTL rule engine. It does **NOT** expose a
public "behavior classifier plugin" hook today, but it has a clean internal seam we can
splice into. All claims below are quoted from the actual source on `main` (raw.github).

> NOTE on method: file *signatures and field names* below are quoted verbatim from the
> raw source files. A few one-line implementation bodies were summarized from the file
> content (called out as "summary"). Always re-confirm against the live source before
> writing the integration shim, since pykaboo is on a fast `ci-windows` release cadence.

---

## 0. The headline parity / latency risks (read this first)

These are the things that threaten online/offline parity or closed-loop latency. Each is
expanded later.

1. **Keypoint NAME mismatch on the 8th point.** Pykaboo's mask-skeleton extractor names
   its 8 points `("nose","left_ear","right_ear","neck","body","left_hip","right_hip",`
   **`"tail_tip"`**`)` (`mask_skeleton.py: KP_ORDER`). Our offline pipeline names the 8th
   point **`"tail"`** (`src/behavior_segmentation/pose.py:24` `KEYPOINT_ORDER`). **Order
   is identical, only the last name differs.** Our pose feature column names are
   `a{0,1}_{kp}_x/_y` built straight from `KEYPOINT_ORDER` (`free_pose.py:49`
   `_channel_names`), and offline pose features key on the string `"tail"` (e.g.
   `pose_tail_speed` in `wavelet_features.py:33`). The online extractor must **rename
   `tail_tip` -> `tail`** before computing features, or every tail-derived feature
   silently becomes zero / NaN.

2. **The model is NON-CAUSAL (bidirectional) and uses a 12 s window.** The default free
   model is `embtcn_attention` with `causal: bool = False`
   (`models/embtcn_attention.py:46`), centered conv padding
   (`pad = (kernel_size - 1) // 2 * dilation`, line 106), a *fully bidirectional*
   Transformer encoder (module docstring lines 11-12), and `window_seconds: 12.0`
   (`configs/default.yaml:75`). The TCN receptive field is large: dilations
   `(1,2,4,8,16,32)`, kernel 5 -> each side needs `((5-1)/2)*sum(dilations) = 2*63 = 126`
   frames, i.e. **~126 future frames (~5 s at 25 fps, ~4.2 s at 30 fps)** before the
   prediction for "now" is stable. For genuine zero-latency closed loop you must either
   (a) run a sliding window and accept a fixed lag of half the window, or (b) retrain a
   `causal=True` variant (the flag exists and is wired through every conv block, lines
   103-122, "A causal flag is kept for a future real-time mode"). **This is the single
   biggest closed-loop design decision.**

3. **Wavelet (Morlet CWT) features use whole-series statistics.**
   `wavelet_features.py:_cwt_power` does `x = x - x.mean()` over the *entire* identity
   time series before `pywt.cwt(x, scales, "morl")` (lines 57-68). The longest scale is a
   4.0 s period (`DEFAULT_PERIODS_SECONDS = [0.12,0.25,0.5,1.0,2.0,4.0]`, line 42).
   Online you can only mean-center over a finite buffer; the CWT itself is also a global
   convolution. Use a rolling buffer of at least the model window (12 s) plus CWT support
   and a running/rolling mean, and accept small edge differences. The 11 source signals
   are listed in `DEFAULT_WAVELET_SIGNALS` (lines 27-39); each yields 6 `_cwt_p{i}`
   columns.

4. **Feature normalization is FIXED and travels in the checkpoint (good news).** The free
   model uses `FeatureNormalizer.fit(...)` (`free_train.py:160` and `:274`) which stores a
   single global per-channel `mean`/`std` z-score (`normalization.py:18-39`). These are
   serialized into the checkpoint, so online inference does **not** need whole-video
   statistics for the standard free path. CAUTION: the class also supports
   `fit_with_video_coral` / `fit_with_video_robust` modes that DO apply per-video test-time
   stats (`normalization.py:42-103`, and `transform()` branches on `self.video_mean`
   / `self.video_median`, lines 103-123). Confirm the *deployed checkpoint* was trained
   with plain `fit` (the free pipeline uses plain `fit`), not a video-adaptive mode, or
   online parity breaks because there is no full video to estimate per-video stats from.

5. **Pykaboo defaults to `expected_mice=1`.** `LiveInferenceConfig.expected_mouse_count
   = 1` and `LiveIdentityTracker(expected_mice=1)` (`live_inference_worker.py`). For our
   two-mouse social model the consumer/UI must set `expected_mouse_count = 2`; otherwise
   only one identity is tracked and `tracked_mice` will not contain a mouse1/mouse2 pair.

6. **Identities are integer `mouse_id`, not strings `mouse1`/`mouse2`.** `TrackedMouseState`
   carries `mouse_id: int` and a free-text `label: str` (`live_detection_types.py`). Our
   offline vocabulary is COCO `track_id` strings `"1"`/`"2"` mapped from `mouse1`/`mouse2`
   (`pose.py:36` `DEFAULT_ANIMAL_TO_IDENTITY = {"mouse1": "1", "mouse2": "2"}`). The shim
   must map `mouse_id` (and/or `label`) to our `"1"`/`"2"` and, crucially, keep the
   subject/partner assignment **stable across frames** (the directed pair features assume a
   consistent subject_id/object_id).

---

## 1. Live per-frame detection output structure (masks + pose + geometry)

Source: `live_detection_types.py`. The per-frame payload is `LiveDetectionResult`,
emitted once per inference cycle:

```python
@dataclass
class LiveDetectionResult:
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    inference_ms: float
    tracked_mice: list[TrackedMouseState] = field(default_factory=list)
    model_key: str = ""
    status: str = ""
    predict_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    queue_wait_ms: float = 0.0
    end_to_end_ms: float = 0.0
    completed_timestamp_s: float = 0.0
    inference_width: int = 0
    inference_height: int = 0
```

Per-identity detection is `TrackedMouseState`:

```python
@dataclass
class TrackedMouseState:
    mouse_id: int
    class_id: int
    confidence: float
    center: tuple[float, float]                 # (cx, cy) in OUTPUT-frame pixels
    bbox: tuple[float, float, float, float]     # XYXY: (x1, y1, x2, y2) pixels
    mask: Optional[np.ndarray] = None           # (H, W) dtype=bool, output-frame size
    label: str = ""
    keypoints: Optional[np.ndarray] = None       # shape (K, 2) -> (x, y) image coords
    keypoint_scores: Optional[np.ndarray] = None # shape (K,) per-joint confidence
```

Key facts for our extractor:
- **bbox is XYXY** `(x1,y1,x2,y2)`, in *output-frame* pixel coords. Our offline COCO is
  **XYWH** `(x, y, w, h)` (`coco_masks.py:38` `bbox: tuple[float,float,float,float]  # x,y,w,h`).
  Convert: `w = x2 - x1`, `h = y2 - y1`.
- **center** is `(cx, cy)`, computed from mask centroid (mean of True pixels) with bbox
  fallback (`_compute_body_center`, summary). Our offline `center_x/center_y` come from
  image-moment centroid `cx = m10/m00`, `cy = m01/m00` (`features.py:195-196`), which is
  the *same definition* as a boolean-mask centroid -> good parity, but verify pykaboo's
  centroid is over the full mask, not a bbox-padded crop edge case.
- **mask** is a dense boolean `np.ndarray (H,W)` already at output-frame resolution (no
  RLE/polygon to decode at the consumer; see section 4).
- **No `area` field.** Pykaboo does not pass mask area in the payload. We must compute
  `area = int(mask.sum())` ourselves, which matches offline `area = moments["m00"]`
  (`features.py:188,209`) only if pykaboo's mask is at the *same resolution* we treat as
  pixel space. Resolution matters: see `inference_width/height` vs `width/height` below.
- **Resolution caveat:** inference runs at a capped size (`inference_max_width: 960`,
  `LiveInferenceConfig`), and the result reports both `width/height` (output frame) and
  `inference_width/inference_height`. Masks/keypoints/bboxes are scaled to the output
  frame inside the worker (`record_to_frame_scale` in `_attach_pose_keypoints_in_bboxes`).
  All offline geometry features are in *pixel* units (center_x, area, speed in px/s, etc.).
  **The online resolution must match the resolution the model was trained against, or all
  scale-dependent features (area, speed, distances, body_length) drift.** Offline features
  were computed at the COCO `width`/`height` (full-res RF-DETR export). The safest fix:
  compute features in a *resolution-normalized* space or rescale pykaboo masks to the
  training resolution before feature extraction. Confirm what resolution the deployed
  checkpoint's features were computed at and replicate it exactly.

---

## 2. The closed-loop integration seam (where to receive frames + emit a trigger)

There are two real seams. Use BOTH: subscribe to detections at (A), and emit a trigger at
(C). (B) is the existing rule engine you can mimic or bypass.

### (A) Receive each frame's detections -- Qt signal on the worker

`live_inference_worker.py`:

```python
class LiveInferenceWorker(QThread):
    """Run live segmentation on the newest preview frame only."""
    result_ready = Signal(object)    # emits a LiveDetectionResult
    status_changed = Signal(str)
    error_occurred = Signal(str)
```

The integration is: `worker.result_ready.connect(our_slot)`, where `our_slot(result:
LiveDetectionResult)` runs our online feature extractor + streaming EmbTCN inference. This
is the cleanest seam: it gives the full per-frame `tracked_mice` (mask + pose + geometry +
identity) with timing metadata, on the GUI/event thread.

> Important: the worker "runs on the newest preview frame only" -- it **drops frames**
> under load. For our streaming model that assumes a contiguous, evenly-sampled time
> series, dropped frames are a parity hazard. Use `result.frame_index` and
> `result.timestamp_s` to detect gaps and either hold-last / interpolate, or feed the
> model real `dt` so kinematics (speed = dx/dt) stay correct. Offline kinematics use the
> COCO-derived fps (`build_social_features` overrides config fps with `video.fps`,
> `social_pipeline.py:257-263`), so online must use measured per-frame `dt`, not a fixed
> nominal fps.

### (B) Existing rule -> trigger engine (the closed-loop pathway to mimic)

`live_detection_logic.py`:

```python
class LiveRuleEngine:
    """Evaluate ROI, proximity, and mask-contact rules against tracked live detections."""

    def __init__(self) -> None: ...

    def evaluate(self, result: Optional[LiveDetectionResult],
                 now_ms: int) -> LiveRuleEvaluation: ...

    def _rule_truth(self, rule: LiveTriggerRule,
                    mouse_lookup: dict[int, TrackedMouseState]) -> Optional[bool]: ...

    def _trigger_rule_pulse(self, rule: LiveTriggerRule, output_id: str,
                            now_ms: int) -> tuple[str, int, int, float]: ...

# module-level helper
def occupied_roi_names(rois: dict[str, BehaviorROI],
                       result: Optional[LiveDetectionResult]) -> set[str]: ...
```

`evaluate()` returns a `LiveRuleEvaluation` with fields `active_rule_ids`,
`triggered_pulses`, `output_states`, `level_output_states`. Built-in rule types
(`_rule_truth`): `"roi_occupancy"`, `"mouse_proximity"`, `"mask_contact"`.

**Our behavior model is a 4th rule kind that does not exist yet.** Two ways to add it:
- *Subclass / monkeypatch* `LiveRuleEngine._rule_truth` to add a `rule_type ==
  "behavior_class"` case that reads a per-frame behavior probability our model already
  computed (set as an attribute on the `result` or in a side-channel keyed by
  `frame_index`), then reuse the existing `_trigger_rule_pulse` path. This keeps all the
  debounce/pulse-train/arbiter machinery.
- *Bypass*: in our `result_ready` slot, run the model and call the Arduino worker directly
  (section C). Simpler, but we lose the entry/exit/continuous debouncing and the
  multi-output arbiter.

### (C) Emit the TTL/trigger -- Arduino worker

`arduino_output.py`:

```python
class ArduinoOutputWorker(QThread):
    def connect_to_port(self, port_name: str) -> bool: ...
    def start_live_output_pulse(self, output_id: str, duration_ms: int): ...
    def start_live_output_pulse_train(self, output_id: str, duration_ms: int,
                                      pulse_count: int = 1,
                                      pulse_frequency_hz: float = 1.0): ...
    # internal: _write_output_signal_locked(signal_key, value),
    #           _write_live_output_locked(output_key, value, force=False)
```

- Transport: `pyfirmata` (StandardFirmata / custom `StandardFirmataBarcode.ino`),
  `pyserial` for port discovery, **57600 baud**.
- Latency budget (from `arduino_output.py`): Firmata sampling interval 2 ms, main refresh
  loop 5 ms sleep, live-pulse scheduler daemon thread 0.5-4 ms precision, GUI state
  emission throttled to 200 ms. So the *hardware* path adds only a few ms; the dominant
  latency is the bidirectional model window (risk #2), not the TTL.
- Output IDs normalize to `"DO1"`, `"DO2"`, ... (digital outputs). The
  `StandardFirmataBarcode.ino` offers Timer1-ISR microsecond barcode pulses for sync.

**Recommended integration seam (concrete):** connect `LiveInferenceWorker.result_ready`
-> our `OnlineBehaviorEngine.on_detection(result)`. Inside, maintain rolling per-identity
feature buffers, run the streaming EmbTCN, and when a target behavior crosses threshold,
call `arduino_worker.start_live_output_pulse("DO1", duration_ms)` (or feed a synthetic
`rule_truth` into `LiveRuleEngine`). This mirrors pykaboo's own dataflow exactly.

---

## 3. Pose skeleton: names + order (parity check)

Pykaboo mask-geometry skeleton (`mask_skeleton.py`):

```python
KP_ORDER: tuple[str, ...] = (
    "nose", "left_ear", "right_ear", "neck",
    "body", "left_hip", "right_hip", "tail_tip",
)
```

Class `MaskSkeletonExtractor` with:

```python
def estimate(self, mask: np.ndarray, *, track_id: object = "default",
             offset: tuple[float, float] | np.ndarray = (0.0, 0.0)) -> Optional[Skeleton]
```

returning a `Skeleton` with `keypoints (8,2)`, `scores (8,)`, `orientation_confidence:
float`. Geometry: PCA principal axis of the tail-removed body core; nose/tail = extrema
along axis; ears at ~18% and hips at ~74% of length as perpendicular cross-section edges;
neck = section center at ~30%; body = mask centroid; tail_tip via morphological opening to
isolate the thin tail filament then farthest-from-base point. This is essentially the same
method our offline `experiments/mask_keypoints_geom.py` uses (PCA on tail-removed core,
centroid for body), per project memory `finding-mask-geometry-keypoints` -- so the
*algorithm* is aligned; the *names* are the only divergence.

Our offline order (`src/behavior_segmentation/pose.py:24`):

```python
KEYPOINT_ORDER = ["nose","left_ear","right_ear","neck","body","left_hip","right_hip","tail"]
```

**Verdict: ORDER MATCHES exactly (1-to-1). Only the 8th NAME differs (`tail_tip` vs
`tail`).** This is a low-effort but easy-to-miss parity fix. Because our pose feature code
builds column names from the strings (`a0_tail_x`, `pose_tail_speed`,
`(body, tail)` skeleton edge in `free_pose.py:40`, etc.), the shim MUST rename `tail_tip`
-> `tail` (or simply index by position 7, which is safest). Note pykaboo also offers a
`yolo_pose` source (`keypoint_source: "yolo_pose"` in `LiveInferenceConfig`); if YOLO pose
is used instead of mask geometry, **verify the YOLO model's keypoint count and order** --
it may not be these same 8 in this order. Prefer `keypoint_source = "mask_geometry"` for
guaranteed name/order parity with this skeleton, OR pin the YOLO pose checkpoint to one
trained with our 8-keypoint order.

---

## 4. Mask representation and deriving bbox / area / center

Pykaboo internal normalization (`live_inference_worker.py`):

```python
def _normalize_detections(self, detections) -> dict[str, np.ndarray]:
    # {"xyxy": (N,4), "confidence": (N,), "class_id": (N,), "mask": (N,H,W) bool}
```

- The **delivered** mask in `TrackedMouseState.mask` is a **dense boolean `np.ndarray
  (H,W)`** at output-frame resolution. No RLE/polygon decode needed at the consumer.
- RF-DETR-Seg gives masks directly; Ultralytics/YOLO-Seg polygons are rasterized via
  `_masks_from_ultralytics_polygons` -> `cv2.fillPoly` to `(H,W)` bool (mirrors our
  offline `decode_polygon` which also uses `cv2.fillPoly`, `coco_masks.py:183-197`).
- bbox: pykaboo XYXY -> our XYWH: `x=x1, y=y1, w=x2-x1, h=y2-y1`.
- area: pykaboo does **not** provide it; compute `area = float(mask.sum())`. Offline area
  = `cv2.moments(mask_u8, binaryImage=True)["m00"]` (`features.py:187-188,209`), which for
  a boolean mask equals `mask.sum()`. Parity holds **only at matching resolution**.
- center: pykaboo gives `center=(cx,cy)` from mask centroid; offline uses moment centroid
  `(m10/m00, m01/m00)` (`features.py:195-196`). Same definition. To be safe and fully
  parity-exact, **recompute center yourself with `cv2.moments` on pykaboo's boolean mask**
  rather than trusting `TrackedMouseState.center` (eliminates any bbox-fallback / padded-
  crop discrepancy in `_compute_body_center`).

The other offline geometric features (`perimeter`, `convex_area`, `solidity`,
`eccentricity`, `major/minor_axis_length`, `orientation`, `equivalent_diameter`, `extent`,
`aspect_ratio`) are all derived from the SAME boolean mask via `cv2.moments`,
`cv2.findContours`, `cv2.convexHull` (`features.py:164-228`). Our online extractor should
re-run that exact `mask_shape_features` code on pykaboo's `mask` to guarantee parity --
**reuse the function, do not reimplement it.**

Recommended online per-mouse record to feed our existing offline feature code (mirrors
`MaskRecord`, `coco_masks.py:32-48`):

```python
record = MaskRecord(
    frame_idx=result.frame_index,
    identity="1" or "2",                  # mapped from TrackedMouseState.mouse_id/label
    bbox=(x1, y1, x2-x1, y2-y1),          # XYWH
    score=float(tracked.confidence),
    height=result.height, width=result.width,
    segmentation=None,                    # we pass the decoded mask directly
    category_id=tracked.class_id,
)
mask_bool = np.asarray(tracked.mask, dtype=bool)   # (H,W)
```

Then call `mask_shape_features(mask_bool, record, result.width, result.height)` from
`features.py` unchanged. This is the cleanest way to guarantee mask-geometry parity.

---

## 5. Runtime / environment / GPU assumptions

From `requirements.txt` and `README.md`:
- OS: **Windows 10/11** (pykaboo is a Windows desktop app; CI is `ci-windows.yml`;
  `run_pykaboo.bat`, `PySpin` win_amd64 wheel, PowerShell build scripts). Our package will
  likely run **on the same Windows host** that runs pykaboo for closed loop, OR pykaboo
  sends detections to us over IPC. Plan for Windows.
- Python: **3.10** ("Python 3.10 recommended"; PySpin wheel is `cp310`).
- Torch: `torch>=2.2`, `torchvision>=0.17`. NVIDIA GPU; "install PyTorch with CUDA 12.6
  support first". Our EmbTCN must load with a compatible torch.
- Detection backends: `rfdetr>=1.6.0`, `ultralytics>=8.3.0`, and **TensorRT**
  (`tensorrt-cu12==10.13.3.9.post1`, `onnx>=1.16`, `onnxslim`). The shipped detection +
  pose models are `.engine` TensorRT files (`models/checkpoint_best_total.engine`,
  `models/poseModel_largebest.engine`) -- **TensorRT engines must be rebuilt on target
  hardware** (README + `scripts/build_rfdetr_engine.py`, `build_yolo_pose_engine.py`).
- numpy pinned `>=1.24,<2` (numpy 1.x). opencv `>=4.8,<4.12`, scipy `>=1.10`, pandas
  `>=2.0`. GUI: `PySide6>=6.6`, `pyqtgraph>=0.13`. Hardware: `pyserial>=3.5`,
  `pyfirmata>=1.1`, `pypylon` (Basler), `flirpy`/PySpin (FLIR).
- **numpy 1.x constraint is a real integration risk:** our project deliberately requires
  the *system* `python3` with its own numpy/pandas (CLAUDE.md: anaconda raises ABI
  errors). If our online package runs *in pykaboo's env*, it must be numpy<2 compatible.
  Verify our EmbTCN inference path + feature code work under numpy 1.24-1.x and pandas 2.x.
- **Existing downstream/online plugin to mirror:** there is no generic "analysis plugin"
  API, but `LiveRuleEngine` IS the reference pattern for "consume per-frame detections ->
  decide -> trigger". The dev driver `scripts/dev_drive_tracking.py` /
  `scripts/dev_drive_app.py` and `tests/test_live_detection_logic.py`,
  `tests/test_live_inference_worker.py` show how to instantiate the worker/engine
  headless and feed synthetic `LiveDetectionResult`s -- use those tests as the template
  for our integration harness.

---

## 6. Does pykaboo have a documented behavior-classifier plugin API? -- NO. Define our INPUT CONTRACT.

Pykaboo has a live detection API (`LiveInferenceWorker.result_ready`) and a closed-loop
rule->TTL engine (`LiveRuleEngine` + `ArduinoOutputWorker`), but it has **no documented
extension point for a custom temporal behavior classifier**. The supported rule types are
geometric only (`roi_occupancy`, `mouse_proximity`, `mask_contact`). Therefore **our
package should define and own a clean per-frame INPUT CONTRACT**, and we adapt to pykaboo
by connecting to `result_ready` (a 1-line `connect`) and mapping `TrackedMouseState` ->
that contract. Pykaboo does not need to change; we adapt to it.

### Proposed minimal per-frame input contract (mirrors offline COCO+DLC, 2 mice)

A streaming consumer receives one of these dicts per frame, in frame order, with stable
identities. It is intentionally a superset of what `TrackedMouseState` already provides so
the pykaboo adapter is trivial.

```python
PerFrameDetections = {
    "frame_idx": int,                 # monotonic; gaps allowed but reported
    "timestamp_s": float,             # wall-clock for real dt (kinematics)
    "width": int, "height": int,      # frame resolution masks/coords live in
    "fps_hint": float,                # nominal fps; real dt taken from timestamps
    "mice": {
        "1": MousePayload,            # subject_id "1" == pykaboo mouse1
        "2": MousePayload,            # subject_id "2" == pykaboo mouse2
    },
}

MousePayload = {
    "present": bool,                  # False -> emit missing_mask_flag=1, hold/interp
    "bbox_xywh": (x: float, y: float, w: float, h: float),   # COCO XYWH (px)
    "score": float,                  # detection confidence
    "mask": np.ndarray | None,       # (H,W) bool at (width,height); None if absent
    # OR, if mask transport is expensive over IPC, send a precomputed geometry block:
    "geometry": {                    # optional: pre-run mask_shape_features() output
        "center_x": float, "center_y": float, "area": float,
        "perimeter": float, "convex_area": float, "solidity": float,
        "eccentricity": float, "major_axis_length": float,
        "minor_axis_length": float, "orientation": float,
        "equivalent_diameter": float, "extent": float, "aspect_ratio": float,
        "missing_mask_flag": float,
    } | None,
    "keypoints": np.ndarray,         # (8,2) float, x,y in px, ORDER EXACTLY:
                                     # [nose, left_ear, right_ear, neck,
                                     #  body, left_hip, right_hip, tail]
    "keypoint_scores": np.ndarray,   # (8,) float; <0.3 -> treated as NaN offline
}
```

Hard requirements baked into this contract (each enforces an offline-parity rule):
- **Keypoints are exactly 8 in our order, with the 8th called `tail` (not `tail_tip`).**
  The adapter renames `tail_tip` -> `tail` / indexes by position 7.
- **bbox is XYWH** (COCO), not XYXY. Adapter converts from pykaboo XYXY.
- **Identities are the string keys `"1"`/`"2"`**, matching
  `DEFAULT_ANIMAL_TO_IDENTITY` and stable across frames. Adapter maps `mouse_id`/`label`,
  set pykaboo `expected_mouse_count = 2`.
- **Coordinates/masks at the SAME resolution the deployed checkpoint's features were
  computed at** (rescale pykaboo output if needed) -- otherwise area/speed/distance drift.
- **Keypoint likelihood threshold 0.3** to match offline `load_pose_csv(...,
  min_likelihood=0.3)` (`social_pipeline.py:271`); points below -> NaN.
- Either ship the boolean `mask` (preferred -- lets us run the *exact* offline
  `mask_shape_features`) or a precomputed `geometry` block produced by that same function.

### Streaming engine obligations (on top of the contract)

To preserve offline/online parity our streaming engine must:
1. Maintain a per-identity rolling buffer >= model window (12 s) + CWT support; compute
   wavelet channels on the buffer with a rolling/running mean (approximates offline
   `x - x.mean()`).
2. Use **real per-frame dt** from `timestamp_s` for all velocity/acceleration/jerk
   features (offline uses video.fps from COCO timestamps, `social_pipeline.py:257-263`).
3. Apply the checkpoint's stored `FeatureNormalizer` (global mean/std) -- **confirm the
   deployed checkpoint used plain `fit`, not `fit_with_video_coral/robust`**.
4. Assemble columns in the EXACT trained order via `build_feature_column_list` /
   `build_inference_tracks` / `build_free_inference_tracks` (subject + partner + pair,
   namespaced `features:` / `pose:`), then select the checkpoint's columns by name
   (`free_infer.py:31-101`). Missing columns -> zeros, same as offline inference.
5. Decide on the latency model for the **non-causal** net: either accept a fixed lag of
   ~half the receptive field (~5 s) by predicting frame t only after t+126 frames arrive,
   or train/deploy a `causal=True` EmbTCN (flag already plumbed,
   `models/embtcn_attention.py:46,103-122`) to make the closed loop near-real-time.

---

## Quick file map (pykaboo, `main`)

| File | Role for us |
|---|---|
| `live_detection_types.py` | `LiveDetectionResult`, `TrackedMouseState`, ROI/rule dataclasses |
| `live_inference_worker.py` | `LiveInferenceWorker(QThread)` -> `result_ready` signal (the input seam) |
| `live_detection_logic.py` | `LiveRuleEngine.evaluate()` -> rule->pulse engine (closed-loop core) |
| `arduino_output.py` | `ArduinoOutputWorker` -> `start_live_output_pulse[_train]` (TTL out) |
| `mask_skeleton.py` | `MaskSkeletonExtractor`, `KP_ORDER` (8 kp; `tail_tip` naming) |
| `live_tracking.py` | `LiveIdentityTracker` (identity persistence across frames) |
| `requirements.txt` / `README.md` | env: Win, py3.10, torch>=2.2, TensorRT, numpy<2 |
| `tests/test_live_inference_worker.py`, `tests/test_live_detection_logic.py` | headless harness templates |
| `StandardFirmataBarcode/StandardFirmataBarcode.ino` | Arduino firmware (Timer1 ISR sync) |

## Quick file map (ours, parity anchors)

| File:line | Role |
|---|---|
| `src/behavior_segmentation/pose.py:24` | `KEYPOINT_ORDER` (8th = `tail`) |
| `src/behavior_segmentation/pose.py:36` | `DEFAULT_ANIMAL_TO_IDENTITY` mouse1/2 -> "1"/"2" |
| `src/behavior_segmentation/free_pose.py:49` | `_channel_names()` -> `a{i}_{kp}_x/_y` |
| `src/behavior_segmentation/coco_masks.py:32-48` | `MaskRecord` (bbox XYWH), `decode_segmentation` |
| `src/behavior_segmentation/features.py:164-228` | `mask_shape_features` (centroid/area/geometry) -- REUSE |
| `src/behavior_segmentation/social_pipeline.py:240-348` | `build_social_features` (the offline entry to replicate) |
| `src/behavior_segmentation/wavelet_features.py:27-69` | CWT signals, scales, whole-series mean-center |
| `src/behavior_segmentation/normalization.py:18-123` | `FeatureNormalizer` (global vs per-video stats) |
| `src/behavior_segmentation/free_infer.py:31-101` | `build_free_inference_tracks` (column selection/order) |
| `src/behavior_segmentation/models/embtcn_attention.py:46,103-122` | `causal` flag, RF, bidirectional |
| `configs/default.yaml:75-76` | `window_seconds: 12.0`, `stride_seconds: 3.0` |
