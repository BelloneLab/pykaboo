# A. Feature Ingest, In-Memory Entry Point, Dependency Footprint

Research target: `behavior-segmentation` repo at
`/home/andry/tracking_project/unsupervised_mask_behavior` (system `python3`, `PYTHONPATH=src`).
Goal: drive the free-interaction EmbTCN-Attention model in real time from pykaboo's
per-frame masks + pose, with online features matching the offline batch pipeline.

All file:line references are to `src/behavior_segmentation/`.

---

## 0. TL;DR / headline findings (read this first)

1. **The deployed free model is NOT lean and NOT pose-namespaced.** The shipped checkpoint
   `outputs/free_social/free_embtcn_attention.pt` stores **432 plain feature names**
   (verified: `len(feature_names)==432`, no `features:`/`pose:` namespacing). Those 432 are
   exactly the **full social-feature surface (2688 cols) minus every rolling `_w` column**
   (2688 − 2256 = 432). So the live engine must reproduce the *mask-core + nearest-animal +
   pair + pose + wavelet* channels but can **skip all centered rolling statistics**. Confirmed
   programmatically: the 432 are a strict, order-preserving subset of the full 2688.

2. **`build_social_features` is path-only at its public signature** (it calls
   `load_coco_videos(coco_path, ...)` and `load_pose_csv(pose_path, ...)` internally), but
   **clean in-memory seams exist one layer down**: `extract_video_features(video, config)` takes
   an already-built `CocoVideo`, and `extract_pose_feature_tables(pose, W, H, fps, egocentric)`
   takes an already-built `PoseData`. You can construct `CocoVideo`/`PoseData` directly from
   in-memory numpy/dicts with **no temp files**. This is the recommended seam.

3. **Latency is the dominant risk, not parity.** Mask feature extraction measured at
   **~8.4 ms/frame** (75.9 s for an 8999-frame video, single thread, CPU), dominated by per-frame
   RLE mask decode + `cv2.moments`/`cv2.findContours`/`cv2.convexHull`. Pose features are
   ~0.02 ms/frame (negligible). For 30 fps closed loop you have a 33 ms budget; the mask geometry
   stage alone uses a quarter of it, before motion derivatives, wavelets, normalization and the
   torch forward pass. The offline code is **whole-video vectorized**; an online port must redo
   this incrementally per frame.

4. **Three offline computations are non-causal and are real online/offline parity threats:**
   (a) **wavelet CWT** (Morlet, scales up to 97.5, longest period 4 s → effectively needs ±4 s of
   future/past, applied over the whole series); (b) **`area_norm = area / global_median(area)`**
   (`normalize_geometry`, features.py:368) divides by the *whole-video* median area — undefined
   online; (c) `np.unwrap` of orientation in `compute_motion_features` (features.py:485) is a
   running, history-dependent transform. All three are in the 432 deployed channels.

5. **Minimal vendor set is 13 internal modules + 6 pip deps** (`numpy pandas cv2 pycocotools
   pywt pydantic+yaml`; pydantic/yaml only via `config.py`, droppable). `torch` enters only via
   `dataset.py` (used by the windowed Dataset, not by feature math) — feature extraction itself is
   torch-free.

6. **Two tracks per frame, confirmed.** A 2-mouse dyad yields exactly TWO directed `TrackData`
   (subject=1/partner=2 and subject=2/partner=1). Each track = `[subject 77+ feats] + [obj_
   partner feats] + [pair_ geometry]` (+ pose + wavelets). The free model is **scene-symmetric**:
   both tracks carry the same labels; at inference you run both and take per-class **max prob**
   across the two tracks for the scene decision.

---

## 1. `build_social_features` — signature, return object, schemas

### Signature (`social_pipeline.py:240-248`)
```python
def build_social_features(
    coco_path: str | Path,
    config: AppConfig,
    pose_path: str | Path | None = None,
    use_pose: bool = True,
    use_wavelets: bool = True,
    wavelet_signals: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> SocialFeatures:
```

### Return object `SocialFeatures` (`social_pipeline.py:228-237`)
```python
@dataclass
class SocialFeatures:
    video_id: str
    identity_df: pd.DataFrame            # per-identity (per-frame) feature table
    pair_df: pd.DataFrame                # ordered-pair (directed) feature table
    frame_indices: list[int]             # sorted COCO frame indices = the time grid
    identities: list[str]                # e.g. ["1", "2"]
    frame_rate: float = 25.0             # detected from COCO timestamps (overrides config)
    frame_times: list[float] = []        # per-frame wall-clock timestamps
    pose_quality: dict[str, Any] = {}    # currently unused/empty here
```

### What it does, step by step (`social_pipeline.py:255-348`)
1. `videos = load_coco_videos(coco_path, config.data)`; picks the video with most frames (`:255-256`).
2. **Detects fps from COCO timestamps** (`video.fps`, median of `1/dt`) and *overrides*
   `config.data.frame_rate` if it differs by >0.2, deep-copying config (`:259-264`). **Parity note:**
   features are computed at the *detected* fps, not the config default — your online engine must
   feed the true camera fps so kinematics match.
3. `identity_df, pair_df = extract_video_features(video, config)` (`:265`) — the mask-feature core.
4. If `use_pose`: finds/loads pose CSV, computes `extract_pose_feature_tables(...)`, merges onto
   `identity_df` (single-animal `pose_*`) and `pair_df` (directed `pp_*`) on
   `["video_id","frame_idx","subject_id","object_id"]` (`:267-300`). If
   `config.features.contact_geometry` is True (default **False**) it also merges
   `free_mask_contact.contact_pair_dataframe` (the `mc_*` columns). Default path does NOT touch
   `free_mask_contact`.
5. If `use_wavelets`: per `subject_id`, attaches a few pair signals, runs
   `add_wavelet_features` (Morlet CWT) producing `<signal>_cwt_p{i}` columns, drops temp pair
   signals (`:304-334`).
6. `replace([inf,-inf], nan).fillna(0.0)` on both tables (`:336-338`).

### Exact column schema

The 4 meta columns are always `["video_id","frame_idx","subject_id","object_id"]`
(`META_COLUMNS`, social_pipeline.py:37; dataset.py:30). For per-identity rows `object_id == ""`.

**`identity_df` numeric columns (subject mask-core, in canonical order)** — these are the 77 names
the deployed model actually uses per side (verified against the checkpoint):
```
GEOMETRIC (features.py:31-50):
 center_x, center_y, bbox_x, bbox_y, bbox_width, bbox_height, area, perimeter,
 convex_area, solidity, extent, eccentricity, major_axis_length, minor_axis_length,
 orientation, aspect_ratio, equivalent_diameter, mask_score
 (+ missing_mask_flag, area_norm)        # area_norm only if normalize_by_body_size
MOTION (features.py:52-81):
 velocity_x, velocity_y, speed, speed_body_norm, forward_velocity, lateral_velocity,
 movement_bearing_sin, movement_bearing_cos, heading_alignment, acceleration,
 acceleration_x, acceleration_y, acceleration_magnitude, acceleration_body_norm,
 jerk, jerk_x, jerk_y, jerk_magnitude, jerk_body_norm, angular_velocity,
 angular_acceleration, area_velocity, area_acceleration, orientation_sin,
 orientation_cos, delta_center_x, delta_center_y, mask_iou_with_previous_frame
INTERANIMAL / nearest-other-animal (features.py:83-113):
 nearest_neighbor_count, nearest_centroid_distance, nearest_centroid_dx/dy,
 nearest_centroid_abs_dx/dy, nearest_centroid_distance_velocity/acceleration,
 nearest_bbox_edge_distance_velocity/acceleration, nearest_subject_speed,
 nearest_neighbor_speed, nearest_relative_velocity_x/y, nearest_relative_speed,
 nearest_relative_acceleration_x/y, nearest_relative_acceleration_magnitude,
 nearest_radial_velocity, nearest_radial_acceleration, nearest_lateral_speed,
 nearest_approach_speed, nearest_bbox_edge_distance, nearest_bbox_iou,
 nearest_bbox_contact_flag, nearest_area_ratio, nearest_bearing_sin/cos,
 nearest_heading_alignment
```
Then (offline default) every numeric column above except `missing_mask_flag` and the position
columns gets **6 rolling summaries × 4 windows** (`_{mean,std,min,max,median,slope}_w{0..3}`,
features.py:158, add_rolling_features:409). **The deployed model drops ALL of these.**
Plus pose single-animal columns `pose_*` (merged from `extract_pose_feature_tables`) and
their wavelet `*_cwt_p{i}`.

**`pair_df` numeric columns (directed subject→object)** — `PAIRWISE_DISTANCE_FEATURES`
(features.py:115-156), 40 cols used by the model:
```
pair_centroid_distance, pair_centroid_dx/dy, pair_centroid_abs_dx/dy,
pair_centroid_distance_body_norm, pair_relative_speed, pair_relative_acceleration,
pair_approaching_score, pair_following_score, pair_relative_orientation,
pair_bbox_edge_distance, pair_bbox_edge_distance_body_norm, pair_bbox_iou,
pair_contact_flag, pair_bbox_contact_flag, pair_area_ratio, pair_area_abs_difference,
pair_subject_speed, pair_object_speed, pair_speed_ratio, pair_subject_acceleration,
pair_object_acceleration, pair_relative_velocity_x/y, pair_relative_motion_speed,
pair_relative_motion_speed_body_norm, pair_radial_velocity,
pair_radial_velocity_body_norm, pair_lateral_velocity, pair_lateral_speed,
pair_lateral_speed_body_norm, pair_distance_acceleration, pair_approach_acceleration,
pair_relative_acceleration_x/y, pair_relative_acceleration_magnitude,
pair_radial_acceleration, pair_bbox_edge_distance_velocity/acceleration
```
Plus directed pose-pair `pp_*` (pose_features.py:375-394), e.g. `pp_nose_nose`, `pp_nose_tail`,
`pp_nose_body`, `pp_body_body`, `pp_tail_nose`, `pp_min_internose`, `pp_facing_cos/sin`,
`pp_approach_speed`, `pp_rel_heading_cos/sin`, `pp_contact_flag`, `pp_body_body_vel` (egocentric
variant adds `pp_obj_<kp>_ego_x/y`, `pp_relative_velocity_*`, `pp_relative_speed`).

**To obtain the schema programmatically:** `feature_columns(df)` (features.py:979-987) returns the
ordered numeric non-meta columns. The canonical *concatenated* order for a pair track is
`build_feature_column_list(identity_df, pair_df, is_pair=True)` (dataset.py:50-59):
`identity_cols + ["obj_"+c for c in identity_cols] + pair_cols`.

### Verified counts (one real video, `760_adu`, 8999 frames, 1080×1080, 30.000 fps)
- `identity_df.shape == (17998, 1312)` (2 identities × 8999 frames; 1312 incl meta + rolling +
  pose + wavelets).
- `pair_df.shape == (17998, 76)`.
- `build_feature_column_list(..., is_pair=True)` length = **2688**.
- Deployed checkpoint `feature_names` length = **432**, all ⊂ the 2688, order-preserving.
- Deployed 432 family breakdown: mask_core 77 (subj) + 77 (obj_); pair_ 40; pp_ 32;
  pose_ 37 (subj) + 37 (obj_); wavelet 66 (subj) + 66 (obj_). Zero rolling `_w` columns.

---

## 2. What it reads from COCO JSON and from the DLC pose CSV

### COCO masks (`coco_masks.py`)
Loader: `load_coco_videos(path, data_config)` (`:230-339`), reads via `read_json` (plain
`json.load`, storage.py:72). Requires top-level `"images"` (else `CocoParseError`).

Per **image** (`:251-267`): `id`, `file_name` (used both for sort order and frame-index regex),
frame index via `resolve_frame_index` (tries `frame_idx`/`frame_index`/`frame`, then regex
`data_config.frame_index_regex` default `r"frame(\d+)"`, else enumeration order), `video_id` via
`resolve_video_id` (field `data_config.video_id_field` default `"video_id"`, else default = file
stem), `width`, `height`, and optional `timestamp` (float; used for fps detection).

Per **annotation** → `MaskRecord` (`:272-292`, dataclass at `:32-48`):
- `image_id` → maps to the image's `frame_idx`, `width`, `height`.
- **identity** via `resolve_identity` (`:120-136`) using `data_config.identity_field_priority`,
  default order `["track_id","identity","attributes.identity","attributes.track_id","category_id"]`.
  Supports dotted keys via `get_nested`. Falls back to `id_<ann_id>` if none. **In the real data,
  `track_id` is `1`/`2` (int) and `identity` is `"mouse1"`/`"mouse2"`; `track_id` wins, so COCO
  identities are `"1"`/`"2"`.**
- `bbox = (x, y, w, h)` from `annotation["bbox"]` (default `[0,0,0,0]`).
- `score = float(annotation.get("score", 1.0))` (used to pick best record per frame/identity, and
  gates nearest-animal features).
- `segmentation = annotation.get("segmentation")` (kept raw, decoded lazily).
- `category_id = annotation.get("category_id")`.
- **NOT read:** `annotation["area"]` and `annotation["center"]` are ignored — area is recomputed
  from the decoded mask via `cv2.moments` (`m00`), and center from moments (`m10/m00`, `m01/m00`).
  **Parity note:** if pykaboo already has area/center, the offline pipeline does NOT use them; it
  recomputes from the raster, so feed the same raster or replicate the moment math.

**Segmentation format in the real dataset: compressed COCO RLE** — `segmentation` is a dict
`{"size":[H,W], "counts": "<ascii string>"}`. Decoded by `decode_segmentation` →
`decode_rle` (`:200-213`) which calls `pycocotools.mask.decode` (string counts are
`.encode("ascii")`; list counts go through `frPyObjects`). Polygon lists are handled by
`decode_polygon` via `cv2.fillPoly` (`:183-197`). **`pycocotools` is therefore a hard runtime dep
for this dataset.** The boolean mask is what every mask feature consumes.

`CocoVideo.fps` (`:78-89`): median `1/dt` over `frame_timestamps`, else 25.0.
`records_by_frame()` (`:102-106`) groups records per frame.

### DLC pose CSV (`pose.py:74-130`)
`load_pose_csv(path, video_id=None, animal_to_identity=None, min_likelihood=0.0)`:
- Header structure: `pd.read_csv(path, header=[1,2,3], index_col=0)` — i.e. it **skips row 0
  (`scorer`)** and uses rows `individuals / bodyparts / coords` as a 3-level column MultiIndex;
  **column 0 is the frame index** (becomes the row index). (`pose.py:92-95`)
- `animals = unique level-0` (the `individuals`, e.g. `mouse1`, `mouse2`).
- `bodyparts = unique level-1`; reordered to canonical `KEYPOINT_ORDER`
  = `[nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail]` (pose.py:24-33), unknown
  parts appended.
- For each (animal, bodypart) reads the `("x","y","likelihood")` triple into
  `coords[F, A, K, 3]` (`:107-114`).
- **Identity remap:** `DEFAULT_ANIMAL_TO_IDENTITY = {"mouse1":"1","mouse2":"2"}` (pose.py:36) so
  pose identities match COCO `track_id` strings.
- `min_likelihood` (0.3 in the social pipeline call, `social_pipeline.py:271`): points with
  `likelihood < thresh` have their x/y set to NaN (`:116-121`); downstream `_interp_xy` fills
  short gaps.

Returns `PoseData(video_id, frame_indices[F], identities[A], keypoint_names[K], coords[F,A,K,3])`
(`pose.py:39-52`).

---

## 3. Can it run from IN-MEMORY data (no temp files)? Where is the seam?

**Yes.** The public `build_social_features` only takes file paths, but the file-reading is
isolated at the very top (`load_coco_videos` and `load_pose_csv`). Everything below operates on
the in-memory `CocoVideo` and `PoseData` objects. There are two clean seams:

### Recommended seam (lowest-level pure functions, zero file I/O)
Construct the two in-memory objects yourself from pykaboo's rolling buffer, then call the
table builders directly:

```python
from behavior_segmentation.coco_masks import CocoVideo, MaskRecord
from behavior_segmentation.pose import PoseData
from behavior_segmentation.features import extract_video_features          # (video, config) -> id_df, pair_df
from behavior_segmentation.pose_features import extract_pose_feature_tables # (pose, W, H, fps, egocentric) -> pose_id, pose_pair
from behavior_segmentation.wavelet_features import add_wavelet_features

# 1. Build CocoVideo from the last-N-frames buffer (no JSON):
recs = []
for frame_idx, per_id in buffer.items():           # per_id: {"1": {...}, "2": {...}}
    for ident, d in per_id.items():
        recs.append(MaskRecord(
            frame_idx=frame_idx, identity=ident,
            bbox=d["bbox"], score=d.get("score", 1.0),
            height=H, width=W,
            segmentation=d["segmentation_or_None"],   # see note below
            category_id=d.get("category_id")))
video = CocoVideo(video_id="live", width=W, height=H,
                  frame_indices=sorted(buffer), records=recs,
                  frame_timestamps={fi: ts for fi, ts in timestamps.items()})

# 2. Build PoseData (coords [F, A, K, 3]):
pose = PoseData(video_id="live", frame_indices=np.array(frames),
                identities=["1","2"],
                keypoint_names=KEYPOINT_ORDER,
                coords=coords)                        # last axis (x, y, likelihood)

# 3. Mask features + pose features + merge + wavelets (mirror social_pipeline.py:265-334).
id_df, pair_df = extract_video_features(video, config)
pose_id, pose_pair = extract_pose_feature_tables(pose, W, H, fps, egocentric=True)
# merge on the 4 meta keys, then add_wavelet_features per subject.
```

This is exactly what `build_social_features` does internally after its two `load_*` calls, so
results are identical. **The single cleanest injection point is the pair
`extract_video_features(CocoVideo, config)` + `extract_pose_feature_tables(PoseData, ...)`.**

**Important raster note for `MaskRecord.segmentation`:** the feature code calls
`record.decode_mask()` → `decode_segmentation`. It accepts (a) a polygon `list[list[float]]`
(`cv2.fillPoly`), or (b) an RLE dict `{"size","counts"}` (`pycocotools`). If pykaboo already has a
boolean/`uint8` numpy mask, the cleanest match is to **encode it to RLE once**
(`pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))`) and store that as
`segmentation` — or, to avoid the encode/decode round trip, subclass/monkeypatch a `MaskRecord`
whose `decode_mask()` returns the raw boolean array. The shape features (`mask_shape_features`,
features.py:164) only need the boolean mask + the bbox + score, so a thin adapter that returns the
in-memory mask is the lowest-latency option.

### Alternative seam (mid-level)
`extract_video_features(video, config)` (features.py:990-999) and the
`identity_df/pair_df → tracks` assembly via `assemble_track_matrix`/`build_inference_tracks`
(dataset.py) are both pure-DataFrame, no I/O. The fully-assembled reference path the GUI uses is
`free_infer.build_free_inference_tracks(coco_path, config, feature_names, pose_path)`
(free_infer.py:31-98) — but that one is path-based again, so prefer the lower seam for live.

### What there is NO clean seam for (must reimplement causally)
`extract_identity_features` (features.py:680-756) is written as a whole-track vectorized pass:
`interpolate_missing` (uses pandas `.interpolate`/ffill/bfill over the whole series),
`normalize_geometry` (global-median `area_norm`), `compute_motion_features` (`np.unwrap`,
`temporal_derivative` over full arrays), `add_rolling_features` (centered windows), then
`add_nearest_animal_features` (groups by frame). For a streaming engine you will **reimplement the
per-frame math incrementally** but **validate numerically against this function** on a buffered
window (see parity threats in §0.4 and §5).

---

## 4. Transitive import footprint + minimal vendor set

### External pip deps reachable on the default feature path
Verified by AST trace over the module-level imports of the strict feature path
(`social_pipeline → coco_masks, features, config, storage, pose, pose_features,
wavelet_features, labels, dataset, normalization, windows, roles`):

| pip package | required by | needed for live features? |
|---|---|---|
| `numpy` | all | **yes** |
| `pandas` | features, pose, pose_features, wavelet_features, social_pipeline, labels, dataset, storage | **yes** (tables) |
| `cv2` (opencv) | coco_masks (polygon), features (moments/contours/hull) | **yes** |
| `pycocotools` | coco_masks.decode_rle | **yes for RLE masks** (this dataset) |
| `pywt` (PyWavelets) | wavelet_features (Morlet CWT) | **yes if you keep wavelet channels** (has a pure-numpy fallback, see below) |
| `pydantic` + `yaml` | config only | optional (only to build `AppConfig`) |
| `torch` | dataset only (Dataset/collate); NOT feature math | optional for feature extraction |
| `pyarrow` | storage.write_table/parquet only | NOT needed (CSV fallback / never written live) |

Per-module third-party imports (measured): `coco_masks → cv2,numpy,pycocotools`;
`features → cv2,numpy,pandas`; `pose/pose_features/labels/social_pipeline → numpy,pandas`;
`wavelet_features → numpy,pandas,pywt`; `normalization/windows/roles → numpy`;
`config → pydantic,yaml`; `dataset → numpy,pandas,torch`; `storage → numpy,pandas,pyarrow(lazy)`.

`pywt` is imported with a try/except in wavelet_features.py:19-24 and has a difference-of-gaussian
**numpy fallback** (`_cwt_power`, :59-67). **WARNING:** the fallback is NOT numerically identical
to the real Morlet CWT, so for parity you should ship `pywt`, not rely on the fallback.

### Minimal internal modules to vendor (ZERO dependency on the rest of the repo)
To compute the 432 deployed features you need these **13 files** from `src/behavior_segmentation/`:

```
coco_masks.py        # CocoVideo/MaskRecord + RLE/polygon decode  (imports: config.DataConfig, storage.read_json)
features.py          # all mask geometric/motion/nearest/pair features (imports: coco_masks, config, storage)
pose.py              # PoseData + DLC CSV loader + KEYPOINT_ORDER   (no internal deps)
pose_features.py     # single + pair pose features                 (imports: pose)
wavelet_features.py  # Morlet CWT channels                         (no internal deps)
social_pipeline.py   # build_social_features orchestration         (imports: config, coco_masks, features, labels, pose, pose_features, wavelet_features)
dataset.py           # TrackData, build_feature_column_list, assemble_track_matrix, build_inference_tracks (imports: features, config, labels, normalization, roles, windows; torch)
normalization.py     # FeatureNormalizer (z-score from checkpoint) (no internal deps)
windows.py           # WindowSpan, build_windows, seconds_to_frames (no internal deps)
config.py            # AppConfig/FeaturesConfig                     (pydantic, yaml)
labels.py            # LabelMap, read_label_csv (only used at module-import time by social_pipeline/dataset) (imports: config, storage)
storage.py           # read_json (json), write_table (pyarrow lazy) (no internal deps)
roles.py             # constrain/expand label helpers (imported by dataset) (no internal deps)
```
`free_mask_contact.py` is **only** needed if `config.features.contact_geometry=True` (default
False) — skip it. `social_train.py`/`free_social.py` (lean selection, dataset registry) are **not**
needed for live inference once you have the checkpoint's 432 names. `metrics.py`, `export.py`,
`models/temporal_tcn.py` come in via `labels.py`/`features.py` lazy or unused paths and are not
required for the feature math.

**Trim opportunities for a tiny live package:**
- Drop `config.py` (and `pydantic`,`yaml`) by passing a small frozen dataclass/dict exposing only
  the `config.data.*` and `config.features.*` fields the feature code reads
  (`frame_rate`, `identity_field_priority`, `video_id_field`, `frame_index_regex`,
  `normalize_by_frame_size`, `normalize_by_body_size`, `egocentric`, `include_pairwise_features`,
  `rolling_windows_seconds`, `missing_mask_policy`, `max_interpolation_gap_frames`).
- Drop `torch` from the *feature* package by not importing `dataset.py` for assembly and instead
  re-implementing the simple `assemble_track_matrix` concatenation (subject + `obj_` partner +
  `pair_`) directly; torch is then only needed for the model forward pass.
- Drop `labels.py`/`roles.py`/`storage(pyarrow)` if you bypass `social_pipeline.read_wide_csv`
  (label alignment is training-only; live inference has no labels).

So the **irreducible feature-math core is 5 files**: `coco_masks.py`, `features.py`, `pose.py`,
`pose_features.py`, `wavelet_features.py` (+ a 30-line config shim), with pip deps
`numpy pandas opencv-python pycocotools pywt`.

---

## 5. `build_tracks`: identity_df + pair_df → per-identity `TrackData` (two tracks confirmed)

### `TrackData` (dataset.py:33-47)
```python
@dataclass
class TrackData:
    video_id: str; subject_id: str; object_id: str
    frame_indices: np.ndarray
    features: np.ndarray          # [T, C] float32
    feature_names: list[str]
    labels: np.ndarray | None     # [K, T] multilabel (or [T] multiclass)
    valid_mask: np.ndarray | None # [T] bool, False where mask missing
    mask_clip: np.ndarray | None  # optional CNN crop branch (unused here)
```

### How a track's feature matrix is assembled — `assemble_track_matrix` (dataset.py:104-152)
For a pair track `(video_id, subject_id, object_id)`:
1. `subj` = identity_df rows for the subject, sorted by `frame_idx`; `frame_indices` = its grid;
   `valid = 1 - missing_mask_flag`.
2. `identity_cols = feature_columns(identity_df)` (the per-identity numeric columns).
3. **Partner attachment (`:133-140`):** `obj` = identity_df rows for the *object_id* animal,
   `set_index("frame_idx").reindex(frame_indices).ffill().bfill()` (aligned to the subject's
   grid). The partner block is appended after the subject block — these become the `obj_`-prefixed
   columns in `build_feature_column_list`.
4. **Pair geometry (`:141-149`):** if `pair_df` non-empty, the rows for
   `(video_id, subject_id, object_id)` are `reindex(frame_indices).ffill().bfill().fillna(0)` and
   appended.
5. `matrix = concatenate([subject, partner, pair], axis=1)`, `nan_to_num(...,0.0)`.
   Final column order = `identity_cols + [obj_+c] + pair_cols` (matches
   `build_feature_column_list(..., is_pair=True)`, dataset.py:50-59).

### `build_tracks` (dataset.py:155-206) and the 2-track guarantee
`build_tracks` is label-driven (training). For **inference** the relevant function is
`build_inference_tracks(identity_df, pair_df, feature_names, is_pair=True)` (dataset.py:209-240):
it loops every `subject` × every `obj` with `subject != obj`. For a 2-mouse dyad
`{subjects} == {"1","2"}` this yields exactly:
- track A: `subject_id="1"`, `object_id="2"` (mouse1 as subject, mouse2 partner geometry),
- track B: `subject_id="2"`, `object_id="1"` (mouse2 as subject, mouse1 partner geometry).

**Confirmed: TWO directed tracks per dyad, each carrying the partner's features (`obj_*`) and the
directed pair geometry (`pair_*`, `pp_*`).** This is exactly what the live engine must build every
frame so it can run the model on both and take the scene-level per-class max-prob.

### Scene-level decision (free model is symmetric)
`free_social.load_scene_label_tracks` (free_social.py:109-170) assigns **the same scene multi-hot
labels to both identity tracks** — the only difference between the two tracks is whose egocentric
frame / partner geometry it is. Inference runs the model on both tracks and combines (per-class
**max** across the two directed views) for the scene prediction. The reference assembly used by the
GUI is `free_infer.build_free_inference_tracks` (free_infer.py:31-98): it builds the full ordered
column list, builds both inference tracks, then **selects the checkpoint's `feature_names` by name**
(missing → zero-filled), and re-applies the checkpoint normalizer per track. The windowed forward
pass + overlap-averaging pattern is in `embtcn_infer.encode_track` (embtcn_infer.py:128-156):
sliding windows of `window_seconds` (16 s = 480 frames at 30 fps) with stride, averaged where they
overlap.

---

## 6. Parity + latency threats for the live port (consolidated)

1. **fps must equal the real camera rate.** Offline overrides config with COCO-timestamp-derived
   fps (social_pipeline.py:259-264); all derivatives use `frame_rate` as the per-second scale
   (`temporal_derivative`, features.py:250). Feed the true fps.
2. **`area_norm` uses the whole-video median area** (`normalize_geometry`, features.py:368-371) and
   IS in the 432 deployed channels. Online you have no full-video median → use a **running median**
   (and accept early-frame drift) or warm up on a buffer. This is the worst parity offender.
3. **Wavelet CWT is whole-series & non-causal.** `add_wavelet_features` runs Morlet CWT over the
   entire identity series; scales up to 97.5 (4 s period). 132 of the 432 deployed channels are
   `_cwt_p`. Online you must run CWT on a trailing buffer ≥ a few seconds; expect edge effects at
   the buffer boundary (the current frame is always at the right edge — exactly where CWT is least
   accurate). Consider a causal wavelet or a longer trailing buffer; validate against offline.
4. **`np.unwrap` of orientation** (features.py:485) and pandas `.interpolate(..., limit_area=
   "inside")` + ffill/bfill in `interpolate_missing` (features.py:375-393) are history/边界-dependent.
   The `missing_mask_policy` default is `interpolate_short_gaps` with `max_gap=5` frames — replicate
   the short-gap interpolation causally (you cannot use future frames online; ffill only).
5. **Rolling stats are centered (`center=True`, features.py:419)** — but the deployed model drops
   ALL of them, so this is a non-issue for the 432-channel model. (If a future model re-includes
   rolling, you'd need causal/trailing windows instead.)
6. **`add_nearest_animal_features` groups all identities per frame** (features.py:536) and then
   differentiates per subject over time (features.py:660-673) — fine online if you keep both mice's
   per-frame geometry and a short history.
7. **Latency budget.** Measured ~8.4 ms/frame for the mask geometry stage alone (RLE decode +
   `cv2.moments`+`findContours`+`convexHull`+`arcLength`), single-thread CPU, 1080×1080 masks.
   Mitigations: crop masks to bbox before moments/contours; reuse pykaboo's already-decoded raster
   (skip RLE decode); precompute area/centroid if you trust pykaboo's (offline ignores them, so
   that changes parity — validate); drop wavelet channels if the F1 hit is acceptable. Pose +
   pair-geometry stages are sub-millisecond.
8. **The model's `frame_rate` in the checkpoint is `30.000000000001705`** — i.e. trained at ~30 fps
   with 16 s windows. The streaming engine should buffer ≥480 frames (16 s) for a full window or
   use the same overlap-average scheme on shorter trailing windows (accept lower accuracy until the
   buffer fills).

---

## Appendix: exact verification commands used
```
cd /home/andry/tracking_project/unsupervised_mask_behavior && PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python3 ...
# - loaded outputs/free_social/free_embtcn_attention.pt -> 432 plain feature_names
# - build_social_features on dataset/inference/760_adu_*  -> id_df (17998,1312), pair_df (17998,76)
# - build_feature_column_list(..., is_pair=True) == 2688; 432 is order-preserving subset; diff = all _w rolling
# - timing: extract_video_features 75.9s / 8999 frames = 8.4 ms/frame; pose 0.21s
# - segmentation in COCO = compressed RLE dict {"size","counts":str}; track_id=1/2, identity=mouse1/2
```
