# B. Feature Taxonomy, Temporality, and Normalization (Online/Offline Parity)

Scope: exactly what the EmbTCN-Attention free-interaction model consumes, in what
order, how each column is computed, whether it is causal, and how it is normalized.
Everything here is read from the repo at
`/home/andry/tracking_project/unsupervised_mask_behavior` and cross-checked against
the trained checkpoint `outputs/free_social/free_embtcn_attention_optimized.pt` and
the cache `outputs/free_social/cache/*.npz`. All facts verified by running the code.

## 0. TL;DR for the implementer (parity + latency risks)

1. **432 lean columns, fixed order, stored in the checkpoint.** The model is fed a
   `[T, 432]` matrix whose column order is `feature_names` in the checkpoint
   (`free_embtcn_attention_optimized.pt['feature_names']`). I verified the cache
   `feature_names`, the checkpoint `feature_names`, and the normalizer
   `feature_names` are byte-identical and in the same order. Build your online
   matrix in exactly that order. The full ordered list is in section 1.

2. **The model itself is NON-CAUSAL.** `embtcn_attention.py:11-12` and the saved
   `model_config['causal'] = False`. The TCN blocks use centered padding
   (`pad = (kernel_size-1)//2 * dilation`, `embtcn_attention.py:106`) and the
   Transformer encoder is bidirectional self-attention. So *even with perfect
   features*, a single forward pass needs frames on BOTH sides of the target frame.
   Offline inference runs 16-second windows (`window_seconds=16`, fps 30 -> 480
   frames) and averages overlapping windows. For closed loop you must either (a)
   accept a fixed latency and feed a trailing+small-leading buffer, or (b) retrain
   with `causal=True` (the flag exists and the conv block already implements the
   causal trim, `embtcn_attention.py:115-119`). This is a model-architecture parity
   risk that is SEPARATE from the feature parity risk below.

3. **Wavelets (132 of 432 cols, ~30%) are the #1 feature parity risk.** They are a
   full-signal Morlet CWT (`pywt.cwt(x, scales, "morl")`,
   `wavelet_features.py:68`). The most-recent-frame coefficient sits at the right
   boundary, where the Morlet wavelet integrates FUTURE samples that do not exist
   yet online. Empirically (section 3.4): the fast scales (p0=0.12s, p1=0.25s,
   p2=0.5s) reach offline parity with ~8-16 frames (0.27-0.53s) of look-ahead; the
   slow scales (p4=2s, p5=4s) need ~2s+ of future and never match at zero latency
   (right-edge MAE ~0.27-0.37 in log1p units, comparable to the channel's own std).
   Plan: a trailing ring buffer of >=512 frames PLUS a tunable look-ahead latency.

4. **Two more non-causal feature ops besides wavelets:**
   - `_cwt_power` mean-subtracts the WHOLE signal (`x = x - x.mean()`,
     `wavelet_features.py:58`). Online you must use a causal running mean (or the
     buffer mean) and accept a small DC offset.
   - Centered rolling features (`rolling(center=True)`, `features.py:419`) and
     `rolling_slope` (centered ramp kernel, `features.py:396-406`) ARE NON-CAUSAL,
     **but NONE of them survive into the lean set** (group `maskroll` is excluded;
     see section 2). So rolling is not a live concern for this model.

5. **Normalization is plain global z-score**, `(x - mean)/std` per column, fitted on
   the training tracks only, stored in the checkpoint. The per-video robust/CORAL
   branches are NOT used by this checkpoint (`video_median`, `video_mean` are empty
   dicts -> `None`). So online: `x_norm = (feat - normalizer.mean) / normalizer.std`,
   then `nan_to_num(...,0)`. Exact math in section 4. This is causal and trivial.

6. **frame_rate is baked in at 30 fps** (`frame_rate = 30.000000000001705` in the
   checkpoint; the offline pipeline overrides config with the COCO-detected fps,
   `social_pipeline.py:259-263`). Every velocity/derivative is "per second", scaled
   by fps. If pykaboo runs at a different fps you must scale derivatives by the
   actual fps, but the model was trained at 30 — feed it 30 fps semantics.

---

## 1. The exact 432 lean columns (verified, in order)

Source of truth: `outputs/free_social/cache/225_adu.npz['feature_names']`
(== checkpoint `feature_names` == normalizer `feature_names`, all confirmed equal
and identically ordered). The 432 columns decompose by `build_feature_column_list`
(`dataset.py:50-59`) into three blocks:

```
matrix = [ subject identity cols ]  ++  [ "obj_" + same identity cols ]  ++  [ pair_ cols ]
```

For the FULL (pre-lean) column list this is `identity_cols + obj_<identity_cols> +
pair_cols`. The lean subset keeps groups {maskcore, pose, wavelet} (drops
`maskroll`) for both the subject block and the `obj_` block, and keeps the
`maskpair` (`pair_`) block (section 2 explains the group rules). After subsetting,
the surviving order is preserved, giving the 432 below.

Group / prefix census (verified):
- by fine group (`feature_group_of`): maskcore 154, pose 106, wavelet 132, maskpair 40
- by prefix: subject-mask/core (no prefix) 95, `pose_` 67, `pp_` 50, `obj_` 180, `pair_` 40
- `_cwt_p` columns: 132 (= 22 source signals x 6 scales; 11 signals x 6 for subject
  block + 11 x 6 for `obj_` block)

### 1a. Subject block (indices 0-179): identity + pose + wavelet for the subject mouse

Mask-core geometric + motion + nearest-animal (indices 0-76):
```
  0 center_x            1 center_y            2 bbox_x              3 bbox_y
  4 bbox_width          5 bbox_height         6 area               7 perimeter
  8 convex_area         9 solidity           10 extent            11 eccentricity
 12 major_axis_length  13 minor_axis_length  14 orientation       15 aspect_ratio
 16 equivalent_diameter 17 mask_score        18 missing_mask_flag 19 mask_iou_with_previous_frame
 20 area_norm          21 delta_center_x     22 delta_center_y    23 velocity_x
 24 velocity_y         25 speed             26 speed_body_norm    27 acceleration
 28 acceleration_x     29 acceleration_y     30 acceleration_magnitude 31 acceleration_body_norm
 32 jerk               33 jerk_x             34 jerk_y            35 jerk_magnitude
 36 jerk_body_norm     37 area_velocity      38 area_acceleration 39 angular_velocity
 40 angular_acceleration 41 orientation_sin  42 orientation_cos   43 movement_bearing_sin
 44 movement_bearing_cos 45 forward_velocity 46 lateral_velocity  47 heading_alignment
 48 nearest_neighbor_count          49 nearest_centroid_distance
 50 nearest_centroid_dx             51 nearest_centroid_dy
 52 nearest_centroid_abs_dx         53 nearest_centroid_abs_dy
 54 nearest_centroid_distance_velocity        55 nearest_centroid_distance_acceleration
 56 nearest_bbox_edge_distance_velocity       57 nearest_bbox_edge_distance_acceleration
 58 nearest_subject_speed           59 nearest_neighbor_speed
 60 nearest_relative_velocity_x     61 nearest_relative_velocity_y
 62 nearest_relative_speed          63 nearest_relative_acceleration_x
 64 nearest_relative_acceleration_y 65 nearest_relative_acceleration_magnitude
 66 nearest_radial_velocity         67 nearest_radial_acceleration
 68 nearest_lateral_speed           69 nearest_approach_speed
 70 nearest_bbox_edge_distance      71 nearest_bbox_iou
 72 nearest_bbox_contact_flag       73 nearest_area_ratio
 74 nearest_bearing_sin             75 nearest_bearing_cos
 76 nearest_heading_alignment
```
Pose (single-animal egocentric) (indices 77-113):
```
 77 pose_body_length   78 pose_nose_neck     79 pose_neck_body    80 pose_body_tail
 81 pose_ear_span      82 pose_hip_span      83 pose_straightness 84 pose_curvature
 85 pose_mean_likelihood
 86 pose_nose_ego_x    87 pose_nose_ego_y    88 pose_left_ear_ego_x  89 pose_left_ear_ego_y
 90 pose_right_ear_ego_x 91 pose_right_ear_ego_y 92 pose_neck_ego_x 93 pose_neck_ego_y
 94 pose_body_ego_x    95 pose_body_ego_y    96 pose_left_hip_ego_x 97 pose_left_hip_ego_y
 98 pose_right_hip_ego_x 99 pose_right_hip_ego_y 100 pose_tail_ego_x 101 pose_tail_ego_y
102 pose_nose_speed   103 pose_nose_accel   104 pose_body_speed   105 pose_body_accel
106 pose_tail_speed   107 pose_tail_accel   108 pose_ego_forward_velocity
109 pose_ego_lateral_velocity 110 pose_ego_speed 111 pose_ego_accel
112 pose_heading_angvel 113 pose_body_length_vel
```
Wavelet (subject) (indices 114-179) — 11 signals x 6 scales (`_cwt_p0`.._cwt_p5`):
```
114-119 speed_cwt_p0..p5
120-125 area_velocity_cwt_p0..p5
126-131 angular_velocity_cwt_p0..p5
132-137 pose_nose_speed_cwt_p0..p5
138-143 pose_body_speed_cwt_p0..p5
144-149 pose_tail_speed_cwt_p0..p5
150-155 pose_body_length_cwt_p0..p5
156-161 pose_heading_angvel_cwt_p0..p5
162-167 pp_body_body_cwt_p0..p5
168-173 pp_nose_nose_cwt_p0..p5
174-179 pp_approach_speed_cwt_p0..p5
```

### 1b. Object block (indices 180-359): the SAME columns for the partner, prefixed `obj_`

Indices 180-359 are exactly `obj_` + every name in 1a in the same order:
`obj_center_x`(180) ... `obj_pp_approach_speed_cwt_p5`(359). These are the partner
mouse's own per-identity features (mask-core+pose+wavelet), reindexed onto the
subject's frame grid with forward/back-fill (`assemble_track_matrix`,
`dataset.py:133-140`). NOTE: even the partner's `nearest_*`, `pp_*` and wavelet
channels are computed from the PARTNER as subject; in a 2-mouse scene `obj_nearest_*`
== the mirror of the subject's nearest, and `obj_pp_*` is the partner->subject
directed pose pair. The `_cwt` channels in this block are the partner's own
wavelets (so 11 more signals -> 66 more cwt cols, total 132).

### 1c. Pair block (indices 360-431): directed pairwise + dyadic pose (`pair_` and `pp_`)

`pair_` mask-pairwise (indices 360-399, 40 cols, from `extract_pairwise_features`):
```
360 pair_centroid_distance        361 pair_centroid_dx
362 pair_centroid_dy              363 pair_centroid_abs_dx
364 pair_centroid_abs_dy          365 pair_centroid_distance_body_norm
366 pair_relative_speed           367 pair_relative_acceleration
368 pair_subject_speed            369 pair_object_speed
370 pair_speed_ratio              371 pair_subject_acceleration
372 pair_object_acceleration      373 pair_relative_velocity_x
374 pair_relative_velocity_y      375 pair_relative_motion_speed
376 pair_relative_motion_speed_body_norm  377 pair_radial_velocity
378 pair_radial_velocity_body_norm 379 pair_lateral_velocity
380 pair_lateral_speed            381 pair_lateral_speed_body_norm
382 pair_distance_acceleration    383 pair_approach_acceleration
384 pair_relative_acceleration_x  385 pair_relative_acceleration_y
386 pair_relative_acceleration_magnitude 387 pair_radial_acceleration
388 pair_approaching_score        389 pair_following_score
390 pair_relative_orientation     391 pair_bbox_edge_distance
392 pair_bbox_edge_distance_body_norm 393 pair_bbox_edge_distance_velocity
394 pair_bbox_edge_distance_acceleration 395 pair_bbox_iou
396 pair_contact_flag             397 pair_bbox_contact_flag
398 pair_area_ratio               399 pair_area_abs_difference
```
`pp_` dyadic pose pair (indices 400-431, 32 cols, from
`pair_pose_features_egocentric`). NOTE these are classified group "pose" (the rule
treats `pp_` and `pose_` as pose), so they survive lean even though they live in the
pair block:
```
400 pp_nose_nose      401 pp_nose_tail      402 pp_nose_body     403 pp_body_body
404 pp_tail_nose      405 pp_min_internose  406 pp_facing_cos    407 pp_facing_sin
408 pp_approach_speed 409 pp_rel_heading_cos 410 pp_rel_heading_sin
411 pp_relative_velocity_forward 412 pp_relative_velocity_lateral 413 pp_relative_speed
414 pp_obj_nose_ego_x   415 pp_obj_nose_ego_y   416 pp_obj_left_ear_ego_x 417 pp_obj_left_ear_ego_y
418 pp_obj_right_ear_ego_x 419 pp_obj_right_ear_ego_y 420 pp_obj_neck_ego_x 421 pp_obj_neck_ego_y
422 pp_obj_body_ego_x   423 pp_obj_body_ego_y   424 pp_obj_left_hip_ego_x 425 pp_obj_left_hip_ego_y
426 pp_obj_right_hip_ego_x 427 pp_obj_right_hip_ego_y 428 pp_obj_tail_ego_x 429 pp_obj_tail_ego_y
430 pp_contact_flag   431 pp_body_body_vel
```

(The full machine-readable JSON list is also at `/tmp/lean_cols.json` from the dump
run; if you need it embedded, re-run the npz dump in section 0.)

---

## 2. What "lean" keeps and drops (`select_feature_indices`)

`free_social.py:63`: `LEAN_GROUPS = ["maskcore", "pose", "wavelet"]`.
`build_and_cache_video` (`free_social.py:226-239`) builds the FULL ordered column
list first, builds tracks, then subsets with
`select_feature_indices(full_names, LEAN_GROUPS)` and `subset_tracks`.

Group classifier `feature_group_of(col)` (`social_train.py:48-60`):
```python
base = col[4:] if col.startswith("obj_") else col   # strip obj_ first
if "_cwt_p" in base:                       return "wavelet"
if base.startswith("pp_") or base.startswith("pose_"): return "pose"
if base.startswith("pair_"):               return "maskpair"
if _ROLL_RE.search(base):                  return "maskroll"   # _(mean|std|min|max|median|slope)_w\d+$
return "maskcore"
```
Aliases (`social_train.py:64-70`):
```python
GROUP_ALIASES = {
  "mask":     {"maskcore","maskroll","maskpair"},
  "maskcore": {"maskcore","maskpair"},     # <-- KEY: "maskcore" ALSO pulls in maskpair
  "maskroll": {"maskroll"},
  "pose":     {"pose"},
  "wavelet":  {"wavelet"},
}
```
So `LEAN_GROUPS` -> wanted = {maskcore, maskpair, pose, wavelet}. **Kept:** mask-core
geometric/motion/nearest, all `pair_` (maskpair), all `pose_`+`pp_` (pose), all
`_cwt_` (wavelet) — for both subject and `obj_` blocks. **Dropped:** group
`maskroll`, i.e. every `*_mean_w*`, `*_std_w*`, `*_min_w*`, `*_max_w*`,
`*_median_w*`, `*_slope_w*` rolling column. The centered-rolling non-causal features
therefore DO NOT reach this model. Confirmed: 0 lean columns match the rolling regex.

If `select_feature_indices` ever returns empty it falls back to all columns
(`social_train.py:82`); not a concern here.

---

## 3. Temporality of every feature group

Conventions used by the pipeline:
- `frame_rate` (fps) = 30. Derivatives are "per second": value-delta x fps.
- `temporal_derivative` (`features.py:250-270`): a **backward finite difference**
  (`np.diff`, then `out[1:] = delta * fps/frame_delta`, `out[0]=0`). CAUSAL: frame t
  uses t and t-1 only. Respects frame gaps (scales by actual frame spacing).
- pose kinematics use `np.diff` along axis 0 with `d[0]=0` then `*frame_rate`
  (`pose_features.py:301-305, 319-325`). Also CAUSAL backward diff.
- `np.unwrap` on heading/orientation is used before differencing
  (`features.py:485`, `pose_features.py:319`). `np.unwrap` is technically a
  whole-array op but the unwrap branch only depends on the cumulative sign of
  past jumps; for a streaming heading you must keep a running unwrap state (carry
  previous unwrapped angle) — a small, solvable causal-state detail, NOT look-ahead.

### 3a. INSTANTANEOUS (current frame's mask + pose only) — fully online, zero history

These need only the current frame's mask/pose/geometry. Safe to compute per-frame.
- **Mask geometry** (`mask_shape_features`, `features.py:164-228`): center_x/y
  (cv2.moments centroid), bbox_x/y/width/height, area (m00), perimeter
  (cv2.arcLength of largest contour), convex_area (cv2.convexHull),
  solidity=area/convex_area, extent=area/(bw*bh), eccentricity & major/minor axis
  & orientation (from central moments mu20/mu02/mu11, eqs at `features.py:201-213`),
  aspect_ratio=bw/bh, equivalent_diameter=sqrt(4*area/pi), mask_score,
  missing_mask_flag. Indices 0-18 (minus the ones below).
- **Geometry-normalize** (`normalize_geometry`, `features.py:351-372`): divides
  spatial cols by frame_width/height and area by W*H IF `normalize_by_frame_size`.
  `area_norm` = area / median(area) over the whole track — **needs the whole-video
  median** (parity note: online use the running/buffer median or the offline
  per-video median; small drift). Check the config flags: see section 5.
- **orientation_sin/cos** (`features.py:489-490`): sin/cos of current orientation.
- **Pose single-frame egocentric geom** (`single_pose_features_egocentric`,
  `pose_features.py:234-297`): pose_body_length, pose_nose_neck, pose_neck_body,
  pose_body_tail, pose_ear_span, pose_hip_span, pose_straightness, pose_curvature,
  pose_mean_likelihood, and all `pose_*_ego_x/y` (8 keypoints rotated into the
  subject's own neck->nose frame, scaled by nose-tail length). All current-frame.
- **Pose pair geometry** (`pair_pose_features_egocentric`, `pose_features.py:397-471`):
  pp_nose_nose, pp_nose_tail, pp_nose_body, pp_body_body, pp_tail_nose,
  pp_min_internose, pp_facing_cos/sin, pp_rel_heading_cos/sin, pp_obj_*_ego_x/y,
  pp_contact_flag. Current-frame dyadic geometry in the subject's egocentric frame.
- **Mask pairwise instantaneous** (`extract_pairwise_features`, `features.py:759-976`):
  pair_centroid_distance/dx/dy/abs_dx/abs_dy, pair_centroid_distance_body_norm,
  pair_relative_orientation, pair_bbox_edge_distance(+body_norm), pair_bbox_iou,
  pair_bbox_contact_flag, pair_area_ratio, pair_area_abs_difference,
  pair_subject_speed/object_speed/speed_ratio (these reuse per-id speed, which is
  causal), pair_relative_velocity_x/y, pair_relative_motion_speed(+body_norm),
  pair_radial_velocity(+body_norm), pair_lateral_velocity/speed(+body_norm),
  pair_following_score. NOTE one parity hazard here: `pair_contact_flag` uses
  `contact_threshold = nanmedian(dist) * 0.5` over the WHOLE track
  (`features.py:906-907`) -> non-causal threshold. Online: use a fixed or
  running-median contact threshold (drift risk; flag is binary).
- **Nearest-animal instantaneous** (`add_nearest_animal_features`,
  `features.py:509-658`): for 2 mice the "nearest" is just the partner. dist, dx,
  dy, abs, bearing_sin/cos, heading_alignment, bbox_edge_distance, bbox_iou,
  bbox_contact_flag, area_ratio, neighbor_count(=1), subject/neighbor speed,
  relative velocity/accel, radial vel/accel, lateral speed, approach speed
  (=-radial_velocity). The velocity/accel pieces are causal (built from per-id
  velocity_x/y which are backward diffs).

### 3b. CAUSAL-HISTORY (needs only PAST frames) — needs a trailing buffer

All use backward finite differences (`np.diff`, `out[0]=0`), so frame t uses
{t, t-1, ...}. None look ahead. Quote of the core op (`features.py:250-270`):
```python
def temporal_derivative(values, frame_indices, frame_rate):
    frame_delta = np.diff(frames);  value_delta = np.diff(arr)
    scale = frame_rate / frame_delta          # per-second, gap-aware
    out[1:] = where(valid, value_delta * scale, 0.0);  out[0]=0
```
- **Subject motion** (`compute_motion_features`, `features.py:433-506`):
  - delta_center_x/y = `np.diff(center)` (1-frame), `d[0]=0` (`features.py:445-447`).
  - velocity_x/y = temporal_derivative(center). speed=hypot(vx,vy).
  - acceleration = d(speed); acceleration_x/y = d(velocity); acceleration_magnitude
    = hypot(ax,ay). jerk = d(acceleration); jerk_x/y = d(accel_x/y);
    jerk_magnitude. (Each extra derivative widens the causal footprint by 1 frame:
    speed needs t-1, accel needs t-2, jerk needs t-3.)
  - `*_body_norm` = value / body_scale, body_scale from current geometry
    (major_axis_length else equiv_diameter else bbox diag, `features.py:273-288`) —
    instantaneous denominator, so the body_norm vels are causal-history overall.
  - area_velocity = d(area); area_acceleration = d(area_velocity).
  - angular_velocity = d(unwrap(2*orientation)/2); angular_acceleration =
    d(angular). Needs running unwrap state (carry last orientation).
  - movement_bearing_sin/cos = sin/cos(atan2(vy,vx)) when speed>1e-9 (causal,
    depends on velocity).
  - forward_velocity = vx*cos(o)+vy*sin(o); lateral_velocity = -vx*sin(o)+vy*cos(o);
    heading_alignment = forward_velocity/speed. (o = current orientation.)
- **Pose kinematics** (`pose_features.py:298-325`): pose_{nose,body,tail}_speed
  (backward diff of ego-kp * fps), pose_{...}_accel (diff of speed * fps),
  pose_ego_forward/lateral_velocity, pose_ego_speed, pose_ego_accel (from
  center backward-diff rotated into ego frame), pose_heading_angvel
  (d(unwrap(heading)) — running unwrap state), pose_body_length_vel
  (d(scale)/scale * fps). All causal-history.
- **Pairwise/dyadic derivatives**: pair_relative_speed=d(dist)
  (`features.py:793`), pair_relative_acceleration=pair_distance_acceleration=
  d(d(dist)), pair_approaching_score=-d(dist), pair_approach_acceleration=
  -d(d(dist)), pair_subject/object_acceleration, pair_relative_acceleration_x/y/mag,
  pair_radial_acceleration, pair_bbox_edge_distance_velocity/acceleration. From
  pp: pp_approach_speed=-diff(body_body)*fps (`pose_features.py:440-441`),
  pp_body_body_vel=diff(body_body)*fps (`pose_features.py:471`),
  pp_relative_velocity_forward/lateral and pp_relative_speed (diff of partner
  position in ego frame, `pose_features.py:442-444`). All backward-diff -> causal.
- **Nearest derived velocities** (`features.py:660-673`):
  nearest_centroid_distance_velocity/acceleration,
  nearest_bbox_edge_distance_velocity/acceleration — temporal_derivative of the
  per-frame distances; causal.

Implementation note for the online extractor: maintain the last ~4 frames of every
base quantity (centroid, area, orientation, ego-kp, body_body distance, centroid
distance) plus running unwrap state for heading/orientation. Jerk is the deepest at
3 frames of history; everything else is <=2.

### 3c. NON-CAUSAL (needs FUTURE frames or whole-signal transforms)

- **Wavelets (132 cols)** — full-signal Morlet CWT. See section 3.4. #1 risk.
- **`_cwt_power` mean subtraction**: `x = x - x.mean()` over the whole 1-D channel
  (`wavelet_features.py:58`). Whole-signal stat. Online use a running/buffer mean.
- **Centered rolling features** (`add_rolling_features`, `features.py:409-430`):
  `rolling(window, center=True)` (mean/std/min/max/median) and `rolling_slope`
  (centered ramp kernel via `np.convolve(..., mode="same")`, `features.py:396-406`)
  are non-causal. **NOT in the lean set** (group maskroll dropped) — no live impact.
- **`np.unwrap`** (`features.py:485`, `pose_features.py:319,224`): a whole-array op;
  resolvable with running state (see 3b), not true look-ahead.
- **Whole-track stats baked into "instantaneous" features**: `area_norm` median
  (`features.py:369-371`) and `pair_contact_flag` median threshold
  (`features.py:906`); both are whole-track scalars. Minor parity drift; use
  offline per-video constants or running medians.

---

## 3.4. WAVELETS — the #1 parity risk, in full

Library / function: **PyWavelets** (`import pywt`, `wavelet_features.py:19-24`),
`pywt.cwt(x, scales, "morl")` (`wavelet_features.py:68`). Mother wavelet = `"morl"`
(Morlet). pywt is installed (v1.8.0). There is a NumPy DoG fallback if pywt is
missing (`wavelet_features.py:59-67`) — make sure pykaboo has pywt so you do not
silently get the fallback (different values!).

Output per channel (`_cwt_power`, `wavelet_features.py:54-69`):
```python
x = np.nan_to_num(signal, nan=0.0)
x = x - x.mean()                       # WHOLE-SIGNAL mean subtraction (non-causal)
coeffs, _ = pywt.cwt(x, scales, "morl")
return np.log1p(np.abs(coeffs))        # [num_scales, T], log1p of |coeff|
```
So each column value = `log1p(|CWT coefficient|)`.

Scales (`periods_to_scales`, `wavelet_features.py:45-51`):
```python
central = pywt.central_frequency("morl")  # = 0.8125
scales = [central * frame_rate * period   for period in periods]
```
Periods (`DEFAULT_PERIODS_SECONDS`, `wavelet_features.py:42`):
`[0.12, 0.25, 0.5, 1.0, 2.0, 4.0]` seconds (fast tremor -> slow bout envelope).
At fps=30 the scales are (verified):
```
period 0.12s -> scale 2.925     period 0.25s -> scale 6.094
period 0.5s  -> scale 12.188    period 1.0s  -> scale 24.375
period 2.0s  -> scale 48.750    period 4.0s  -> scale 97.500
```
Column suffix `_cwt_p{i}` for i=0..5 maps to those periods in order
(`add_wavelet_features`, `wavelet_features.py:90-93`).

Input signals (THE 8/11 SIGNALS) (`DEFAULT_WAVELET_SIGNALS`,
`wavelet_features.py:27-39`), filtered to those present:
```
speed, area_velocity, angular_velocity,
pose_nose_speed, pose_body_speed, pose_tail_speed,
pose_body_length, pose_heading_angvel,
pp_body_body, pp_nose_nose, pp_approach_speed
```
The first 8 are per-identity signals already in identity_df; the last 3
(`pp_body_body`, `pp_nose_nose`, `pp_approach_speed`) are PAIR signals temporarily
merged from pair_df onto each subject before the CWT, then dropped (their cwt cols
kept) (`social_pipeline.py:304-334`, `pair_signal_map` at 308-312). So 11 signals
total per identity x 6 scales = 66 cwt cols per identity. Subject block + obj block
= 132 cwt cols. (The module docstring says "8 signals"; the actual present count is
11 once pose+pair signals are available, which is the real free-interaction case.)

Is the CWT computed over the FULL signal array per channel? **Yes** — `add_wavelet_features`
is called once per contiguous identity track over the whole ~9000-frame array
(`social_pipeline.py:328`). Never across identity/video boundaries.

### Minimum trailing buffer + look-ahead (measured)

The Morlet wavelet has effective half-support ~4*scale samples on EACH side. So the
most-recent frame sits at the right boundary and the offline coefficient there
integrates samples up to ~4*scale frames into the FUTURE:
```
p0 scale 2.9  -> ~12 future samples (~0.4s)     p3 scale 24.4 -> ~98  (~3.3s)
p1 scale 6.1  -> ~24            (~0.8s)          p4 scale 48.8 -> ~195 (~6.5s)
p2 scale 12.2 -> ~49            (~1.6s)          p5 scale 97.5 -> ~390 (~13s)
```
Empirical right-edge error (MAE in log1p|coef| units; channel std shown for scale),
trailing buffer only, zero look-ahead:
```
                p0     p1     p2     p3     p4     p5
right-edge MAE  0.138  0.105  0.278  0.217  0.270  0.370
channel std     0.092  0.132  0.260  0.240  0.361  0.496
```
The error is dominated by the missing-future boundary, NOT by buffer length
(B=64..768 barely changes it). Accepting a look-ahead latency L collapses the error
for the faster scales:
```
L (frames / s)   p0     p1     p2     p3     p4     p5
0  / 0.00s       0.138  0.105  0.278  0.217  0.270  0.370
4  / 0.13s       0.055  0.092  0.296  0.222  0.343  0.373
8  / 0.27s       0.003  0.037  0.179  0.244  0.408  0.430
16 / 0.53s       ~0     0.003  0.069  0.125  0.416  0.467
32 / 1.07s       ~0     ~0     0.003  0.060  0.159  0.466
64 / 2.13s       ~0     ~0     ~0     0.004  0.068  0.254
```
(Numbers from a synthetic AR+oscillation signal; treat as order-of-magnitude.)

**Recommendations for online wavelets:**
- Keep a trailing ring buffer of at least ~512 frames (~17s) per CWT input signal so
  the LEFT side of all scales (incl. p5 4s) is fully supported.
- For each new frame, recompute `pywt.cwt(buffer, scales, "morl")` and read the
  column at index `len(buffer) - 1 - L` for a chosen look-ahead L. With L = 0 you
  match the model's training-time fast-scale dynamics decently but the slow scales
  (p4,p5 = ~16 of 132 cols x2 blocks) will be biased.
- Three options to manage the slow-scale bias: (a) accept ~1-2s closed-loop latency
  (L=32-64) so p3/p4 converge; (b) zero-out / freeze p5 (and maybe p4) online and
  document the train/serve gap; (c) retrain with a one-sided (causal) CWT or drop
  the >=2s periods from `DEFAULT_PERIODS_SECONDS` so the feature is causal by
  construction. Option (c) gives the cleanest parity for a real-time trigger.
- Use the buffer mean (not a global mean) for the `x - x.mean()` step; recompute it
  incrementally. The DC term mostly cancels in `|coeff|` for non-DC scales but
  affects edge behavior.
- Cost: `pywt.cwt` on a 512-length buffer x 6 scales x 22 signals every frame is
  cheap (sub-ms to a few ms on CPU). Latency budget is dominated by the model
  forward + the chosen look-ahead, not the CWT compute.

---

## 4. FeatureNormalizer (exact math, keys, inference usage)

File: `normalization.py`. Dataclass fields: `feature_names, mean, std,
video_median, video_iqr, video_mean, video_std` (`normalization.py:18-25`).

### Fit (what this checkpoint used)

`FeatureNormalizer.fit` (`normalization.py:27-39`) — plain global z-score:
```python
stacked = concatenate(matrices, axis=0)        # all training tracks stacked over time
mean = np.nanmean(stacked, axis=0)             # per-column
std  = np.nanstd(stacked, axis=0)
std  = np.where(std < 1e-6, 1.0, std)          # guard tiny std
```
Called in `free_train.train_supervised` at `free_train.py:274`:
`FeatureNormalizer.fit([t.features for t in train_tracks], feature_names)`. So the
stored `mean`/`std` are computed on the LEAN 432-col TRAIN tracks only. No
clipping of inputs; no robust median/IQR; per-column mean/std only.

(There are also `fit_with_video_robust` (median/IQR) and `fit_with_video_coral`
(per-video mean/std) variants, `normalization.py:41-101`, but the trained
checkpoint does NOT use them — see "Confirmed in checkpoint" below.)

### Transform (apply at inference)

`transform` (`normalization.py:103-135`):
```python
x = matrix.astype(float64)
# optional per-video robust step (only if video_median/iqr present) -> SKIPPED here
# optional per-video CORAL step (only if video_mean/std present)    -> SKIPPED here
out = (x - self.mean) / self.std
return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(float32)
```
For this checkpoint the two optional branches are inert (their dicts are None), so
online normalization is exactly:
```python
x_norm = np.nan_to_num((feat_row_432 - mean_432) / std_432, nan=0.0,
                       posinf=0.0, neginf=0.0).astype(np.float32)
```
Fully causal, per-frame, vectorized. No state.

How it is applied at inference: in `predict_track_probs`
(`social_train.py:157-194`) the WHOLE track is transformed once
(`feats = normalizer.transform(track.features)`, line 169) then sliced into
windows; in `encode_track` (`embtcn_infer.py:142`) likewise. For free supervised
inference the path is `with_video_stats_from_tracks` (`embtcn_infer.py:198`) which,
when there are no video stats, just returns a plain copy (`normalization.py:140-150`)
— so still pure global z-score. Online: apply the global transform per frame; do NOT
attempt per-video re-fitting (this checkpoint has no per-video stats and re-fitting
on a short live buffer would diverge from training).

### to_dict / from_dict keys

`to_dict` (`normalization.py:203-220`) keys:
`feature_names, mean, std, video_median, video_iqr, video_mean, video_std`
(arrays -> lists; the four video dicts -> {video_id: list}). `from_dict`
(`normalization.py:222-248`) reads them back, turning empty video dicts into None.

### Confirmed in checkpoint

`free_embtcn_attention_optimized.pt['normalizer']` has keys
`[feature_names, mean, std, video_median, video_iqr, video_mean, video_std]`,
`len(mean)==len(std)==len(feature_names)==432`, and `video_median`/`video_mean`
are EMPTY (-> None) -> plain global z-score. Top-level checkpoint also stores:
`architecture='embtcn_attention'`, `frame_rate=30.0`, `window_seconds=16.0`,
`eval_stride_seconds=4.0`, `smooth_win=9`, `min_bout_frames=20`,
`merge_gap_frames=30`, `thresholds=[0.60,0.65,0.65,0.70,0.60,0.60,0.60]` (per the 7
classes: background, nose-to-nose, nose-to-body, anogenital, passive, rearing,
mounting), `label_map`, and full `model_config` (causal=False, max_len=4096,
tcn_dilations=(1,2,4,8,16,32), kernel_size=5, num_encoder_layers=4, d_model=160,
embedding_dim=96, num_refinement_stages=1). Saved by `save_free_checkpoint`
(`free_train.py:388-423`).

NOTE the optimized checkpoint's smooth_win/min_bout/merge_gap (9/20/30) differ from
the FreeTrainConfig dataclass defaults (15/12/6, `social_train.py:413-415`). Use the
values FROM THE CHECKPOINT, not the dataclass.

---

## 5. windows.py + window/stride semantics

`seconds_to_frames(seconds, frame_rate)` (`windows.py:29-30`):
```python
return max(int(round(seconds * frame_rate)), 1)
```
`build_windows(num_frames, window_length, stride, ...)` (`windows.py:33-56`):
sliding half-open `[start, end)` windows, `window_length = min(window_length,
num_frames)`, `stride = max(stride,1)`, advancing `start += stride`, with a final
partial window allowed unless `drop_last_partial`.

Window settings actually used:
- Training (this checkpoint): `window_seconds=16.0` -> 480 frames @30fps;
  `stride_seconds=2.0` (train) / `eval_stride_seconds=4.0` (eval) ->
  `seconds_to_frames` = 480 / 60 / 120 frames. (`TrainConfig`,
  `social_train.py:396-398`; checkpoint window_seconds=16, eval_stride=4.)
- `predict_track_probs` (`social_train.py:175-194`): builds overlapping windows of
  `window_frames` at `stride_frames`, runs the model on each, and AVERAGES
  overlapping per-frame probabilities (`prob_sum[:,s:e]+=p; counts[s:e]+=1;
  probs=prob_sum/counts`). Always appends a final window ending at T so the tail is
  covered (`social_train.py:176-177`). So offline each frame's prediction is the
  mean of up to ceil(window/stride) windows, each a bidirectional pass — heavy
  look-ahead/look-behind by design.

**Streaming implication:** you cannot reproduce the overlap-averaged 16s windows in
real time. For closed loop, run a SINGLE rolling window of fixed length W ending at
(or slightly past) the current frame and read the prediction at the target column.
A bidirectional model on a W-frame buffer means the target frame's prediction
depends on up to W/2 frames of future inside the buffer — so to predict frame t you
must wait until you have ~W/2 frames after t (or place t near the right edge and
accept degraded context). Practical recipe: keep a trailing buffer of length W
(e.g. 256-480), put the decision frame at index `W-1-L` for look-ahead L, take that
column from the forward pass. Tie L to the same value chosen for the wavelet
look-ahead so feature parity and model-context parity use one latency budget.

Post-processing (offline, after probs): `smooth_probs` (moving-average over
`smooth_win`, `social_train.py:224-231`, CENTERED via `np.convolve mode="same"`),
threshold per class, `enforce_min_bout` (drops runs < min_bout_frames), merge gaps.
These shape the reported bouts but are NON-CAUSAL (centered smoothing + min-bout
needs the run to complete). For a real-time trigger, replace with a causal
hysteresis/debounce (e.g. require N consecutive frames over threshold) rather than
the offline smooth+min-bout. The raw per-frame `probs` vs `thresholds` is the causal
signal to trigger on.

---

## 6. Online feature-extractor checklist (parity contract)

1. Emit a `[432]` vector per frame in the EXACT order of section 1 (subject block,
   then `obj_` block = same features for the partner, then `pair_` + `pp_` block).
2. Compute mask geometry from the current mask with cv2 moments/contours EXACTLY as
   `mask_shape_features` (centroid from moments, axes from central moments, perimeter
   from largest contour, etc.). Match `missing_mask_flag` and the NaN-fallback paths.
3. Apply the same frame-size normalization as `normalize_geometry` IF the config used
   for training had `normalize_by_frame_size`/`normalize_by_body_size` set — verify
   in `configs/default.yaml` (`features` block). The cache was built via
   `build_or_load_cache(config_path="configs/default.yaml")`. Use those flags; W/H =
   the COCO video size.
4. Pose features in the subject EGOCENTRIC frame (translate to body kp, rotate
   neck->nose, scale by nose-tail length) as `single_pose_features_egocentric` /
   `pair_pose_features_egocentric`. Interpolate short keypoint gaps with the same
   linear+ffill+bfill (`_interp_xy`, max_gap=10) — online use causal ffill only.
   `egocentric` flag: confirm `config.features.egocentric` (used at
   `social_pipeline.py:278`); the lean cols (`*_ego_*`, `pp_relative_velocity_*`)
   imply egocentric=True.
5. All derivatives = backward finite diff x fps (per-second), `out[0]=0`. Keep up to
   4 frames of history + running unwrap state for heading/orientation.
6. Wavelets: trailing buffer >=512 frames per signal, `pywt.cwt(buf-buf.mean(),
   scales, "morl")`, `log1p(|.|)`, read column `len-1-L`. scales from
   `0.8125*30*period`, periods `[0.12,0.25,0.5,1,2,4]`. Ensure pywt is installed.
7. Normalize: `(x - mean)/std` from the checkpoint normalizer, `nan_to_num->0`.
8. Feed to the model in a fixed rolling window; read decision column at look-ahead L;
   threshold with the checkpoint `thresholds`; debounce causally for the trigger.

## 7. Key file:line index

- 432 order: `dataset.py:50-59` build_feature_column_list; cache/checkpoint
  `feature_names` (verified equal).
- Lean selection: `free_social.py:63,226-239`; `social_train.py:48-95`.
- Mask geom: `features.py:164-228`; motion (causal diffs): `features.py:250-270,
  433-506`; nearest: `features.py:509-677`; pairwise: `features.py:759-976`;
  rolling (dropped): `features.py:396-430`.
- Pose: `pose_features.py:234-326` (single ego), `397-472` (pair ego),
  `129-146` egocentric_transform.
- Dyadic (NOT in lean — `dy_*` columns are absent from the 432): `dyadic_features.py`
  (built only if separately requested; the free model's pair block is `pair_`+`pp_`).
- Wavelets: `wavelet_features.py:27-69` (signals, scales, cwt); merged
  `social_pipeline.py:304-334`.
- Normalizer: `normalization.py:27-39` fit, `103-135` transform, `203-248`
  dict; used `social_train.py:169`, `embtcn_infer.py:142,198`.
- Windows: `windows.py:29-56`; inference overlap-average `social_train.py:157-194`.
- Model non-causality: `embtcn_attention.py:11-12,98-121,230-255`;
  `model_config['causal']=False`.
- Config defaults: `config.py:28 frame_rate=30`, `164-165 train 12/3`,
  `237-241 inference 12/6`; TrainConfig `social_train.py:396-419 (16/2/4,
  smooth 15/min 12/merge 6)`; CHECKPOINT overrides 16/4, smooth 9/min 20/merge 30.
- build_social_features: `social_pipeline.py:240-348` (fps override 259-263,
  pose merge 267-300, wavelet 304-334).
