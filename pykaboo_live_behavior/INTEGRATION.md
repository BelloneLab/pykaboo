# Live behavior detection in pykaboo (closed-loop TTL)

This wires live social-behavior detection into pykaboo's live detection + TTL trigger
system, so a detected behavior can drive an Arduino output for closed-loop optogenetics.

## Two backends (pick one in the UI: "Detector")

1. **Rule-based (fast, default)** -- `rule_based_social.py`. Per-frame geometric +
   kinematic tests over keypoints + mask contours: nose2nose, sidebyside, sidereside,
   nose2anogenital, nose2body, oriented_toward, following, chasing, approach,
   withdrawal_from_partner, escape, withdrawal_after_contact, fighting. Booleans are
   trailing-smoothed (uniform window, keep > 0.4). **Sub-ms per frame, torch-free, no
   checkpoint** -- this is the backend for genuine closed-loop triggering.

   **Contact model:** a social contact is gated by **mask contour overlap** (masks
   within +/-5% of body length, `mask_contact_frac`); the contact TYPE is then read
   from the **closest inter-animal keypoints** -- nose<->nose (mutual) = nose2nose,
   a leading nose -> partner tail = nose2anogenital, leading nose -> partner
   body/flank = nose2body, bodies touching with no leading nose = sidebyside /
   sidereside (by heading). When a mask is missing for a frame it falls back to
   keypoint proximity. The chip shows the **highest-priority active** behavior
   (fighting > chasing > anogenital > nose2nose > nose2body > ...), not the noisiest
   weak cue. All tolerances scale with body length (resolution/arena invariant); knobs
   live in `RuleParams`. Tune offline against a recording with
   `analyze_recording.py <tracking_dlc.csv> --masks <masks_coco.json>`.
2. **ML model (EmbTCN-Attention)** -- the trained free-interaction temporal model
   (7 classes: background, nose-to-nose, nose-to-body, anogenital, passive, rearing,
   mounting). More holistic but ~0.3 s/decision (not real-time); see the latency
   section.

Both backends run in the same `LiveBehaviorWorker` and emit the same `BehaviorFrameState`,
so the on/off toggle, the per-mouse subtitles, and the `behavior_class` TTL rules work
identically regardless of backend.

## Dataflow

```
camera -> LiveInferenceWorker.result_ready (LiveDetectionResult: masks + pose + ids)
       -> MainWindow._on_live_detection_result
            -> LiveBehaviorWorker.submit_result(result)        [non-blocking queue]
                 (worker thread) OnlineFeatureExtractor -> EmbTCN-AT forward (both mice)
                 -> causal post-proc -> scene reduction (OR/MAX over mice)
            -> behavior_ready(BehaviorFrameState)
       -> MainWindow._on_behavior_state
            -> LiveRuleEngine.set_behavior_state(active, probs)
            -> _apply_live_rule_evaluation -> behavior_class rule fires
            -> ArduinoOutputWorker.start_live_output_pulse_train / set_live_output_level
```

The model runs entirely on a **background QThread** (`LiveBehaviorWorker`); the GUI never
blocks. The worker keeps the temporal buffer **contiguous** (it pushes every frame) and
**coalesces scoring** when it falls behind, so the decision rate degrades gracefully under
load while the 480-frame window never develops gaps.

## What was added / changed

New (package `pykaboo_live_behavior/`, all pre-existing except the worker + this doc):
- `pykaboo_behavior_worker.py` - `LiveBehaviorWorker(QThread)` + `BehaviorFrameState` +
  `read_checkpoint_labels`. Owns the engine, runs it off-thread, emits scene state.
- `smoke_test.py`, `profile_extractor.py` - synthetic validation / profiling.
- `live_features.py` - added a **parity-exact `mask_iou` cache** (the second per-frame
  hotspot): `compute_windows` over a 480-frame buffer dropped from ~527 ms to ~288 ms,
  byte-identical output.
- `live_engine.py` - split `on_detection` into `push_only` + `score_latest` so the worker
  can push every frame for contiguity but score only the newest.

New / changed (pykaboo app root):
- `live_behavior_integration.py` (new) - import shim exposing the worker behind
  `BEHAVIOR_AVAILABLE`; degrades gracefully if torch / the model are missing. Imports
  torch in the correct DLL order (matches `main.py`).
- `live_detection_types.py` - `LiveTriggerRule.behavior_name` field (+ (de)serialize).
- `live_detection_logic.py` - `LiveRuleEngine.set_behavior_state()` / `behavior_state()`,
  a `behavior_class` case in `_rule_truth`, behavior-aware `evaluate(result=None)`, and a
  `build_rule_label` branch.
- `main_window_enhanced.py` - lazy worker lifecycle (`_ensure/_start/_stop/_sync_behavior_worker`),
  `_on_behavior_state` / `_on_behavior_labels` / `_on_behavior_status` slots, result
  submission, and shutdown.
- `live_detection_panel.py` - "Add Behavior Rule" builder section (behavior class combo,
  output, mode, pulse params, activation) + a live behavior status label.

## How to use (in the app)

1. Start the camera preview, then **Start Live Inference**. Set **expected mice = 2** and
   (recommended) **keypoint source = Mask geometry** so the 8-keypoint order matches the
   model.
2. **Pick the detector**: in the "Add Behavior Rule" section, set **Detector** to
   *Rule-based (fast)* for real-time closed loop (default) or *ML model (EmbTCN)*.
3. **Run it + see subtitles**: tick the **Behavior** overlay checkbox (next to
   Masks / Boxes / Keypoints). This starts the behavior worker even without any TTL rule
   and draws a per-mouse subtitle chip on the preview, e.g. `1: chasing (88%)` /
   `2: following (61%)` (per-mouse = directional rules / the model's two directed tracks;
   the chip shows each mouse's top class). Untick to stop and hide subtitles.
3. **Closed-loop TTL**: open the rules group -> **Add Behavior Rule**: pick a behavior
   (e.g. `mounting`), an output (`DO1`), a mode (Gate / Level / Pulse) and pulse
   parameters, then **Add**. A behavior rule also auto-starts the worker.
4. The behavior worker loads the model on first use (~16 s warm-up to fill the 480-frame /
   16 s window). The status line shows the active behavior + per-frame latency.
5. When the behavior crosses threshold (debounced), the rule fires the TTL exactly like the
   ROI / proximity / contact rules.

The worker runs whenever the **Behavior** toggle is on OR a `behavior_class` rule exists.
The toggle state persists in QSettings (`live_show_behavior`).

The per-mouse subtitle chips are also burned into the **recorded overlay video** ("Rec MP4")
when the Behavior overlay is on (shared renderer `draw_behavior_subtitle_bgr` in
`live_overlay_quality.py`, used by both the live preview and `overlay_video_export.py`).

The default checkpoint is
`pykaboo_live_behavior/checkpoints/free_embtcn_attention_optimized.pt`
(7 classes: background, nose-to-nose, nose-to-body, anogenital, passive, rearing, mounting).

## Latency reality (important)

- Model forward (both mice, GPU): ~5-40 ms. Feature extraction over the 480-frame buffer:
  **~290 ms/score** after the `mask_iou` cache (was ~530 ms). So the closed-loop decision
  rate on this GPU is ~3 Hz, not 30 Hz. The worker coalescing keeps the buffer contiguous;
  only the decision cadence drops.
- The shipped model is **bidirectional (`causal=False`)** with a 16 s window. The engine
  reads the decision at `T-1-L` (look-ahead `L`, default 8 frames). Onset-to-trigger floor
  is roughly `L/fps + (min_bout-1)/fps + smooth_win/2/fps` (~0.8 s) **plus** the ~0.3 s
  feature compute. Budget ~1-1.5 s of onset latency for second-scale social behaviors.
- Path to true 30 fps real-time (future work): an **incremental feature extractor** (only
  recompute the tail rows that change instead of the whole buffer each frame), and/or a
  **`causal=True` retrained checkpoint** (the flag is already plumbed) to drop the
  look-ahead and the bidirectional right-context dependence.

## Parity caveats (online vs offline)

Documented in `research/A_..D_*.md`. The online extractor reuses the exact offline feature
functions (vendored `behavior_segmentation`), so most of the 432 channels are byte-exact.
The known drift channels are: the slow wavelet scales at the right edge (covered by the
look-ahead `L`), `area_norm` (whole-video vs buffer median), and `pair_contact_flag`
(median threshold). Normalization is plain global z-score from the checkpoint (streamable).

## Tests

```
# fast, no GPU (rule-engine behavior_class logic):
python -m pytest tests/test_live_detection_logic.py -q

# headless end-to-end closed loop (needs torch + GPU + checkpoint; auto-skips otherwise):
python -m pytest tests/test_live_behavior_worker.py -q

# synthetic engine smoke + latency (standalone):
python pykaboo_live_behavior/smoke_test.py --device cuda --min-window 240

# offline/online parity on a real recording (needs a COCO masks JSON + DLC pose CSV):
python pykaboo_live_behavior/replay_parity_test.py --coco <masks.json> --pose <pose.csv>
```
