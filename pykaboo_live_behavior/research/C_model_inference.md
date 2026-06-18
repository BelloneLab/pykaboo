# C. Model forward, windowed inference, postproc causality, checkpoint rebuild, latency

Research notes for the pykaboo live-inference engine. Source repo:
`/home/andry/tracking_project/unsupervised_mask_behavior` (package `behavior-segmentation`).
Read-only investigation; nothing in the repo was modified.

Target model for closed-loop: the free-interaction **EmbTCN-Attention** checkpoint
`outputs/free_social/free_embtcn_attention_optimized.pt`
(`architecture: "embtcn_attention"`, 432 features, 7 classes).

TL;DR of the parity / latency threats (full detail below):

- The encoder is **fully bidirectional** and the conv blocks use **centered (non-causal)
  padding** (`causal: False` in the saved config). Frame `t` legitimately reads future
  frames inside its window. End-aligning the target frame (zero-latency live) is NOT the
  same computation as the training/offline center placement. On unstructured input I
  measured an end-vs-center mean abs prob difference of ~0.15 and up to ~0.30 for single
  behaviors (worst case). On real autocorrelated features it will be smaller, but this is
  the single biggest online/offline parity risk and must be measured on real data.
- Offline `predict_track_probs` does **overlap averaging** of per-window probabilities,
  not center-crop. Empirically the averaged offline value tracks the center-of-window
  value closely (MAE ~0.039) and the end-aligned value poorly (MAE ~0.12). So the
  "honest" live approximation of offline is **center placement with ~win/2 latency
  (~8 s)**, which is unacceptable for closed loop. The practical compromise is end-align
  with a short trailing context and accept a calibrated accuracy drop.
- The normalizer in the real checkpoint is **plain global z-score** (no per-video
  median/IQR or CORAL stats), so normalization is a pure per-channel `(x-mean)/std`. This
  is the one big thing that is genuinely streamable with zero look-ahead.
- Postproc as shipped is **non-causal**: `smooth_probs` is a centered moving average,
  `enforce_min_bout` and `merge_short_gaps_binary` both scan whole runs. All three need
  causal redesign (formulas given in §3).
- Latency budget at 30 fps is 33.3 ms/frame. One full `[1,432,480]` forward is ~59 ms on
  CPU and ~2.4 ms on CUDA. CPU cannot keep up frame-by-frame; you need CUDA, or score on
  a stride, or shrink the window. 2.52 M params.

---

## 1. Model: forward signature, config, output dataclass, shapes, causality

File: `src/behavior_segmentation/models/embtcn_attention.py`.

### 1.1 `EmbTCNConfig` (dataclass, lines 33-69)

```python
@dataclass
class EmbTCNConfig:
    num_features: int = 250          # D, input feature channels
    num_classes: int = 17            # K, behaviors for the optional supervised head
    d_model: int = 160
    embedding_dim: int = 96          # E
    tcn_dilations: tuple = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    num_encoder_layers: int = 4
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.15
    temperature: float = 0.5         # tau for temporal attention
    causal: bool = False             # offline -> non-causal (bidirectional)
    max_len: int = 4096
    use_supervised_head: bool = True
    use_fault_head: bool = True
    use_decoder: bool = True         # SSL reconstruction head
    multilabel: bool = True
    num_refinement_stages: int = 0
    refinement_hidden: int = 64
    refinement_dilations: tuple = (1, 2, 4, 8)
```

`from_dict` (lines 60-66) keeps only known fields and re-tuples `tcn_dilations`;
`to_dict` (lines 68-69) is `dataclasses.asdict`. **NOTE `from_dict` does NOT re-tuple
`refinement_dilations`**; after loading it is a list, which is fine because
`RefinementStage` wraps it with `tuple(...)` (line 267).

**Actual saved `model_config` for the real EmbTCN-AT checkpoint** (verified by loading
`outputs/free_social/free_embtcn_attention_optimized.pt`):

```python
{'num_features': 432, 'num_classes': 7, 'd_model': 160, 'embedding_dim': 96,
 'tcn_dilations': (1, 2, 4, 8, 16, 32), 'kernel_size': 5, 'num_encoder_layers': 4,
 'num_heads': 8, 'ffn_mult': 4, 'dropout': 0.3, 'temperature': 0.5, 'causal': False,
 'max_len': 4096, 'use_supervised_head': True, 'use_fault_head': False,
 'use_decoder': True, 'multilabel': True, 'num_refinement_stages': 1,
 'refinement_hidden': 64, 'refinement_dilations': (1, 2, 4, 8)}
```

So the deployed model has **1 refinement stage**, fault head OFF, decoder ON (only used
for SSL; harmless at inference but it does run, see §6 for skipping it).

### 1.2 `EmbTCNAttention.forward` (lines 307-343)

```python
def forward(
    self,
    x: torch.Tensor,                       # [B, D, T]
    mask: torch.Tensor | None = None,      # [B, T] bool, SSL span mask (NOT needed for inference)
    padding_mask: torch.Tensor | None = None,  # [B, T] bool, True = padded (NOT needed for B=1 full window)
) -> EmbTCNOutput:
```

Flow: `encode()` -> `[B,E,T]`; then optionally decoder reconstruction `[B,D,T]`, fault
`[B,T]`, and the supervised head `cls_head` -> `[B,K,T]` followed by `num_refinement_stages`
refiner passes (line 334-340). `out.logits` is the **final refined** stage,
`out.stages` is the list of all stages, `out.probabilities = sigmoid(out.logits)`
(line 342).

### 1.3 `EmbTCNOutput` (lines 72-87)

```python
@dataclass
class EmbTCNOutput:
    embeddings: torch.Tensor                      # [B, E, T]
    reconstruction: torch.Tensor | None = None    # [B, D, T]
    fault: torch.Tensor | None = None             # [B, T]
    logits: torch.Tensor | None = None            # [B, K, T]
    probabilities: torch.Tensor | None = None     # [B, K, T] = sigmoid(logits)
    temporal_weights: torch.Tensor | None = None  # [B, T]
    channel_gate: torch.Tensor | None = None      # [B, D_model]
    stages: list | None = None

    @property
    def stage_logits(self) -> list[torch.Tensor]:
        if self.stages: return self.stages
        return [self.logits] if self.logits is not None else []
```

Inference only needs `out.probabilities` (`[1,K,T]`) and, if you want UMAP/debug,
`out.embeddings` (`[1,E,T]`).

### 1.4 Shapes (verified by running)

Input `[1, 432, 480]` -> `out.probabilities (1,7,480)`, `out.logits (1,7,480)`,
`out.embeddings (1,96,480)`, `out.stages len = 2` (base + 1 refiner). Convention is
channel-major `[B, D, T] -> [B, K, T]`. `forward_time_major` (lines 345-353) accepts
`[B,T,D]` and just transposes; the pipeline always feeds `[B,D,T]`.

### 1.5 Does inference REQUIRE `padding_mask`? NO.

`predict_track_probs` calls `model(x)` with **only `x`** (line 182). `mask` and
`padding_mask` default to `None`. With `padding_mask=None` the DBA pools over all frames
and the encoder gets `src_key_padding_mask=None`. For live B=1 inference on a fully valid
window you pass nothing. You would only need `padding_mask` if you batch ragged windows
or feed a partially-filled warmup window and want to exclude the empty slots; for live,
just feed a full-length window of real frames (warmup handling in §2.4).

### 1.6 Is the encoder BIDIRECTIONAL over the window? YES (this is the core causality fact)

Two independent sources of future-looking dependence inside the window:

1. **TransformerEncoder** is built with no causal attention mask (line 295:
   `self.encoder(h, src_key_padding_mask=padding_mask)` — `src_mask` is never passed).
   So self-attention is fully bidirectional: frame `t` attends to every other frame in
   the window, including future ones.
2. **Dilated conv blocks** use centered padding when `causal=False`
   (line 106: `pad = (kernel_size - 1) // 2 * dilation`), so each `DilatedResidualBlock`
   reads symmetric past+future context. The `causal=True` branch (lines 115-119) trims
   right padding to make it past-only, but the saved config has `causal=False`.
3. The single **RefinementStage** (lines 204-227) also uses
   `DilatedResidualBlock(..., causal=False)` (line 218), so refinement adds another
   centered receptive field on top of the logits (kernel 3, dilations (1,2,4,8) ->
   half-width 15 frames each side).

Implication for live: placing the target frame at the window **END** (causal, zero
look-ahead) is a different computation from placing it near the **CENTER** (as effectively
happens during training and offline overlap averaging). See §2.5 for the measured impact.

### 1.7 DBA caveat for variable windows (parity subtlety)

`DualBranchAttention.forward` (lines 175-201) computes a softmax-normalized temporal
attention `alpha` over the **whole window** and an SE channel gate from the **window mean**
(line 191: `pooled = h.mean(dim=1)`). Both are window-global. Therefore the embedding (and
hence logits) at frame `t` depends on the statistics of the entire window. If you change
the window length or contents in live mode, `t`'s output shifts even without new
information at `t`. Keep the window length fixed at the trained 480 frames for parity.

---

## 2. `predict_track_probs`: windowing, stitching, and the live implication

File: `src/behavior_segmentation/social_train.py`, lines 157-194 (quoted fully):

```python
def predict_track_probs(
    model: TemporalTcnModel,
    track,
    normalizer: FeatureNormalizer,
    window_frames: int,
    stride_frames: int,
    device: str,
    want_embeddings: bool = False,
):
    """Sliding-window multilabel inference -> probs ``[K, T]`` (+ optional emb)."""
    model.eval()
    feats = normalizer.transform(track.features).astype(np.float32)
    T = feats.shape[0]
    K = model.num_classes
    prob_sum = np.zeros((K, T), dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)
    emb_sum = None
    starts = list(range(0, max(T - window_frames, 0) + 1, max(stride_frames, 1)))
    if not starts or starts[-1] + window_frames < T:
        starts.append(max(T - window_frames, 0))
    for s in starts:
        e = min(s + window_frames, T)
        chunk = feats[s:e]
        x = torch.from_numpy(chunk.T[None]).to(device)
        out = model(x)
        p = out.probabilities[0].cpu().numpy()  # [K, e-s]
        prob_sum[:, s:e] += p
        counts[s:e] += 1.0
        if want_embeddings:
            emb = out.embeddings[0].cpu().numpy()
            if emb_sum is None:
                emb_sum = np.zeros((emb.shape[0], T), dtype=np.float64)
            emb_sum[:, s:e] += emb
    counts = np.clip(counts, 1.0, None)
    probs = prob_sum / counts
    embeddings = (emb_sum / counts) if emb_sum is not None else None
    return probs, embeddings
```

### 2.1 Tiling

`starts = range(0, max(T-win,0)+1, stride)` then **always appends a final
`max(T-win,0)` start** if the last window does not reach `T`. With the real checkpoint
`win = seconds_to_frames(16.0, 30) = 480`, `stride = seconds_to_frames(4.0, 30) = 120`
(`seconds_to_frames` at `windows.py:29-30` is `max(round(seconds*fps), 1)`). So windows
are `[0..480), [120..600), [240..720), ...` plus a tail-aligned `[T-480..T)`.
Short tracks (`T < win`) yield a single window `[0..T)` (so the model can run on any
length; the conv/attention are length-agnostic).

### 2.2 Stitching: OVERLAP AVERAGING (not center-crop, not last-value)

Each frame accumulates the probabilities from **every window that covers it**
(`prob_sum[:, s:e] += p`, `counts[s:e] += 1`) and the final per-frame value is the
arithmetic mean (`probs = prob_sum / counts`). A frame in the interior is covered by
`win/stride = 480/120 = 4` windows, at offsets {0, 120, 240, 360} within those windows.
So the offline value at frame `t` is the average of that frame scored at 4 different
window positions, ranging from near-start to near-center. Embeddings are averaged the
same way.

### 2.3 Live sliding-window equivalent

To score the newest frame `t` live you must run the model on a window of `win=480`
frames and read out one column. Two placements:

- **Center placement (offline-faithful):** window `[t-240 .. t+239]`, read column 240.
  Requires 239 future frames -> **latency ~win/2 = 240 frames = 8.0 s @30fps**. This is
  the closest match to training/offline but is unusable for closed-loop optogenetics.
- **End placement (causal live):** window `[t-479 .. t]`, read the **last** column.
  Zero look-ahead -> **latency ~0** (plus compute time). But the model never saw frame
  `t` at the window end during training, and the bidirectional encoder/centered conv lose
  the right-context they normally use, so outputs differ (see §2.5).

Offline averaging mixes positions but is dominated by center-ish placements; it is NOT
end-placement. So you cannot reproduce offline exactly in real time without buffering the
future.

### 2.4 Warmup

For the first `win-1` frames you do not yet have 480 past frames. Options:
(a) wait until `win` frames are buffered before emitting (cold start ~16 s); or
(b) run on the partial buffer (`T<win` is supported, single window `[0..T)`) and read the
last column, accepting that short-window stats differ (§2.5 stability table shows
W=120 already within MAE ~0.09 of W=480-end on noise). Recommendation: emit nothing for
the first ~window/4 frames (a few seconds), then run end-aligned on the growing buffer
capped at 480.

### 2.5 Measured divergence (empirical, this model, synthetic autocorrelated input)

I loaded the real checkpoint and scored the same true frame `t=600` three ways
(center column of a centered window, last column of an end-aligned window, and the offline
overlap-averaged value). Mean absolute probability differences:

| comparison | MAE over 7 classes |
|---|---|
| offline-vs-center | **0.039** |
| offline-vs-end    | **0.123** |
| center-vs-end     | **0.146** |

Per-class center-vs-end gaps reached 0.30 (anogenital 0.297, passive 0.312). This was on
unstructured cumulative-noise features (deliberate worst case); on real smooth pose/mask
features the gap will be smaller, but the ordering is robust: **offline ~= center, and end
is the outlier.** This is the parity threat to flag loudest.

End-aligned stability vs window length (same true frame at the last column):

| W (frames) | MAE vs W=480-end |
|---|---|
| 360 | 0.026 |
| 240 | 0.083 |
| 180 | 0.109 |
| 120 | 0.091 |
| 60  | 0.093 |

So shrinking the window below ~360 noticeably changes end-aligned outputs; keep W>=360
(ideally the trained 480).

### 2.6 Recommended live strategy + how to measure the drop

**Recommended:** a small fixed look-ahead, NOT pure end-alignment and NOT full center.
Buffer `L` future frames (latency `L/fps`), feed window `[t-(win-1-L) .. t+L]`, read
column `win-1-L`. With `L = 0` you get end placement (0 latency, max drop); with
`L = 239` you get center (8 s latency, min drop). Pick `L` to trade latency for accuracy
based on the closed-loop deadline (e.g. `L=30` -> 1 s latency often recovers most of the
right-context benefit because the centered conv half-width plus refiner is ~15+15 frames
and attention decays).

**How to measure the drop (do this on REAL pykaboo features before deploying):**
1. Run the offline pipeline (`run_social_inference`) on a held-out recording -> reference
   `probs_offline[K,T]` and binarized `labels_offline[K,T]`.
2. Re-run the model in the live emulation (end or `L`-lookahead, fixed window, causal
   postproc from §3) -> `probs_live[K,T]`.
3. Report per-class MAE `|probs_live - probs_offline|`, agreement of binarized labels
   (Jaccard / per-frame accuracy), and bout-level F1 (use `bout_f1` /
   `intervals_from_binary` from `social_train.py`, IoU 0.5). Sweep `L` to plot
   latency-vs-agreement and pick the operating point.

---

## 3. Postprocessing causality and causal online equivalents

The offline order in `infer_track` (`social_infer.py` lines 308-316):
`predict_track_probs -> smooth_probs(sp, smooth_win) -> threshold ->
merge_short_gaps_binary(merge_gap_frames) -> enforce_min_bout(min_bout_frames)`.

Real-checkpoint params (`optimized.pt`): `smooth_win=9`, `min_bout_frames=20`,
`merge_gap_frames=30`. Thresholds (per class, after softmax/sigmoid):
`[0.60, 0.65, 0.65, 0.70, 0.60, 0.60, 0.60]` for
`[background, nose-to-nose, nose-to-body, anogenital, passive, rearing, mounting]`.
(The non-optimized `free_embtcn_attention.pt` uses smooth_win=15, min_bout=12, merge_gap=6,
different thresholds — always read these from the checkpoint, never hardcode.)

### 3.1 `smooth_probs` — CENTERED moving average (NEEDS future)

`social_train.py` lines 224-231:

```python
def smooth_probs(probs: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return probs
    kernel = np.ones(win) / win
    out = np.empty_like(probs)
    for k in range(probs.shape[0]):
        out[k] = np.convolve(probs[k], kernel, mode="same")
    return out
```

`np.convolve(..., mode="same")` is a **centered** box filter of width `win`: output `t` =
mean of `probs[t-(win//2) .. t+(win-1-win//2)]`. With win=9 that is `t-4..t+4`, i.e. it
reads **4 future frames**. (Edge note: `mode="same"` zero-pads at the boundaries, so the
first/last `win//2` frames are biased toward 0; a live trailing filter avoids that
boundary bias entirely.)

**Causal online equivalent — pick one:**

- Trailing box (matches the centered filter up to a `win//2` shift):
  `s_t[k] = mean(prob[k, t-win+1 .. t])`. Implement with a length-`win` ring buffer per
  class and a running sum: `running_sum += p_t - p_{t-win}; s_t = running_sum / min(t+1, win)`.
  This is the most faithful causal analogue; it lags the centered version by ~`win//2`
  frames (~4 frames = 133 ms at win=9).
- EMA (cheaper, smoother, configurable lag): `s_t = a*p_t + (1-a)*s_{t-1}`. To roughly
  match a box of width `win` set `a = 2/(win+1)` (so win=9 -> a=0.2). EMA has effective
  delay ~`(win-1)/2` like the box but no hard buffer.

Recommendation: trailing box with running sum (exact-width, deterministic, O(1)/frame).

### 3.2 `merge_short_gaps_binary` — bridges short OFF gaps (NEEDS future)

`social_train.py` lines 257-281:

```python
def merge_short_gaps_binary(binary, max_gap_frames):
    if max_gap_frames <= 0:
        return binary
    out = binary.copy()
    for k in range(out.shape[0]):
        row = out[k]; idx = 0; n = len(row)
        while idx < n:
            if row[idx] == 0:
                j = idx
                while j < n and row[j] == 0:
                    j += 1
                gap = j - idx
                bounded_by_positive = (idx > 0 and j < n and row[idx-1] == 1 and row[j] == 1)
                if bounded_by_positive and gap <= max_gap_frames:
                    row[idx:j] = 1
                idx = j
            else:
                idx += 1
    return out
```

Fills a 0-run with 1s **only if it is bounded by 1 on both sides and length <= max_gap**.
The right boundary (`row[j]==1`) is in the future, so this is non-causal: it cannot decide
to fill a gap until the behavior resumes.

**Causal online equivalent (debounce / hold-off after offset):** when a class turns OFF,
do NOT immediately emit 0; keep emitting 1 for up to `max_gap` frames. If the class turns
back ON within that window, the gap is retroactively bridged (consistent with offline). If
it stays OFF past `max_gap`, the offset is committed (and you can backfill the held frames
as 1, matching offline, at the cost of `max_gap` latency on the OFF transition).
Per-class state machine:

```
on entering OFF at frame g (was ON at g-1):
    hold_counter = 0
    while still OFF:
        hold_counter += 1
        emit 1            # provisional hold
        if turns ON again (within max_gap):  # gap <= max_gap and bounded both sides -> matches offline fill
            keep ON; reset
        elif hold_counter > max_gap:
            commit OFF starting at frame g   # (frames g..now stay 1 only if you backfilled; for strictly causal emit, see note)
```

Note the tension: offline fills the whole gap as 1; a strictly-causal stream that already
emitted 0 cannot retroactively change it. To match offline you must **delay the OFF
decision by `max_gap` frames** (emit the held 1s, then if it stays off, the trailing
`max_gap` frames were correctly 1; if it turns on, they stay 1). This adds `max_gap`
(=30 frames = 1.0 s) latency to OFF transitions only. For closed loop where you typically
trigger on ONSET, this OFF-latency is usually acceptable.

### 3.3 `enforce_min_bout` — removes short ON runs (NEEDS future)

`social_train.py` lines 234-254:

```python
def enforce_min_bout(binary, min_frames):
    if min_frames <= 1:
        return binary
    out = binary.copy()
    for k in range(out.shape[0]):
        row = out[k]; idx = 0; n = len(row)
        while idx < n:
            if row[idx] == 1:
                j = idx
                while j < n and row[j] == 1:
                    j += 1
                if j - idx < min_frames:
                    row[idx:j] = 0
                idx = j
            else:
                idx += 1
    return out
```

Deletes any positive run shorter than `min_frames`. Non-causal: you cannot know a run's
length until it ends.

**Causal online equivalent (onset-confirmation delay):** a behavior is only *emitted* as
ON after it has persisted `min_frames` consecutive frames. Per-class counter:

```
if raw_t == 1:
    run_len += 1
    emit (run_len >= min_frames)     # withhold ON until confirmed
else:
    run_len = 0
    emit 0
```

This **exactly** reproduces `enforce_min_bout` for the streaming case (short runs are
suppressed because they never reach `min_frames`; long runs become ON at frame
`onset + min_frames - 1`). **Added latency = min_frames - 1 = 19 frames = 0.63 s** on
every ON transition (with `min_bout_frames=20`). This is the dominant onset latency of the
whole pipeline and must be budgeted in the closed-loop timing. If 0.63 s is too long for
the manipulation, lower `min_bout_frames` (a deploy-time knob) and re-measure F1.

### 3.4 Ordering and combined onset latency

Offline order is smooth -> threshold -> merge_gap -> min_bout. Causal stream order should
be: trailing-smooth (lag ~`win//2`=4) -> threshold (per-class from checkpoint) -> onset
min-bout confirm (lag `min_bout-1`=19) -> optional gap debounce on offset (lag up to
`merge_gap`=30 on OFF only). **Total ONSET latency ~= 4 + 19 = 23 frames ~= 0.77 s**
beyond the model's own window latency choice (§2.6). Spell this out to the closed-loop
designer: the floor on "behavior X started -> trigger" is roughly `0.77 s + L/fps`.

---

## 4. Scene-level reduction (combine the two identity tracks)

The live engine must build **two** per-identity tracks (mouse1 as subject with mouse2 as
object, and vice versa), score each, then reduce to a scene ethogram. Two reductions are
used and they must both be supported:

### 4.1 Probability reduction = MAX across identities

`experiments/free_fused_rule_check.py` line 53:
```python
scene_prob = np.maximum(scene_prob, p.probabilities[:, :T])
```
Per class, per frame: `scene_prob[k,t] = max over identities of track_prob[k,t]`.

### 4.2 Binary reduction = OR across identities

`free_fused_rule_check.py` line 52: `scene_lab |= p.labels[:, :T].astype(np.int8)`.
And the training/eval scene report does the same — `scene_level_report`
(`social_train.py` lines 621-651):

```python
def scene_level_report(model, tracks, normalizer, win, stride, device, thr, label_map, tc):
    by_video = {}
    for t in tracks:
        by_video.setdefault(t.video_id, []).append(t)
    scene_preds, scene_tgts = [], []
    for vid, ts in by_video.items():
        n = min(t.labels.shape[1] for t in ts)
        pred_or = np.zeros((label_map.num_classes, n), dtype=np.int8)
        tgt_or  = np.zeros((label_map.num_classes, n), dtype=np.int8)
        for t in ts:
            probs, _ = predict_track_probs(model, t, normalizer, win, stride, device)
            sp = smooth_probs(probs[:, :n], tc.smooth_win)
            pred = (sp >= thr[:, None]).astype(np.int8)
            pred = merge_short_gaps_binary(pred, tc.merge_gap_frames)
            pred = enforce_min_bout(pred, tc.min_bout_frames)
            pred_or |= pred
            tgt_or  |= t.labels[:, :n].astype(np.int8)
        scene_preds.append(pred_or)
        scene_tgts.append(tgt_or)
    return per_behavior_report(scene_preds, scene_tgts, thr, label_map, 0,
                               smooth_win=1, min_bout=1, merge_gap=0)
```

Key points for the live engine:
- The per-track binary is produced **independently** (own smooth/threshold/merge/min-bout),
  THEN OR-ed. So in live mode: run the full causal postproc (§3) per identity, then OR the
  two binaries and MAX the two prob vectors each frame.
- `per_behavior_report` is called with `smooth_win=1, min_bout=1, merge_gap=0` on the
  already-postprocessed scene preds, i.e. no double smoothing. (`weighted_macro` at lines
  382-387 then computes support-weighted and macro F1; `per_behavior_report` lines 284-335
  computes per-class framewise P/R/F1 plus `bout_f1` at IoU 0.5. These are metrics-only,
  not needed at deploy time, but reuse them to validate live-vs-offline.)
- `background` is excluded from metrics (`per_behavior_report` skips `k == bg`), but
  background_id resolves to 0 here because the saved label map has
  `background_label="background"` at index 0 (`LabelMap.background_id` = `name_to_id.get("background", 0)`,
  `labels.py:78-79`). For closed-loop you only care about the 6 real behaviors anyway.

`_collect` (lines 609-618) is just the per-track gather used for metrics; it truncates
probs/targets to `min(probs.shape[1], labels.shape[1])`. Not needed live (no labels), but
note the length-truncation habit: always align the two tracks to a common `T` before
OR/MAX (`T = min(p.labels.shape[1] for p in preds)`), since the two identity tracks can
differ in frame count.

### 4.3 Labels and order (class index -> name)

7 classes in this fixed order: `['background', 'nose-to-nose', 'nose-to-body',
'anogenital', 'passive', 'rearing', 'mounting']` (index = class id). Build a
`name_to_k = {n:i for i,n in enumerate(label_map.names)}` for trigger logic.

---

## 5. Standalone checkpoint rebuild (no free_train / social_infer imports)

The checkpoint is a plain `torch.save` dict (top-level keys verified):
`architecture, model_config, feature_names, label_map, normalizer, thresholds,
frame_rate, window_seconds, eval_stride_seconds, smooth_win, min_bout_frames,
merge_gap_frames, metadata, model_state`. `model_state` is a normal `state_dict` that
`EmbTCNAttention(EmbTCNConfig.from_dict(model_config)).load_state_dict(...)` loads with
**zero missing/unexpected keys** (verified). The only dependency is the model module
(`embtcn_attention.py`) + torch + numpy. You do **not** need `free_train`, `social_infer`,
`config.py`, or the feature pipeline to rebuild and run the network.

`load_social_checkpoint` (`social_infer.py` lines 251-284) and
`SocialCheckpoint.build_model` (lines 50-71) show the canonical path; the relevant branch
is just:

```python
m = EmbTCNAttention(EmbTCNConfig.from_dict(self.model_config)).to(device)
m.load_state_dict(self.model_state)
m.eval()
```

`load_social_checkpoint` uses `torch.load(path, map_location="cpu", weights_only=False)`
(line 252) — you need `weights_only=False` because the payload contains a non-tensor
LabelMap dict and numpy lists (actually all numpy is serialized to lists/dicts, so it is
safe; but the loader uses `weights_only=False`). Minimal self-contained loader for the
live package (copy `embtcn_attention.py` into your package, then):

```python
import torch, numpy as np
from your_pkg.embtcn_attention import EmbTCNAttention, EmbTCNConfig

class LiveModel:
    def __init__(self, ckpt_path, device="cuda"):
        p = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = EmbTCNConfig.from_dict(p["model_config"])  # 432/7/d_model 160/...
        self.model = EmbTCNAttention(self.cfg).to(device).eval()
        self.model.load_state_dict(p["model_state"])          # strict=True works (0 missing)
        # everything the streaming engine needs:
        self.feature_names   = list(p["feature_names"])       # 432 names, exact order
        self.labels          = list(p["label_map"]["names"])  # 7 class names
        self.thresholds      = np.asarray(p["thresholds"], dtype=np.float64)  # [7]
        self.mean            = np.asarray(p["normalizer"]["mean"], dtype=np.float64)  # [432]
        self.std             = np.asarray(p["normalizer"]["std"],  dtype=np.float64)  # [432]
        self.frame_rate      = float(p["frame_rate"])         # ~30.0
        self.win             = max(round(p["window_seconds"]      * self.frame_rate), 1)  # 480
        self.eval_stride     = max(round(p["eval_stride_seconds"] * self.frame_rate), 1)  # 120
        self.smooth_win      = int(p["smooth_win"])           # 9
        self.min_bout_frames = int(p["min_bout_frames"])      # 20
        self.merge_gap_frames= int(p["merge_gap_frames"])     # 30
        self.device = device

    @torch.no_grad()
    def probs(self, feats_TxD):      # feats already z-scored, [T, 432]
        x = torch.from_numpy(np.ascontiguousarray(feats_TxD.T[None], dtype=np.float32)).to(self.device)
        return self.model(x).probabilities[0].cpu().numpy()   # [7, T]

    def normalize(self, raw_TxD):    # plain global z-score (matches FeatureNormalizer.transform)
        out = (raw_TxD.astype(np.float64) - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

`EmbTCNConfig.from_dict` / `to_dict` are quoted in §1.1. Caveats:
- `seconds_to_frames` (the repo helper) is `max(int(round(seconds*fps)),1)`. Reproduce it
  exactly (note `frame_rate` in the checkpoint is `30.000000000001705`, so
  `round(16.0*30.000...) = 480`, `round(4.0*30.000...) = 120` — fine).
- The **normalizer in the real checkpoint has no per-video stats**
  (`video_median`/`video_mean` are empty/None), so `FeatureNormalizer.transform` reduces
  to `nan_to_num((x - mean)/std)` (`normalization.py` line 134-135). The `normalize()`
  above is byte-for-byte equivalent. **If you ever deploy a checkpoint whose normalizer
  DOES carry per-video median/IQR or CORAL mean/std, parity breaks**: that path computes
  test-time per-video statistics over the WHOLE clip (`transform` lines 105-133), which is
  inherently non-streaming. Assert at load time that `video_median`/`video_iqr`/
  `video_mean`/`video_std` are empty.
- You can skip the decoder/fault compute to save time only by editing the model; with the
  saved config `use_fault_head=False` already (fault not computed), but `use_decoder=True`
  so the reconstruction MLP runs every forward. It is cheap relative to the encoder, but
  if you want to drop it, set `cfg.use_decoder=False` BEFORE constructing the model and
  load with `strict=False` (the decoder weights become unexpected keys and are ignored).
  Verified the encoder/cls path does not depend on decoder output.
- `EmbTCNConfig.from_dict` does not coerce `refinement_dilations` to a tuple but
  `RefinementStage` tuples it internally, so it is safe.

The `nemba:embtcn` fused model (`models/free_fused_pose.pt`, 464 feats) needs the
`free_backbones.NembaSupervised` class instead and is a different code path; if pykaboo
targets that one, you must port `NembaSupervised` too. For the EmbTCN-AT optimized model,
only `embtcn_attention.py` is required.

---

## 6. Latency and parameter budget (measured on this box)

Model: 2,515,634 parameters (**2.52 M**). State_dict loads with 0 missing / 0 unexpected
keys.

Single forward, mean of 50 runs after warmup (system python3, this workstation):

| input | CPU | CUDA |
|---|---|---|
| `[1, 432, 480]` (full trained window) | **59.4 ms** | **2.37 ms** |
| `[1, 432, 240]` | 27.5 ms | - |
| `[1, 432, 120]` | 29.4 ms | - |
| `[1, 432, 64]`  | 12.2 ms | - |

30 fps budget = **33.3 ms/frame**.

Implications for the live loop:
- **CPU full-window per-frame inference is over budget** (59 ms > 33 ms) and you have TWO
  identity tracks per frame -> ~120 ms/frame on CPU. Not viable frame-by-frame.
- **CUDA is comfortably in budget**: 2.4 ms/window, ~4.8 ms for both tracks, leaving ~28 ms
  for feature extraction and postproc. Use CUDA for closed loop.
- The forward is essentially **constant per window regardless of how many output columns
  you read** (you always pay for the whole window). So scoring only the newest frame still
  costs a full forward. To keep up on slower hardware, options: (a) run inference every
  `eval_stride`=120 frames and hold the last result (matches offline stride, but adds up to
  4 s of staleness — bad for closed loop); (b) run every N frames with N small (e.g. N=3 ->
  10 Hz decisions, ~20 ms/window on CPU is still too slow, so this only helps on GPU);
  (c) shrink the window to 360 (smallest length with MAE<0.03 vs 480-end per §2.5) — CPU
  ~45 ms, still over budget, GPU fine. Net: **plan for a GPU.**
- There is a sub-linear oddity (W=240 and W=120 both ~28 ms, W=480 ~59 ms): the transformer
  attention is O(T^2) so the 480 window dominates; below ~240 you hit fixed overhead. Do
  not infer that 120 is cheaper than 240; both are ~28 ms.

Memory/throughput tip: batch the two identity windows into one `[2, 432, 480]` forward
(the model is fully batched; DBA/encoder handle B>1) to roughly halve per-frame GPU
overhead vs two separate calls.

---

## Appendix: exact deploy constants for `free_embtcn_attention_optimized.pt`

```
architecture        = "embtcn_attention"
num_features (D)     = 432    (feature_names[0:6] = center_x, center_y, bbox_x, bbox_y,
                              bbox_width, bbox_height ; [-4:] = pp_obj_tail_ego_x,
                              pp_obj_tail_ego_y, pp_contact_flag, pp_body_body_vel)
num_classes (K)      = 7
classes (index order)= background, nose-to-nose, nose-to-body, anogenital, passive,
                       rearing, mounting
d_model              = 160 ; embedding_dim (E) = 96 ; heads = 8 ; encoder layers = 4
tcn_dilations        = (1,2,4,8,16,32) ; kernel_size = 5 ; refinement_stages = 1
causal               = False  (BIDIRECTIONAL - the central live caveat)
frame_rate           = 30.0
window_seconds       = 16.0  -> win = 480 frames
eval_stride_seconds  = 4.0   -> stride = 120 frames (offline overlap = 4x interior coverage)
smooth_win           = 9     (centered box -> trailing box, lag ~4 frames)
min_bout_frames      = 20    (onset confirm delay = 19 frames = 0.63 s)
merge_gap_frames     = 30    (offset debounce hold = up to 30 frames = 1.0 s, OFF only)
thresholds           = [0.60, 0.65, 0.65, 0.70, 0.60, 0.60, 0.60]
normalizer           = plain global z-score (no per-video median/IQR or CORAL stats)
params               = 2.52 M ; forward [1,432,480] = 59 ms CPU / 2.4 ms CUDA
```

(The non-optimized sibling `free_embtcn_attention.pt` shares architecture/features but has
smooth_win=15, min_bout=12, merge_gap=6 and different thresholds — read all constants from
the checkpoint at load time.)
