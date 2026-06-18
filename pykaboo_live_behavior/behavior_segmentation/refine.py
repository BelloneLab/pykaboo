"""Refine a checkpoint by self-training on rule-corrected pseudo-labels.

The idea (user-requested): the model violates logical constraints (predicts
mutually-exclusive behaviors at once) and misses some detections. Apply the rule
pipeline to the model's own predictions on a set of videos to get cleaner labels,
then fine-tune the model on those labels (warm-started from the current weights,
reusing the original feature normalizer so the warm start is valid). Save the
result next to the original as ``<stem>_refined.pt`` in the same format, so it
loads through the same path.

Honest caveat: self-training on pseudo-labels can reinforce the model's own
errors (confirmation bias). The rule correction makes the pseudo-labels logically
consistent, which helps, but this is not a substitute for human labels. The GUI
reports whether the refined model actually changed/improved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .rule_correction import RuleCorrectionConfig, correct_track
from .social_infer import (
    infer_track,
    load_social_checkpoint,
    runtime_config_for_checkpoint,
)

# Pseudo-label rule pipeline: remove impossible combinations (mutex) AND recover
# missed contact detections by lowering their decision threshold (the user flagged
# anogenital / nose-to-nose as under-detected). These cleaner, fuller labels are
# what the model is fine-tuned on.
PSEUDO_LABEL_CONFIG = RuleCorrectionConfig(
    enabled=True,
    mutex=True,
    threshold_overrides={"anogenital": 0.42, "nose-to-nose": 0.42, "nose-to-body": 0.42},
)


def _align_to_feature_names(tracks, feature_names):
    for t in tracks:
        name_to_idx = {n: i for i, n in enumerate(t.feature_names)}
        new = np.zeros((t.features.shape[0], len(feature_names)), dtype=np.float32)
        for j, name in enumerate(feature_names):
            src = name_to_idx.get(name)
            if src is not None:
                new[:, j] = t.features[:, src]
        t.features = new
        t.feature_names = list(feature_names)
    return tracks


def refine_checkpoint(
    checkpoint_path: str | Path,
    video_specs: list[tuple[str, str | None]],
    config,
    epochs: int = 20,
    learning_rate: float = 3e-4,
    device: str = "auto",
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Self-train ``checkpoint_path`` on rule-corrected pseudo-labels of
    ``video_specs`` (list of (coco_json, pose_csv|None)). Returns the path of the
    saved ``<stem>_refined.pt``."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from .dataset import WindowDataset, build_inference_tracks, collate_windows
    from .free_infer import build_free_inference_tracks, is_fused_feature_names
    from .social_pipeline import build_social_features
    from .social_train import (
        focal_bce,
        multilabel_window_weights,
        pos_weights_from_tracks,
    )

    def emit(frac: float, msg: str) -> None:
        if log:
            log(msg)
        if progress:
            progress(frac, msg)

    if device in (None, "auto", "", "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_social_checkpoint(str(checkpoint_path))
    cfg = runtime_config_for_checkpoint(ckpt, config)
    mode = cfg.behavior_roles.interaction_mode
    bg = ckpt.label_map.background_id
    model = ckpt.build_model(device)
    is_fused = str(ckpt.architecture).startswith("nemba:") or is_fused_feature_names(ckpt.feature_names)

    # 1. build rule-corrected pseudo-labeled tracks from the model's own predictions
    pseudo = []
    n_vid = max(len(video_specs), 1)
    changed_total, frame_total = 0, 0
    for i, (coco, pose) in enumerate(video_specs):
        if should_stop and should_stop():
            break
        emit(0.05 + 0.25 * i / n_vid, f"Predicting + rule-correcting {Path(coco).stem} ...")
        if is_fused:
            _, tracks = build_free_inference_tracks(coco, cfg, ckpt.feature_names, pose_path=pose, log=None)
        else:
            feats = build_social_features(coco, cfg, pose_path=pose, use_pose=True, use_wavelets=True, log=None)
            tracks = build_inference_tracks(feats.identity_df, feats.pair_df, ckpt.feature_names, is_pair=True)
            tracks = _align_to_feature_names(tracks, ckpt.feature_names)
        for t in tracks:
            sp = infer_track(ckpt, t, model, device, want_embeddings=False)
            corrected, info = correct_track(
                np.asarray(sp.labels), np.asarray(sp.probabilities),
                ckpt.label_map, PSEUDO_LABEL_CONFIG, interaction_mode=mode, background_id=bg,
            )
            t.labels = corrected.astype(np.int8)
            pseudo.append(t)
            changed_total += int(info["magnitude"] * corrected.shape[1])
            frame_total += corrected.shape[1]
    if not pseudo:
        raise RuntimeError("No tracks to refine on (no usable videos).")
    emit(0.32, f"Rule-corrected {changed_total}/{frame_total} "
               f"({100*changed_total/max(frame_total,1):.1f}%) frames into pseudo-labels.")

    # 2. fine-tune (warm-started; reuse the ORIGINAL normalizer so the warm start is valid)
    ds = WindowDataset(pseudo, 16.0, 2.0, ckpt.frame_rate, ckpt.normalizer, multilabel=True)
    w = multilabel_window_weights(ds, ckpt.label_map.num_classes, bg)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, collate_fn=collate_windows, num_workers=0)
    pos_weight = pos_weights_from_tracks(pseudo, ckpt.label_map.num_classes, 20.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    model.train()
    for ep in range(epochs):
        if should_stop and should_stop():
            break
        tot, nb = 0.0, 0
        for batch in loader:
            feats = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            try:
                out = model(feats, padding_mask=~mask.bool())
            except TypeError:
                out = model(feats)
            loss = focal_bce(out.logits, labels, mask, pos_weight, 2.0)
            for sl in getattr(out, "stage_logits", [])[:-1]:
                loss = loss + 0.5 * focal_bce(sl, labels, mask, pos_weight, 2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        emit(0.35 + 0.55 * (ep + 1) / max(epochs, 1),
             f"fine-tune epoch {ep + 1}/{epochs} loss={tot / max(nb, 1):.4f}")

    # 3. save <stem>_refined.pt next to the original, same format
    model.eval()
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    payload["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta = dict(payload.get("metadata", {}) or {})
    meta["refined_from"] = str(checkpoint_path)
    meta["refined_videos"] = [Path(c).stem for c, _ in video_specs]
    meta["refined_epochs"] = int(epochs)
    payload["metadata"] = meta
    src = Path(checkpoint_path)
    refined = src.with_name(f"{src.stem}_refined{src.suffix}")
    torch.save(payload, refined)
    emit(1.0, f"Saved refined checkpoint -> {refined}")
    return str(refined)
