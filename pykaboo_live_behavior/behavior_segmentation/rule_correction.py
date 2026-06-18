"""Rule-based post-inference correction using logical behavior constraints.

The model predicts each behavior with an independent sigmoid, so it can emit
combinations the biology forbids (e.g. nose-to-body AND nose-to-nose at once: the
nose can only be at one place). The ground-truth co-occurrence matrix (see
experiments/free_data_audit.py) shows which behaviors are mutually exclusive in
practice; this module enforces those constraints on the per-frame predictions and
measures how much correction was needed (a cheap, label-free signal of model
quality on a given video, used to surface a "Refine model" action).

Operates on per-track arrays: ``labels`` and ``probs`` are ``[K, T]`` (K behaviors
incl. a background row, T frames). Pure numpy; no model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mutually-exclusive behavior groups, keyed by interaction mode. Members essentially
# never co-occur in ground truth, so at most one may fire per frame. Resolution:
# keep the highest-probability member, zero the rest. Behaviors absent from the
# model's label map are skipped, so the same registry is safe for any model.
MUTEX_GROUPS: dict[str, list[list[str]]] = {
    # free-interaction model: contact LOCATION is one-hot (nose is at one place);
    # mounting and rearing are postural opposites.
    "classic_free_interaction": [
        ["nose-to-nose", "nose-to-body", "anogenital"],
        ["mounting", "rearing"],
    ],
    # aggression model: attack-family hierarchy is NOT mutually exclusive
    # (attack superset of biting/wrestling), so no mutex groups by default.
    "aggression_cd1_bl6": [],
}


# Contact-geometry rules (experiments/geom_rule_audit.py + validate_contact_gate.py,
# validated vs human GT on 3 videos). Two directions, both keyed by interaction mode.
#
# GATE (precision): suppress a contact behaviour when the relevant raw nose-to-region
# distance is beyond the validated contact range (p95 of the GT-positive distance) -
# the nose is demonstrably NOT at the partner's region. Removes premature / far false
# positives (anogenital while the mice are apart, anogenital on a reared-up mouse) at
# ~87% precision of removed frames. Adopted for anogenital + nose-to-nose; NOT
# nose-to-body (its distance to a large body target does not separate).
#   behaviour -> (raw geometry channel, far-distance threshold)
CONTACT_GATES: dict[str, dict[str, tuple[str, float]]] = {
    "classic_free_interaction": {
        "anogenital": ("pp_nose_tail", 0.335),
        "nose-to-nose": ("pp_nose_nose", 0.369),
    },
    "aggression_cd1_bl6": {},
}

# RECOVER (recall + disambiguation): where a behaviour's geometry is VERY tight,
# geometry is authoritative - assert the behaviour and clear the other triad members.
# Only nose-to-nose qualifies for the SINGLE-channel form: when the two noses are
# touching (<= 0.18) GT is nose-to-nose 75% / nose-to-body 7% / anogenital 0.1%, so
# asserting it and clearing competitors recovers missed nose-to-nose AND fixes
# nose-to-body<->nose-to-nose confusions (nose-to-nose F1 0.693->0.774, nose-to-body
# 0.509->0.544). Recovering anogenital this way with a SINGLE channel was REFUTED
# (floods FPs: the nose is also near the tail during nose-to-body / when merely
# close); use CONTACT_RECOVER_ARGMIN below instead.
#   behaviour -> (raw geometry channel, tight-distance threshold, [members to clear])
CONTACT_RECOVER: dict[str, dict[str, tuple[str, float, list[str]]]] = {
    "classic_free_interaction": {
        "nose-to-nose": ("pp_nose_nose", 0.18, ["nose-to-body", "anogenital"]),
    },
    "aggression_cd1_bl6": {},
}

# ARGMIN RECOVER (precision-safe recall): assert a contact when its nose-to-region
# distance is the SMALLEST of the three contact distances (the nose is at THIS region,
# not just near it) AND within the contact range. This is the disciplined fix for the
# refuted single-channel anogenital recovery: requiring the region to beat its
# competitors removes the false positives that plagued the naive rule.
#
# Validated vs human GT on 3 videos (free_fused_pose.pt, experiments/
# anogenital_recovery_eval.py): anogenital recall 0.638 -> 0.925, F1 0.619 -> 0.696 at
# thr 0.28, recovering 7/9 user-flagged misses; argmin holds higher precision than the
# single-channel rule at every threshold (869 vs 1019 false at 0.28). The recovery is
# directional and runs per track, so the acting animal gets anogenital while the
# competing contacts are cleared; the scene mutex then resolves cross-animal conflicts.
#   behaviour -> {channel, closer_than:[channels], margins:{channel:m}, thr, clear:[...]}
CONTACT_RECOVER_ARGMIN: dict[str, dict[str, dict]] = {
    "classic_free_interaction": {
        "anogenital": {
            "channel": "pp_nose_tail",
            "closer_than": ["pp_nose_nose", "pp_nose_body"],
            "margins": {},                       # require tail strictly <= competitors
            "thr": 0.28,
            "clear": ["nose-to-nose", "nose-to-body"],
        },
    },
    "aggression_cd1_bl6": {},
}


# Refine-trigger thresholds (design-rule-correction workflow, GT-validated): surface
# a "Refine model" action when the prediction is internally self-contradicting
# (logical-violation rate >= C) OR mushy/low-confidence (mean decision margin < M).
REFINE_C_THRESHOLD = 0.10
REFINE_MARGIN_THRESHOLD = 0.22


@dataclass
class RuleCorrectionConfig:
    """Tunable rule pipeline.

    EMPIRICAL FINDING (design-rule-correction workflow, validated twice vs human GT):
    the mutual-exclusion rules HURT macro-F1 (mutex-contact -0.021, mutex-mountrear
    -0.020) because the model's per-class probabilities are not calibrated against
    each other, so keep-argmax deletes more correct labels than wrong ones (mounting
    is wiped). Min-bout is a no-op (bouts already long). So correction is OFF by
    default; the deliverable is the refine-trigger SIGNAL, not the correction. The
    mutex rules remain available as an opt-in for users who want logically-consistent
    output and accept the accuracy cost, or for a future better-calibrated model."""

    enabled: bool = False                   # master switch; correction off by default
    mutex: bool = False                     # enforce mutually-exclusive groups (HURTS now)
    # per-behavior decision-threshold overrides applied to probs (over-predicted
    # classes get raised). Empty = use the labels as given.
    threshold_overrides: dict[str, float] = field(default_factory=dict)
    min_bout_frames: int = 0                # drop foreground bouts shorter than this (0 = off)


def _name_index(label_map) -> dict[str, int]:
    names = getattr(label_map, "names", None) or list(label_map)
    return {str(n): i for i, n in enumerate(names)}


def apply_mutex(labels: np.ndarray, probs: np.ndarray, label_map,
                groups: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Enforce mutually-exclusive groups: in any frame where >1 group member is
    active, keep only the highest-probability member. Returns (corrected_labels,
    per_frame_changed_bool)."""
    idx = _name_index(label_map)
    out = labels.copy()
    T = labels.shape[1]
    changed = np.zeros(T, dtype=bool)
    for group in groups:
        members = [idx[n] for n in group if n in idx]
        if len(members) < 2:
            continue
        sub = out[members]                       # [g, T]
        conflict = sub.sum(axis=0) > 1
        if not conflict.any():
            continue
        keep = np.argmax(probs[members], axis=0)  # [T] index within the group
        for gi, m in enumerate(members):
            drop = conflict & (sub[gi] > 0) & (keep != gi)
            out[m, drop] = 0
        changed |= conflict
    return out, changed


def _group_by_video(predictions) -> dict[str, list[int]]:
    by_video: dict[str, list[int]] = {}
    for i, p in enumerate(predictions):
        by_video.setdefault(str(getattr(p, "video_id", "")), []).append(i)
    return by_video


def _scene_frame_grid(predictions, members_i):
    """Union of frame indices across the given tracks + per-track position maps onto
    that grid. Returns (all_frames, [pos_array_per_track])."""
    frame_lists = [np.asarray(predictions[i].frame_indices) for i in members_i]
    if not frame_lists:
        return np.empty(0, dtype=int), []
    all_frames = np.unique(np.concatenate(frame_lists))
    lookup = {int(f): j for j, f in enumerate(all_frames)}
    track_pos = [
        np.fromiter((lookup[int(f)] for f in np.asarray(predictions[i].frame_indices)),
                    dtype=int, count=len(predictions[i].frame_indices))
        for i in members_i
    ]
    return all_frames, track_pos


def apply_scene_mutex(predictions, label_map, groups: list[list[str]]) -> list[np.ndarray]:
    """Enforce dyad / scene-level mutual exclusion ACROSS the tracks of each video.

    The free-interaction contact behaviours are scene-level states: ground truth
    essentially never has two members of a group active at once, even across the two
    animals (verified on the labelled videos: triad co-occurrence ~0.00-0.03%). But
    independent per-track sigmoids can disagree (track A says nose-to-nose, track B
    says nose-to-body), so the merged scene shows an impossible combination. This is
    the level at which the user actually sees the conflict (the per-track arrays are
    each already clean).

    For each video and each frame, if more than one group member is active across all
    its tracks, keep only the member with the highest probability anywhere in the
    scene and zero the others in EVERY track. Returns a list of corrected label arrays
    (one per input prediction, same order); arrays that did not change are copies.
    """
    out = [np.asarray(p.labels).astype(np.int8, copy=True) for p in predictions]
    if not groups:
        return out
    idx = _name_index(label_map)
    for _vid, members_i in _group_by_video(predictions).items():
        all_frames, track_pos = _scene_frame_grid(predictions, members_i)
        F = all_frames.shape[0]
        if F == 0:
            continue
        for group in groups:
            gmembers = [idx[n] for n in group if n in idx]
            if len(gmembers) < 2:
                continue
            G = len(gmembers)
            scene_active = np.zeros((G, F), dtype=bool)
            scene_prob = np.full((G, F), -1.0, dtype=float)
            for ti, i in enumerate(members_i):
                lab = out[i]
                prob = np.asarray(predictions[i].probabilities)
                tp = track_pos[ti]
                for gi, m in enumerate(gmembers):
                    scene_active[gi, tp] |= lab[m] > 0
                    scene_prob[gi, tp] = np.maximum(scene_prob[gi, tp], prob[m])
            conflict = scene_active.sum(axis=0) > 1          # [F]
            if not conflict.any():
                continue
            winner = np.argmax(scene_prob, axis=0)           # [F] winning group index
            for ti, i in enumerate(members_i):
                tp = track_pos[ti]
                lab = out[i]
                conf_t = conflict[tp]
                win_t = winner[tp]
                for gi, m in enumerate(gmembers):
                    drop = conf_t & (lab[m] > 0) & (win_t != gi)
                    lab[m, drop] = 0
    return out


def apply_contact_gate(labels: np.ndarray, geometry: dict | None, label_map,
                       interaction_mode: str = "classic_free_interaction",
                       argmin_recover: bool = True) -> np.ndarray:
    """Apply the contact-geometry rules to one track, both directions.

    GATE: suppress a contact behaviour where the relevant raw nose-to-region distance
    is beyond its contact range (nose not at the partner -> false/premature firing).
    RECOVER: where a behaviour's geometry is very tight, assert it and clear the other
    triad members (geometry is authoritative in the tight-contact regime; recovers
    missed nose-to-nose and fixes nose-to-body<->nose-to-nose confusions).
    ARGMIN RECOVER (``argmin_recover``, on by default): recover missed anogenital
    where the nose-to-tail distance is the closest region and within range. This
    trades precision for recall on anogenital (GT: R 0.74->0.91, P 0.88->0.67),
    so it is exposed as a switch; set False for the higher-precision gate-only mode.

    ``geometry`` maps channel name -> [T] raw distances for THIS track. No-op when
    geometry is unavailable (e.g. predictions restored from a project). Returns
    corrected labels [K, T]."""
    if not geometry:
        return labels
    idx = _name_index(label_map)
    out = labels.copy()
    T = out.shape[1]

    def _dist(channel):
        if channel not in geometry:
            return None
        d = np.asarray(geometry[channel], dtype=float)
        return d if d.shape[0] == T else None

    # GATE: drop far contacts.
    for beh, (channel, thr) in CONTACT_GATES.get(interaction_mode, {}).items():
        d = _dist(channel)
        if beh in idx and d is not None:
            out[idx[beh], np.isfinite(d) & (d > float(thr))] = 0

    # RECOVER (single-channel): assert tight contacts and clear competing members.
    for beh, (channel, thr, clear) in CONTACT_RECOVER.get(interaction_mode, {}).items():
        d = _dist(channel)
        if beh not in idx or d is None:
            continue
        close = np.isfinite(d) & (d <= float(thr))
        out[idx[beh], close] = 1
        for other in clear:
            if other in idx:
                out[idx[other], close] = 0

    # ARGMIN RECOVER: assert a contact where its region is the closest one and within
    # range (precision-safe recall for anogenital). Requires the channel to beat each
    # competitor (by an optional margin) so it only fires when the nose is at THIS
    # region, not merely near it. The argmin must be VERIFIABLE: if a competitor
    # channel is missing entirely we skip the recovery rather than fall back to the
    # refuted single-channel rule, and per frame we only fire where every competitor
    # distance is finite (a NaN competitor means we cannot confirm the argmin).
    argmin_specs = CONTACT_RECOVER_ARGMIN.get(interaction_mode, {}) if argmin_recover else {}
    for beh, spec in argmin_specs.items():
        if beh not in idx:
            continue
        d = _dist(spec["channel"])
        if d is None:
            continue
        competitors = [(c, _dist(c)) for c in spec.get("closer_than", [])]
        if any(od is None for _, od in competitors):
            continue                              # cannot verify argmin -> do not fire
        ok = np.isfinite(d) & (d <= float(spec["thr"]))
        for other_ch, od in competitors:
            margin = float(spec.get("margins", {}).get(other_ch, 0.0))
            ok &= np.isfinite(od) & (d <= od - margin)
        out[idx[beh], ok] = 1
        for other in spec.get("clear", []):
            if other in idx:
                out[idx[other], ok] = 0
    return out


def scene_violation_rate(predictions, label_map,
                         interaction_mode: str = "classic_free_interaction") -> float:
    """Fraction of scene frames (pooled over videos) where the merged tracks assert
    >=2 members of a mutex group at once. This is the conflict the user sees in the
    scene/correction view; the per-track ``refine_signal`` rate misses it because
    each track is individually consistent."""
    idx = _name_index(label_map)
    groups = MUTEX_GROUPS.get(interaction_mode, [])
    if not groups:
        return 0.0
    tot_conf = 0
    tot_frames = 0
    for _vid, members_i in _group_by_video(predictions).items():
        all_frames, track_pos = _scene_frame_grid(predictions, members_i)
        F = all_frames.shape[0]
        if F == 0:
            continue
        conflict_any = np.zeros(F, dtype=bool)
        for group in groups:
            gmembers = [idx[n] for n in group if n in idx]
            if len(gmembers) < 2:
                continue
            active = np.zeros((len(gmembers), F), dtype=bool)
            for ti, i in enumerate(members_i):
                lab = np.asarray(predictions[i].labels)
                tp = track_pos[ti]
                for gi, m in enumerate(gmembers):
                    active[gi, tp] |= lab[m] > 0
            conflict_any |= active.sum(axis=0) > 1
        tot_conf += int(conflict_any.sum())
        tot_frames += F
    return tot_conf / tot_frames if tot_frames else 0.0


def _apply_min_bout(labels: np.ndarray, min_frames: int, background_id: int) -> np.ndarray:
    """Drop foreground bouts shorter than ``min_frames`` per behavior independently."""
    if min_frames <= 1:
        return labels
    out = labels.copy()
    for k in range(out.shape[0]):
        if k == background_id:
            continue
        row = out[k]
        padded = np.concatenate(([0], row, [0]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            if e - s < min_frames:
                out[k, s:e] = 0
    return out


def correct_track(labels: np.ndarray, probs: np.ndarray, label_map,
                  config: RuleCorrectionConfig | None = None,
                  interaction_mode: str = "classic_free_interaction",
                  background_id: int = 0) -> tuple[np.ndarray, dict]:
    """Apply the rule pipeline to one track. Returns (corrected_labels[K,T], info).

    ``info`` carries the correction magnitude (fraction of frames changed) and a
    per-rule breakdown, for the refine-trigger.
    """
    config = config or RuleCorrectionConfig()
    raw = labels.astype(np.int8, copy=True)
    out = raw.copy()
    if not config.enabled:
        return out, {"magnitude": 0.0, "mutex_fraction": 0.0, "raw_violation_fraction": 0.0}

    idx = _name_index(label_map)
    # raw logical-violation rate (independent of whether we correct): a clean,
    # label-free model-quality signal for the refine-trigger.
    raw_violation = np.zeros(raw.shape[1], dtype=bool)
    for group in MUTEX_GROUPS.get(interaction_mode, []):
        members = [idx[n] for n in group if n in idx]
        if len(members) >= 2:
            raw_violation |= raw[members].sum(axis=0) > 1

    # per-behavior threshold overrides (re-threshold over-predicted classes)
    if config.threshold_overrides:
        for name, thr in config.threshold_overrides.items():
            if name in idx:
                k = idx[name]
                out[k] = (probs[k] >= float(thr)).astype(np.int8)

    mutex_changed = np.zeros(raw.shape[1], dtype=bool)
    if config.mutex:
        out, mutex_changed = apply_mutex(out, probs, label_map,
                                         MUTEX_GROUPS.get(interaction_mode, []))

    if config.min_bout_frames > 1:
        out = _apply_min_bout(out, config.min_bout_frames, background_id)

    changed_frames = (out != raw).any(axis=0)
    info = {
        "magnitude": float(changed_frames.mean()),
        "mutex_fraction": float(mutex_changed.mean()),
        "raw_violation_fraction": float(raw_violation.mean()),
    }
    return out, info


def refine_signal(labels: np.ndarray, probs: np.ndarray, label_map,
                  interaction_mode: str = "classic_free_interaction") -> dict:
    """Label-free model-quality signal for the refine-trigger (one track).

    Returns the logical-violation rate ``C`` (fraction of frames where the
    prediction asserts mutually-exclusive behaviors at once = internal self-
    contradiction), the mean decision ``margin`` (mean |prob-0.5|, decisiveness),
    and ``should_refine``. GT-validated thresholds: C >= 0.10 OR margin < 0.22.
    """
    idx = _name_index(label_map)
    T = labels.shape[1]
    viol = np.zeros(T, dtype=bool)
    for group in MUTEX_GROUPS.get(interaction_mode, []):
        members = [idx[n] for n in group if n in idx]
        if len(members) >= 2:
            viol |= labels[members].sum(axis=0) > 1
    C = float(viol.mean()) if T else 0.0
    bg = getattr(label_map, "background_id", None)
    fg = [k for k in range(probs.shape[0]) if k != bg]
    margin = float(np.mean(np.abs(probs[fg] - 0.5))) if fg and T else 0.5
    return {"violation_rate": C, "margin": margin,
            "should_refine": (C >= REFINE_C_THRESHOLD) or (margin < REFINE_MARGIN_THRESHOLD)}


def refine_signal_for_predictions(predictions, label_map,
                                  interaction_mode: str = "classic_free_interaction") -> dict:
    """Aggregate the refine signal over all tracks of a video (the worst track
    drives the decision: max violation rate, min margin)."""
    sigs = [refine_signal(np.asarray(p.labels), np.asarray(p.probabilities),
                          label_map, interaction_mode)
            for p in predictions if getattr(p, "labels", None) is not None
            and np.asarray(p.labels).ndim == 2]
    if not sigs:
        return {"violation_rate": 0.0, "margin": 0.5, "should_refine": False}
    C = max(s["violation_rate"] for s in sigs)
    margin = min(s["margin"] for s in sigs)
    return {"violation_rate": C, "margin": margin,
            "should_refine": (C >= REFINE_C_THRESHOLD) or (margin < REFINE_MARGIN_THRESHOLD)}
