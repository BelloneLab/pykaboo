"""Export per-frame (or pooled per-bout) embeddings from a trained model.

Embeddings are the penultimate temporal feature tensor of the TCN. They can be
exported framewise for dense motif discovery, or pooled per behavior bout for a
compact representation suited to bout-level clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint import load_checkpoint
from .config import load_config
from .export import TrackPrediction
from .infer import run_inference
from .metrics import extract_bouts
from .storage import write_table


def pool_bout_embeddings(
    predictions: list[TrackPrediction], background_id: int
) -> pd.DataFrame:
    """Pool framewise embeddings into mean/max/std vectors per behavior bout."""

    rows: list[dict] = []
    for pred in predictions:
        embed_dim = pred.embeddings.shape[0]
        for bout in extract_bouts(pred.labels, background_id):
            segment = pred.embeddings[:, bout.start : bout.end + 1]
            row = {
                "video_id": pred.video_id,
                "subject_id": pred.subject_id,
                "object_id": pred.object_id,
                "behavior_id": bout.behavior,
                "start_frame": int(pred.frame_indices[bout.start]),
                "end_frame": int(pred.frame_indices[bout.end]),
                "duration_frames": bout.length,
            }
            mean = segment.mean(axis=1)
            std = segment.std(axis=1)
            mx = segment.max(axis=1)
            for i in range(embed_dim):
                row[f"mean_{i:03d}"] = float(mean[i])
                row[f"std_{i:03d}"] = float(std[i])
                row[f"max_{i:03d}"] = float(mx[i])
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export model embeddings.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--level", choices=["frame", "bout"], default="frame", help="Embedding level."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    outputs = run_inference(
        checkpoint, args.coco_json, config, log=lambda m: print(m, flush=True)
    )
    if args.level == "bout":
        df = pool_bout_embeddings(
            outputs.predictions, checkpoint.label_map.background_id
        )
    else:
        df = outputs.embeddings
    path = write_table(df, args.output)
    print(f"Wrote {args.level}-level embeddings: {path}  shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
