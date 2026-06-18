"""Render a before/after comparison chart from two keypoint-bench result sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    base = json.loads((REPO_ROOT / "dev_screenshots" / "keypoint_bench" / "results.json").read_text())
    opt = json.loads((REPO_ROOT / "dev_screenshots" / "keypoint_bench_opt" / "results.json").read_text())

    modes = ["yolo_pose + mask", "mask_geometry + mask"]
    x = np.arange(len(modes))
    w = 0.36

    base_tot = [base[m]["mean_ms"] for m in modes]
    opt_tot = [opt[m]["mean_ms"] for m in modes]
    base_geo = [base[m]["breakdown_ms"].get("geometry", 0.0) for m in modes]
    opt_geo = [opt[m]["breakdown_ms"].get("geometry", 0.0) for m in modes]

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=130)
    fig.patch.set_facecolor("#0d1626")
    ax.set_facecolor("#0d1626")

    b1 = ax.bar(x - w / 2, base_tot, w, label="before", color="#37506f", edgecolor="#0d1626")
    b2 = ax.bar(x + w / 2, opt_tot, w, label="after (cropped geometry)", color="#3aa0ff", edgecolor="#0d1626")
    # Geometry portion overlay (the part that changed).
    ax.bar(x - w / 2, base_geo, w, bottom=[t - g for t, g in zip(base_tot, base_geo)],
           color="#6fe06e", alpha=0.55, edgecolor="#0d1626")
    ax.bar(x + w / 2, opt_geo, w, bottom=[t - g for t, g in zip(opt_tot, opt_geo)],
           color="#6fe06e", edgecolor="#0d1626")

    for rect, tot in list(zip(b1, base_tot)) + list(zip(b2, opt_tot)):
        fps = 1000.0 / tot if tot else 0.0
        ax.text(rect.get_x() + rect.get_width() / 2, tot + 0.8,
                f"{tot:.0f} ms\n{fps:.0f} fps", ha="center", va="bottom",
                color="#e6eef8", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(modes, color="#cfe0f5")
    ax.set_ylabel("latency per frame (ms)", color="#cfe0f5")
    ax.set_title("Live keypoints: before vs after cropping mask geometry to ROI\n"
                 "(green = geometry portion)  ·  rfdetr-seg-medium @ 512px · 2 mice · max_gpu",
                 color="#e6eef8", fontsize=11)
    ax.tick_params(colors="#cfe0f5")
    for spine in ax.spines.values():
        spine.set_color("#26344a")
    ax.legend(facecolor="#0b1626", edgecolor="#26344a", labelcolor="#cfe0f5", loc="upper right")
    ax.set_ylim(0, max(base_tot) * 1.25)
    fig.tight_layout()
    out = REPO_ROOT / "dev_screenshots" / "keypoint_bench_compare.png"
    fig.savefig(str(out))
    print(f"saved {out}")
    # Also print a concise table.
    for m in modes:
        print(f"{m:24s} before {base[m]['mean_ms']:6.1f} ms ({1000/base[m]['mean_ms']:4.1f} fps)  "
              f"-> after {opt[m]['mean_ms']:6.1f} ms ({1000/opt[m]['mean_ms']:4.1f} fps)  "
              f"| geometry {base[m]['breakdown_ms'].get('geometry',0):5.1f} -> {opt[m]['breakdown_ms'].get('geometry',0):4.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
