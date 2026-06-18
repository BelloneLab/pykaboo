"""Download PyKaboo's default model weights from the GitHub Release.

The portable weights (.pth/.pt) are too large for the git repo, so they are hosted
as release assets. This fetches them into pykaboo/models/ next to the prebuilt
.engine files. After downloading, rebuild the TensorRT engines for your machine:

    python scripts/build_rfdetr_engine.py   --checkpoint pykaboo/models/checkpoint_best_total.pth
    python scripts/build_yolo_pose_engine.py --pose       pykaboo/models/poseModel_largebest.pt

Usage:
    python scripts/download_models.py            # download the default weights
    python scripts/download_models.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_REPO = "BelloneLab/pykaboo"
DEFAULT_TAG = "models-v1"
ASSETS = ("checkpoint_best_total.pth", "poseModel_largebest.pt")


def _download(url: str, dest: Path) -> None:
    print(f"  {url}\n   -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = 100.0 * done / total_size
            sys.stdout.write(f"\r   {done / 1e6:7.1f} / {total_size / 1e6:7.1f} MB ({pct:5.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, _hook)
    sys.stdout.write("\n")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="owner/name of the GitHub repo hosting the release")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag")
    ap.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{args.repo}/releases/download/{args.tag}"
    print(f"Downloading default weights from {args.repo}@{args.tag} into {MODELS_DIR}", flush=True)

    ok = True
    for name in ASSETS:
        dest = MODELS_DIR / name
        if dest.is_file() and not args.force:
            print(f"  {name} already present ({dest.stat().st_size / 1e6:.1f} MB) — skipping (use --force)", flush=True)
            continue
        try:
            _download(f"{base}/{name}", dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {name}: {exc}", flush=True)
            ok = False

    if ok:
        print(
            "\nDone. Now rebuild engines for this machine:\n"
            "  python scripts/build_rfdetr_engine.py   --checkpoint pykaboo/models/checkpoint_best_total.pth\n"
            "  python scripts/build_yolo_pose_engine.py --pose       pykaboo/models/poseModel_largebest.pt",
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
