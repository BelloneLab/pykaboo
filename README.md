<h1 align="center">PyKaboo 🐭📷</h1>

<p align="center">
  <img src="assets/pykaboo_big.png" alt="PyKaboo logo" width="420">
</p>

<p align="center">
  <b>Point a camera at your mice and... <i>peekaboo!</i></b><br>
  PyKaboo sees them, draws their pose, <b>names what they're doing</b>, and fires your
  TTLs in real time — all from one tidy Windows app.
</p>

<p align="center">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?logo=windows&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="GUI" src="https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/Accel-CUDA%20%2F%20TensorRT-76B900?logo=nvidia&logoColor=white">
  <img alt="Cameras" src="https://img.shields.io/badge/Cameras-Basler%20%7C%20FLIR%20%7C%20USB-ff8a00">
</p>

---

PyKaboo is a Windows desktop app for **synchronized camera acquisition, planner-driven
recording, live animal detection, and closed-loop Arduino TTL control**. Basler, FLIR,
and USB cameras all live under one interface, and the whole recording workflow stays tied
to a multi-trial session plan so your filenames, metadata, and triggers are always aligned.

It started as a careful acquisition tool. It grew a brain. 🧠

## 🎬 See it in action

Two mice, live, on plain bedding — masks, pose skeletons, and a **per-mouse behavior
subtitle** drawn straight onto the video (and burned into the recorded overlay MP4):

<table>
  <tr>
    <td width="33%"><img src="docs/screenshots/behavior-nose2nose.png" alt="nose-to-nose"></td>
    <td width="33%"><img src="docs/screenshots/behavior-nose2body.png" alt="nose-to-body"></td>
    <td width="33%"><img src="docs/screenshots/behavior-following.png" alt="following"></td>
  </tr>
  <tr>
    <td align="center"><b>nose2nose</b> — both heads meet, both chips agree at 100%.</td>
    <td align="center"><b>nose2body</b> — the actor (pink) sniffs the partner's flank; the recipient reads <i>none</i>.</td>
    <td align="center"><b>following</b> — one mouse pursues, the other shows <i>withdrawal_after_contact</i>.</td>
  </tr>
</table>

> Every chip is computed **per mouse** (the two directed views of the scene), so you can
> tell who is doing what. The label, the colour, and the connector line all match the
> animal's identity.

## ✨ Highlights

- 🗂️ **Planner-first workflow** — trial rows drive filename, metadata, and recording duration; the planner auto-restores on launch and auto-advances to the next pending trial.
- 🎥 **Many cameras, one window** — Basler, FLIR (machine-vision + thermal), and USB; add up to three auxiliary cameras that all record in sync.
- 🧩 **Live segmentation + pose** — RF-DETR-Seg / YOLO-Seg instance masks with an 8-keypoint skeleton, accelerated with CUDA or TensorRT.
- 🧠 **Live behavior detection** — name social behaviors frame-by-frame (see below) and overlay them on the preview *and* the recorded video.
- ⚡ **Closed-loop TTL** — turn a detected behavior (or an ROI / proximity / mask-contact event) into an Arduino pulse for optogenetics and stimulation, in real time.
- 🧾 **Frame-aligned exports** — MP4 + metadata/TTL/behavior CSVs, all zero-referenced to the first recorded frame so every clock lines up.

## 🧠 Live behavior detection (the fun part)

Pick a **Behavior method** right next to the overlay toggles, flip **Behavior** on, and
PyKaboo starts naming social behaviors for the dyad. Two engines, same output and same
overlay — choose per experiment:

| Engine | What it is | Speed | Use it for |
|---|---|---|---|
| 🟢 **Rule-based** (default) | Pure geometry/kinematics on keypoints + mask contours | **sub-millisecond** | real-time closed-loop TTL |
| 🔵 **ML model** | A trained EmbTCN-Attention temporal network (7 classes) | ~0.3 s / decision | holistic offline-style scoring |

**How the rule engine reads a frame:** a *social contact* is gated by **mask-contour
overlap** (the two segmentation masks touching within ~5% of body length); the contact
**type** is then read from the **closest inter-animal keypoints** — mutual noses →
`nose2nose`, a leading nose at the partner's tail → `nose2anogenital`, a leading nose at
the flank → `nose2body`, bodies touching with no leading nose → `sidebyside` /
`sidereside`. On top of contact it scores locomotor behaviors — `following`, `chasing`,
`approach`, `withdrawal`, `escape`, `fighting` — all temporally smoothed, all with
tolerances that scale to the animal's body length so it works at any zoom or arena size.

Behaviors flow everywhere you'd want them:

- 🖍️ **On screen + in the MP4** — per-mouse subtitle chips on the live preview and the recorded overlay video.
- 🧾 **In the CSVs** — per-frame, per-mouse behavior columns in the detection exports.
- ⚡ **Into TTLs** — add a `behavior_class` trigger rule (e.g. *mounting → DO1*) and PyKaboo pulses the Arduino the instant the behavior crosses threshold.

Full design notes live in [`pykaboo_live_behavior/INTEGRATION.md`](pykaboo_live_behavior/INTEGRATION.md).

## 🖥️ The interface

PyKaboo keeps you in one place from setup to acquisition — connect hardware, plan trials,
watch the live stream, and record with metadata and TTLs already aligned.

<p align="center">
  <img src="docs/screenshots/workspace-overview.png" alt="PyKaboo workspace and session planner" width="85%">
</p>

*The session planner stays central: every trial row drives the active filename and
metadata, and finished trials auto-advance to the next pending one.*

<p align="center">
  <img src="docs/screenshots/settings-and-live-panels.png" alt="Live detection panel" width="85%">
</p>

*The Live Detection panel: choose a segmentation model, set the mouse count, toggle
overlays (masks / boxes / keypoints / **behavior**), pick the **behavior method**, draw
behavioural ROIs, map DO pins, and wire up trigger rules.*

<p align="center">
  <img src="docs/screenshots/general-settings.png" alt="General settings and planner" width="49%">
  <img src="docs/screenshots/advanced-camera-controls.png" alt="Advanced camera controls" width="33%">
</p>

*Left: filename assembly, nested session storage, and behavior/TTL defaults. Right: the
advanced camera pop-up — live preview pipeline, gain, white balance, and ROI cropping.*

## 🚀 Quick start

**Requirements**

- Windows 10 or 11, Python 3.10 recommended
- `ffmpeg` on `PATH`
- Camera SDKs for vendor hardware: Basler (Pylon + `pypylon`), FLIR machine-vision (Spinnaker + `PySpin`), FLIR thermal (`flirpy`)
- An NVIDIA GPU is strongly recommended for live detection (CUDA / TensorRT)

**Install (conda)**

```powershell
conda env create -f environment.yaml
conda activate CamApp
```

**Install (venv)**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Run**

```powershell
python main.py
```

On Windows prefer `run_pykaboo.bat`, which finds a Python runtime that can import
`PySide6`, `cv2`, and `PySpin` (so FLIR cameras don't vanish when launched from the wrong
interpreter, and torch loads in the right DLL order).

## 🗂️ Planner workflow

- Import a CSV plan or build rows directly in the Recording Planner.
- The current planner is saved automatically and restored on the next launch.
- Select a row to load its metadata into the session form.
- `Ctrl+C` / `Ctrl+V` on a row copies trial content onto other rows.
- Right-click rows for duplicate, copy, paste, move up/down, apply, and remove.
- When a trial finishes recording it is marked `Acquired` and the next pending row is selected.

## 📦 Outputs

Each recording can produce:

- `<name>.mp4` and `<name>_overlay.mp4` (masks + pose + behavior subtitles)
- `<name>_metadata.csv` / `.json` / `.txt`
- `<name>_ttl_states.csv`, `<name>_ttl_counts.csv`
- `<name>_live_detections.csv` and `<name>_tracking_dlc.csv` (DLC-style keypoints)
- `<name>_masks_coco.json` (segmentation masks)
- `<name>_behavior_summary.csv`
- `<name>_<backend>_<n>.mp4` (+ metadata) per auxiliary camera

Timestamp columns (`timestamp_software`, `timestamp_camera`, `timestamp_ticks`) are
elapsed seconds starting at 0 on the first recorded frame; `camera_frame_id` is rebased to
0 so it can be compared with `frame_id` to spot dropped frames.

## 🔌 Arduino & TTL

PyKaboo supports:

- `StandardFirmata` for generic TTL monitoring and output control
- `StandardFirmataBarcode` for the custom barcode/sync workflow ([StandardFirmataBarcode](StandardFirmataBarcode))

**Additional Arduino devices:** the primary board (gate/sync/barcode + DO1-8 live outputs)
is unchanged. To drive extra outputs or log extra inputs, use **Additional Arduino
Devices** at the bottom of the Arduino Setup panel — add a board, pick its COM port, and
set each pin to **Input** (sampled per camera frame) or **Output**. Each input/output is
logged as a frame-aligned `dev<id>_<label>_ttl` column in `*_metadata.csv`. Auxiliary
boards must each use their own COM port, and the roster persists between launches.

## 🛠️ Build a Windows EXE

```powershell
python -m pip install -r requirements.txt pyinstaller
.\scripts\build_release.ps1 -Version v2026.04.12 -PythonExe python -Clean
```

See [camApp-live-detection.spec](camApp-live-detection.spec) and
[scripts/build_release.ps1](scripts/build_release.ps1).

## 🧰 Troubleshooting

- **`ffmpeg` not found** — add FFmpeg to `PATH` and restart the shell.
- **`PySpin` import errors** — use a Spinnaker-compatible wheel and keep `numpy<2`.
- **No live inference output** — check the checkpoint path and that the ML/behavior packages are installed; a GPU is strongly recommended.
- **Empty / 1-frame overlay MP4** — fixed: a degenerate detection used to crash the overlay writer; it now skips bad frames and keeps recording.
- **Vendor camera won't connect** — confirm it opens in the vendor SDK viewer first.

---

<p align="center"><i>Made for behavioural neuroscience rigs that need to see, decide, and act — frame by frame.</i></p>
