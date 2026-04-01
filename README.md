# CamApp Live Detection
<img width="1602" height="932" alt="image" src="https://github.com/user-attachments/assets/68a5923f-c5ad-464c-badf-b71bb7b47f44" />

This standalone app records Basler, FLIR, or USB camera video, runs realtime mouse segmentation on the live stream, and can drive Arduino TTL outputs from live ROI or proximity rules while still supporting synchronized camera metadata and classic gate/barcode/sync generation.

## Features

- Basler, FLIR, and USB camera support
- Basler via Pylon, FLIR machine-vision cameras via Spinnaker / `PySpin`, FLIR thermal cameras via `flirpy`, USB via OpenCV
- Live view with optional ROI cropping
- Live detection panel for RF-DETR Seg and YOLO Seg checkpoints
- Stable mouse identity overlay from tracker mode or model-class mode
- Behavioural ROI drawing on the live preview with rectangle, circle, and polygon tools
- Live trigger rules for ROI occupancy or mouse-mouse proximity
- Generic Arduino logical outputs `DO1..DO8` with level or pulse trigger modes
- Collapsible bottom control strip so acquisition and recording panels can be hidden while the record button stays visible
- Recording with FFmpeg (GPU or CPU encoders)
- Per-frame metadata logging (timestamp, exposure, thermal statistics, GPIO line status when available)
- Arduino TTL I/O via pyFirmata with live TTL plot
- Metadata templates saved to JSON plus TTL history saved to CSV

## Requirements

- Windows 10/11
- Python 3.10+ (recommended: Anaconda or Miniconda)
- FFmpeg in PATH
- Arduino with a Firmata sketch flashed (recommended: `StandardFirmata`)

Optional:
- Basler Pylon SDK + `pypylon` (camera drivers)
- FLIR Spinnaker SDK + `PySpin` for FLIR machine-vision cameras
- `flirpy` for FLIR Boson, Lepton, and TeAx/Tau integrations

Compatibility note:

- `PySpin` 3.2.x is not compatible with NumPy 2.x in this project. Use `numpy<2` in the CamApp Live Detection environment when you need FLIR Spinnaker support.

## Environment Setup

Create a fresh environment and install dependencies:

```powershell
conda env create -f environment.yml

```

If you are using a system Python:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## FLIR Spinnaker Setup

CamApp Live Detection supports two different FLIR paths:

- FLIR machine-vision cameras through Teledyne FLIR Spinnaker + `PySpin`
- FLIR thermal cameras through `flirpy`

For FLIR machine-vision cameras such as Blackfly / Chameleon / Grasshopper:

1. Install the Spinnaker SDK for your Windows/Python build from Teledyne FLIR.
2. Install the matching `PySpin` wheel provided with Spinnaker.
3. Confirm `import PySpin` works in the same environment used to launch CamApp Live Detection.
4. Restart CamApp Live Detection and scan cameras again.

Notes:

- `PySpin` is not installed from `requirements.txt`; the wheel must come from the Spinnaker SDK package and must match your Python version and architecture.
- The repository's top-level `PySpin/` folder is only a local cache for vendor wheels and docs; it is not the installed Spinnaker Python package.
- If `PySpin` imports fail with `_ARRAY_API not found` or `numpy.core.multiarray failed to import`, the environment is using NumPy 2.x; downgrade to `numpy<2`.
- `simple_pyspin` and `EasyPySpin` are useful for standalone diagnostics, but CamApp Live Detection uses raw `PySpin` directly so the app can manage acquisition and GenICam nodes itself.

## FFmpeg (Required)

FFmpeg is used for encoding and saving video. The app uses raw frames piped to FFmpeg, so FFmpeg must be on PATH.

### Install FFmpeg on Windows

Option A (static build):
1. Download a static build from https://www.gyan.dev/ffmpeg/builds/ (full or essentials).
2. Extract to a folder like `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your PATH.

Option B (BtbN build):
1. Download a release from https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract and add `bin` to PATH.

### Verify FFmpeg

```powershell
ffmpeg -version
```

### GPU Encoders (Optional)

The app supports these encoders:
- `h264_nvenc` (NVIDIA GPU)
- `h264_qsv` (Intel QuickSync)
- `libx264` (CPU)

Notes:
- `h264_nvenc` requires a compatible NVIDIA GPU and recent drivers.
- If GPU encoding fails, switch to `libx264` in the UI.

## Run the App

```powershell
python main.py
```

## Arduino Firmata Setup (Optional)

1. Open Arduino IDE.
2. Load `File > Examples > Firmata > StandardFirmata`.
3. Flash it to the Arduino.
4. Use the CamApp Live Detection UI to scan and connect to the correct COM port.

Pin mapping is configured directly in the app (`Gate`, `Sync`, `Barcode`, `Lever`, `Cue`, `Reward`, `ITI`, plus live `DO1..DO8`) and uses Firmata digital pin read/write.

Barcode timing notes:
- `Gap After Code` is the silent interval after one barcode word finishes.
- Full barcode cycle time = `start pulse + start low + (bits * bit duration) + gap`.
- The app no longer depends on legacy custom Arduino sketches in this repo; use `StandardFirmata`.

## Build CamApp Live Detection.exe

CamApp Live Detection ships with a checked-in PyInstaller spec at `camApp-live-detection.spec` plus a reusable build script at `scripts/build_release.ps1`.

Local build from the `camapp-live-detection` Python environment:

```powershell
python -m pip install -r requirements.txt pyinstaller
.\scripts\build_release.ps1 -Version dev -PythonExe python -Clean
```

If FLIR Spinnaker support matters in the compiled EXE, build with the same interpreter that already passes `import PySpin` and can see the camera, for example:

```powershell
.\scripts\build_release.ps1 -Version dev -PythonExe C:\Users\bellone\.conda\envs\camapp-live-detection\python.exe -Clean
```

The build script prints the exact interpreter path plus a `PySpin` preflight result before packaging.

That produces:

- `dist/CamAppLiveDetection.exe`
- `release/camApp-live-detection-dev-windows-x64.zip`
- `release/camApp-live-detection-dev-windows-x64.sha256`
- `release/camApp-live-detection-dev-windows-x64-warn.txt`

Note: the compiled EXE still requires FFmpeg available on PATH at runtime. Local builds bundle `PySpin` only when the selected build interpreter already has the real vendor package installed. The GitHub build still does not bundle the Spinnaker SDK because the CI environment does not install the vendor wheel.

## GitHub Release Workflow

There is a manual GitHub Actions workflow at `.github/workflows/release-windows.yml`.

Use `Actions > Build And Release CamApp Live Detection > Run workflow` and provide:

- `tag`: release tag such as `v1.0.0`
- `release_name`: optional display title
- `prerelease`: mark the GitHub release as prerelease
- `draft`: publish as draft instead of a public release

The workflow builds the Windows EXE on `windows-latest`, zips it, generates a SHA-256 checksum, uploads the build artifacts, and then publishes a GitHub release with those assets attached.

## Troubleshooting

- "FFmpeg not found": add FFmpeg to PATH and restart the terminal.
- "Failed to start Arduino TTLs": make sure no other app is holding the COM port, then reconnect.
- "No Basler camera found": verify Pylon is installed and the camera is detected by Pylon Viewer.
- FLIR Spinnaker camera missing from the list: verify Spinnaker is installed, `import PySpin` works, and the camera is visible in SpinView.
- FLIR thermal camera missing from the list: verify `flirpy` is installed and Windows still sees the device as a USB video device.
- "No USB camera found": verify the device is connected and not in use by another app.
