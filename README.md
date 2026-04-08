# CamApp Live Detection

CamApp Live Detection is a Windows desktop app for synchronized camera acquisition, live segmentation, Arduino TTL control, and per-frame metadata export. It is built around PySide6, FFmpeg, Basler / FLIR / USB camera backends, and a session planner that drives filenames and recording metadata.

## Screenshots

### Workspace Overview

![Workspace overview](docs/screenshots/workspace-overview.png)

### General Settings And Filename Builder

![General settings](docs/screenshots/general-settings.png)

### Advanced Camera Controls

![Advanced camera controls](docs/screenshots/advanced-camera-controls.png)

### Live Detection Panel

![Live detection panel](docs/screenshots/settings-and-live-panels.png)

## What It Does

- Connects to Basler, FLIR, and generic USB cameras from one UI.
- Records MP4 video through FFmpeg while exporting frame-by-frame metadata to CSV.
- Runs live RF-DETR Seg and YOLO-based inference for ROI occupancy and mouse proximity rules.
- Drives Arduino digital outputs through pyFirmata, including barcode, sync, gate, and live DO mappings.
- Uses a planner-driven metadata flow so `Animal ID`, `Session`, `Trial`, `Experiment`, and `Condition` can drive the recording filename automatically.
- Supports organized recording folders such as `animal/session`.
- Exposes camera-native controls such as pixel format, bit depth, gain, white balance, offsets, and ROI cropping.
- Saves live detections and merged recording metadata after each acquisition.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 recommended
- FFmpeg available on `PATH`
- A compatible camera SDK when using vendor cameras:
  - Basler: Pylon SDK plus `pypylon`
  - FLIR machine vision: Spinnaker SDK plus `PySpin`
  - FLIR thermal: `flirpy`

Notes:

- `PySpin` is a vendor wheel and is not installed from `requirements.txt`.
- `numpy<2` is recommended when FLIR Spinnaker support matters.
- The compiled EXE still expects FFmpeg on `PATH` at runtime.

## Install

Conda:

```powershell
conda env create -f environment.yml
conda activate CamApp
```

Virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run From Source

```powershell
python main.py
```

## Arduino Setup

CamApp Live Detection supports two practical Arduino modes:

- `StandardFirmata` for generic TTL monitoring and output control.
- `StandardFirmataBarcode` for the custom barcode / sync / TM1638 counter workflow included in this repo.

The custom sketch lives in [StandardFirmataBarcode](StandardFirmataBarcode) and adds:

- onboard barcode generation
- D5 sync relay monitoring
- TM1638 timer display
- custom Firmata SysEx commands for counter start, stop, and restart

## Metadata And Recording Flow

- Planner rows can populate the active metadata form.
- The filename preview is generated from the selected filename order and skips empty fields automatically.
- A manual filename override is still available when needed.
- Final frame metadata export now normalizes `timestamp_software` and `timestamp_ticks` to elapsed seconds from frame `0`.
- Low-value raw diagnostic columns are removed from the final per-frame metadata CSV.

## Build A Windows EXE

The repo ships with:

- [camApp-live-detection.spec](camApp-live-detection.spec)
- [scripts/build_release.ps1](scripts/build_release.ps1)

Local build from the `CamApp` environment:

```powershell
python -m pip install -r requirements.txt pyinstaller
.\scripts\build_release.ps1 -Version v2026.04.08 -PythonExe python -Clean
```

When you need vendor SDK support inside the EXE, build with the same interpreter that already imports the required packages:

```powershell
.\scripts\build_release.ps1 -Version v2026.04.08 -PythonExe C:\Users\bellone\.conda\envs\CamApp\python.exe -Clean
```

The build script produces:

- `dist/CamAppLiveDetection.exe`
- `release/camApp-live-detection-<version>-windows-x64.zip`
- `release/camApp-live-detection-<version>-windows-x64.sha256`
- `release/camApp-live-detection-<version>-windows-x64-warn.txt`

## GitHub Actions

### Continuous Integration

[.github/workflows/ci-windows.yml](.github/workflows/ci-windows.yml) runs on pushes, pull requests, and manual dispatch. It:

- installs dependencies on `windows-latest`
- runs unit tests
- runs `py_compile` checks on the main app modules
- builds the Windows release artifact with PyInstaller
- uploads the ZIP, checksum, and warning log as workflow artifacts

### Release Publishing

[.github/workflows/release-windows.yml](.github/workflows/release-windows.yml) publishes a GitHub release on:

- pushed tags matching `v*`
- manual workflow dispatch

It rebuilds the Windows package, uploads release assets, and publishes a GitHub release with the ZIP, SHA-256 checksum, and PyInstaller warning log attached.

## Troubleshooting

- `FFmpeg not found`: add FFmpeg to `PATH` and restart the shell.
- `Failed to start native camera backend`: verify the vendor SDK is installed and the camera opens in the vendor viewer.
- `PySpin` import errors: use Python 3.10 with a Spinnaker-compatible wheel and keep `numpy<2`.
- Pixel format controls disabled: connect a GenICam-style camera first; the control is camera-native and not available for generic USB sources.
- No live inference output: verify the selected checkpoint exists and the environment contains `torch`, `rfdetr`, and `ultralytics`.
