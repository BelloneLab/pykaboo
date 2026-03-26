# CamApp

This app records Basler, FLIR, or USB camera video with synchronized Arduino TTL outputs (gate, barcode, 1 Hz sync), and logs metadata.

## Features

- Basler, FLIR, and USB camera support
- Basler via Pylon, FLIR thermal cameras via `flirpy`, USB via OpenCV
- Live view with optional ROI cropping
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
- `flirpy` for FLIR Boson, Lepton, and TeAx/Tau integrations

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

## FLIR Support Notes

- FLIR discovery is provided through `flirpy` and appears in the same source list as Basler and generic USB devices.
- Basler exposes camera GPIO state through chunk metadata when supported by the device.
- FLIR backends do not expose Basler-style per-frame line status chunks, so CamApp keeps Arduino as the general GPIO read/write and synchronization layer for FLIR workflows.
- FLIR thermal frames are normalized to 8-bit for preview and MP4 recording; raw thermal min/max/mean values are still logged per frame in CSV metadata.

## Arduino Firmata Setup (Optional)

1. Open Arduino IDE.
2. Load `File > Examples > Firmata > StandardFirmata`.
3. Flash it to the Arduino.
4. Use the CamApp UI to scan and connect to the correct COM port.

Pin mapping is configured directly in the app (`Gate`, `Sync`, `Barcode`, `Lever`, `Cue`, `Reward`, `ITI`) and uses Firmata digital pin read/write.

Barcode timing notes:
- `Gap After Code` is the silent interval after one barcode word finishes.
- Full barcode cycle time = `start pulse + start low + (bits * bit duration) + gap`.
- The app no longer depends on legacy custom Arduino sketches in this repo; use `StandardFirmata`.

## Build CamApp.exe

Install build tool:

```powershell
pip install pyinstaller
```

Build a single-file EXE:

```powershell
pyinstaller --onefile --noconsole --name CamApp main.py
```

The output will be in `dist/CamApp.exe`. Copy it to the project root if desired:

```powershell
copy dist\CamApp.exe .\
```

Note: The compiled EXE still requires FFmpeg available on PATH at runtime.

## Troubleshooting

- "FFmpeg not found": add FFmpeg to PATH and restart the terminal.
- "Failed to start Arduino TTLs": make sure no other app is holding the COM port, then reconnect.
- "No Basler camera found": verify Pylon is installed and the camera is detected by Pylon Viewer.
- FLIR camera missing from the list: verify `flirpy` is installed and Windows still sees the device as a USB video device.
- "No USB camera found": verify the device is connected and not in use by another app.
