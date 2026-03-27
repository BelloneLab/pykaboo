# CamApp
<img width="1602" height="932" alt="image" src="https://github.com/user-attachments/assets/68a5923f-c5ad-464c-badf-b71bb7b47f44" />

This app records Basler, FLIR, or USB camera video with synchronized Arduino TTL outputs (gate, barcode, 1 Hz sync), and logs metadata.

## Features

- Basler, FLIR, and USB camera support
- Basler via Pylon, FLIR machine-vision cameras via Spinnaker / `PySpin`, FLIR thermal cameras via `flirpy`, USB via OpenCV
- Live view with optional ROI cropping
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

- `PySpin` 3.2.x is not compatible with NumPy 2.x in this project. Use `numpy<2` in the CamApp environment when you need FLIR Spinnaker support.

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

CamApp now supports two different FLIR paths:

- FLIR machine-vision cameras through Teledyne FLIR Spinnaker + `PySpin`
- FLIR thermal cameras through `flirpy`

For FLIR machine-vision cameras such as Blackfly / Chameleon / Grasshopper:

1. Install the Spinnaker SDK for your Windows/Python build from Teledyne FLIR.
2. Install the matching `PySpin` wheel provided with Spinnaker.
3. Confirm `import PySpin` works in the same environment used to launch CamApp.
4. Restart CamApp and scan cameras again.

Notes:

- `PySpin` is not installed from `requirements.txt`; the wheel must come from the Spinnaker SDK package and must match your Python version and architecture.
- If `PySpin` imports fail with `_ARRAY_API not found` or `numpy.core.multiarray failed to import`, the environment is using NumPy 2.x; downgrade to `numpy<2`.
- `simple_pyspin` and `EasyPySpin` are useful for standalone diagnostics, but CamApp uses raw `PySpin` directly so the app can manage acquisition and GenICam nodes itself.

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

- FLIR Spinnaker cameras appear in the same source list as Basler and USB cameras when `PySpin` is installed.
- FLIR thermal cameras discovered via `flirpy` also appear in the same source list.
- Basler exposes camera GPIO state through chunk metadata when supported by the device.
- FLIR Spinnaker cameras now use direct camera-node control in the app for:
  frame rate via `AcquisitionFrameRate`
  exposure via `ExposureAuto` + `ExposureMode` + `ExposureTime`
  ROI / resolution via `Width`, `Height`, `OffsetX`, and `OffsetY`
  manual gain via `GainAuto` + `Gain`
- Recording `Max Length` is now applied against the active recording FPS used by the worker, and changing the limit during an active trial updates the current trial immediately.
- On color FLIR Spinnaker cameras, CamApp can preview and record `BGR8` even when the camera is streaming a native Bayer format such as `BayerRG8`; the app debayers on the host side when direct in-camera RGB output is unavailable.
- FLIR thermal backends do not expose Basler-style per-frame line status chunks, so CamApp keeps Arduino as the general GPIO read/write and synchronization layer for those workflows.
- FLIR thermal frames are normalized to 8-bit for preview and MP4 recording; raw thermal min/max/mean values are still logged per frame in CSV metadata.
- On Spinnaker cameras, frame rate, exposure time, ROI size, and gain remain coupled by sensor and bus limits; if a requested value is outside the camera limits, CamApp clamps it to the nearest valid node value.
- For stable high-rate acquisition on FLIR Spinnaker cameras, prefer:
  shorter exposures than the frame period
  smaller ROI / resolution when you need higher FPS
  `Mono8` when color is not needed, because it reduces bandwidth and host conversion load

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
- FLIR Spinnaker camera missing from the list: verify Spinnaker is installed, `import PySpin` works, and the camera is visible in SpinView.
- FLIR thermal camera missing from the list: verify `flirpy` is installed and Windows still sees the device as a USB video device.
- "No USB camera found": verify the device is connected and not in use by another app.
