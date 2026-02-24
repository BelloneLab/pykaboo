# Camera Control App (camApp)

This app records Basler or USB camera video with synchronized Arduino TTL outputs (gate, barcode, 1 Hz sync), and logs metadata.

## Features

- Basler and USB camera support (Basler via Pylon, USB via OpenCV)
- Live view with optional ROI cropping
- Recording with FFmpeg (GPU or CPU encoders)
- Per-frame metadata logging (timestamp, exposure, GPIO line status when available)
- Arduino TTL generation and live TTL plot
- Metadata templates saved to JSON plus TTL history saved to CSV

## Requirements

- Windows 10/11
- Python 3.10+ (recommended: Anaconda or Miniconda)
- FFmpeg in PATH
- Arduino with the `arduino_ttl_generator` sketch flashed

Optional (Basler cameras only):
- Basler Pylon SDK + `pypylon` (camera drivers)

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

## Arduino TTL Generator (Optional)

1. Open `arduino_ttl_generator/arduino_ttl_generator.ino` in Arduino IDE.
2. Flash to the Arduino.
3. Use the UI to scan and connect to the correct port.

Pin configuration (firmware defaults):
- Gate: 6, 7
- 1 Hz Sync: 8, 9
- Barcode: 10, 11

## Build camApp.exe

Install build tool:

```powershell
pip install pyinstaller
```

Build a single-file EXE:

```powershell
pyinstaller --onefile --noconsole --name camApp main.py
```

The output will be in `dist/camApp.exe`. Copy it to the project root if desired:

```powershell
copy dist\camApp.exe .\
```

Note: The compiled EXE still requires FFmpeg available on PATH at runtime.

## Troubleshooting

- "FFmpeg not found": add FFmpeg to PATH and restart the terminal.
- "Failed to start Arduino TTLs": make sure no other app is holding the COM port, then reconnect.
- "No Basler camera found": verify Pylon is installed and the camera is detected by Pylon Viewer.
- "No USB camera found": verify the device is connected and not in use by another app.
