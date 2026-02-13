"""
Camera Worker Thread - Clean Implementation
Handles camera acquisition, GPU-accelerated recording, and metadata logging.
"""
import time
import subprocess
import os
import threading
import numpy as np
import pandas as pd
import cv2
from typing import Optional, Dict, List
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from pypylon import pylon


class CameraWorker(QThread):
    """
    Worker thread for camera operations.
    Manages frame acquisition, FFmpeg encoding, and metadata collection.
    """

    # Signals
    frame_ready = Signal(np.ndarray)
    status_update = Signal(str)
    fps_update = Signal(float)
    buffer_update = Signal(int)
    error_occurred = Signal(str)
    recording_stopped = Signal()
    frame_recorded = Signal(dict)  # Signal for each recorded frame with metadata

    def __init__(self):
        super().__init__()

        # Camera
        self.camera: Optional[pylon.InstantCamera] = None
        self.usb_capture: Optional[cv2.VideoCapture] = None
        self.camera_type: Optional[str] = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Thread control
        self.running = False
        self.mutex = QMutex()

        # Recording
        self.is_recording = False
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.ffmpeg_stderr_thread: Optional[threading.Thread] = None
        self.metadata_buffer: List[Dict] = []
        self.recording_filename = ""
        self.frame_counter = 0

        # FFmpeg encoder settings
        self.encoder = "h264_nvenc"  # Default to NVIDIA GPU
        self.encoder_preset = "p4"
        self.bitrate = "5M"

        # Camera config
        self.trigger_mode = "FreeRun"
        self.width = 0
        self.height = 0
        self.fps_target = 30.0
        self.image_format = "Mono8"
        self.roi = None

        # FPS calculation
        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        self.line_label_map: Dict[str, str] = {}

    def set_encoder(self, encoder: str, preset: str = "p4", bitrate: str = "5M"):
        """Set FFmpeg encoder and parameters."""
        self.encoder = encoder
        self.encoder_preset = preset
        self.bitrate = bitrate

    def set_image_format(self, image_format: str):
        """Set image format for recording/display."""
        self.image_format = image_format
        if self.camera_type == "basler":
            if image_format == "BGR8":
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            else:
                self.converter.OutputPixelFormat = pylon.PixelType_Mono8

    def set_target_fps(self, fps: float):
        """Set target FPS for recording output."""
        self.fps_target = float(fps)

    def update_resolution(self, width: int, height: int):
        """Update cached resolution for recording output."""
        self.width = int(width)
        self.height = int(height)

    def set_roi(self, roi: Optional[dict]):
        """Set software ROI (x, y, w, h) for cropping."""
        self.roi = roi

    def set_line_label_map(self, label_map: Dict[str, str]):
        """Set optional suffixes for line status column labels."""
        self.line_label_map = label_map or {}

    def sync_camera_fps(self):
        """Sync FPS target from camera if possible."""
        if self.camera_type == "basler":
            if not self.camera or not self.camera.IsOpen():
                return
            fps = self._read_camera_fps()
            if fps:
                self.fps_target = fps
        elif self.camera_type == "usb" and self.usb_capture:
            usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
            if usb_fps and usb_fps > 0:
                self.fps_target = float(usb_fps)

    def connect_camera(self, camera_info: Optional[dict] = None) -> bool:
        """Connect to a Basler or USB camera."""
        try:
            camera_info = camera_info or {"type": "basler", "index": 0}
            if camera_info.get("type") == "usb":
                index = int(camera_info.get("index", 0))
                self.usb_capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
                if not self.usb_capture or not self.usb_capture.isOpened():
                    self.error_occurred.emit("No USB camera found!")
                    return False

                self.camera_type = "usb"
                self.usb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FPS, self.fps_target)

                self.width = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
                if usb_fps and usb_fps > 0:
                    self.fps_target = float(usb_fps)

                self.status_update.emit(f"USB camera connected: {self.width}x{self.height}")
                return True

            # Get Basler camera
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) == 0:
                self.error_occurred.emit("No Basler camera found!")
                return False

            index = int(camera_info.get("index", 0))
            index = max(0, min(index, len(devices) - 1))
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[index]))
            self.camera.Open()
            self.camera_type = "basler"

            # Configure camera
            self.camera.MaxNumBuffer = 10

            # Apply image format
            self.set_image_format(self.image_format)

            # Get camera resolution
            self.width = self.camera.Width.GetValue()
            self.height = self.camera.Height.GetValue()

            # Enable chunk mode for metadata
            self.camera.ChunkModeActive.SetValue(True)
            self.camera.ChunkSelector.SetValue('Timestamp')
            self.camera.ChunkEnable.SetValue(True)
            self.camera.ChunkSelector.SetValue('ExposureTime')
            self.camera.ChunkEnable.SetValue(True)

            # Try to enable line status
            try:
                self.camera.ChunkSelector.SetValue('LineStatusAll')
                self.camera.ChunkEnable.SetValue(True)
            except:
                pass  # Not all cameras support this

            self.status_update.emit(f"Basler camera connected: {self.width}x{self.height}")
            return True

        except Exception as e:
            self.error_occurred.emit(f"Camera connection error: {str(e)}")
            return False

    def disconnect_camera(self):
        """Disconnect camera."""
        with QMutexLocker(self.mutex):
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
                self.camera = None
            if self.usb_capture:
                self.usb_capture.release()
                self.usb_capture = None
            self.camera_type = None

    def set_trigger_mode(self, mode: str):
        """Set trigger mode: FreeRun or ExternalTrigger."""
        self.trigger_mode = mode
        if self.camera and self.camera.IsOpen():
            try:
                if mode == "ExternalTrigger":
                    self.camera.TriggerMode.SetValue("On")
                    self.camera.TriggerSource.SetValue("Line1")
                else:
                    self.camera.TriggerMode.SetValue("Off")
                self.status_update.emit(f"Trigger mode: {mode}")
            except Exception as e:
                self.error_occurred.emit(f"Trigger mode error: {str(e)}")

    def start_recording(self, filename: str) -> bool:
        """Start recording video and metadata."""
        with QMutexLocker(self.mutex):
            if self.is_recording:
                return False

            try:
                if self.camera_type == "basler" and self.camera and self.camera.IsOpen():
                    fps = self._read_camera_fps()
                    if fps:
                        self.fps_target = fps
                    try:
                        self.width = int(self.camera.Width.GetValue())
                        self.height = int(self.camera.Height.GetValue())
                    except Exception:
                        pass
                elif self.camera_type == "usb" and self.usb_capture:
                    usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
                    if usb_fps and usb_fps > 0:
                        self.fps_target = float(usb_fps)
                    self.width = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self.recording_filename = filename
                self.metadata_buffer = []
                self.frame_counter = 0

                # Start FFmpeg
                self._start_ffmpeg()

                self.is_recording = True
                self.status_update.emit(f"Recording: {Path(filename).name}.mp4")
                return True

            except Exception as e:
                self.error_occurred.emit(f"Recording start error: {str(e)}")
                return False

    def _start_ffmpeg(self):
        """Start FFmpeg process for video encoding."""
        output_file = f"{self.recording_filename}.mp4"
        effective_width, effective_height = self._get_effective_dimensions()

        # Build FFmpeg command based on encoder
        if self.encoder == "h264_nvenc":
            # NVIDIA GPU encoder
            codec_args = [
                '-c:v', 'h264_nvenc',
                '-preset', self.encoder_preset,
                '-b:v', self.bitrate,
            ]
        elif self.encoder == "libx264":
            # CPU encoder (software)
            codec_args = [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
            ]
        elif self.encoder == "h264_qsv":
            # Intel QuickSync
            codec_args = [
                '-c:v', 'h264_qsv',
                '-preset', 'fast',
                '-b:v', self.bitrate,
            ]
        else:
            # Fallback to libx264
            codec_args = [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
            ]

        pixel_format = 'gray'
        if self.image_format == "BGR8":
            pixel_format = 'bgr24'

        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{effective_width}x{effective_height}',
            '-pix_fmt', pixel_format,
            '-r', f'{self.fps_target:.3f}',
            '-i', '-',
        ] + codec_args + [
            '-pix_fmt', 'yuv420p', # Ensure compatible output format
            '-an',  # No audio
            output_file
        ]

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10**8,
                creationflags=creationflags
            )
            self._start_ffmpeg_stderr_thread()
            # Check if process died immediately
            time.sleep(0.1)
            if self.ffmpeg_process.poll() is not None:
                stderr = b""
                try:
                    _, stderr = self.ffmpeg_process.communicate(timeout=1)
                except Exception:
                    pass
                raise Exception(f"FFmpeg failed to start: {stderr.decode(errors='replace')}")

            self.status_update.emit(f"FFmpeg started: {self.encoder}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found! Install FFmpeg and add to PATH")
        except Exception as e:
            raise Exception(f"FFmpeg error: {str(e)}")

    def stop_recording(self):
        """Stop recording and save metadata."""
        with QMutexLocker(self.mutex):
            if not self.is_recording:
                return

            try:
                self.is_recording = False

                # Close FFmpeg
                if self.ffmpeg_process:
                    try:
                        self.ffmpeg_process.stdin.close()
                        self.ffmpeg_process.wait(timeout=10)
                    except:
                        self.ffmpeg_process.kill()
                    finally:
                        try:
                            if self.ffmpeg_process.stderr:
                                self.ffmpeg_process.stderr.close()
                        except Exception:
                            pass
                        self.ffmpeg_process = None
                        self.ffmpeg_stderr_thread = None

                # Save metadata
                self._save_metadata()

                self.status_update.emit("Recording stopped")
                self.recording_stopped.emit()

            except Exception as e:
                self.error_occurred.emit(f"Stop recording error: {str(e)}")
                self.recording_stopped.emit()

    def _save_metadata(self):
        """Save metadata buffer to CSV."""
        if not self.metadata_buffer:
            return

        try:
            df = pd.DataFrame(self.metadata_buffer)
            if self.line_label_map:
                rename_map = {
                    key: f"{key}_{suffix}"
                    for key, suffix in self.line_label_map.items()
                    if key in df.columns and suffix
                }
                if rename_map:
                    df = df.rename(columns=rename_map)
            csv_file = f"{self.recording_filename}_metadata.csv"
            df.to_csv(csv_file, index=False)
            self.status_update.emit(f"Metadata saved: {len(df)} frames")
        except Exception as e:
            self.error_occurred.emit(f"Metadata save error: {str(e)}")

    def run(self):
        """Main acquisition loop."""
        if self.camera_type == "basler":
            if not self.camera or not self.camera.IsOpen():
                self.error_occurred.emit("Camera not connected!")
                return
        elif self.camera_type == "usb":
            if not self.usb_capture or not self.usb_capture.isOpened():
                self.error_occurred.emit("USB camera not connected!")
                return
        else:
            self.error_occurred.emit("Camera not connected!")
            return

        try:
            self.running = True
            if self.camera_type == "basler":
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                while self.running:
                    try:
                        grab_result = self.camera.RetrieveResult(
                            5000,
                            pylon.TimeoutHandling_ThrowException
                        )

                        if grab_result.GrabSucceeded():
                            self._process_frame(grab_result)
                            self._update_fps()

                            # Update buffer
                            try:
                                num_queued = self.camera.NumQueuedBuffers.GetValue()
                                num_ready = self.camera.NumReadyBuffers.GetValue()
                                buffer_usage = int((num_queued / (num_queued + num_ready + 1)) * 100)
                                self.buffer_update.emit(buffer_usage)
                            except:
                                pass

                        grab_result.Release()

                    except pylon.TimeoutException:
                        self.status_update.emit("Frame timeout...")
                        continue

                self.camera.StopGrabbing()
            else:
                while self.running:
                    ok, frame = self.usb_capture.read()
                    if not ok:
                        self.status_update.emit("USB frame timeout...")
                        time.sleep(0.01)
                        continue

                    self._process_usb_frame(frame)
                    self._update_fps()

        except Exception as e:
            self.error_occurred.emit(f"Acquisition error: {str(e)}")
        finally:
            self.running = False

    def _process_frame(self, grab_result):
        """Process grabbed frame."""
        # Convert to Mono8
        image = self.converter.Convert(grab_result)
        img_array = image.GetArray()
        record_array = self._apply_roi(img_array)
        if self.image_format == "BGR8":
            display_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            display_array = img_array

        # Extract metadata
        metadata = self._extract_chunk_data(grab_result)
        metadata['timestamp_software'] = time.time()

        # If recording, write to FFmpeg and emit signal for TTL sync
        if self.is_recording and self.ffmpeg_process:
            try:
                # Write frame to FFmpeg
                self.ffmpeg_process.stdin.write(record_array.tobytes())

                # Add frame ID
                metadata['frame_id'] = self.frame_counter
                self.frame_counter += 1

                # Buffer metadata
                self.metadata_buffer.append(metadata.copy())

                # Emit for TTL sampling (sync with camera frames)
                self.frame_recorded.emit(metadata)

            except Exception as e:
                self.error_occurred.emit(f"Frame write error: {str(e)}")
                self.stop_recording()

        # Emit for display (every other frame to reduce GUI load)
        if self.fps_frame_count % 2 == 0:
            self.frame_ready.emit(display_array.copy())

    def _process_usb_frame(self, frame: np.ndarray):
        """Process USB camera frame."""
        metadata = {
            'timestamp_software': time.time(),
        }

        if self.image_format == "Mono8":
            record_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = record_frame
        else:
            record_frame = frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        record_frame = self._apply_roi(record_frame)

        if self.is_recording and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(record_frame.tobytes())

                metadata['frame_id'] = self.frame_counter
                self.frame_counter += 1

                self.metadata_buffer.append(metadata.copy())
                self.frame_recorded.emit(metadata)

            except Exception as e:
                self.error_occurred.emit(f"Frame write error: {str(e)}")
                self.stop_recording()

        if self.fps_frame_count % 2 == 0:
            self.frame_ready.emit(display_frame.copy())

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Apply software ROI cropping to a frame."""
        if not self.roi:
            return frame
        x = int(self.roi.get('x', 0))
        y = int(self.roi.get('y', 0))
        w = int(self.roi.get('w', frame.shape[1]))
        h = int(self.roi.get('h', frame.shape[0]))

        max_w = frame.shape[1]
        max_h = frame.shape[0]
        x = max(0, min(x, max_w - 1))
        y = max(0, min(y, max_h - 1))
        w = max(1, min(w, max_w - x))
        h = max(1, min(h, max_h - y))
        return frame[y:y + h, x:x + w]

    def _get_effective_dimensions(self) -> tuple[int, int]:
        """Get output dimensions after ROI."""
        if not self.roi:
            return self.width, self.height
        x = int(self.roi.get('x', 0))
        y = int(self.roi.get('y', 0))
        w = int(self.roi.get('w', self.width))
        h = int(self.roi.get('h', self.height))
        w = max(1, min(w, self.width - x))
        h = max(1, min(h, self.height - y))
        return w, h

    def _extract_chunk_data(self, grab_result) -> Dict:
        """Extract chunk metadata from frame."""
        metadata = {}

        try:
            # Timestamp
            if hasattr(grab_result, 'ChunkTimestamp'):
                try:
                    if callable(getattr(grab_result.ChunkTimestamp, 'IsReadable', None)):
                        if grab_result.ChunkTimestamp.IsReadable():
                            metadata['timestamp_ticks'] = grab_result.ChunkTimestamp.GetValue()
                    else:
                        metadata['timestamp_ticks'] = grab_result.ChunkTimestamp.GetValue()
                except:
                    metadata['timestamp_ticks'] = None
            else:
                metadata['timestamp_ticks'] = None

            # Exposure time
            if hasattr(grab_result, 'ChunkExposureTime'):
                try:
                    if callable(getattr(grab_result.ChunkExposureTime, 'IsReadable', None)):
                        if grab_result.ChunkExposureTime.IsReadable():
                            metadata['exposure_time_us'] = grab_result.ChunkExposureTime.GetValue()
                    else:
                        metadata['exposure_time_us'] = grab_result.ChunkExposureTime.GetValue()
                except:
                    metadata['exposure_time_us'] = None
            else:
                metadata['exposure_time_us'] = None

            # Line status (GPIO)
            if hasattr(grab_result, 'ChunkLineStatusAll'):
                try:
                    if callable(getattr(grab_result.ChunkLineStatusAll, 'IsReadable', None)):
                        if grab_result.ChunkLineStatusAll.IsReadable():
                            line_status = grab_result.ChunkLineStatusAll.GetValue()
                        else:
                            line_status = grab_result.ChunkLineStatusAll.GetValue()
                    else:
                        line_status = grab_result.ChunkLineStatusAll.GetValue()

                    metadata['line_status_all'] = line_status
                    # Correct extraction based on standard Basler Line Status All chunk
                    # Bit 0 = Line 1, Bit 1 = Line 2, etc.
                    metadata['line1_status'] = (line_status >> 0) & 0x01
                    metadata['line2_status'] = (line_status >> 1) & 0x01
                    metadata['line3_status'] = (line_status >> 2) & 0x01
                    metadata['line4_status'] = (line_status >> 3) & 0x01
                except:
                    metadata['line_status_all'] = None
                    metadata['line1_status'] = None
                    metadata['line2_status'] = None
                    metadata['line3_status'] = None
                    metadata['line4_status'] = None
            else:
                metadata['line_status_all'] = None
                metadata['line1_status'] = None
                metadata['line2_status'] = None
                metadata['line3_status'] = None
                metadata['line4_status'] = None

        except Exception as e:
            self.error_occurred.emit(f"Chunk data error: {str(e)}")

        return metadata

    def _update_fps(self):
        """Update FPS counter."""
        self.fps_frame_count += 1

        if self.fps_frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_last_time

            if elapsed > 0:
                fps = 30.0 / elapsed
                self.fps_update.emit(fps)
                self.fps_last_time = current_time

    def _read_camera_fps(self) -> Optional[float]:
        """Read the camera's actual or configured frame rate."""
        if not self.camera:
            return None

        candidates = [
            "ResultingFrameRate",
            "ResultingFrameRateAbs",
            "AcquisitionFrameRate",
            "AcquisitionFrameRateAbs",
        ]
        for name in candidates:
            try:
                node = getattr(self.camera, name)
            except Exception:
                continue
            try:
                if hasattr(node, "IsReadable") and not node.IsReadable():
                    continue
            except Exception:
                pass
            try:
                value = float(node.GetValue())
            except Exception:
                continue
            if value > 0:
                return value

        return None

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.is_recording:
            self.stop_recording()

    def _start_ffmpeg_stderr_thread(self):
        """Drain FFmpeg stderr to avoid blocking on full buffers."""
        if not self.ffmpeg_process or not self.ffmpeg_process.stderr:
            return

        def _drain():
            try:
                for _ in self.ffmpeg_process.stderr:
                    pass
            except Exception:
                pass

        self.ffmpeg_stderr_thread = threading.Thread(target=_drain, daemon=True)
        self.ffmpeg_stderr_thread.start()

