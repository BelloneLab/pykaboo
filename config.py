"""
Configuration file for camera and recording settings.
Modify these values to customize the application behavior.
"""

# === Camera Configuration ===
CAMERA_CONFIG = {
    # Acquisition settings
    'frame_rate_enable': True,
    'max_frame_rate': None,  # None = camera maximum, or specify value (e.g., 60.0)

    # Trigger settings
    'default_trigger_mode': 'FreeRun',  # 'FreeRun' or 'ExternalTrigger'
    'trigger_source': 'Line1',  # Input line for external trigger
    'trigger_activation': 'RisingEdge',  # 'RisingEdge', 'FallingEdge', 'AnyEdge'

    # Buffer settings
    'grab_strategy': 'LatestImageOnly',  # 'LatestImageOnly' or 'OneByOne'
    'grab_timeout_ms': 5000,  # Timeout for frame retrieval

    # GigE specific (if using GigE camera)
    'gige_packet_size': 1500,  # MTU size for GigE cameras
}

# === Recording Configuration ===
RECORDING_CONFIG = {
    # FFmpeg encoding
    'codec': 'h264_nvenc',  # 'h264_nvenc' (GPU) or 'libx264' (CPU)
    'preset': 'p4',  # For nvenc: 'p1'-'p7', for libx264: 'ultrafast', 'fast', 'medium'
    'bitrate': '5M',  # Video bitrate (e.g., '5M', '10M', '20M')
    'frame_rate': 30,  # Output video frame rate
    'pixel_format': 'gray',  # 'gray' for mono, 'rgb24' for color

    # File settings
    'output_format': 'mp4',  # Container format
    'default_output_dir': '.',  # Default save directory

    # Performance
    'ffmpeg_buffer_size': 10**8,  # Buffer size for FFmpeg pipe
}

# === GUI Configuration ===
GUI_CONFIG = {
    # Display settings
    'window_width': 1280,
    'window_height': 800,
    'video_display_min_width': 800,
    'video_display_min_height': 600,

    # Performance
    'display_frame_skip': 2,  # Display every Nth frame (1=all, 2=every other)
    'fps_update_interval': 1.0,  # FPS calculation interval in seconds

    # UI refresh
    'status_message_timeout': 5000,  # Status bar message timeout (ms)
}

# === Chunk Data Configuration ===
CHUNK_CONFIG = {
    # Enable/disable specific chunks
    'enable_timestamp': True,
    'enable_exposure_time': True,
    'enable_line_status': True,
    'enable_frame_counter': False,  # Not all cameras support this
    'enable_gain': False,  # Not all cameras support this
}

# === Metadata Configuration ===
METADATA_CONFIG = {
    # CSV export settings
    'include_software_timestamp': True,
    'decode_individual_lines': True,  # Decode Line1, Line2, Line3 from bitmask
    'timestamp_format': 'ticks',  # 'ticks' or 'seconds' (requires conversion)
}

# === Debug Configuration ===
DEBUG_CONFIG = {
    'verbose_logging': False,
    'log_chunk_data': False,
    'log_frame_timing': False,
}

