"""
Arduino TTL communication worker for CamApp Live Detection using pyFirmata.

This worker keeps the same GUI-facing API as the former pyserial-based worker,
but reads/writes digital signals through Firmata pins.
"""
import time
import sys
import inspect
import threading
from typing import Dict, List, Optional, Any

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

try:
    import pyfirmata
    from pyfirmata import util
    PYFIRMATA_IMPORT_ERROR = None
except Exception as exc:
    pyfirmata = None
    util = None
    PYFIRMATA_IMPORT_ERROR = exc

try:
    from serial.tools import list_ports
    PYSERIAL_IMPORT_ERROR = None
except Exception as exc:
    list_ports = None
    PYSERIAL_IMPORT_ERROR = exc

from PySide6.QtCore import QMutex, QMutexLocker, QSettings, QThread, Signal


class ArduinoOutputWorker(QThread):
    """
    Arduino TTL I/O worker.

    Communication is handled through Firmata pin reads/writes so the GUI can
    monitor and drive TTL lines without custom serial text commands.
    """

    SIGNAL_KEYS = ("gate", "sync", "barcode0", "barcode1", "lever", "cue", "reward", "iti")
    LIVE_OUTPUT_KEYS = tuple(f"do{i}" for i in range(1, 9))
    COUNT_KEY_MAP = {
        "gate": "gate_count",
        "sync": "sync_count",
        "barcode0": "barcode_count",
        "barcode1": "barcode_count",
        "lever": "lever_count",
        "cue": "cue_count",
        "reward": "reward_count",
        "iti": "iti_count",
    }
    EDGE_KEY_MAP = {
        "gate": "gate_edge_ms",
        "sync": "sync_edge_ms",
        "barcode0": "barcode_edge_ms",
        "barcode1": "barcode_edge_ms",
        "lever": "lever_edge_ms",
        "cue": "cue_edge_ms",
        "reward": "reward_edge_ms",
        "iti": "iti_edge_ms",
    }
    DEFAULT_PIN_CONFIG = {
        "gate": [3],
        "sync": [9],
        "barcode": [18],
        "lever": [14],
        "cue": [45],
        "reward": [21],
        "iti": [46],
        "do1": [],
        "do2": [],
        "do3": [],
        "do4": [],
        "do5": [],
        "do6": [],
        "do7": [],
        "do8": [],
    }
    DEFAULT_SIGNAL_ROLES = {
        "gate": "Output",
        "sync": "Output",
        "barcode": "Output",
        "lever": "Input",
        "cue": "Output",
        "reward": "Output",
        "iti": "Output",
        "do1": "Output",
        "do2": "Output",
        "do3": "Output",
        "do4": "Output",
        "do5": "Output",
        "do6": "Output",
        "do7": "Output",
        "do8": "Output",
    }
    # Host-side TTL timing parameters.
    SYNC_1HZ_PERIOD_S = 1.0
    SYNC_1HZ_PULSE_S = 0.05
    BARCODE_BITS = 32
    BARCODE_START_PULSE_S = 0.1
    BARCODE_START_LOW_S = 0.1
    BARCODE_BIT_S = 0.1
    BARCODE_INTERVAL_S = 5.0  # Gap after one barcode word before the next starts.
    BC_START_HI = 0
    BC_START_LO = 1
    BC_BITS = 2
    BC_GAP = 3
    # Lower sampling interval improves reliability for short input pulses.
    FIRMATA_SAMPLING_INTERVAL_MS = 2

    # Hardware barcode (StandardFirmataBarcode) sysex protocol.
    # When hw_barcode_enabled is True the Arduino Timer1 ISR drives D8 (word
    # sync) and D9 (data, LSB-first) autonomously — Python only sends
    # start / stop / configure commands via sysex.
    HW_BARCODE_SYSEX_CMD = 0x7D
    HW_BARCODE_START = 0x01
    HW_BARCODE_STOP = 0x02
    HW_BARCODE_SET_BITWIDTH = 0x03
    HW_BARCODE_RESET_COUNTER = 0x04
    HW_BARCODE_QUERY_STATUS = 0x05
    HW_BARCODE_PIN_SYNC = 8   # Timer1 OC1B — word-sync pulse
    HW_BARCODE_PIN_DATA = 9   # Timer1 OC1A — data (LSB first)

    COUNTER_SYSEX_CMD = 0x0F
    COUNTER_START_CMD = 0x01
    COUNTER_STOP_CMD = 0x02
    COUNTER_RESTART_CMD = 0x03

    # Signals
    port_list_updated = Signal(list)
    connection_status = Signal(bool, str)
    ttl_states_updated = Signal(dict)
    pin_config_received = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()

        self.board: Optional[Any] = None
        self.iterator: Optional[util.Iterator] = None
        # Compatibility attribute used by some legacy diagnostics.
        self.serial_port = None

        self.running = False
        self.mutex = QMutex()
        self.is_generating = False

        # Configuration
        self.port_name = ""
        self.baud_rate = 57600  # Standard Firmata default
        self.pin_config = {key: pins.copy() for key, pins in self.DEFAULT_PIN_CONFIG.items()}
        self.signal_roles = self.DEFAULT_SIGNAL_ROLES.copy()
        self.pin_handles: Dict[str, List[Any]] = {key: [] for key in self.DEFAULT_PIN_CONFIG}
        self.output_shadow = {key: False for key in self.SIGNAL_KEYS}
        self.live_output_shadow = {key: False for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_levels = {key: False for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_pulses_until = {key: 0.0 for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_pulse_trains = {key: [] for key in self.LIVE_OUTPUT_KEYS}
        self.live_pulse_scheduler_stop = threading.Event()
        self.live_pulse_scheduler_wake = threading.Event()
        self.live_pulse_scheduler_thread: Optional[threading.Thread] = None

        self.generation_mode = "idle"
        self.generation_start_time = 0.0
        self.sync_output_state = False
        self.sync_next_time = 0.0
        self.barcode_state = self.BC_GAP
        self.barcode_next_time = 0.0
        self.barcode_bit_index = 0
        self.barcode_word = 0

        # Hardware barcode (Timer1 ISR on Arduino)
        self.hw_barcode_enabled = False
        self.hw_barcode_running = False
        self.hw_barcode_bit_width_ms = 80
        self.hw_barcode_status: Optional[Dict] = None
        self._hw_barcode_last_query = 0.0

        self.last_serial_error_message = ""
        self.last_serial_error_time = 0.0
        self.serial_error_cooldown_s = 1.0

        # Current states
        self.current_states = {key: False for key in self.SIGNAL_KEYS}
        self.last_state_packet = self.current_states.copy()

        # TTL history for frame-synced logging
        self.ttl_history: List[Dict] = []
        self.last_state_emit = 0.0
        self.ttl_event_history: List[Dict] = []
        self.ttl_pulse_counts = {key: 0 for key in self.SIGNAL_KEYS}
        self.last_event_state = self.current_states.copy()
        self.live_state_history: List[Dict] = []
        self.max_live_state_history = 50000

        # Maintain old public attributes used by UI code
        self.gate_pins = self.pin_config["gate"].copy()
        self.sync_pins = self.pin_config["sync"].copy()
        self.barcode_pins = self.pin_config["barcode"].copy()

        self.settings = QSettings("CamApp Live Detection", "CamApp Live Detection")
        self._migrate_legacy_settings()
        self.load_settings()

    # ===== Settings / Config =====

    def _migrate_legacy_settings(self):
        """Copy saved settings from the previous app identity once."""
        legacy_settings = QSettings("BaslerCam", "CameraApp")
        existing_keys = set(self.settings.allKeys())
        copied = False

        for key in legacy_settings.allKeys():
            if key in existing_keys:
                continue
            self.settings.setValue(key, legacy_settings.value(key))
            copied = True

        if copied:
            self.settings.sync()

    def load_settings(self):
        """Load saved Arduino/Firmata settings."""
        self.port_name = self.settings.value("arduino_port", "")
        self.baud_rate = int(self.settings.value("arduino_baud_rate", 57600))

        for key, default_pins in self.DEFAULT_PIN_CONFIG.items():
            raw_value = self.settings.value(f"behavior_pin_{key}", None)
            pins = self._parse_pin_setting_value(raw_value)
            if pins:
                self.pin_config[key] = pins
            else:
                self.pin_config[key] = default_pins.copy()

        for key, default_role in self.DEFAULT_SIGNAL_ROLES.items():
            raw_role = self.settings.value(f"behavior_role_{key}", default_role)
            self.signal_roles[key] = self._normalize_signal_role(raw_role, default_role)

        sync_period = self._safe_float(
            self.settings.value("sync_period_s", self.SYNC_1HZ_PERIOD_S),
            self.SYNC_1HZ_PERIOD_S,
        )
        sync_pulse = self._safe_float(
            self.settings.value("sync_pulse_s", self.SYNC_1HZ_PULSE_S),
            self.SYNC_1HZ_PULSE_S,
        )
        if sync_period <= 0.0:
            sync_period = 1.0
        sync_pulse = min(max(sync_pulse, 0.001), max(0.001, sync_period - 0.001))
        self.SYNC_1HZ_PERIOD_S = float(sync_period)
        self.SYNC_1HZ_PULSE_S = float(sync_pulse)

        barcode_bits = self._safe_int(
            self.settings.value("barcode_bits", self.BARCODE_BITS),
            default=self.BARCODE_BITS,
        )
        self.BARCODE_BITS = max(1, min(64, int(barcode_bits)))
        self.BARCODE_START_PULSE_S = max(
            0.001,
            self._safe_float(
                self.settings.value("barcode_start_pulse_s", self.BARCODE_START_PULSE_S),
                self.BARCODE_START_PULSE_S,
            ),
        )
        self.BARCODE_START_LOW_S = max(
            0.001,
            self._safe_float(
                self.settings.value("barcode_start_low_s", self.BARCODE_START_LOW_S),
                self.BARCODE_START_LOW_S,
            ),
        )
        self.BARCODE_BIT_S = max(
            0.001,
            self._safe_float(
                self.settings.value("barcode_bit_s", self.BARCODE_BIT_S),
                self.BARCODE_BIT_S,
            ),
        )
        self.BARCODE_INTERVAL_S = max(
            0.0,
            self._safe_float(
                self.settings.value("barcode_interval_s", self.BARCODE_INTERVAL_S),
                self.BARCODE_INTERVAL_S,
            ),
        )

        # Hardware barcode settings
        self.hw_barcode_enabled = bool(
            int(self.settings.value("hw_barcode_enabled", 0))
        )
        self.hw_barcode_bit_width_ms = max(
            20,
            min(500, self._safe_int(
                self.settings.value("hw_barcode_bit_width_ms", 80), 80
            )),
        )

        self._refresh_legacy_pin_attributes()

    def save_settings(self):
        """Persist current board settings."""
        self.settings.setValue("arduino_port", self.port_name)
        self.settings.setValue("arduino_baud_rate", int(self.baud_rate))
        for key, pins in self.pin_config.items():
            self.settings.setValue(f"behavior_pin_{key}", ",".join(str(int(pin)) for pin in pins))
        for key, role in self.signal_roles.items():
            self.settings.setValue(f"behavior_role_{key}", role)
        self.settings.setValue("sync_period_s", float(self.SYNC_1HZ_PERIOD_S))
        self.settings.setValue("sync_pulse_s", float(self.SYNC_1HZ_PULSE_S))
        self.settings.setValue("barcode_bits", int(self.BARCODE_BITS))
        self.settings.setValue("barcode_start_pulse_s", float(self.BARCODE_START_PULSE_S))
        self.settings.setValue("barcode_start_low_s", float(self.BARCODE_START_LOW_S))
        self.settings.setValue("barcode_bit_s", float(self.BARCODE_BIT_S))
        self.settings.setValue("barcode_interval_s", float(self.BARCODE_INTERVAL_S))
        self.settings.setValue("hw_barcode_enabled", int(self.hw_barcode_enabled))
        self.settings.setValue("hw_barcode_bit_width_ms", int(self.hw_barcode_bit_width_ms))
        self.settings.sync()

    def set_manual_pin_config(self, pin_config: Dict[str, List[int]]):
        """Apply manual pin mapping from GUI configuration."""
        if not isinstance(pin_config, dict):
            return

        with QMutexLocker(self.mutex):
            updated = self.pin_config.copy()
            for raw_key, raw_pins in pin_config.items():
                key = self._normalize_pin_key(str(raw_key))
                if not key:
                    continue
                pins = self._normalize_pin_list(raw_pins)
                if pins:
                    updated[key] = pins

            self.pin_config = updated
            self._refresh_legacy_pin_attributes()
            self.save_settings()
            if self.board is not None:
                self._configure_pin_handles_locked()

        self.pin_config_received.emit(self.pin_config.copy())

    def set_signal_roles(self, role_config: Dict[str, str]):
        """Apply per-signal Input/Output role mapping."""
        if not isinstance(role_config, dict):
            return

        with QMutexLocker(self.mutex):
            updated = self.signal_roles.copy()
            for raw_key, raw_role in role_config.items():
                key = self._normalize_pin_key(str(raw_key))
                if key is None:
                    continue
                updated[key] = self._normalize_signal_role(raw_role, updated.get(key, "Output"))
            self.signal_roles = updated
            self.save_settings()
            if self.board is not None:
                self._configure_pin_handles_locked()

    def get_sync_parameters(self):
        """Return sync timing parameters."""
        with QMutexLocker(self.mutex):
            return float(self.SYNC_1HZ_PERIOD_S), float(self.SYNC_1HZ_PULSE_S)

    def set_sync_parameters(self, period_s: float, pulse_s: float):
        """Update sync timing parameters."""
        with QMutexLocker(self.mutex):
            period = max(0.01, float(period_s))
            pulse = max(0.001, float(pulse_s))
            pulse = min(pulse, max(0.001, period - 0.001))
            self.SYNC_1HZ_PERIOD_S = period
            self.SYNC_1HZ_PULSE_S = pulse
            self.save_settings()
            if self.is_generating:
                self._reset_signal_generators_locked(time.time())

    def get_barcode_parameters(self) -> Dict[str, Any]:
        """Return barcode state-machine parameters."""
        with QMutexLocker(self.mutex):
            word_duration = self._barcode_word_duration_seconds()
            return {
                "bits": int(self.BARCODE_BITS),
                "start_pulse_s": float(self.BARCODE_START_PULSE_S),
                "start_low_s": float(self.BARCODE_START_LOW_S),
                "bit_s": float(self.BARCODE_BIT_S),
                "interval_s": float(self.BARCODE_INTERVAL_S),
                "word_duration_s": float(word_duration),
                "cycle_duration_s": float(word_duration + self.BARCODE_INTERVAL_S),
                "output_pins": self.pin_config.get("barcode", []).copy(),
            }

    def set_barcode_parameters(
        self,
        bits: int,
        start_pulse_s: float,
        start_low_s: float,
        bit_s: float,
        interval_s: float,
    ):
        """Update barcode state-machine parameters."""
        with QMutexLocker(self.mutex):
            self.BARCODE_BITS = max(1, min(64, int(bits)))
            self.BARCODE_START_PULSE_S = max(0.001, float(start_pulse_s))
            self.BARCODE_START_LOW_S = max(0.001, float(start_low_s))
            self.BARCODE_BIT_S = max(0.001, float(bit_s))
            self.BARCODE_INTERVAL_S = max(0.0, float(interval_s))
            self.save_settings()
            if self.is_generating:
                self._reset_signal_generators_locked(time.time())

    # ===== Hardware Barcode (Timer1 ISR) Control =====

    def get_hw_barcode_enabled(self) -> bool:
        with QMutexLocker(self.mutex):
            return bool(self.hw_barcode_enabled)

    def set_hw_barcode_enabled(self, enabled: bool):
        with QMutexLocker(self.mutex):
            self.hw_barcode_enabled = bool(enabled)
            self.settings.setValue("hw_barcode_enabled", int(self.hw_barcode_enabled))
            self.settings.sync()

    def get_hw_barcode_bit_width_ms(self) -> int:
        with QMutexLocker(self.mutex):
            return int(self.hw_barcode_bit_width_ms)

    def set_hw_barcode_bit_width_ms(self, ms: int):
        ms = max(20, min(500, int(ms)))
        with QMutexLocker(self.mutex):
            self.hw_barcode_bit_width_ms = ms
            self.settings.setValue("hw_barcode_bit_width_ms", ms)
            self.settings.sync()
            if self.board is not None:
                self._send_hw_barcode_set_bitwidth_locked(ms)

    def get_hw_barcode_status(self) -> Optional[Dict]:
        with QMutexLocker(self.mutex):
            return self.hw_barcode_status.copy() if self.hw_barcode_status else None

    def _register_hw_barcode_sysex_locked(self):
        """Register the sysex handler for barcode status replies on the current board."""
        if self.board is None:
            return
        try:
            self.board.add_cmd_handler(self.HW_BARCODE_SYSEX_CMD, self._handle_hw_barcode_sysex)
        except Exception:
            pass

    def _handle_hw_barcode_sysex(self, *data):
        """Parse a sysex status reply from StandardFirmataBarcode."""
        if len(data) < 10 or data[0] != self.HW_BARCODE_QUERY_STATUS:
            return
        running = bool(data[1])
        bit_width = int(data[2]) | (int(data[3]) << 7)
        counter = (int(data[4]) | (int(data[5]) << 7) |
                   (int(data[6]) << 14) | (int(data[7]) << 21) |
                   (int(data[8]) << 28))
        bit_index = int(data[9])
        with QMutexLocker(self.mutex):
            self.hw_barcode_status = {
                "running": running,
                "bit_width_ms": bit_width,
                "counter": counter,
                "bit_index": bit_index,
            }
            self.hw_barcode_running = running

    def _send_hw_barcode_sysex_locked(self, *payload):
        """Send a sysex message to the barcode module (must hold mutex)."""
        if self.board is None:
            return
        try:
            self.board.send_sysex(self.HW_BARCODE_SYSEX_CMD, list(payload))
        except Exception:
            pass

    def _send_counter_sysex_locked(self, command: int):
        """Send a TM1638 counter control command (must hold mutex)."""
        if self.board is None:
            return
        try:
            self.board.send_sysex(self.COUNTER_SYSEX_CMD, [int(command) & 0x7F])
        except Exception:
            pass

    def _send_hw_barcode_set_bitwidth_locked(self, ms: int):
        ms = max(20, min(500, int(ms)))
        low7 = ms & 0x7F
        high7 = (ms >> 7) & 0x7F
        self._send_hw_barcode_sysex_locked(self.HW_BARCODE_SET_BITWIDTH, low7, high7)

    def _start_hw_barcode_locked(self):
        """Start hardware barcode generation via sysex (must hold mutex)."""
        self._send_hw_barcode_set_bitwidth_locked(self.hw_barcode_bit_width_ms)
        self._send_hw_barcode_sysex_locked(self.HW_BARCODE_RESET_COUNTER)
        self._send_hw_barcode_sysex_locked(self.HW_BARCODE_START)
        self.hw_barcode_running = True

    def _stop_hw_barcode_locked(self):
        """Stop hardware barcode generation via sysex (must hold mutex)."""
        self._send_hw_barcode_sysex_locked(self.HW_BARCODE_STOP)
        self.hw_barcode_running = False

    def _query_hw_barcode_status_locked(self):
        """Send a status query; reply arrives asynchronously via sysex handler."""
        self._send_hw_barcode_sysex_locked(self.HW_BARCODE_QUERY_STATUS)

    def _refresh_legacy_pin_attributes(self):
        self.gate_pins = self.pin_config.get("gate", []).copy()
        self.sync_pins = self.pin_config.get("sync", []).copy()
        self.barcode_pins = self.pin_config.get("barcode", []).copy()

    def _reserved_output_pins_locked(self) -> set[int]:
        reserved: set[int] = set()
        for key in ("gate", "sync", "barcode"):
            reserved.update(int(pin) for pin in self.pin_config.get(key, []) if pin is not None)
        return reserved

    def _normalize_signal_role(self, role_value, default_role: str) -> str:
        role_text = str(role_value).strip().lower()
        if role_text in ("input", "in", "i"):
            return "Input"
        if role_text in ("output", "out", "o"):
            return "Output"
        return default_role

    # ===== Port / Board Management =====

    def scan_ports(self) -> List[str]:
        """Scan available COM ports."""
        if list_ports is None:
            message = self._missing_serial_dependency_message()
            self.error_occurred.emit(message)
            self.port_list_updated.emit([])
            return []

        ports = list_ports.comports()
        port_list = []

        for port in ports:
            if "Arduino" in port.description or "CH340" in port.description or "USB" in port.description:
                port_list.append(f"{port.device} - {port.description}")

        if not port_list:
            port_list = [f"{port.device} - {port.description}" for port in ports]

        self.port_list_updated.emit(port_list)
        return port_list

    def connect_to_port(self, port_name: str) -> bool:
        """Connect to an Arduino running Firmata on the selected COM port."""
        if " - " in str(port_name):
            port_name = str(port_name).split(" - ")[0]

        port_name = str(port_name).strip()
        if not port_name:
            self.error_occurred.emit("No COM port selected.")
            self.connection_status.emit(False, "No COM port selected")
            return False

        if pyfirmata is None or util is None or PYFIRMATA_IMPORT_ERROR is not None:
            message = self._missing_firmata_dependency_message()
            self.error_occurred.emit(message)
            self.connection_status.emit(False, message)
            return False

        if list_ports is None or PYSERIAL_IMPORT_ERROR is not None:
            message = self._missing_serial_dependency_message()
            self.error_occurred.emit(message)
            self.connection_status.emit(False, message)
            return False

        available_ports = self._available_port_names()
        if available_ports and port_name.upper() not in {port.upper() for port in available_ports}:
            message = (
                f"{port_name} is not currently listed by Windows. "
                f"Available serial ports: {', '.join(available_ports)}. "
                "Unplug/replug the Arduino, click Scan, then select the listed Arduino COM port."
            )
            self.error_occurred.emit(message)
            self.connection_status.emit(False, message)
            return False

        board = None
        iterator = None

        with QMutexLocker(self.mutex):
            previous_board = self._detach_board_locked()
        if previous_board is not None:
            self._close_board(previous_board)

        try:
            constructors = []
            mega_ctor = getattr(pyfirmata, "ArduinoMega", None)
            std_ctor = getattr(pyfirmata, "Arduino", None)
            if callable(mega_ctor):
                constructors.append(mega_ctor)
            if callable(std_ctor) and std_ctor not in constructors:
                constructors.append(std_ctor)
            if not constructors:
                raise RuntimeError("pyFirmata board constructors not available")

            last_error = None
            for ctor in constructors:
                try:
                    board = ctor(port_name)
                    break
                except Exception as exc:
                    last_error = exc
                    board = None
                    if self._is_access_denied_error(exc):
                        break

            if board is None:
                raise RuntimeError(f"Unable to connect to {port_name}: {last_error}")

            iterator = util.Iterator(board)
            iterator.start()
            time.sleep(0.2)

            with QMutexLocker(self.mutex):
                self.board = board
                self.iterator = iterator
                self.serial_port = getattr(board, "sp", None)
                self.port_name = port_name
                self.baud_rate = 57600
                self.is_generating = False
                self.generation_mode = "idle"
                self.generation_start_time = 0.0
                self.current_states = {key: False for key in self.SIGNAL_KEYS}
                self.output_shadow = {key: False for key in self.SIGNAL_KEYS}
                self.live_output_shadow = {key: False for key in self.LIVE_OUTPUT_KEYS}
                self.live_output_levels = {key: False for key in self.LIVE_OUTPUT_KEYS}
                self.live_output_pulses_until = {key: 0.0 for key in self.LIVE_OUTPUT_KEYS}
                self.live_output_pulse_trains = {key: [] for key in self.LIVE_OUTPUT_KEYS}
                self._reset_ttl_event_tracking()
                self._configure_pin_handles_locked()
                self._register_hw_barcode_sysex_locked()
                if self.hw_barcode_enabled:
                    self._send_hw_barcode_set_bitwidth_locked(self.hw_barcode_bit_width_ms)
                self.save_settings()

            self.pin_config_received.emit(self.pin_config.copy())
            self.connection_status.emit(True, f"Connected to {port_name} via Firmata")
            return True

        except Exception as e:
            if iterator is not None:
                try:
                    iterator.join(timeout=0.1)
                except Exception:
                    pass
            if board is not None:
                self._close_board(board)
            message = self._connection_error_message(port_name, e)
            self.error_occurred.emit(message)
            self.connection_status.emit(False, message)
            return False

    def get_live_output_mapping(self) -> Dict[str, List[int]]:
        with QMutexLocker(self.mutex):
            return {
                key.upper(): self.pin_config.get(key, []).copy()
                for key in self.LIVE_OUTPUT_KEYS
            }

    def configure_live_output_mapping(self, mapping: Dict[str, List[int] | str]):
        """Persist DO1..DO8 logical output pin mappings with reserved-pin validation."""
        if not isinstance(mapping, dict):
            return

        with QMutexLocker(self.mutex):
            updated = self.pin_config.copy()
            reserved = self._reserved_output_pins_locked()
            for raw_key, raw_pins in mapping.items():
                key = self._normalize_pin_key(str(raw_key))
                if key not in self.LIVE_OUTPUT_KEYS:
                    continue
                pins = self._normalize_pin_list(raw_pins)
                overlap = sorted(set(pins) & reserved)
                if overlap:
                    raise ValueError(
                        f"{key.upper()} cannot reuse reserved gate/sync/barcode pin(s): "
                        + ", ".join(str(pin) for pin in overlap)
                    )
                updated[key] = pins

            self.pin_config = updated
            self.save_settings()
            if self.board is not None:
                self._configure_pin_handles_locked()

        self.pin_config_received.emit(self.pin_config.copy())

    def set_live_output_level(self, output_id: str, active: bool):
        """Hold one logical live output high or low until changed again."""
        key = self._normalize_pin_key(output_id)
        if key not in self.LIVE_OUTPUT_KEYS:
            return
        with QMutexLocker(self.mutex):
            self.live_output_levels[key] = bool(active)
            if self.board is not None:
                self._apply_live_output_state_locked(key, time.time())

    def start_live_output_pulse(self, output_id: str, duration_ms: int):
        """Pulse one logical live output high for the requested duration."""
        self.start_live_output_pulse_train(output_id, duration_ms, pulse_count=1, pulse_frequency_hz=1.0)

    def start_live_output_pulse_train(
        self,
        output_id: str,
        duration_ms: int,
        pulse_count: int = 1,
        pulse_frequency_hz: float = 1.0,
    ):
        """Start a pulse train on one logical live output."""
        key = self._normalize_pin_key(output_id)
        if key not in self.LIVE_OUTPUT_KEYS:
            return
        with QMutexLocker(self.mutex):
            duration_s = max(0.001, float(duration_ms) / 1000.0)
            count = max(1, int(pulse_count))
            if count > 1:
                frequency_hz = max(0.001, float(pulse_frequency_hz))
                period_s = max(duration_s, 1.0 / frequency_hz)
            else:
                period_s = duration_s
            now = time.time()
            train = (now, duration_s, period_s, count)
            trains = self._pruned_live_output_pulse_trains_locked(key, now)
            trains.append(train)
            self.live_output_pulse_trains[key] = trains
            self.live_output_pulses_until[key] = max(
                [self._live_output_pulse_train_end(train_entry) for train_entry in trains] or [0.0]
            )
            if self.board is not None:
                self._apply_live_output_state_locked(key, now)
                self._ensure_live_pulse_scheduler_locked()

    def clear_live_outputs(self):
        """Drop all live-detection output levels and pending pulses."""
        with QMutexLocker(self.mutex):
            self._set_all_live_outputs_low_locked()

    def disconnect_port(self):
        """Disconnect from Arduino/Firmata board."""
        with QMutexLocker(self.mutex):
            if self.board is not None:
                self._send_counter_sysex_locked(self.COUNTER_STOP_CMD)
                if self.hw_barcode_running:
                    self._stop_hw_barcode_locked()
                self._set_all_outputs_low_locked()
                self._set_all_live_outputs_low_locked()
            board = self._detach_board_locked()

        if board is not None:
            self._close_board(board)

        self.connection_status.emit(False, "Disconnected")

    def _detach_board_locked(self):
        """Detach board object from worker state. Must be called with mutex held."""
        self._stop_live_pulse_scheduler_locked()
        board = self.board
        self.board = None
        self.iterator = None
        self.serial_port = None
        self.pin_handles = {key: [] for key in self.DEFAULT_PIN_CONFIG}
        self.is_generating = False
        self.generation_mode = "idle"
        self.generation_start_time = 0.0
        self.current_states = {key: False for key in self.SIGNAL_KEYS}
        self.output_shadow = {key: False for key in self.SIGNAL_KEYS}
        self.live_output_shadow = {key: False for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_levels = {key: False for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_pulses_until = {key: 0.0 for key in self.LIVE_OUTPUT_KEYS}
        self.live_output_pulse_trains = {key: [] for key in self.LIVE_OUTPUT_KEYS}
        self.hw_barcode_running = False
        self.hw_barcode_status = None
        self._reset_signal_generators_locked(time.time())
        self._reset_ttl_event_tracking()
        return board

    def _close_board(self, board):
        """Best-effort board shutdown."""
        try:
            board.exit()
            return
        except Exception:
            pass

        try:
            serial_handle = getattr(board, "sp", None)
            if serial_handle is not None:
                serial_handle.close()
        except Exception:
            pass

    def _set_firmata_sampling_interval_locked(self, interval_ms: int):
        """Best-effort Firmata sampling interval configuration."""
        if self.board is None:
            return

        interval = max(1, int(interval_ms))
        board = self.board

        try:
            sampling_on = getattr(board, "samplingOn", None)
            if callable(sampling_on):
                sampling_on(interval)
                return
        except Exception:
            pass

        try:
            set_sampling = getattr(board, "setSamplingInterval", None)
            if callable(set_sampling):
                set_sampling(interval)
                return
        except Exception:
            pass

        try:
            # Standard Firmata SAMPLING_INTERVAL sysex command.
            board.send_sysex(0x7A, [interval & 0x7F, (interval >> 7) & 0x7F])
        except Exception:
            pass

    def _board_digital_pin_handle_locked(self, pin: int):
        """Return board.digital[pin] if available."""
        if self.board is None:
            return None

        try:
            bank = getattr(self.board, "digital", None)
            if bank is None:
                return None
            if pin < 0 or pin >= len(bank):
                return None
            return bank[pin]
        except Exception:
            return None

    def _configure_input_pin_locked(self, pin: int, handle):
        """Set pin to INPUT and enable reporting."""
        input_mode = getattr(pyfirmata, "INPUT", 0)

        try:
            handle.mode = input_mode
        except Exception:
            pass
        try:
            handle.enable_reporting()
        except Exception:
            pass

        # Some pyFirmata board layouts only report reliably via board.digital.
        board_pin = self._board_digital_pin_handle_locked(pin)
        if board_pin is None or board_pin is handle:
            return

        try:
            board_pin.mode = input_mode
        except Exception:
            pass
        try:
            board_pin.enable_reporting()
        except Exception:
            pass

    def _configure_output_pin_locked(self, pin: int, handle):
        """Set pin to OUTPUT and drive LOW."""
        output_mode = getattr(pyfirmata, "OUTPUT", 1)

        try:
            handle.mode = output_mode
        except Exception:
            pass
        try:
            handle.write(0)
        except Exception:
            pass

        board_pin = self._board_digital_pin_handle_locked(pin)
        if board_pin is None or board_pin is handle:
            return

        try:
            board_pin.mode = output_mode
        except Exception:
            pass
        try:
            board_pin.write(0)
        except Exception:
            pass

    def _create_digital_pin_handle_locked(self, pin: int, mode: str):
        """Create/configure one digital pin handle with fallback paths."""
        handle = self._board_digital_pin_handle_locked(pin)

        if handle is None and self.board is not None:
            descriptor = f"d:{int(pin)}:{mode}"
            try:
                handle = self.board.get_pin(descriptor)
            except Exception:
                handle = None

        if handle is None:
            return None

        if mode == "i":
            self._configure_input_pin_locked(pin, handle)
        else:
            self._configure_output_pin_locked(pin, handle)

        return handle

    def _configure_pin_handles_locked(self):
        """Create Firmata pin handles for current pin map and role map."""
        self.pin_handles = {key: [] for key in self.DEFAULT_PIN_CONFIG}
        if self.board is None:
            return

        self._set_firmata_sampling_interval_locked(self.FIRMATA_SAMPLING_INTERVAL_MS)
        unresolved = []

        for base_key, pins in self.pin_config.items():
            role = self.signal_roles.get(base_key, self.DEFAULT_SIGNAL_ROLES.get(base_key, "Output"))
            mode = "i" if role == "Input" else "o"
            handles = []
            for raw_pin in pins:
                pin = self._safe_int(raw_pin, default=None)
                if pin is None or pin < 0:
                    continue
                handle = self._create_digital_pin_handle_locked(int(pin), mode)
                if handle is None:
                    unresolved.append(f"{base_key}:{int(pin)}")
                    continue
                handles.append(handle)
            self.pin_handles[base_key] = handles

        if unresolved:
            self._emit_error_throttled(
                "Could not configure Firmata pin(s): " + ", ".join(unresolved[:8])
            )

        self._refresh_legacy_pin_attributes()
        self._sync_output_shadow_to_states_locked()

    # ===== Control Commands =====

    def start_recording(self):
        """Arm recording mode output generation (Firmata)."""
        with QMutexLocker(self.mutex):
            if self.board is None:
                return False
            now = time.time()
            self._set_all_outputs_low_locked()
            self.is_generating = True
            self.generation_mode = "recording"
            self.generation_start_time = now
            self._reset_signal_generators_locked(now)
            self.ttl_history.clear()
            self._reset_ttl_event_tracking()
            self._send_counter_sysex_locked(self.COUNTER_RESTART_CMD)
            if self.hw_barcode_enabled:
                self._start_hw_barcode_locked()
            return True

    def stop_recording(self):
        """Stop recording mode output generation without discarding collected telemetry."""
        with QMutexLocker(self.mutex):
            if self.board is None:
                return False
            self._send_counter_sysex_locked(self.COUNTER_STOP_CMD)
            self._stop_generation_locked()
            return True

    def start_test(self):
        """Start test mode output generation/monitoring."""
        with QMutexLocker(self.mutex):
            if self.board is None:
                return False
            now = time.time()
            self._set_all_outputs_low_locked()
            self.is_generating = True
            self.generation_mode = "test"
            self.generation_start_time = now
            self._reset_signal_generators_locked(now)
            self._reset_ttl_event_tracking()
            if self.hw_barcode_enabled:
                self._start_hw_barcode_locked()
            return True

    def stop_test(self):
        """Stop test mode generation/monitoring without clearing counters immediately."""
        with QMutexLocker(self.mutex):
            if self.board is None:
                return False
            self._stop_generation_locked()
            return True

    # ===== Data Sampling =====

    def sample_ttl_state(self, frame_metadata: Dict):
        """
        Sample TTL state synchronized with camera frame.
        Called when each camera frame is recorded.
        """
        with QMutexLocker(self.mutex):
            now = time.time()
            self._update_live_outputs_locked(now)
            states = self.current_states.copy()
            live_outputs = self.live_output_shadow.copy()
            counts = self.ttl_pulse_counts.copy()

        if not states:
            return

        ttl_data = {
            "frame_id": frame_metadata.get("frame_id", 0),
            "timestamp_camera": frame_metadata.get("timestamp_ticks", None),
            "timestamp_software": frame_metadata.get("timestamp_software", None),
            "exposure_time_us": frame_metadata.get("exposure_time_us", None),
            "line1_status": frame_metadata.get("line1_status", None),
            "line2_status": frame_metadata.get("line2_status", None),
            "line3_status": frame_metadata.get("line3_status", None),
            "line4_status": frame_metadata.get("line4_status", None),
            "gate_ttl": int(states["gate"]),
            "sync_1hz_ttl": int(states["sync"]),
            "sync_10hz_ttl": int(states["sync"]),
            "barcode_pin0_ttl": int(states["barcode0"]),
            "barcode_pin1_ttl": int(states["barcode1"]),
            "lever_ttl": int(states["lever"]),
            "cue_ttl": int(states["cue"]),
            "reward_ttl": int(states["reward"]),
            "iti_ttl": int(states["iti"]),
            "gate_count": counts["gate"],
            "sync_count": counts["sync"],
            "barcode_count": max(counts["barcode0"], counts["barcode1"]),
            "lever_count": counts["lever"],
            "cue_count": counts["cue"],
            "reward_count": counts["reward"],
            "iti_count": counts["iti"],
        }
        for output_key in self.LIVE_OUTPUT_KEYS:
            ttl_data[f"{output_key}_ttl"] = int(bool(live_outputs.get(output_key, False)))

        with QMutexLocker(self.mutex):
            self.ttl_history.append(ttl_data)

    def get_ttl_states(self) -> Optional[Dict]:
        """Read current Firmata signal states and return normalized packet."""
        with QMutexLocker(self.mutex):
            if self.board is None:
                return None

            now = time.time()
            if self.is_generating:
                self._update_generated_outputs_locked(now)
            self._update_live_outputs_locked(now)
            self._refresh_input_states_locked()
            self._sync_output_shadow_to_states_locked()
            return self._build_state_packet(passive=False)

    def run(self):
        """Worker loop: refresh signal states and stream packets to GUI."""
        self.running = True

        while self.running:
            emit_packet = None
            emit_ts = None
            try:
                with QMutexLocker(self.mutex):
                    if self.board is not None:
                        now = time.time()

                        if self.is_generating:
                            self._update_generated_outputs_locked(now)
                        else:
                            if any(
                                bool(self.output_shadow.get(key, False))
                                for key in self.SIGNAL_KEYS
                                if self._is_output_role(self._base_key_for_signal(key))
                            ):
                                self._set_all_outputs_low_locked()

                        self._update_live_outputs_locked(now)
                        self._refresh_input_states_locked()
                        self._sync_output_shadow_to_states_locked()

                        previous_packet = self.last_state_packet.copy()
                        packet = self._build_state_packet(passive=False)

                        state_changed = any(
                            bool(packet.get(key, False)) != bool(previous_packet.get(key, False))
                            for key in self.SIGNAL_KEYS
                        )
                        if state_changed:
                            self._record_ttl_event(packet, previous_packet)
                            packet = self._build_state_packet(passive=False)

                        count_changed = self._counts_changed(packet, previous_packet)
                        if state_changed or count_changed or (now - self.last_state_emit) >= 0.2:
                            packet["pulse_counts"] = self.ttl_pulse_counts.copy()
                            emit_packet = packet.copy()
                            emit_ts = now
                            self.last_state_emit = now

                        self.last_state_packet = packet.copy()

                if emit_packet is not None:
                    self.ttl_states_updated.emit(emit_packet)
                    self._record_live_state_sample(emit_packet, emit_ts)

                time.sleep(0.005)

            except Exception as e:
                self._handle_firmata_io_failure(e, context="Arduino Firmata read/write error")
                time.sleep(0.1)

    # ===== Firmata Read/Write =====

    def _base_key_for_signal(self, signal_key: str) -> str:
        if signal_key in ("barcode0", "barcode1"):
            return "barcode"
        return signal_key

    def _is_output_role(self, base_key: str) -> bool:
        role = self.signal_roles.get(base_key, self.DEFAULT_SIGNAL_ROLES.get(base_key, "Output"))
        return role == "Output"

    def _write_output_signal_locked(self, signal_key: str, value: bool):
        """Write one logical signal to configured output pin(s)."""
        base_key = self._base_key_for_signal(signal_key)
        if not self._is_output_role(base_key):
            return

        handles = self.pin_handles.get(base_key, [])
        value_int = 1 if bool(value) else 0

        if base_key == "barcode":
            if not handles:
                self.output_shadow[signal_key] = bool(value)
                self.current_states[signal_key] = bool(value)
                return

            index = 0 if signal_key == "barcode0" else 1
            if index >= len(handles):
                index = 0

            try:
                handles[index].write(value_int)
            except Exception:
                pass

            self.output_shadow[signal_key] = bool(value)
            self.current_states[signal_key] = bool(value)
            if len(handles) == 1:
                mirror_key = "barcode1" if signal_key == "barcode0" else "barcode0"
                self.output_shadow[mirror_key] = bool(value)
                self.current_states[mirror_key] = bool(value)
            return

        for handle in handles:
            try:
                handle.write(value_int)
            except Exception:
                pass

        self.output_shadow[signal_key] = bool(value)
        self.current_states[signal_key] = bool(value)

    def _write_live_output_locked(self, output_key: str, value: bool, force: bool = False):
        if output_key not in self.LIVE_OUTPUT_KEYS:
            return
        if not self._is_output_role(output_key):
            return

        if not force and bool(self.live_output_shadow.get(output_key, False)) == bool(value):
            return

        handles = self.pin_handles.get(output_key, [])
        value_int = 1 if bool(value) else 0
        for handle in handles:
            try:
                handle.write(value_int)
            except Exception:
                pass
        self.live_output_shadow[output_key] = bool(value)

    @staticmethod
    def _live_output_pulse_train_end(train):
        start_s, duration_s, period_s, count = train
        return float(start_s) + ((int(count) - 1) * float(period_s)) + float(duration_s)

    @staticmethod
    def _live_output_pulse_train_active(train, now: float) -> bool:
        start_s, duration_s, period_s, count = train
        elapsed_s = float(now) - float(start_s)
        if elapsed_s < 0:
            return False
        if float(now) >= ArduinoOutputWorker._live_output_pulse_train_end(train):
            return False
        pulse_index = int(elapsed_s // float(period_s))
        if pulse_index >= int(count):
            return False
        return (elapsed_s - (pulse_index * float(period_s))) < float(duration_s)

    def _pruned_live_output_pulse_trains_locked(self, output_key: str, now: float):
        trains = [
            train
            for train in self.live_output_pulse_trains.get(output_key, [])
            if self._live_output_pulse_train_end(train) > float(now)
        ]
        self.live_output_pulse_trains[output_key] = trains
        self.live_output_pulses_until[output_key] = max(
            [self._live_output_pulse_train_end(train) for train in trains] or [0.0]
        )
        return trains

    @staticmethod
    def _live_output_pulse_train_next_transition(train, now: float) -> Optional[float]:
        start_s, duration_s, period_s, count = train
        start_s = float(start_s)
        duration_s = float(duration_s)
        period_s = max(0.0001, float(period_s))
        count = max(1, int(count))
        now = float(now)
        end_s = ArduinoOutputWorker._live_output_pulse_train_end(train)
        if now < start_s:
            return start_s
        if now >= end_s:
            return None
        elapsed_s = now - start_s
        pulse_index = int(elapsed_s // period_s)
        if pulse_index >= count:
            return end_s
        pulse_start_s = start_s + (pulse_index * period_s)
        pulse_end_s = pulse_start_s + duration_s
        if now < pulse_end_s:
            return pulse_end_s
        next_pulse_index = pulse_index + 1
        if next_pulse_index < count:
            return start_s + (next_pulse_index * period_s)
        return end_s

    def _next_live_output_transition_locked(self, now: float) -> Optional[float]:
        next_transition = None
        for output_key in self.LIVE_OUTPUT_KEYS:
            for train in self.live_output_pulse_trains.get(output_key, []):
                transition = self._live_output_pulse_train_next_transition(train, now)
                if transition is None:
                    continue
                if next_transition is None or transition < next_transition:
                    next_transition = transition
        return next_transition

    def _has_live_pulse_trains_locked(self) -> bool:
        return any(bool(self.live_output_pulse_trains.get(output_key)) for output_key in self.LIVE_OUTPUT_KEYS)

    def _ensure_live_pulse_scheduler_locked(self):
        self.live_pulse_scheduler_stop.clear()
        if self.live_pulse_scheduler_thread is not None and self.live_pulse_scheduler_thread.is_alive():
            self.live_pulse_scheduler_wake.set()
            return
        self.live_pulse_scheduler_wake.clear()
        self.live_pulse_scheduler_thread = threading.Thread(
            target=self._live_pulse_scheduler_loop,
            name="CamAppLivePulseScheduler",
            daemon=True,
        )
        self.live_pulse_scheduler_thread.start()

    def _stop_live_pulse_scheduler_locked(self):
        self.live_pulse_scheduler_stop.set()
        self.live_pulse_scheduler_wake.set()

    def _live_pulse_scheduler_loop(self):
        current_thread = threading.current_thread()
        try:
            while not self.live_pulse_scheduler_stop.is_set():
                sleep_s = 0.002
                with QMutexLocker(self.mutex):
                    if self.board is None or not self._has_live_pulse_trains_locked():
                        break

                    now = time.time()
                    for output_key in self.LIVE_OUTPUT_KEYS:
                        self._apply_live_output_state_locked(output_key, now)

                    next_transition = self._next_live_output_transition_locked(now)
                    if next_transition is not None:
                        sleep_s = max(0.0005, min(0.004, float(next_transition) - now))

                if self.live_pulse_scheduler_wake.wait(sleep_s):
                    self.live_pulse_scheduler_wake.clear()
        finally:
            with QMutexLocker(self.mutex):
                if self.live_pulse_scheduler_thread is current_thread:
                    self.live_pulse_scheduler_thread = None

    def _apply_live_output_state_locked(self, output_key: str, now: float):
        active = bool(self.live_output_levels.get(output_key, False))
        trains = self._pruned_live_output_pulse_trains_locked(output_key, now)
        if any(self._live_output_pulse_train_active(train, now) for train in trains):
            active = True
        self._write_live_output_locked(output_key, active)

    def _update_live_outputs_locked(self, now: float):
        for output_key in self.LIVE_OUTPUT_KEYS:
            self._apply_live_output_state_locked(output_key, now)

    def _set_all_outputs_low_locked(self):
        """Drive all output-configured logical signals to LOW."""
        for signal_key in self.SIGNAL_KEYS:
            base_key = self._base_key_for_signal(signal_key)
            if self._is_output_role(base_key):
                self._write_output_signal_locked(signal_key, False)

    def _set_all_live_outputs_low_locked(self):
        for output_key in self.LIVE_OUTPUT_KEYS:
            self.live_output_levels[output_key] = False
            self.live_output_pulses_until[output_key] = 0.0
            self.live_output_pulse_trains[output_key] = []
            self._write_live_output_locked(output_key, False, force=True)
        self._stop_live_pulse_scheduler_locked()

    def _sync_output_shadow_to_states_locked(self):
        """Keep output signal states aligned to last written values."""
        for signal_key in self.SIGNAL_KEYS:
            base_key = self._base_key_for_signal(signal_key)
            if self._is_output_role(base_key):
                self.current_states[signal_key] = bool(self.output_shadow.get(signal_key, False))

    def _read_pin_bool(self, pin_handle) -> Optional[bool]:
        try:
            raw_value = pin_handle.read()
        except Exception:
            raw_value = None

        if raw_value is None:
            raw_value = getattr(pin_handle, "value", None)

        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            return raw_value

        try:
            return float(raw_value) >= 0.5
        except Exception:
            return bool(raw_value)

    def _refresh_input_states_locked(self):
        """Read all input-configured signal pins."""
        for base_key, handles in self.pin_handles.items():
            if self._is_output_role(base_key):
                continue
            if not handles:
                continue

            values = []
            for handle in handles:
                bit = self._read_pin_bool(handle)
                if bit is not None:
                    values.append(bool(bit))

            if not values:
                continue

            if base_key == "barcode":
                barcode0 = bool(values[0])
                barcode1 = bool(values[1]) if len(values) > 1 else barcode0
                self.current_states["barcode0"] = barcode0
                self.current_states["barcode1"] = barcode1
            else:
                self.current_states[base_key] = any(values)

    def _update_generated_outputs_locked(self, now: float):
        """Generate TTL patterns while test/recording mode is active."""
        if self.generation_start_time <= 0.0:
            self.generation_start_time = now

        elapsed = max(0.0, now - self.generation_start_time)

        # Core TTL lines
        self._write_output_signal_locked("gate", True)
        self._update_sync_state_machine_locked(now)

        if self.hw_barcode_enabled:
            # Hardware barcode: Timer1 ISR handles D8/D9 on the Arduino.
            # Periodically query status so the GUI can display counter/bit.
            if (now - self._hw_barcode_last_query) >= 1.0:
                self._query_hw_barcode_status_locked()
                self._hw_barcode_last_query = now
        else:
            self._update_barcode_state_machine_locked(now)

        # Behavior lines in test mode only
        if self.generation_mode == "test":
            cycle = elapsed % 2.0
            cue_state = 0.20 <= cycle < 0.30
            reward_state = 0.80 <= cycle < 0.90
            iti_state = 1.40 <= cycle < 1.50
        else:
            cue_state = False
            reward_state = False
            iti_state = False

        self._write_output_signal_locked("cue", cue_state)
        self._write_output_signal_locked("reward", reward_state)
        self._write_output_signal_locked("iti", iti_state)

    def _reset_signal_generators_locked(self, now: Optional[float] = None):
        """Reset sync and barcode timing engines to initial state."""
        if now is None:
            now = time.time()
        self.sync_output_state = False
        self.sync_next_time = float(now)
        self.barcode_state = self.BC_START_HI
        self.barcode_next_time = float(now)
        self.barcode_bit_index = 0
        self.barcode_word = 0

    def _stop_generation_locked(self):
        """Stop generated outputs but keep collected history/counters available to the GUI."""
        if self.hw_barcode_enabled and self.hw_barcode_running:
            self._stop_hw_barcode_locked()
        self.is_generating = False
        self.generation_mode = "idle"
        self.generation_start_time = 0.0
        self._set_all_outputs_low_locked()
        self._reset_signal_generators_locked(time.time())

    def _barcode_word_duration_seconds(self) -> float:
        """Return the active duration of one barcode word excluding the configured inter-code gap."""
        return (
            float(self.BARCODE_START_PULSE_S)
            + float(self.BARCODE_START_LOW_S)
            + (float(self.BARCODE_BITS) * float(self.BARCODE_BIT_S))
        )

    def _barcode_gap_seconds(self) -> float:
        """
        Return the silent gap after one barcode word.

        The UI exposes this as the interval between barcode words, so it should
        be interpreted literally rather than as a total start-to-start period.
        """
        return max(0.0, float(self.BARCODE_INTERVAL_S))

    def _set_barcode_level_locked(self, value: bool):
        self._write_output_signal_locked("barcode0", value)
        self._write_output_signal_locked("barcode1", value)

    def _update_sync_state_machine_locked(self, now: float):
        """Mirror ttl_generator sync behavior: 50ms HIGH every 1s."""
        if self.sync_next_time <= 0.0:
            self.sync_next_time = now

        guard = 0
        while now >= self.sync_next_time and guard < 16:
            if self.sync_output_state:
                self.sync_output_state = False
                self.sync_next_time += (self.SYNC_1HZ_PERIOD_S - self.SYNC_1HZ_PULSE_S)
            else:
                self.sync_output_state = True
                self.sync_next_time += self.SYNC_1HZ_PULSE_S
            guard += 1
        if guard >= 16:
            self.sync_next_time = now + self.SYNC_1HZ_PERIOD_S

        self._write_output_signal_locked("sync", self.sync_output_state)

    def _update_barcode_state_machine_locked(self, now: float):
        """Mirror ttl_generator barcode state machine (start, bits, gap)."""
        if self.barcode_next_time <= 0.0:
            self.barcode_next_time = now

        guard = 0
        while now >= self.barcode_next_time and guard < 256:
            if self.barcode_state == self.BC_START_HI:
                self._set_barcode_level_locked(True)
                self.barcode_next_time += self.BARCODE_START_PULSE_S
                self.barcode_state = self.BC_START_LO

            elif self.barcode_state == self.BC_START_LO:
                self._set_barcode_level_locked(False)
                self.barcode_next_time += self.BARCODE_START_LOW_S
                self.barcode_state = self.BC_BITS
                self.barcode_bit_index = 0

            elif self.barcode_state == self.BC_BITS:
                shift = self.BARCODE_BITS - 1 - self.barcode_bit_index
                bit_val = ((self.barcode_word >> shift) & 0x1) != 0
                self._set_barcode_level_locked(bit_val)
                self.barcode_bit_index += 1
                self.barcode_next_time += self.BARCODE_BIT_S
                if self.barcode_bit_index >= self.BARCODE_BITS:
                    self.barcode_state = self.BC_GAP

            else:
                self._set_barcode_level_locked(False)
                self.barcode_word = (self.barcode_word + 1) & ((1 << self.BARCODE_BITS) - 1)
                gap_s = self._barcode_gap_seconds()
                self.barcode_state = self.BC_START_HI
                if gap_s > 0.0:
                    self.barcode_next_time += gap_s

            guard += 1
        if guard >= 256:
            self.barcode_next_time = now + self.BARCODE_BIT_S

    # ===== Packet / Event Tracking =====

    def _build_state_packet(self, passive: bool = False) -> Dict:
        """Build one normalized packet consumed by GUI plots/counters."""
        packet = self.current_states.copy()
        packet["gate_count"] = int(self.ttl_pulse_counts["gate"])
        packet["sync_count"] = int(self.ttl_pulse_counts["sync"])
        packet["barcode_count"] = int(max(self.ttl_pulse_counts["barcode0"], self.ttl_pulse_counts["barcode1"]))
        packet["lever_count"] = int(self.ttl_pulse_counts["lever"])
        packet["cue_count"] = int(self.ttl_pulse_counts["cue"])
        packet["reward_count"] = int(self.ttl_pulse_counts["reward"])
        packet["iti_count"] = int(self.ttl_pulse_counts["iti"])
        packet["pulse_counts"] = self.ttl_pulse_counts.copy()
        packet["passive_mode"] = bool(passive)
        for output_key in self.LIVE_OUTPUT_KEYS:
            packet[output_key] = int(bool(self.live_output_shadow.get(output_key, False)))
        # Hardware barcode status (None when software mode or no reply yet)
        packet["hw_barcode_enabled"] = bool(self.hw_barcode_enabled)
        packet["hw_barcode_running"] = bool(self.hw_barcode_running)
        packet["hw_barcode_status"] = self.hw_barcode_status.copy() if self.hw_barcode_status else None
        return packet

    def _reset_ttl_event_tracking(self):
        """Reset TTL edge tracking and counters."""
        self.ttl_event_history.clear()
        self.live_state_history.clear()
        self.ttl_pulse_counts = {key: 0 for key in self.ttl_pulse_counts}
        self.last_event_state = self.current_states.copy()
        self.last_state_packet = self.current_states.copy()

    def _record_live_state_sample(self, states: Dict, timestamp: Optional[float] = None):
        """Record periodic live TTL/behavior state samples for CSV export."""
        ts = float(timestamp if timestamp is not None else time.time())
        sample = {
            "timestamp_software": ts,
            "passive_mode": int(bool(states.get("passive_mode", False))),
        }
        for key in self.SIGNAL_KEYS:
            sample[key] = int(bool(states.get(key, False)))
        for key in self.LIVE_OUTPUT_KEYS:
            sample[key] = int(bool(states.get(key, False)))

        sample["gate_count"] = int(self.ttl_pulse_counts.get("gate", 0))
        sample["sync_count"] = int(self.ttl_pulse_counts.get("sync", 0))
        sample["barcode_count"] = int(max(self.ttl_pulse_counts.get("barcode0", 0), self.ttl_pulse_counts.get("barcode1", 0)))
        sample["lever_count"] = int(self.ttl_pulse_counts.get("lever", 0))
        sample["cue_count"] = int(self.ttl_pulse_counts.get("cue", 0))
        sample["reward_count"] = int(self.ttl_pulse_counts.get("reward", 0))
        sample["iti_count"] = int(self.ttl_pulse_counts.get("iti", 0))

        with QMutexLocker(self.mutex):
            self.live_state_history.append(sample)
            if len(self.live_state_history) > self.max_live_state_history:
                del self.live_state_history[: len(self.live_state_history) - self.max_live_state_history]

    def _record_ttl_event(self, states: Dict, previous_states: Dict):
        """Record TTL edge events and update pulse counters."""
        timestamp = time.time()
        edge_ms = int(timestamp * 1000)

        for key in self.SIGNAL_KEYS:
            current_state = bool(states.get(key, False))
            previous_state = bool(previous_states.get(key, False))
            if current_state == previous_state:
                continue

            edge = "rising" if current_state else "falling"
            if edge == "rising":
                self.ttl_pulse_counts[key] += 1

            self.ttl_event_history.append(
                {
                    "timestamp_software": timestamp,
                    "timestamp_arduino_ms": edge_ms,
                    "signal": key,
                    "edge": edge,
                    "state": int(current_state),
                    "count": int(self.ttl_pulse_counts[key]),
                }
            )

        self.last_event_state = {key: bool(states.get(key, False)) for key in self.SIGNAL_KEYS}

    def _counts_changed(self, states: Dict, previous_states: Dict) -> bool:
        """Detect counter changes between packets."""
        for count_key in set(self.COUNT_KEY_MAP.values()):
            if int(states.get(count_key, 0)) != int(previous_states.get(count_key, 0)):
                return True
        return False

    # ===== Helpers =====

    def _emit_error_throttled(self, message: str):
        """Emit error signal while suppressing rapid duplicate spam."""
        now = time.time()
        if (
            message == self.last_serial_error_message
            and (now - self.last_serial_error_time) < self.serial_error_cooldown_s
        ):
            return
        self.last_serial_error_message = message
        self.last_serial_error_time = now
        self.error_occurred.emit(message)

    def _is_access_denied_error(self, error: Exception) -> bool:
        """Detect Windows permission/port-lock style failures."""
        message = str(error).lower()
        return ("access is denied" in message) or ("permission" in message and "denied" in message)

    def _missing_firmata_dependency_message(self) -> str:
        return (
            "pyFirmata is not installed in the Python interpreter running this app. "
            f"Install it with: \"{sys.executable}\" -m pip install pyfirmata pyserial"
        )

    def _missing_serial_dependency_message(self) -> str:
        return (
            "pySerial is not installed in the Python interpreter running this app. "
            f"Install it with: \"{sys.executable}\" -m pip install pyserial pyfirmata"
        )

    def _available_port_names(self) -> List[str]:
        if list_ports is None:
            return []
        try:
            return [str(port.device) for port in list_ports.comports()]
        except Exception:
            return []

    def _connection_error_message(self, port_name: str, error: Exception) -> str:
        if self._is_access_denied_error(error):
            return (
                f"{port_name} is locked or Windows denied access to it. "
                "Close Arduino IDE Serial Monitor/Plotter, Thonny, PuTTY, other Python sessions, "
                "and any previous CamApp instance, then unplug/replug the Arduino and click Scan. "
                "Running as Administrator usually does not fix a COM port that is already open."
            )
        return f"Arduino connection error on {port_name}: {str(error)}"

    def _handle_firmata_io_failure(self, error: Exception, context: str):
        """Close lost board once and surface an actionable error."""
        error_text = f"{context}: {str(error)}"
        if self._is_access_denied_error(error):
            status_message = self._connection_error_message(self.port_name or "Arduino COM port", error)
        else:
            status_message = f"Firmata connection lost: {str(error)}"

        with QMutexLocker(self.mutex):
            board = self._detach_board_locked()

        if board is not None:
            self._close_board(board)

        self._emit_error_throttled(error_text)
        self.connection_status.emit(False, status_message)

    def _normalize_pin_key(self, raw_key: str) -> Optional[str]:
        key = str(raw_key).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        mapping = {
            "gate": "gate",
            "sync": "sync",
            "sync1hz": "sync",
            "barcode": "barcode",
            "barcode0": "barcode",
            "barcode1": "barcode",
            "lever": "lever",
            "cue": "cue",
            "cueled": "cue",
            "ledgreen": "cue",
            "reward": "reward",
            "rewardled": "reward",
            "ledblue": "reward",
            "iti": "iti",
            "itled": "iti",
            "ledred": "iti",
            "do1": "do1",
            "digitaloutput1": "do1",
            "output1": "do1",
            "do2": "do2",
            "digitaloutput2": "do2",
            "output2": "do2",
            "do3": "do3",
            "digitaloutput3": "do3",
            "output3": "do3",
            "do4": "do4",
            "digitaloutput4": "do4",
            "output4": "do4",
            "do5": "do5",
            "digitaloutput5": "do5",
            "output5": "do5",
            "do6": "do6",
            "digitaloutput6": "do6",
            "output6": "do6",
            "do7": "do7",
            "digitaloutput7": "do7",
            "output7": "do7",
            "do8": "do8",
            "digitaloutput8": "do8",
            "output8": "do8",
        }
        return mapping.get(key, None)

    def _normalize_pin_list(self, value) -> List[int]:
        if isinstance(value, (list, tuple)):
            pins = []
            for entry in value:
                parsed = self._safe_int(entry)
                if parsed is not None:
                    pins.append(parsed)
            return pins

        if value is None:
            return []

        raw = str(value).replace(";", ",")
        pins = []
        for token in raw.split(","):
            parsed = self._safe_int(token.strip())
            if parsed is not None:
                pins.append(parsed)
        return pins

    def _parse_pin_setting_value(self, value) -> List[int]:
        return self._normalize_pin_list(value)

    def _safe_int(self, value, default=None):
        try:
            return int(str(value).strip())
        except Exception:
            return default

    def _safe_float(self, value, default=None):
        try:
            return float(str(value).strip())
        except Exception:
            return default

    # ===== Public Data Accessors =====

    def stop(self):
        """Stop the worker thread and disconnect board."""
        self.running = False
        self.disconnect_port()

    def get_ttl_history(self) -> List[Dict]:
        """Get TTL history for CSV export."""
        with QMutexLocker(self.mutex):
            return self.ttl_history.copy()

    def clear_ttl_history(self):
        """Clear TTL history."""
        with QMutexLocker(self.mutex):
            self.ttl_history.clear()

    def get_ttl_event_history(self) -> List[Dict]:
        """Get TTL edge event history."""
        with QMutexLocker(self.mutex):
            return self.ttl_event_history.copy()

    def get_live_state_history(self) -> List[Dict]:
        """Get periodic live TTL/behavior state samples."""
        with QMutexLocker(self.mutex):
            return self.live_state_history.copy()

    def get_ttl_pulse_counts(self) -> Dict:
        """Get TTL pulse counts based on detected rising edges."""
        with QMutexLocker(self.mutex):
            return self.ttl_pulse_counts.copy()

    def clear_ttl_event_history(self):
        """Clear TTL event and live sample history."""
        with QMutexLocker(self.mutex):
            self.ttl_event_history.clear()
            self.live_state_history.clear()
