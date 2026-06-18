from __future__ import annotations

"""
Auxiliary Arduino support for PyKaboo.

The primary board (``ArduinoOutputWorker`` in ``arduino_output.py``) keeps its
full TTL / barcode / sync / live-output feature set. This module adds *extra*
boards on top of that, using a deliberately simple generic per-pin model:

    * each pin has a number, a user label, and a role:
        - Input  -> sampled once per camera frame and logged to the CSV
        - Output -> driven manually (hold high/low) or pulsed from the UI

``AuxiliaryArduinoWorker`` talks to one board over Firmata (mirroring the proven
connection pattern from the primary worker). ``ArduinoDeviceManager`` owns a
roster of these workers, persists their configuration, and merges their
per-frame samples into the recording's frame CSV as prefixed columns
(``dev<id>_<label>_ttl``) so every auxiliary signal stays frame-aligned with the
primary TTL columns.
"""

import inspect
import sys
import time
from typing import Any, Dict, List, Optional

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

try:
    import pyfirmata
    from pyfirmata import util
    PYFIRMATA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    pyfirmata = None
    util = None
    PYFIRMATA_IMPORT_ERROR = exc

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas always present in app env
    pd = None

from PySide6.QtCore import QMutex, QMutexLocker, QObject, QThread, Signal

from arduino_output import scan_serial_ports

try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover - environment dependent
    list_ports = None


def _missing_firmata_dependency_message() -> str:
    return (
        "pyFirmata is not installed in the Python interpreter running this app. "
        f"Install it with: \"{sys.executable}\" -m pip install pyfirmata pyserial"
    )


def _safe_int(value, default=None):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def sanitize_column_token(value: str) -> str:
    """Reduce a label to a CSV-column-safe token (alnum + underscore)."""
    cleaned = []
    for char in str(value).strip():
        if char.isalnum() or char == "_":
            cleaned.append(char)
        elif char in (" ", "-"):
            cleaned.append("_")
    token = "".join(cleaned).strip("_")
    return token


class AuxiliaryArduinoWorker(QThread):
    """One auxiliary board with a generic set of input/output digital pins."""

    connection_status = Signal(str, bool, str)   # device_id, connected, message
    states_updated = Signal(dict)                 # {device_id, name, pins:{label:bool}, connected}
    error_occurred = Signal(str)

    POLL_INTERVAL_S = 0.005
    EMIT_INTERVAL_S = 0.15

    def __init__(self, device_id: str, name: str = "", port_name: str = "",
                 pins: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.device_id = str(device_id)
        self.name = str(name or f"Device {device_id}")
        self.port_name = str(port_name or "")

        self.board: Optional[Any] = None
        self.iterator: Optional[Any] = None
        self.running = False
        self.mutex = QMutex()

        # Each pin: {"pin":int, "label":str, "role":"Input"|"Output", "key":str}
        self.pins: List[Dict[str, Any]] = []
        self.pin_handles: Dict[int, Any] = {}
        self.input_states: Dict[int, bool] = {}
        self.output_levels: Dict[int, bool] = {}
        self.output_shadow: Dict[int, bool] = {}
        # pin -> (start_s, duration_s, period_s, count)
        self.output_pulses: Dict[int, tuple] = {}

        self.is_recording = False
        self.history: List[Dict[str, Any]] = []
        self._last_emit = 0.0

        self.set_pins(pins or [])

    # ===== Configuration =====

    @staticmethod
    def normalize_pins(raw_pins: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Validate/clean a pin list and assign unique CSV column keys."""
        normalized: List[Dict[str, Any]] = []
        used_keys: set[str] = set()
        seen_pins: set[int] = set()
        for entry in (raw_pins or []):
            if not isinstance(entry, dict):
                continue
            pin = _safe_int(entry.get("pin"), default=None)
            if pin is None or pin < 0 or pin in seen_pins:
                continue
            seen_pins.add(pin)
            role = "Input" if str(entry.get("role", "Input")).strip().lower().startswith("in") else "Output"
            label = str(entry.get("label", "") or "").strip() or f"pin{pin}"
            base_key = sanitize_column_token(label) or f"pin{pin}"
            key = base_key
            suffix = 2
            while key in used_keys:
                key = f"{base_key}_{suffix}"
                suffix += 1
            used_keys.add(key)
            normalized.append({"pin": pin, "label": label, "role": role, "key": key})
        return normalized

    def set_pins(self, raw_pins: List[Dict[str, Any]]):
        """Replace the pin configuration (and reconfigure handles if connected)."""
        normalized = self.normalize_pins(raw_pins)
        with QMutexLocker(self.mutex):
            self.pins = normalized
            self.output_levels = {
                p["pin"]: bool(self.output_levels.get(p["pin"], False))
                for p in normalized if p["role"] == "Output"
            }
            self.output_pulses = {
                pin: train for pin, train in self.output_pulses.items()
                if pin in self.output_levels
            }
            self.input_states = {
                p["pin"]: bool(self.input_states.get(p["pin"], False))
                for p in normalized if p["role"] == "Input"
            }
            if self.board is not None:
                self._configure_pin_handles_locked()

    def snapshot(self) -> Dict[str, Any]:
        """Serializable description for persistence."""
        with QMutexLocker(self.mutex):
            return {
                "id": self.device_id,
                "name": self.name,
                "port": self.port_name,
                "pins": [
                    {"pin": p["pin"], "label": p["label"], "role": p["role"]}
                    for p in self.pins
                ],
            }

    def pin_labels(self) -> Dict[int, str]:
        with QMutexLocker(self.mutex):
            return {p["pin"]: p["label"] for p in self.pins}

    @property
    def is_connected(self) -> bool:
        return self.board is not None

    # ===== Connection =====

    def connect_to_port(self, port_name: str) -> bool:
        if " - " in str(port_name):
            port_name = str(port_name).split(" - ")[0]
        port_name = str(port_name).strip()

        if not port_name:
            self.error_occurred.emit(f"{self.name}: no COM port selected.")
            self.connection_status.emit(self.device_id, False, "No COM port selected")
            return False

        if pyfirmata is None or util is None or PYFIRMATA_IMPORT_ERROR is not None:
            message = _missing_firmata_dependency_message()
            self.error_occurred.emit(message)
            self.connection_status.emit(self.device_id, False, message)
            return False

        available = self._available_port_names()
        if available and port_name.upper() not in {p.upper() for p in available}:
            message = (
                f"{port_name} is not currently listed by Windows. "
                f"Available serial ports: {', '.join(available)}. "
                "Plug the board in, click Scan, then pick its COM port."
            )
            self.error_occurred.emit(message)
            self.connection_status.emit(self.device_id, False, message)
            return False

        board = None
        iterator = None
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
                self.port_name = port_name
                self.output_shadow = {}
                self._configure_pin_handles_locked()

            self.connection_status.emit(
                self.device_id, True, f"{self.name} connected to {port_name}"
            )
            return True

        except Exception as exc:
            if iterator is not None:
                try:
                    iterator.join(timeout=0.1)
                except Exception:
                    pass
            if board is not None:
                self._close_board(board)
            message = self._connection_error_message(port_name, exc)
            self.error_occurred.emit(message)
            self.connection_status.emit(self.device_id, False, message)
            return False

    def disconnect_port(self):
        with QMutexLocker(self.mutex):
            if self.board is not None:
                self._set_all_outputs_low_locked()
            board = self._detach_board_locked()
        if board is not None:
            self._close_board(board)
        self.connection_status.emit(self.device_id, False, f"{self.name} disconnected")

    def stop(self):
        """Stop the worker thread and release the board."""
        self.running = False
        self.disconnect_port()

    def _detach_board_locked(self):
        board = self.board
        self.board = None
        self.iterator = None
        self.pin_handles = {}
        self.output_shadow = {}
        self.output_pulses = {}
        return board

    def _close_board(self, board):
        try:
            board.exit()
            return
        except Exception:
            pass
        try:
            handle = getattr(board, "sp", None)
            if handle is not None:
                handle.close()
        except Exception:
            pass

    def _available_port_names(self) -> List[str]:
        if list_ports is None:
            return []
        try:
            return [str(port.device) for port in list_ports.comports()]
        except Exception:
            return []

    def _is_access_denied_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return ("access is denied" in message) or ("permission" in message and "denied" in message)

    def _connection_error_message(self, port_name: str, error: Exception) -> str:
        if self._is_access_denied_error(error):
            return (
                f"{port_name} is locked or Windows denied access to it. "
                "Close Arduino IDE Serial Monitor/Plotter, other Python sessions, and any previous "
                "CamApp instance, then unplug/replug the board and click Scan."
            )
        return f"{self.name} connection error on {port_name}: {str(error)}"

    # ===== Pin handles =====

    def _board_digital_pin_handle_locked(self, pin: int):
        if self.board is None:
            return None
        try:
            bank = getattr(self.board, "digital", None)
            if bank is None or pin < 0 or pin >= len(bank):
                return None
            return bank[pin]
        except Exception:
            return None

    def _create_pin_handle_locked(self, pin: int, mode: str):
        handle = self._board_digital_pin_handle_locked(pin)
        if handle is None and self.board is not None:
            try:
                handle = self.board.get_pin(f"d:{int(pin)}:{mode}")
            except Exception:
                handle = None
        if handle is None:
            return None
        input_mode = getattr(pyfirmata, "INPUT", 0)
        output_mode = getattr(pyfirmata, "OUTPUT", 1)
        try:
            if mode == "i":
                handle.mode = input_mode
                handle.enable_reporting()
            else:
                handle.mode = output_mode
                handle.write(0)
        except Exception:
            pass
        return handle

    def _configure_pin_handles_locked(self):
        self.pin_handles = {}
        if self.board is None:
            return
        unresolved = []
        for entry in self.pins:
            pin = int(entry["pin"])
            mode = "i" if entry["role"] == "Input" else "o"
            handle = self._create_pin_handle_locked(pin, mode)
            if handle is None:
                unresolved.append(str(pin))
                continue
            self.pin_handles[pin] = handle
        if unresolved:
            self.error_occurred.emit(
                f"{self.name}: could not configure pin(s): {', '.join(unresolved[:8])}"
            )

    # ===== Outputs =====

    def set_output_level(self, pin: int, active: bool):
        pin = _safe_int(pin)
        if pin is None:
            return
        with QMutexLocker(self.mutex):
            if pin not in self.output_levels:
                return
            self.output_levels[pin] = bool(active)
            if self.board is not None:
                self._apply_output_state_locked(pin, time.time())

    def start_output_pulse(self, pin: int, duration_ms: int, count: int = 1, frequency_hz: float = 1.0):
        pin = _safe_int(pin)
        if pin is None:
            return
        with QMutexLocker(self.mutex):
            if pin not in self.output_levels:
                return
            duration_s = max(0.001, float(duration_ms) / 1000.0)
            count = max(1, int(count))
            if count > 1:
                period_s = max(duration_s, 1.0 / max(0.001, float(frequency_hz)))
            else:
                period_s = duration_s
            self.output_pulses[pin] = (time.time(), duration_s, period_s, count)
            if self.board is not None:
                self._apply_output_state_locked(pin, time.time())

    @staticmethod
    def _pulse_active(train, now: float) -> bool:
        start_s, duration_s, period_s, count = train
        elapsed = float(now) - float(start_s)
        if elapsed < 0:
            return False
        end_s = float(start_s) + ((int(count) - 1) * float(period_s)) + float(duration_s)
        if now >= end_s:
            return False
        pulse_index = int(elapsed // float(period_s))
        if pulse_index >= int(count):
            return False
        return (elapsed - (pulse_index * float(period_s))) < float(duration_s)

    def _apply_output_state_locked(self, pin: int, now: float):
        active = bool(self.output_levels.get(pin, False))
        train = self.output_pulses.get(pin)
        if train is not None:
            end_s = float(train[0]) + ((int(train[3]) - 1) * float(train[2])) + float(train[1])
            if now >= end_s:
                self.output_pulses.pop(pin, None)
            elif self._pulse_active(train, now):
                active = True
        self._write_pin_locked(pin, active)

    def _write_pin_locked(self, pin: int, value: bool):
        handle = self.pin_handles.get(pin)
        if handle is not None and bool(self.output_shadow.get(pin)) != bool(value):
            try:
                handle.write(1 if value else 0)
            except Exception:
                pass
        self.output_shadow[pin] = bool(value)

    def _set_all_outputs_low_locked(self):
        for pin in list(self.output_levels.keys()):
            self.output_levels[pin] = False
            self.output_pulses.pop(pin, None)
            self._write_pin_locked(pin, False)

    def _update_outputs_locked(self, now: float):
        for pin in list(self.output_levels.keys()):
            self._apply_output_state_locked(pin, now)

    # ===== Inputs =====

    def _read_pin_bool(self, handle) -> Optional[bool]:
        try:
            raw = handle.read()
        except Exception:
            raw = None
        if raw is None:
            raw = getattr(handle, "value", None)
        if raw is None:
            return None
        if isinstance(raw, bool):
            return raw
        try:
            return float(raw) >= 0.5
        except Exception:
            return bool(raw)

    def _refresh_inputs_locked(self):
        for entry in self.pins:
            if entry["role"] != "Input":
                continue
            pin = int(entry["pin"])
            handle = self.pin_handles.get(pin)
            if handle is None:
                continue
            bit = self._read_pin_bool(handle)
            if bit is not None:
                self.input_states[pin] = bool(bit)

    def _current_states_locked(self) -> Dict[int, bool]:
        states: Dict[int, bool] = {}
        for entry in self.pins:
            pin = int(entry["pin"])
            if entry["role"] == "Input":
                states[pin] = bool(self.input_states.get(pin, False))
            else:
                states[pin] = bool(self.output_shadow.get(pin, False))
        return states

    # ===== Recording =====

    def start_recording(self):
        with QMutexLocker(self.mutex):
            self.history = []
            self.is_recording = True

    def stop_recording(self):
        with QMutexLocker(self.mutex):
            self.is_recording = False

    def sample_state(self, frame_metadata: Dict):
        """Append one per-frame sample (called from the GUI/recording thread)."""
        with QMutexLocker(self.mutex):
            if self.board is None or not self.pins:
                return
            row: Dict[str, Any] = {"frame_id": int(frame_metadata.get("frame_id", 0) or 0)}
            states = self._current_states_locked()
            for entry in self.pins:
                row[entry["key"]] = int(bool(states.get(int(entry["pin"]), False)))
            self.history.append(row)

    def get_history(self) -> List[Dict[str, Any]]:
        with QMutexLocker(self.mutex):
            return list(self.history)

    def clear_history(self):
        with QMutexLocker(self.mutex):
            self.history = []

    # ===== Worker loop =====

    def run(self):
        self.running = True
        while self.running:
            emit_payload = None
            try:
                with QMutexLocker(self.mutex):
                    if self.board is not None:
                        now = time.time()
                        self._update_outputs_locked(now)
                        self._refresh_inputs_locked()
                        if (now - self._last_emit) >= self.EMIT_INTERVAL_S:
                            self._last_emit = now
                            pins_payload = {
                                entry["label"]: bool(
                                    self.input_states.get(int(entry["pin"]), False)
                                    if entry["role"] == "Input"
                                    else self.output_shadow.get(int(entry["pin"]), False)
                                )
                                for entry in self.pins
                            }
                            emit_payload = {
                                "device_id": self.device_id,
                                "name": self.name,
                                "connected": True,
                                "pins": pins_payload,
                            }
                if emit_payload is not None:
                    self.states_updated.emit(emit_payload)
                time.sleep(self.POLL_INTERVAL_S)
            except Exception as exc:
                self._handle_io_failure(exc)
                time.sleep(0.1)

    def _handle_io_failure(self, error: Exception):
        with QMutexLocker(self.mutex):
            board = self._detach_board_locked()
        if board is not None:
            self._close_board(board)
        self.error_occurred.emit(f"{self.name}: connection lost: {str(error)}")
        self.connection_status.emit(self.device_id, False, f"{self.name}: connection lost")


class ArduinoDeviceManager(QObject):
    """Owns the roster of auxiliary boards and fans out record/merge calls."""

    SETTINGS_KEY = "aux_arduino_devices"

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._devices: "Dict[str, AuxiliaryArduinoWorker]" = {}
        self._order: List[str] = []
        self._next_id = 2  # primary board is conceptually device 1
        # Set by the GUI so newly created workers can have their signals wired.
        self.on_device_created = None

    # ----- roster -----

    def devices(self) -> List[AuxiliaryArduinoWorker]:
        return [self._devices[d] for d in self._order if d in self._devices]

    def get_device(self, device_id: str) -> Optional[AuxiliaryArduinoWorker]:
        return self._devices.get(str(device_id))

    def used_ports(self, exclude_id: Optional[str] = None) -> set:
        ports = set()
        for dev in self.devices():
            if exclude_id is not None and dev.device_id == str(exclude_id):
                continue
            normalized = dev.port_name.split(" - ")[0].strip().upper()
            if normalized:
                ports.add(normalized)
        return ports

    def _allocate_id(self) -> str:
        while str(self._next_id) in self._devices:
            self._next_id += 1
        device_id = str(self._next_id)
        self._next_id += 1
        return device_id

    def add_device(self, name: str = "", port: str = "",
                   pins: Optional[List[Dict[str, Any]]] = None,
                   device_id: Optional[str] = None) -> AuxiliaryArduinoWorker:
        device_id = str(device_id) if device_id is not None else self._allocate_id()
        worker = AuxiliaryArduinoWorker(
            device_id=device_id,
            name=name or f"Device {device_id}",
            port_name=port,
            pins=pins,
        )
        self._devices[device_id] = worker
        if device_id not in self._order:
            self._order.append(device_id)
        if callable(self.on_device_created):
            try:
                self.on_device_created(worker)
            except Exception:
                pass
        return worker

    def remove_device(self, device_id: str):
        device_id = str(device_id)
        worker = self._devices.pop(device_id, None)
        if device_id in self._order:
            self._order.remove(device_id)
        if worker is not None:
            try:
                worker.stop()
                worker.wait(1500)
            except Exception:
                pass
        self.save()

    # ----- persistence -----

    def load(self):
        raw = self.settings.value(self.SETTINGS_KEY, "")
        if not raw:
            return
        try:
            import json
            entries = json.loads(raw)
        except Exception:
            entries = []
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            self.add_device(
                name=str(entry.get("name", "")),
                port=str(entry.get("port", "")),
                pins=entry.get("pins", []),
                device_id=str(entry.get("id")) if entry.get("id") is not None else None,
            )

    def save(self):
        import json
        roster = [dev.snapshot() for dev in self.devices()]
        self.settings.setValue(self.SETTINGS_KEY, json.dumps(roster))
        self.settings.sync()

    # ----- lifecycle -----

    def connect_device(self, device_id: str) -> bool:
        worker = self.get_device(device_id)
        if worker is None:
            return False
        if worker.connect_to_port(worker.port_name):
            if not worker.isRunning():
                worker.start()
            return True
        return False

    def disconnect_device(self, device_id: str):
        worker = self.get_device(device_id)
        if worker is not None:
            worker.stop()
            worker.wait(1500)

    def connect_all(self):
        for dev in self.devices():
            if dev.port_name:
                self.connect_device(dev.device_id)

    def stop_all(self):
        for dev in self.devices():
            try:
                dev.stop()
                dev.wait(1500)
            except Exception:
                pass

    # ----- recording -----

    def start_recording(self):
        for dev in self.devices():
            if dev.is_connected:
                dev.start_recording()

    def stop_recording(self):
        for dev in self.devices():
            dev.stop_recording()

    def sample_state(self, frame_metadata: Dict):
        for dev in self.devices():
            if dev.is_connected:
                dev.sample_state(frame_metadata)

    def clear_history(self):
        for dev in self.devices():
            dev.clear_history()

    def has_recorded_history(self) -> bool:
        return any(dev.get_history() for dev in self.devices())

    def merge_into_frame_df(self, df):
        """Left-merge each device's per-frame samples as dev<id>_<label>_ttl columns."""
        if pd is None or df is None or "frame_id" not in getattr(df, "columns", []):
            return df
        added_columns: List[str] = []
        for dev in self.devices():
            history = dev.get_history()
            if not history:
                continue
            ddf = pd.DataFrame(history)
            if "frame_id" not in ddf.columns:
                continue
            rename = {
                col: f"dev{dev.device_id}_{col}_ttl"
                for col in ddf.columns if col != "frame_id"
            }
            ddf = ddf.rename(columns=rename)
            ddf = ddf.drop_duplicates(subset="frame_id", keep="last")
            df = df.merge(ddf, on="frame_id", how="left")
            added_columns.extend(rename.values())
        for col in added_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        return df
