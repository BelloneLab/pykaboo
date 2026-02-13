"""
Arduino TTL Output Generator - Clean Implementation
Syncs TTL generation with camera frame acquisition.
"""
import serial
import serial.tools.list_ports
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, QSettings
import time
from typing import Optional, List, Dict


class ArduinoOutputWorker(QThread):
    """
    Arduino TTL output generator.
    Syncs with camera frames for precise TTL state logging.
    """

    # Signals
    port_list_updated = Signal(list)
    connection_status = Signal(bool, str)
    ttl_states_updated = Signal(dict)
    pin_config_received = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()

        self.serial_port: Optional[serial.Serial] = None
        self.running = False
        self.mutex = QMutex()
        self.is_generating = False  # True when TTLs are being generated

        # Configuration
        self.port_name = ""
        self.baud_rate = 115200

        # Current states
        self.current_states = {
            'gate': False,
            'sync': False,
            'barcode0': False,
            'barcode1': False
        }

        # Pin configuration (from Arduino)
        self.gate_pins = [6, 7]
        self.sync_pins = [8, 9]
        self.barcode_pins = [10, 11]

        # TTL history for frame-synced logging
        self.ttl_history: List[Dict] = []
        self.last_state_emit = 0.0
        self.ttl_event_history: List[Dict] = []
        self.ttl_pulse_counts = {
            'gate': 0,
            'sync': 0,
            'barcode0': 0,
            'barcode1': 0,
        }
        self.last_event_state = self.current_states.copy()
        self.last_edge_ms = {
            'gate': None,
            'sync': None,
            'barcode0': None,
            'barcode1': None,
        }
        self.last_counts = {
            'gate': 0,
            'sync': 0,
            'barcode0': 0,
            'barcode1': 0,
        }

        # Load settings
        self.settings = QSettings('BaslerCam', 'CameraApp')
        self.load_settings()

    def load_settings(self):
        """Load saved Arduino settings."""
        self.port_name = self.settings.value('arduino_port', '')

    def save_settings(self):
        """Save Arduino settings."""
        self.settings.setValue('arduino_port', self.port_name)

    def scan_ports(self) -> List[str]:
        """Scan for available serial ports."""
        ports = serial.tools.list_ports.comports()
        port_list = []

        for port in ports:
            if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                port_list.append(f"{port.device} - {port.description}")

        if not port_list:
            port_list = [f"{port.device} - {port.description}" for port in ports]

        self.port_list_updated.emit(port_list)
        return port_list

    def connect_to_port(self, port_name: str) -> bool:
        """Connect to Arduino."""
        with QMutexLocker(self.mutex):
            try:
                if ' - ' in port_name:
                    port_name = port_name.split(' - ')[0]

                self.port_name = port_name

                self.serial_port = serial.Serial(
                    port=port_name,
                    baudrate=self.baud_rate,
                    timeout=0.1
                )

                time.sleep(2)  # Wait for Arduino reset

                # Wait for READY message
                start_time = time.time()
                while time.time() - start_time < 5:
                    if self.serial_port.in_waiting > 0:
                        response = self.serial_port.readline().decode().strip()
                        if response == "READY":
                            break
                    time.sleep(0.1)

                # Get pin configuration
                self.serial_port.write(b"GET_PINS\n")
                time.sleep(0.1)

                if self.serial_port.in_waiting > 0:
                    response = self.serial_port.readline().decode().strip()
                    self._parse_pin_config(response)

                self.save_settings()
                self.connection_status.emit(True, f"Connected to {port_name}")
                return True

            except Exception as e:
                self.error_occurred.emit(f"Arduino connection error: {str(e)}")
                self.connection_status.emit(False, str(e))
                return False

    def _parse_pin_config(self, response: str):
        """Parse pin configuration from Arduino."""
        try:
            # Format: GATE:6,7,SYNC:8,9,BARCODE:10,11
            parts = response.split(',')
            config = {}
            current_key = None

            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    current_key = key.strip()
                    config[current_key.lower()] = [int(value)]
                else:
                    if current_key:
                        config[current_key.lower()].append(int(part))

            if 'gate' in config:
                self.gate_pins = config['gate']
            if 'sync' in config:
                self.sync_pins = config['sync']
            if 'barcode' in config:
                self.barcode_pins = config['barcode']

            self.pin_config_received.emit(config)

        except Exception as e:
            self.error_occurred.emit(f"Pin config parse error: {str(e)}")

    def disconnect_port(self):
        """Disconnect from Arduino."""
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"STOP_RECORDING\n")
                    time.sleep(0.1)
                except:
                    pass

                self.serial_port.close()
                self.serial_port = None
                self.is_generating = False
                self.connection_status.emit(False, "Disconnected")

    def start_recording(self):
        """Start TTL generation for recording."""
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                # Retry mechanism
                for attempt in range(3):
                    try:
                        # Flush input buffer before sending command
                        self.serial_port.reset_input_buffer()

                        self.serial_port.write(b"START_RECORDING\n")
                        # Give Arduino some time to process
                        time.sleep(0.1)

                        start_wait = time.time()
                        while time.time() - start_wait < 1.0: # 1 second timeout
                            if self.serial_port.in_waiting > 0:
                                response = self.serial_port.readline().decode().strip()
                                if response == "OK_RECORDING":
                                    self.is_generating = True
                                    self.ttl_history.clear()
                                    self._reset_ttl_event_tracking()
                                    return True
                            time.sleep(0.01)
                    except Exception as e:
                        print(f"Attempt {attempt+1} failed: {e}")
                        time.sleep(0.2)

                self.error_occurred.emit("Failed to receive OK_RECORDING from Arduino after retries")
        return False

    def stop_recording(self):
        """Stop TTL generation."""
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"STOP_RECORDING\n")
                    ok = self._await_response("OK_STOPPED", timeout=0.5)
                    self.is_generating = False
                    self.current_states = {key: False for key in self.current_states}
                    self._reset_ttl_event_tracking()
                    return ok
                except Exception as e:
                    self.error_occurred.emit(f"Stop recording error: {str(e)}")
        return False

    def start_test(self):
        """Start TTL generation for testing."""
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"START_TEST\n")
                    time.sleep(0.05)

                    if self.serial_port.in_waiting > 0:
                        response = self.serial_port.readline().decode().strip()
                        if response == "OK_TEST":
                            self.is_generating = True
                            self._reset_ttl_event_tracking()
                            return True
                except Exception as e:
                    self.error_occurred.emit(f"Start test error: {str(e)}")
        return False

    def stop_test(self):
        """Stop test TTL generation."""
        with QMutexLocker(self.mutex):
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b"STOP_TEST\n")
                    ok = self._await_response("OK_STOPPED", timeout=0.5)
                    self.is_generating = False
                    self.current_states = {key: False for key in self.current_states}
                    self._reset_ttl_event_tracking()
                    return ok
                except Exception as e:
                    self.error_occurred.emit(f"Stop test error: {str(e)}")
        return False

    def sample_ttl_state(self, frame_metadata: Dict):
        """
        Sample TTL state synchronized with camera frame.
        Called when each camera frame is recorded.
        """
        with QMutexLocker(self.mutex):
            states = self.current_states.copy()

        if states:
            # Merge with camera frame metadata
            ttl_data = {
                'frame_id': frame_metadata.get('frame_id', 0),
                'timestamp_camera': frame_metadata.get('timestamp_ticks', None),
                'timestamp_software': frame_metadata.get('timestamp_software', None),
                'exposure_time_us': frame_metadata.get('exposure_time_us', None),
                'line1_status': frame_metadata.get('line1_status', None),
                'line2_status': frame_metadata.get('line2_status', None),
                'line3_status': frame_metadata.get('line3_status', None),
                'line4_status': frame_metadata.get('line4_status', None),
                'gate_ttl': int(states['gate']),
                'sync_1hz_ttl': int(states['sync']),
                'sync_10hz_ttl': int(states['sync']),
                'barcode_pin0_ttl': int(states['barcode0']),
                'barcode_pin1_ttl': int(states['barcode1']),
            }

            self.ttl_history.append(ttl_data)

    def get_ttl_states(self) -> Optional[Dict]:
        """Get current TTL states from Arduino."""
        try:
            with QMutexLocker(self.mutex):
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.write(b"GET_STATES\n")
                    time.sleep(0.01)

                    if self.serial_port.in_waiting > 0:
                        response = self.serial_port.readline().decode().strip()
                        parts = response.split(',')
                        if len(parts) >= 4:
                            states = {
                                'gate': bool(int(parts[0])),
                                'sync': bool(int(parts[1])),
                                'barcode0': bool(int(parts[2])),
                                'barcode1': bool(int(parts[3])),
                            }
                            if len(parts) >= 10:
                                states.update({
                                    'gate_edge_ms': int(parts[4]),
                                    'sync_edge_ms': int(parts[5]),
                                    'barcode_edge_ms': int(parts[6]),
                                    'gate_count': int(parts[7]),
                                    'sync_count': int(parts[8]),
                                    'barcode_count': int(parts[9]),
                                })
                            self.current_states = states
                            return states
        except Exception:
            pass

        return None

    def run(self):
        """Main thread loop - read TTL states for visualization."""
        self.running = True

        while self.running:
            try:
                if self.serial_port and self.serial_port.is_open:
                    previous_states = self.current_states.copy()
                    states = self.get_ttl_states()
                    if states:
                        if states != previous_states or self._counts_changed(states, previous_states):
                            self._record_ttl_event(states, previous_states)
                        now = time.time()
                        if states != previous_states or (now - self.last_state_emit) >= 0.2:
                            self.ttl_states_updated.emit(states)
                            self.last_state_emit = now

                time.sleep(0.01)  # ~100Hz polling to capture TTL edges reliably

            except Exception as e:
                self.error_occurred.emit(f"Arduino read error: {str(e)}")
                time.sleep(0.1)

    def _await_response(self, expected: str, timeout: float = 0.5) -> bool:
        """Wait for a specific response from the Arduino."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self.serial_port.in_waiting > 0:
                response = self.serial_port.readline().decode().strip()
                if response == expected:
                    return True
            time.sleep(0.01)
        return False

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.is_generating:
            self.stop_recording()
        self.disconnect_port()

    def get_ttl_history(self) -> List[Dict]:
        """Get TTL history for CSV export."""
        return self.ttl_history.copy()

    def clear_ttl_history(self):
        """Clear TTL history."""
        self.ttl_history.clear()

    def _reset_ttl_event_tracking(self):
        """Reset TTL edge tracking and counters."""
        self.ttl_event_history.clear()
        self.ttl_pulse_counts = {key: 0 for key in self.ttl_pulse_counts}
        self.last_event_state = self.current_states.copy()
        self.last_edge_ms = {key: None for key in self.last_edge_ms}
        self.last_counts = {key: 0 for key in self.last_counts}

    def _record_ttl_event(self, states: Dict, previous_states: Dict):
        """Record TTL edge events and update pulse counters."""
        timestamp = time.time()
        edge_ms_map = {
            'gate': states.get('gate_edge_ms'),
            'sync': states.get('sync_edge_ms'),
            'barcode0': states.get('barcode_edge_ms'),
            'barcode1': states.get('barcode_edge_ms'),
        }
        for key in ('gate', 'sync', 'barcode0', 'barcode1'):
            if states.get(key) == previous_states.get(key):
                continue
            edge = "rising" if states.get(key) else "falling"
            if edge == "rising":
                self.ttl_pulse_counts[key] += 1
            self.ttl_event_history.append({
                'timestamp_software': timestamp,
                'timestamp_arduino_ms': edge_ms_map.get(key),
                'signal': key,
                'edge': edge,
                'state': int(states.get(key, False)),
                'count': self.ttl_pulse_counts[key],
            })
        self.last_event_state = states.copy()

    def _counts_changed(self, states: Dict, previous_states: Dict) -> bool:
        """Detect counter changes from Arduino if available."""
        for key in ('gate', 'sync'):
            count_key = f"{key}_count"
            if count_key in states and count_key in previous_states:
                if states[count_key] != previous_states[count_key]:
                    return True
        if 'barcode_count' in states and 'barcode_count' in previous_states:
            if states['barcode_count'] != previous_states['barcode_count']:
                return True
        return False

    def get_ttl_event_history(self) -> List[Dict]:
        """Get TTL edge event history."""
        return self.ttl_event_history.copy()

    def get_ttl_pulse_counts(self) -> Dict:
        """Get TTL pulse counts based on detected rising edges."""
        return self.ttl_pulse_counts.copy()

    def clear_ttl_event_history(self):
        """Clear TTL event history."""
        self.ttl_event_history.clear()

