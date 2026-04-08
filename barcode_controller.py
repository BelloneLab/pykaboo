"""
barcode_controller.py — Control the Arduino barcode generator from Python/camApp

Usage:
    from barcode_controller import BarcodeController
    
    bc = BarcodeController('COM3')  # or '/dev/ttyUSB0'
    bc.set_bit_width(80)            # 80ms per bit (2 frames at 25fps)
    bc.start()                      # start barcode on D8/D9
    # ... run acquisition ...
    bc.stop()                       # stop barcode
    bc.close()

The Arduino must be flashed with StandardFirmataBarcode.ino

Sysex protocol:
    Command byte: 0x7D
    Sub-commands: START=0x01, STOP=0x02, SET_BITWIDTH=0x03, 
                  RESET=0x04, QUERY=0x05
"""

import pyfirmata
import time
import struct

BARCODE_SYSEX_CMD = 0x7D

# Sub-commands
BARCODE_START         = 0x01
BARCODE_STOP          = 0x02
BARCODE_SET_BITWIDTH  = 0x03
BARCODE_RESET_COUNTER = 0x04
BARCODE_QUERY_STATUS  = 0x05


class BarcodeController:
    """
    Controls the barcode sync generator on an Arduino running 
    StandardFirmataBarcode.
    
    The barcode encodes a 32-bit word counter on pin D9 (data, LSB first)
    with a word-sync pulse on pin D8 (HIGH during bit 0 of each word).
    
    Timing is handled entirely by the Arduino's Timer1 hardware interrupt,
    so Python jitter does not affect barcode precision.
    
    Parameters
    ----------
    port : str
        Serial port (e.g. 'COM3', '/dev/ttyUSB0')
    bit_width_ms : int
        Duration of each bit in milliseconds. Default 80ms = 2 frames at 25fps.
        Must be between 20 and 500.
    """
    
    def __init__(self, port, bit_width_ms=80):
        self.board = pyfirmata.Arduino(port)
        time.sleep(2)  # wait for Arduino reset
        
        # Start the iterator to handle incoming sysex replies
        self._iterator = pyfirmata.util.Iterator(self.board)
        self._iterator.start()
        
        self.bit_width_ms = bit_width_ms
        self._running = False
        
        # Register sysex handler for status replies
        self.last_status = None
        self.board.add_cmd_handler(
            BARCODE_SYSEX_CMD, 
            self._handle_status_reply
        )
        
        # Set bit width on Arduino
        self.set_bit_width(bit_width_ms)
    
    def _send_sysex(self, *data):
        """Send a sysex message to the barcode module."""
        self.board.send_sysex(BARCODE_SYSEX_CMD, list(data))
    
    def _handle_status_reply(self, *data):
        """Handle status query reply from Arduino."""
        if len(data) >= 10 and data[0] == BARCODE_QUERY_STATUS:
            running = bool(data[1])
            bit_width = data[2] | (data[3] << 7)
            counter = (data[4] | (data[5] << 7) | 
                      (data[6] << 14) | (data[7] << 21) | 
                      (data[8] << 28))
            bit_index = data[9]
            self.last_status = {
                'running': running,
                'bit_width_ms': bit_width,
                'counter': counter,
                'bit_index': bit_index,
            }
    
    def start(self):
        """Start barcode generation. D8=sync, D9=data."""
        self._send_sysex(BARCODE_START)
        self._running = True
    
    def stop(self):
        """Stop barcode generation. Pins go LOW."""
        self._send_sysex(BARCODE_STOP)
        self._running = False
    
    def set_bit_width(self, ms):
        """
        Set bit width in milliseconds (20-500).
        
        Recommended values:
            25 fps → 80ms  (2 frames/bit)  → 2.56s/word
            30 fps → 66ms  (2 frames/bit)  → 2.11s/word
            60 fps → 34ms  (2 frames/bit)  → 1.09s/word
            
        For robust decoding, use at least 2× frame interval.
        3× frame interval is even safer for noisy environments.
        """
        ms = max(20, min(500, int(ms)))
        self.bit_width_ms = ms
        low7  = ms & 0x7F
        high7 = (ms >> 7) & 0x7F
        self._send_sysex(BARCODE_SET_BITWIDTH, low7, high7)
    
    def reset_counter(self):
        """Reset the word counter to 0. Useful at acquisition start."""
        self._send_sysex(BARCODE_RESET_COUNTER)
    
    def query_status(self, timeout=0.5):
        """
        Query current barcode status from Arduino.
        
        Returns dict with: running, bit_width_ms, counter, bit_index
        """
        self.last_status = None
        self._send_sysex(BARCODE_QUERY_STATUS)
        t0 = time.time()
        while self.last_status is None and (time.time() - t0) < timeout:
            time.sleep(0.01)
        return self.last_status
    
    @property
    def is_running(self):
        return self._running
    
    def close(self):
        """Stop barcode and close serial connection."""
        if self._running:
            self.stop()
        time.sleep(0.1)
        self.board.exit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


# ─── Barcode decoding utilities (for timeSy / offline analysis) ──────────────

def decode_barcode_from_edges(edge_times, bit_width_s=0.080, n_bits=32):
    """
    Decode barcode words from edge timestamps (e.g. from CatGT .xd file).
    
    Parameters
    ----------
    edge_times : array-like
        Rising and falling edge times in seconds
    bit_width_s : float
        Expected bit width in seconds (default 0.080 = 80ms)
    n_bits : int
        Bits per word (default 32)
    
    Returns
    -------
    list of dict
        Each dict has: 'word' (int), 'time' (float, time of word start)
    """
    if len(edge_times) < 2:
        return []
    
    # Compute intervals between edges
    intervals = []
    for i in range(1, len(edge_times)):
        intervals.append(edge_times[i] - edge_times[i-1])
    
    # Find word boundaries: intervals >> bit_width indicate gaps between words
    # Within a word, intervals ≈ bit_width
    words = []
    word_start = 0
    current_bits = []
    
    for i, interval in enumerate(intervals):
        if interval > bit_width_s * 1.5:
            # Gap detected — end of previous word
            if len(current_bits) >= n_bits:
                word_val = 0
                for bit_idx, bit in enumerate(current_bits[:n_bits]):
                    word_val |= (bit << bit_idx)
                words.append({
                    'word': word_val,
                    'time': edge_times[word_start],
                })
            current_bits = []
            word_start = i + 1
        else:
            # Within a word: determine if this interval is a 0→1 or 1→0
            # For simple level encoding: sample the state at each bit center
            n_bits_in_interval = round(interval / bit_width_s)
            current_bits.extend([1] * n_bits_in_interval)
    
    return words


def decode_barcode_from_signal(values, sample_rate, bit_width_s=0.080, 
                                n_bits=32, threshold=None):
    """
    Decode barcode from a continuous signal (e.g. camera GPIO line, LED ROI).
    
    Parameters
    ----------
    values : array-like
        Signal values (e.g. pixel intensity or GPIO state per frame)
    sample_rate : float
        Sample rate in Hz (e.g. 25 for 25fps camera)
    bit_width_s : float
        Expected bit width in seconds
    n_bits : int
        Bits per word
    threshold : float or None
        Binarization threshold. If None, uses midpoint of min/max.
    
    Returns
    -------
    list of dict
        Each dict has: 'word' (int), 'sample_index' (int), 'time' (float)
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    
    if threshold is None:
        threshold = (values.min() + values.max()) / 2
    
    binary = (values > threshold).astype(int)
    
    # Find rising edges on sync pin (if available) or segment by bit width
    samples_per_bit = int(round(bit_width_s * sample_rate))
    samples_per_word = samples_per_bit * n_bits
    
    # Detect word boundaries using sync channel or falling-edge gaps
    # Simple approach: scan for long LOW periods (inter-word gaps)
    words = []
    i = 0
    while i < len(binary) - samples_per_word:
        # Skip LOW regions
        if binary[i] == 0:
            i += 1
            continue
        
        # Found a HIGH — try to read a word starting here
        word_val = 0
        for bit in range(n_bits):
            bit_center = i + int(bit * samples_per_bit + samples_per_bit / 2)
            if bit_center >= len(binary):
                break
            if binary[bit_center]:
                word_val |= (1 << bit)
        
        words.append({
            'word': word_val,
            'sample_index': i,
            'time': i / sample_rate,
        })
        
        i += samples_per_word + samples_per_bit  # skip past this word + gap
    
    return words


# ─── Example integration with camApp ─────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Barcode sync controller')
    parser.add_argument('port', help='Arduino serial port (e.g. COM3)')
    parser.add_argument('--bit-width', type=int, default=80,
                       help='Bit width in ms (default: 80)')
    parser.add_argument('--duration', type=float, default=10,
                       help='Run duration in seconds (default: 10)')
    args = parser.parse_args()
    
    print(f"Connecting to Arduino on {args.port}...")
    
    with BarcodeController(args.port, bit_width_ms=args.bit_width) as bc:
        print(f"Bit width: {args.bit_width}ms")
        print(f"Word duration: {args.bit_width * 32 / 1000:.2f}s")
        print(f"Running for {args.duration}s...")
        
        bc.reset_counter()
        bc.start()
        
        t0 = time.time()
        while (time.time() - t0) < args.duration:
            status = bc.query_status()
            if status:
                print(f"\r  Counter: {status['counter']:>8d}  "
                      f"Bit: {status['bit_index']:>2d}/32  "
                      f"Running: {status['running']}", end='')
            time.sleep(0.5)
        
        bc.stop()
        final = bc.query_status()
        print(f"\n\nDone. Final counter: {final['counter'] if final else '?'}")
