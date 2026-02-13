/*
 * Arduino TTL Output Generator for Basler Camera Application
 *
 * Version: 2.0 (1Hz Sync)
 *
 * Updates:
 * - Changed from 10Hz to 1Hz sync
 * - Pulse duration: 50ms pulse every 1 second
 * - Fixed timing logic to prevent frequency drift
 * - Improved state machine robustness
 * - Added GET_PINS command
 */

#include <Arduino.h>

// ============================================================================
// Pin Configuration
// ============================================================================
const uint8_t GATE_PINS[] = {6, 7};
const uint8_t SYNC_1HZ_PINS[] = {8, 9};
const uint8_t BARCODE_PINS[] = {10, 11};

const uint8_t N_GATE_PINS = sizeof(GATE_PINS) / sizeof(GATE_PINS[0]);
const uint8_t N_SYNC_PINS = sizeof(SYNC_1HZ_PINS) / sizeof(SYNC_1HZ_PINS[0]);
const uint8_t N_BARCODE_PINS = sizeof(BARCODE_PINS) / sizeof(BARCODE_PINS[0]);


// ============================================================================
// 1Hz Sync Configuration (Modified)
// ============================================================================
const uint16_t SYNC_1HZ_PERIOD_MS = 1000;  // 1Hz = 1000ms period
const uint16_t SYNC_1HZ_PULSE_MS = 50;     // 50ms HIGH pulse

// ============================================================================
// Barcode Configuration
// ============================================================================
const uint8_t  BARCODE_BITS = 32;
const uint16_t BARCODE_MAX_PULSE_MS = 500;
const uint16_t START_PULSE_MS_RAW = 100;
const uint16_t START_LOW_MS = 100;
const uint16_t BIT_MS_RAW = 100;
const uint16_t START_PULSE_MS = (START_PULSE_MS_RAW > BARCODE_MAX_PULSE_MS)
  ? BARCODE_MAX_PULSE_MS
  : START_PULSE_MS_RAW;
const uint16_t BIT_MS = (BIT_MS_RAW > BARCODE_MAX_PULSE_MS)
  ? BARCODE_MAX_PULSE_MS
  : BIT_MS_RAW;
const uint32_t INTER_BARCODE_INTERVAL_MS = 5000;

// ============================================================================
// State Variables
// ============================================================================
enum BarcodeState : uint8_t { BC_START_HI, BC_START_LO, BC_BITS, BC_GAP };
static BarcodeState bcState = BC_GAP;
static uint32_t bcNextMs = 0;
static uint8_t bcBitIdx = 0;
static uint32_t bcWord = 0;

enum RunMode : uint8_t { MODE_IDLE, MODE_RECORDING, MODE_TEST };
static RunMode currentMode = MODE_IDLE;

// 1Hz Sync State
static uint32_t sync1HzNextMs = 0;
static bool sync1HzState = false;

// Gate State Tracker (for reporting)
static bool gateState = false;
static bool syncPinState = false;
static bool barcodePinState = false;
static uint32_t gateEdgeMs = 0;
static uint32_t syncEdgeMs = 0;
static uint32_t barcodeEdgeMs = 0;
static uint32_t gateCount = 0;
static uint32_t syncCount = 0;
static uint32_t barcodeCount = 0;

// ============================================================================
// Helper Functions
// ============================================================================

static uint32_t barcodeGapMs() {
  uint32_t used = START_PULSE_MS + START_LOW_MS + (uint32_t)BARCODE_BITS * BIT_MS;
  return (INTER_BARCODE_INTERVAL_MS > used) ? (INTER_BARCODE_INTERVAL_MS - used) : 0;
}

void setGatePins(bool state) {
  if (state != gateState) {
    gateEdgeMs = millis();
    if (state) {
      gateCount++;
    }
  }
  gateState = state;
  for (uint8_t i = 0; i < N_GATE_PINS; i++) {
    digitalWrite(GATE_PINS[i], state ? HIGH : LOW);
  }
}
// ...existing code...
void setSyncPins(bool state) {
  if (state != syncPinState) {
    syncEdgeMs = millis();
    if (state) {
      syncCount++;
    }
    syncPinState = state;
  }
  for (uint8_t i = 0; i < N_SYNC_PINS; i++) {
    digitalWrite(SYNC_1HZ_PINS[i], state ? HIGH : LOW);
  }
}

void setBarcodePins(bool state) {
  if (state != barcodePinState) {
    barcodeEdgeMs = millis();
    if (state) {
      barcodeCount++;
    }
    barcodePinState = state;
  }
  for (uint8_t i = 0; i < N_BARCODE_PINS; i++) {
    digitalWrite(BARCODE_PINS[i], state ? HIGH : LOW);
  }
}

void writeBarcodeBit(bool bitVal) {
  if (bitVal != barcodePinState) {
    barcodeEdgeMs = millis();
    if (bitVal) {
      barcodeCount++;
    }
    barcodePinState = bitVal;
  }
  for (uint8_t i = 0; i < N_BARCODE_PINS; i++) {
    digitalWrite(BARCODE_PINS[i], bitVal ? HIGH : LOW);
  }
}

void stopAllOutputs() {
  setGatePins(false);
  setSyncPins(false);
  setBarcodePins(false);

  sync1HzState = false;
}

void startOutputs() {
  // Reset outputs first to ensure clean state
  stopAllOutputs();

  // Start gate
  setGatePins(true);

  // Initialize 1Hz sync
  sync1HzState = false;
  syncPinState = false;
  sync1HzNextMs = millis(); // Start immediately
  setSyncPins(false);

  // Initialize barcode
  bcState = BC_START_HI;
  bcNextMs = millis();
  bcBitIdx = 0;
  barcodePinState = false;
  gateCount = 0;
  syncCount = 0;
  barcodeCount = 0;
}

// ============================================================================
// Barcode State Machine
// ============================================================================
void updateBarcode() {
  uint32_t now = millis();

  if ((int32_t)(now - bcNextMs) < 0) {
    return;  // Not time yet
  }

  switch (bcState) {
    case BC_START_HI:
      setBarcodePins(true);
      bcNextMs += START_PULSE_MS;
      bcState = BC_START_LO;
      break;

    case BC_START_LO:
      setBarcodePins(false);
      bcNextMs += START_LOW_MS;
      bcState = BC_BITS;
      bcBitIdx = 0;
      break;

    case BC_BITS:
      {
        bool bitVal = (bcWord >> (BARCODE_BITS - 1 - bcBitIdx)) & 1;
        writeBarcodeBit(bitVal);
        bcBitIdx++;

        if (bcBitIdx >= BARCODE_BITS) {
          bcState = BC_GAP;
          bcNextMs += barcodeGapMs();
        } else {
          bcNextMs += BIT_MS;
        }
      }
      break;

    case BC_GAP:
      setBarcodePins(false);
      bcWord++;  // Increment for next barcode
      bcState = BC_START_HI;
      // Start next barcode immediately after gap duration
      break;
  }
}

// ============================================================================
// 1Hz Sync Update
// ============================================================================
void update1HzSync() {
  uint32_t now = millis();

  if ((int32_t)(now - sync1HzNextMs) < 0) {
    return;  // Not time yet
  }

  if (sync1HzState) {
    // Currently HIGH, go LOW
    setSyncPins(false);
    sync1HzState = false;
    sync1HzNextMs += (SYNC_1HZ_PERIOD_MS - SYNC_1HZ_PULSE_MS);
  } else {
    // Currently LOW, go HIGH
    setSyncPins(true);
    sync1HzState = true;
    sync1HzNextMs += SYNC_1HZ_PULSE_MS;
  }
}

// ============================================================================
// Setup
// ============================================================================
void setup() {
  Serial.begin(115200);

  // Configure pins as outputs
  for (uint8_t i = 0; i < N_GATE_PINS; i++) pinMode(GATE_PINS[i], OUTPUT);
  for (uint8_t i = 0; i < N_SYNC_PINS; i++) pinMode(SYNC_1HZ_PINS[i], OUTPUT);
  for (uint8_t i = 0; i < N_BARCODE_PINS; i++) pinMode(BARCODE_PINS[i], OUTPUT);

  // Initialize all LOW
  stopAllOutputs();

  // Wait for serial
  while (!Serial) {
    ;
  }

  Serial.println("READY");
}

// ============================================================================
// Main Loop
// ============================================================================
void loop() {
  // Process serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "START_RECORDING") {
      currentMode = MODE_RECORDING;
      bcWord = 0;  // Reset barcode counter
      startOutputs();
      Serial.println("OK_RECORDING");

    } else if (command == "STOP_RECORDING") {
      stopAllOutputs();
      currentMode = MODE_IDLE;
      Serial.println("OK_STOPPED");

    } else if (command == "START_TEST") {
      currentMode = MODE_TEST;
      bcWord = 0;  // Reset barcode counter
      startOutputs();
      Serial.println("OK_TEST");

    } else if (command == "STOP_TEST") {
      stopAllOutputs();
      currentMode = MODE_IDLE;
      Serial.println("OK_STOPPED");

    } else if (command == "GET_STATES") {
      // Return format: GATE,SYNC,BARCODE0,BARCODE1
      // Use tracked state vars for reliability
      Serial.print(gateState ? 1 : 0);
      Serial.print(",");
      // Use sync state tracker
      Serial.print(sync1HzState ? 1 : 0);
      Serial.print(",");
      Serial.print(digitalRead(BARCODE_PINS[0]));
      Serial.print(",");
      Serial.print((N_BARCODE_PINS > 1) ? digitalRead(BARCODE_PINS[1]) : 0);
      Serial.print(",");
      Serial.print(gateEdgeMs);
      Serial.print(",");
      Serial.print(syncEdgeMs);
      Serial.print(",");
      Serial.print(barcodeEdgeMs);
      Serial.print(",");
      Serial.print(gateCount);
      Serial.print(",");
      Serial.print(syncCount);
      Serial.print(",");
      Serial.println(barcodeCount);

    } else if (command == "GET_PINS") {
      // Return full pin config
      // Format: GATE:6,7,SYNC:8,9,BARCODE:10,11
      Serial.print("GATE:");
      Serial.print(GATE_PINS[0]);
      Serial.print(",");
      Serial.print(GATE_PINS[1]);
      Serial.print(",SYNC:");
      Serial.print(SYNC_1HZ_PINS[0]);
      Serial.print(",");
      Serial.print(SYNC_1HZ_PINS[1]);
      Serial.print(",BARCODE:");
      Serial.print(BARCODE_PINS[0]);
      Serial.print(",");
      Serial.println(BARCODE_PINS[1]);
    }
  }

  // Update outputs independently
  if (currentMode == MODE_RECORDING || currentMode == MODE_TEST) {
    // Continuously assert Gate HIGH during recording/test
    // This provides robustness against any electrical noise or state glitches
    setGatePins(true);

    update1HzSync();
    updateBarcode();
  } else {
    // Ensure outputs stay OFF when idle
    // This is the "Generation stops when recording stops" enforcement
    stopAllOutputs();
  }
}

