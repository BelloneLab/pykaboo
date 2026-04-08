/*
  StandardFirmata + Barcode Sync Generator
  =========================================
  Based on StandardFirmata (Firmata 2.5.8)
  
  ADDITIONS:
  - Hardware Timer1-driven barcode output on D8 (word sync) and D9 (data)
  - 32-bit binary barcode encoding a frame/word counter
  - Bit width configurable (default 80ms = 2 frames at 25fps)
  - Controlled via Sysex commands from host (pyfirmata compatible)
  
  BARCODE FORMAT:
  Each "word" = 32 data bits transmitted LSB-first on D9
  D8 pulses HIGH during bit 0 of each word (word sync marker)
  Word counter increments after each complete word transmission
  
  SYSEX COMMANDS (from host):
  0x01 = START barcode generation
  0x02 = STOP barcode generation  
  0x03 = SET bit width in ms (2 bytes: low7, high7)
  0x04 = RESET word counter to 0
  0x05 = QUERY status (replies with counter + running state)
  
  TIMING:
  Timer1 Compare Match A interrupt fires every [bitWidth] ms
  Default bitWidth = 80ms → each bit spans 2 frames at 25fps
  32 bits × 80ms = 2.56s per word
  At 25fps: ~64 frames per word, plenty of margin
  
  Hardware: Arduino Uno / Nano (ATmega328P, 16MHz)
  
  Copyright (C) 2006-2008 Hans-Christoph Steiner.  All rights reserved.
  Copyright (C) 2010-2011 Paul Stoffregen.  All rights reserved.
  Copyright (C) 2009 Shigeru Kobayashi.  All rights reserved.
  Copyright (C) 2009-2016 Jeff Hoefs.  All rights reserved.

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  See file LICENSE.txt for further informations on licensing terms.
*/

#include <Wire.h>
#include <Firmata.h>

/*==============================================================================
 * BARCODE GENERATOR CONFIGURATION
 *============================================================================*/

#define BARCODE_DATA_PIN      9    // D9: barcode data output (LSB first)
#define BARCODE_SYNC_PIN      8    // D8: word sync pulse (HIGH on bit 0)
#define BARCODE_NUM_BITS      32   // bits per word
#define BARCODE_DEFAULT_BIT_MS 80  // ms per bit (2 frames at 25fps)
#define BARCODE_SYSEX_CMD     0x7D // Sysex command ID (unused in standard Firmata)

// Barcode sub-commands (sent as first data byte after sysex cmd)
#define BARCODE_START         0x01
#define BARCODE_STOP          0x02
#define BARCODE_SET_BITWIDTH  0x03
#define BARCODE_RESET_COUNTER 0x04
#define BARCODE_QUERY_STATUS  0x05

// Barcode state — all volatile because modified in ISR
volatile bool     barcodeRunning   = false;
volatile uint32_t barcodeCounter   = 0;     // word counter (the value being encoded)
volatile uint8_t  barcodeBitIndex  = 0;     // current bit position (0..31)
volatile uint32_t barcodeCurrentWord = 0;   // snapshot of counter for current word

// Configurable
uint16_t barcodeBitWidth_ms = BARCODE_DEFAULT_BIT_MS;

/*==============================================================================
 * TIMER1 SETUP AND ISR FOR BARCODE
 * 
 * Timer1 is 16-bit, prescaler 256 → tick = 16μs at 16MHz
 * CTC mode: fires ISR exactly every bitWidth_ms with ~μs jitter
 * No Servo library → Timer1 is free
 *============================================================================*/

void barcodeTimerSetup(uint16_t interval_ms) {
  noInterrupts();
  
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  
  // CTC mode (WGM12)
  TCCR1B |= (1 << WGM12);
  
  // Prescaler 256 → 62500 ticks/sec → 16μs/tick
  TCCR1B |= (1 << CS12);
  
  // Compare match value: 80ms → 80 * 62.5 = 5000
  uint16_t compareVal = (uint16_t)((uint32_t)interval_ms * 625UL / 10UL);
  OCR1A = compareVal;
  
  // Enable Timer1 Compare Match A interrupt
  TIMSK1 |= (1 << OCIE1A);
  
  interrupts();
}

void barcodeTimerStop() {
  noInterrupts();
  TIMSK1 &= ~(1 << OCIE1A);
  TCCR1B = 0;
  interrupts();
}

// Timer1 Compare Match A ISR — fires every bitWidth_ms
ISR(TIMER1_COMPA_vect) {
  if (!barcodeRunning) return;
  
  if (barcodeBitIndex == 0) {
    barcodeCurrentWord = barcodeCounter;
    PORTB |= (1 << PB0);   // D8 HIGH (word sync)
  } else {
    PORTB &= ~(1 << PB0);  // D8 LOW
  }
  
  if (barcodeCurrentWord & ((uint32_t)1 << barcodeBitIndex)) {
    PORTB |= (1 << PB1);   // D9 HIGH
  } else {
    PORTB &= ~(1 << PB1);  // D9 LOW
  }
  
  barcodeBitIndex++;
  if (barcodeBitIndex >= BARCODE_NUM_BITS) {
    barcodeBitIndex = 0;
    barcodeCounter++;
  }
}

/*==============================================================================
 * BARCODE CONTROL FUNCTIONS
 *============================================================================*/

void barcodeStart() {
  // Configure pins as output (direct register for speed)
  pinMode(BARCODE_DATA_PIN, OUTPUT);
  pinMode(BARCODE_SYNC_PIN, OUTPUT);
  digitalWrite(BARCODE_DATA_PIN, LOW);
  digitalWrite(BARCODE_SYNC_PIN, LOW);
  
  // Mark pins in Firmata so they're not overwritten
  Firmata.setPinMode(BARCODE_DATA_PIN, OUTPUT);
  Firmata.setPinMode(BARCODE_SYNC_PIN, OUTPUT);
  
  barcodeBitIndex = 0;
  barcodeRunning = true;
  barcodeTimerSetup(barcodeBitWidth_ms);
}

void barcodeStop() {
  barcodeRunning = false;
  barcodeTimerStop();
  digitalWrite(BARCODE_DATA_PIN, LOW);
  digitalWrite(BARCODE_SYNC_PIN, LOW);
}

void barcodeSendStatus() {
  // Reply with: [cmd, running, counter_b0..b3 (each split into 7-bit pairs)]
  uint8_t reply[12];
  reply[0] = BARCODE_QUERY_STATUS;
  reply[1] = barcodeRunning ? 1 : 0;
  reply[2] = barcodeBitWidth_ms & 0x7F;
  reply[3] = (barcodeBitWidth_ms >> 7) & 0x7F;
  
  uint32_t cnt = barcodeCounter;
  reply[4] = cnt & 0x7F;
  reply[5] = (cnt >> 7) & 0x7F;
  reply[6] = (cnt >> 14) & 0x7F;
  reply[7] = (cnt >> 21) & 0x7F;
  reply[8] = (cnt >> 28) & 0x0F;
  
  reply[9] = barcodeBitIndex;
  
  Firmata.sendSysex(BARCODE_SYSEX_CMD, 10, reply);
}

/*==============================================================================
 * STANDARD FIRMATA DEFINITIONS (unchanged)
 *============================================================================*/

#define I2C_WRITE                   B00000000
#define I2C_READ                    B00001000
#define I2C_READ_CONTINUOUSLY       B00010000
#define I2C_STOP_READING            B00011000
#define I2C_READ_WRITE_MODE_MASK    B00011000
#define I2C_10BIT_ADDRESS_MODE_MASK B00100000
#define I2C_END_TX_MASK             B01000000
#define I2C_STOP_TX                 1
#define I2C_RESTART_TX              0
#define I2C_MAX_QUERIES             8
#define I2C_REGISTER_NOT_SPECIFIED  -1

#define MINIMUM_SAMPLING_INTERVAL   1

/*==============================================================================
 * GLOBAL VARIABLES
 *============================================================================*/

#ifdef FIRMATA_SERIAL_FEATURE
SerialFirmata serialFeature;
#endif

int analogInputsToReport = 0;
byte reportPINs[TOTAL_PORTS];
byte previousPINs[TOTAL_PORTS];
byte portConfigInputs[TOTAL_PORTS];

unsigned long currentMillis;
unsigned long previousMillis;
unsigned int samplingInterval = 19;

struct i2c_device_info {
  byte addr;
  int reg;
  byte bytes;
  byte stopTX;
};

i2c_device_info query[I2C_MAX_QUERIES];
byte i2cRxData[64];
boolean isI2CEnabled = false;
signed char queryIndex = -1;
unsigned int i2cReadDelayTime = 0;

boolean isResetting = false;

void setPinModeCallback(byte, int);
void reportAnalogCallback(byte analogPin, int value);
void sysexCallback(byte, byte, byte*);

void wireWrite(byte data) {
#if ARDUINO >= 100
  Wire.write((byte)data);
#else
  Wire.send(data);
#endif
}

byte wireRead(void) {
#if ARDUINO >= 100
  return Wire.read();
#else
  return Wire.receive();
#endif
}

/*==============================================================================
 * FUNCTIONS (standard Firmata — no servo, barcode uses Timer1)
 *============================================================================*/

void enableI2CPins() {
  byte i;
  for (i = 0; i < TOTAL_PINS; i++) {
    if (IS_PIN_I2C(i)) {
      setPinModeCallback(i, PIN_MODE_I2C);
    }
  }
  isI2CEnabled = true;
  Wire.begin();
}

void disableI2CPins() {
  isI2CEnabled = false;
  queryIndex = -1;
}

void readAndReportData(byte address, int theRegister, byte numBytes, byte stopTX) {
  if (theRegister != I2C_REGISTER_NOT_SPECIFIED) {
    Wire.beginTransmission(address);
    wireWrite((byte)theRegister);
    Wire.endTransmission(stopTX);
    if (i2cReadDelayTime > 0) {
      delayMicroseconds(i2cReadDelayTime);
    }
  } else {
    theRegister = 0;
  }
  Wire.requestFrom(address, numBytes);
  if (numBytes < Wire.available()) {
    Firmata.sendString("I2C: Too many bytes received");
  } else if (numBytes > Wire.available()) {
    Firmata.sendString("I2C: Too few bytes received");
    numBytes = Wire.available();
  }
  i2cRxData[0] = address;
  i2cRxData[1] = theRegister;
  for (int i = 0; i < numBytes && Wire.available(); i++) {
    i2cRxData[2 + i] = wireRead();
  }
  Firmata.sendSysex(SYSEX_I2C_REPLY, numBytes + 2, i2cRxData);
}

void outputPort(byte portNumber, byte portValue, byte forceSend) {
  portValue = portValue & portConfigInputs[portNumber];
  if (forceSend || previousPINs[portNumber] != portValue) {
    Firmata.sendDigitalPort(portNumber, portValue);
    previousPINs[portNumber] = portValue;
  }
}

void checkDigitalInputs(void) {
  if (TOTAL_PORTS > 0 && reportPINs[0]) outputPort(0, readPort(0, portConfigInputs[0]), false);
  if (TOTAL_PORTS > 1 && reportPINs[1]) outputPort(1, readPort(1, portConfigInputs[1]), false);
  if (TOTAL_PORTS > 2 && reportPINs[2]) outputPort(2, readPort(2, portConfigInputs[2]), false);
  if (TOTAL_PORTS > 3 && reportPINs[3]) outputPort(3, readPort(3, portConfigInputs[3]), false);
  if (TOTAL_PORTS > 4 && reportPINs[4]) outputPort(4, readPort(4, portConfigInputs[4]), false);
  if (TOTAL_PORTS > 5 && reportPINs[5]) outputPort(5, readPort(5, portConfigInputs[5]), false);
  if (TOTAL_PORTS > 6 && reportPINs[6]) outputPort(6, readPort(6, portConfigInputs[6]), false);
  if (TOTAL_PORTS > 7 && reportPINs[7]) outputPort(7, readPort(7, portConfigInputs[7]), false);
  if (TOTAL_PORTS > 8 && reportPINs[8]) outputPort(8, readPort(8, portConfigInputs[8]), false);
  if (TOTAL_PORTS > 9 && reportPINs[9]) outputPort(9, readPort(9, portConfigInputs[9]), false);
  if (TOTAL_PORTS > 10 && reportPINs[10]) outputPort(10, readPort(10, portConfigInputs[10]), false);
  if (TOTAL_PORTS > 11 && reportPINs[11]) outputPort(11, readPort(11, portConfigInputs[11]), false);
  if (TOTAL_PORTS > 12 && reportPINs[12]) outputPort(12, readPort(12, portConfigInputs[12]), false);
  if (TOTAL_PORTS > 13 && reportPINs[13]) outputPort(13, readPort(13, portConfigInputs[13]), false);
  if (TOTAL_PORTS > 14 && reportPINs[14]) outputPort(14, readPort(14, portConfigInputs[14]), false);
  if (TOTAL_PORTS > 15 && reportPINs[15]) outputPort(15, readPort(15, portConfigInputs[15]), false);
}

void setPinModeCallback(byte pin, int mode) {
  if (Firmata.getPinMode(pin) == PIN_MODE_IGNORE)
    return;
    
  // *** BARCODE PROTECTION: don't let Firmata reconfigure barcode pins ***
  if (barcodeRunning && (pin == BARCODE_DATA_PIN || pin == BARCODE_SYNC_PIN)) {
    Firmata.sendString("Pin reserved for barcode");
    return;
  }

  if (Firmata.getPinMode(pin) == PIN_MODE_I2C && isI2CEnabled && mode != PIN_MODE_I2C) {
    disableI2CPins();
  }
  if (IS_PIN_ANALOG(pin)) {
    reportAnalogCallback(PIN_TO_ANALOG(pin), mode == PIN_MODE_ANALOG ? 1 : 0);
  }
  if (IS_PIN_DIGITAL(pin)) {
    if (mode == INPUT || mode == PIN_MODE_PULLUP) {
      portConfigInputs[pin / 8] |= (1 << (pin & 7));
    } else {
      portConfigInputs[pin / 8] &= ~(1 << (pin & 7));
    }
  }
  Firmata.setPinState(pin, 0);
  switch (mode) {
    case PIN_MODE_ANALOG:
      if (IS_PIN_ANALOG(pin)) {
        if (IS_PIN_DIGITAL(pin)) {
          pinMode(PIN_TO_DIGITAL(pin), INPUT);
#if ARDUINO <= 100
          digitalWrite(PIN_TO_DIGITAL(pin), LOW);
#endif
        }
        Firmata.setPinMode(pin, PIN_MODE_ANALOG);
      }
      break;
    case INPUT:
      if (IS_PIN_DIGITAL(pin)) {
        pinMode(PIN_TO_DIGITAL(pin), INPUT);
#if ARDUINO <= 100
        digitalWrite(PIN_TO_DIGITAL(pin), LOW);
#endif
        Firmata.setPinMode(pin, INPUT);
      }
      break;
    case PIN_MODE_PULLUP:
      if (IS_PIN_DIGITAL(pin)) {
        pinMode(PIN_TO_DIGITAL(pin), INPUT_PULLUP);
        Firmata.setPinMode(pin, PIN_MODE_PULLUP);
        Firmata.setPinState(pin, 1);
      }
      break;
    case OUTPUT:
      if (IS_PIN_DIGITAL(pin)) {
        if (Firmata.getPinMode(pin) == PIN_MODE_PWM) {
          digitalWrite(PIN_TO_DIGITAL(pin), LOW);
        }
        pinMode(PIN_TO_DIGITAL(pin), OUTPUT);
        Firmata.setPinMode(pin, OUTPUT);
      }
      break;
    case PIN_MODE_PWM:
      if (IS_PIN_PWM(pin)) {
        pinMode(PIN_TO_PWM(pin), OUTPUT);
        analogWrite(PIN_TO_PWM(pin), 0);
        Firmata.setPinMode(pin, PIN_MODE_PWM);
      }
      break;
    case PIN_MODE_I2C:
      if (IS_PIN_I2C(pin)) {
        Firmata.setPinMode(pin, PIN_MODE_I2C);
      }
      break;
    case PIN_MODE_SERIAL:
#ifdef FIRMATA_SERIAL_FEATURE
      serialFeature.handlePinMode(pin, PIN_MODE_SERIAL);
#endif
      break;
    default:
      Firmata.sendString("Unknown pin mode");
  }
}

void setPinValueCallback(byte pin, int value) {
  if (pin < TOTAL_PINS && IS_PIN_DIGITAL(pin)) {
    if (Firmata.getPinMode(pin) == OUTPUT) {
      Firmata.setPinState(pin, value);
      digitalWrite(PIN_TO_DIGITAL(pin), value);
    }
  }
}

void analogWriteCallback(byte pin, int value) {
  if (pin < TOTAL_PINS) {
    switch (Firmata.getPinMode(pin)) {
      case PIN_MODE_PWM:
        if (IS_PIN_PWM(pin))
          analogWrite(PIN_TO_PWM(pin), value);
        Firmata.setPinState(pin, value);
        break;
    }
  }
}

void digitalWriteCallback(byte port, int value) {
  byte pin, lastPin, pinValue, mask = 1, pinWriteMask = 0;
  if (port < TOTAL_PORTS) {
    lastPin = port * 8 + 8;
    if (lastPin > TOTAL_PINS) lastPin = TOTAL_PINS;
    for (pin = port * 8; pin < lastPin; pin++) {
      // *** BARCODE PROTECTION ***
      if (barcodeRunning && (pin == BARCODE_DATA_PIN || pin == BARCODE_SYNC_PIN)) {
        mask = mask << 1;
        continue;
      }
      if (IS_PIN_DIGITAL(pin)) {
        if (Firmata.getPinMode(pin) == OUTPUT || Firmata.getPinMode(pin) == INPUT) {
          pinValue = ((byte)value & mask) ? 1 : 0;
          if (Firmata.getPinMode(pin) == OUTPUT) {
            pinWriteMask |= mask;
          } else if (Firmata.getPinMode(pin) == INPUT && pinValue == 1 && Firmata.getPinState(pin) != 1) {
#if ARDUINO > 100
            pinMode(pin, INPUT_PULLUP);
#else
            pinWriteMask |= mask;
#endif
          }
          Firmata.setPinState(pin, pinValue);
        }
      }
      mask = mask << 1;
    }
    writePort(port, (byte)value, pinWriteMask);
  }
}

void reportAnalogCallback(byte analogPin, int value) {
  if (analogPin < TOTAL_ANALOG_PINS) {
    if (value == 0) {
      analogInputsToReport = analogInputsToReport & ~(1 << analogPin);
    } else {
      analogInputsToReport = analogInputsToReport | (1 << analogPin);
      if (!isResetting) {
        Firmata.sendAnalog(analogPin, analogRead(analogPin));
      }
    }
  }
}

void reportDigitalCallback(byte port, int value) {
  if (port < TOTAL_PORTS) {
    reportPINs[port] = (byte)value;
    if (value) outputPort(port, readPort(port, portConfigInputs[port]), true);
  }
}

/*==============================================================================
 * SYSEX CALLBACK — with barcode command handler
 *============================================================================*/

void sysexCallback(byte command, byte argc, byte *argv) {
  byte mode;
  byte stopTX;
  byte slaveAddress;
  byte data;
  int slaveRegister;
  unsigned int delayTime;

  switch (command) {
  
    // *** BARCODE SYSEX HANDLER ***
    case BARCODE_SYSEX_CMD:
      if (argc < 1) break;
      switch (argv[0]) {
        case BARCODE_START:
          Firmata.sendString("Barcode: START");
          barcodeStart();
          break;
        case BARCODE_STOP:
          Firmata.sendString("Barcode: STOP");
          barcodeStop();
          break;
        case BARCODE_SET_BITWIDTH:
          if (argc >= 3) {
            barcodeBitWidth_ms = argv[1] | (argv[2] << 7);
            if (barcodeBitWidth_ms < 20) barcodeBitWidth_ms = 20;   // min 20ms
            if (barcodeBitWidth_ms > 500) barcodeBitWidth_ms = 500; // max 500ms
            Firmata.sendString("Barcode: bitwidth set");
            // If running, restart timer with new interval
            if (barcodeRunning) {
              barcodeTimerStop();
              barcodeTimerSetup(barcodeBitWidth_ms);
            }
          }
          break;
        case BARCODE_RESET_COUNTER:
          noInterrupts();
          barcodeCounter = 0;
          barcodeBitIndex = 0;
          interrupts();
          Firmata.sendString("Barcode: counter reset");
          break;
        case BARCODE_QUERY_STATUS:
          barcodeSendStatus();
          break;
      }
      break;

    case I2C_REQUEST:
      mode = argv[1] & I2C_READ_WRITE_MODE_MASK;
      if (argv[1] & I2C_10BIT_ADDRESS_MODE_MASK) {
        Firmata.sendString("10-bit addressing not supported");
        return;
      } else {
        slaveAddress = argv[0];
      }
      if (argv[1] & I2C_END_TX_MASK) {
        stopTX = I2C_RESTART_TX;
      } else {
        stopTX = I2C_STOP_TX;
      }
      switch (mode) {
        case I2C_WRITE:
          Wire.beginTransmission(slaveAddress);
          for (byte i = 2; i < argc; i += 2) {
            data = argv[i] + (argv[i + 1] << 7);
            wireWrite(data);
          }
          Wire.endTransmission();
          delayMicroseconds(70);
          break;
        case I2C_READ:
          if (argc == 6) {
            slaveRegister = argv[2] + (argv[3] << 7);
            data = argv[4] + (argv[5] << 7);
          } else {
            slaveRegister = I2C_REGISTER_NOT_SPECIFIED;
            data = argv[2] + (argv[3] << 7);
          }
          readAndReportData(slaveAddress, (int)slaveRegister, data, stopTX);
          break;
        case I2C_READ_CONTINUOUSLY:
          if ((queryIndex + 1) >= I2C_MAX_QUERIES) {
            Firmata.sendString("too many queries");
            break;
          }
          if (argc == 6) {
            slaveRegister = argv[2] + (argv[3] << 7);
            data = argv[4] + (argv[5] << 7);
          } else {
            slaveRegister = (int)I2C_REGISTER_NOT_SPECIFIED;
            data = argv[2] + (argv[3] << 7);
          }
          queryIndex++;
          query[queryIndex].addr = slaveAddress;
          query[queryIndex].reg = slaveRegister;
          query[queryIndex].bytes = data;
          query[queryIndex].stopTX = stopTX;
          break;
        case I2C_STOP_READING:
          byte queryIndexToSkip;
          if (queryIndex <= 0) {
            queryIndex = -1;
          } else {
            queryIndexToSkip = 0;
            for (byte i = 0; i < queryIndex + 1; i++) {
              if (query[i].addr == slaveAddress) {
                queryIndexToSkip = i;
                break;
              }
            }
            for (byte i = queryIndexToSkip; i < queryIndex + 1; i++) {
              if (i < I2C_MAX_QUERIES) {
                query[i].addr = query[i + 1].addr;
                query[i].reg = query[i + 1].reg;
                query[i].bytes = query[i + 1].bytes;
                query[i].stopTX = query[i + 1].stopTX;
              }
            }
            queryIndex--;
          }
          break;
        default:
          break;
      }
      break;
    case I2C_CONFIG:
      delayTime = (argv[0] + (argv[1] << 7));
      if (argc > 1 && delayTime > 0) {
        i2cReadDelayTime = delayTime;
      }
      if (!isI2CEnabled) {
        enableI2CPins();
      }
      break;
    case SAMPLING_INTERVAL:
      if (argc > 1) {
        samplingInterval = argv[0] + (argv[1] << 7);
        if (samplingInterval < MINIMUM_SAMPLING_INTERVAL) {
          samplingInterval = MINIMUM_SAMPLING_INTERVAL;
        }
      }
      break;
    case EXTENDED_ANALOG:
      if (argc > 1) {
        int val = argv[1];
        if (argc > 2) val |= (argv[2] << 7);
        if (argc > 3) val |= (argv[3] << 14);
        analogWriteCallback(argv[0], val);
      }
      break;
    case CAPABILITY_QUERY:
      Firmata.write(START_SYSEX);
      Firmata.write(CAPABILITY_RESPONSE);
      for (byte pin = 0; pin < TOTAL_PINS; pin++) {
        if (IS_PIN_DIGITAL(pin)) {
          Firmata.write((byte)INPUT);
          Firmata.write(1);
          Firmata.write((byte)PIN_MODE_PULLUP);
          Firmata.write(1);
          Firmata.write((byte)OUTPUT);
          Firmata.write(1);
        }
        if (IS_PIN_ANALOG(pin)) {
          Firmata.write(PIN_MODE_ANALOG);
          Firmata.write(10);
        }
        if (IS_PIN_PWM(pin)) {
          Firmata.write(PIN_MODE_PWM);
          Firmata.write(DEFAULT_PWM_RESOLUTION);
        }
        if (IS_PIN_I2C(pin)) {
          Firmata.write(PIN_MODE_I2C);
          Firmata.write(1);
        }
#ifdef FIRMATA_SERIAL_FEATURE
        serialFeature.handleCapability(pin);
#endif
        Firmata.write(127);
      }
      Firmata.write(END_SYSEX);
      break;
    case PIN_STATE_QUERY:
      if (argc > 0) {
        byte pin = argv[0];
        Firmata.write(START_SYSEX);
        Firmata.write(PIN_STATE_RESPONSE);
        Firmata.write(pin);
        if (pin < TOTAL_PINS) {
          Firmata.write(Firmata.getPinMode(pin));
          Firmata.write((byte)Firmata.getPinState(pin) & 0x7F);
          if (Firmata.getPinState(pin) & 0xFF80) Firmata.write((byte)(Firmata.getPinState(pin) >> 7) & 0x7F);
          if (Firmata.getPinState(pin) & 0xC000) Firmata.write((byte)(Firmata.getPinState(pin) >> 14) & 0x7F);
        }
        Firmata.write(END_SYSEX);
      }
      break;
    case ANALOG_MAPPING_QUERY:
      Firmata.write(START_SYSEX);
      Firmata.write(ANALOG_MAPPING_RESPONSE);
      for (byte pin = 0; pin < TOTAL_PINS; pin++) {
        Firmata.write(IS_PIN_ANALOG(pin) ? PIN_TO_ANALOG(pin) : 127);
      }
      Firmata.write(END_SYSEX);
      break;
    case SERIAL_MESSAGE:
#ifdef FIRMATA_SERIAL_FEATURE
      serialFeature.handleSysex(command, argc, argv);
#endif
      break;
  }
}

/*==============================================================================
 * SETUP
 *============================================================================*/

void systemResetCallback() {
  isResetting = true;

#ifdef FIRMATA_SERIAL_FEATURE
  serialFeature.reset();
#endif

  // Stop barcode on system reset
  barcodeStop();
  barcodeCounter = 0;
  barcodeBitIndex = 0;
  barcodeBitWidth_ms = BARCODE_DEFAULT_BIT_MS;

  if (isI2CEnabled) {
    disableI2CPins();
  }

  for (byte i = 0; i < TOTAL_PORTS; i++) {
    reportPINs[i] = false;
    portConfigInputs[i] = 0;
    previousPINs[i] = 0;
  }

  for (byte i = 0; i < TOTAL_PINS; i++) {
    if (IS_PIN_ANALOG(i)) {
      setPinModeCallback(i, PIN_MODE_ANALOG);
    } else if (IS_PIN_DIGITAL(i)) {
      setPinModeCallback(i, OUTPUT);
    }

  }

  analogInputsToReport = 0;

  isResetting = false;
}

void setup() {
  Firmata.setFirmwareVersion(FIRMATA_FIRMWARE_MAJOR_VERSION, FIRMATA_FIRMWARE_MINOR_VERSION);

  Firmata.attach(ANALOG_MESSAGE, analogWriteCallback);
  Firmata.attach(DIGITAL_MESSAGE, digitalWriteCallback);
  Firmata.attach(REPORT_ANALOG, reportAnalogCallback);
  Firmata.attach(REPORT_DIGITAL, reportDigitalCallback);
  Firmata.attach(SET_PIN_MODE, setPinModeCallback);
  Firmata.attach(SET_DIGITAL_PIN_VALUE, setPinValueCallback);
  Firmata.attach(START_SYSEX, sysexCallback);
  Firmata.attach(SYSTEM_RESET, systemResetCallback);

  Firmata.begin(57600);
  while (!Serial) {
    ;
  }

  systemResetCallback();
}

/*==============================================================================
 * LOOP
 *============================================================================*/

void loop() {
  byte pin, analogPin;

  checkDigitalInputs();

  while (Firmata.available())
    Firmata.processInput();

  currentMillis = millis();
  if (currentMillis - previousMillis > samplingInterval) {
    previousMillis += samplingInterval;
    for (pin = 0; pin < TOTAL_PINS; pin++) {
      if (IS_PIN_ANALOG(pin) && Firmata.getPinMode(pin) == PIN_MODE_ANALOG) {
        analogPin = PIN_TO_ANALOG(pin);
        if (analogInputsToReport & (1 << analogPin)) {
          Firmata.sendAnalog(analogPin, analogRead(analogPin));
        }
      }
    }
    if (queryIndex > -1) {
      for (byte i = 0; i < queryIndex + 1; i++) {
        readAndReportData(query[i].addr, query[i].reg, query[i].bytes, query[i].stopTX);
      }
    }
  }

#ifdef FIRMATA_SERIAL_FEATURE
  serialFeature.update();
#endif
}
