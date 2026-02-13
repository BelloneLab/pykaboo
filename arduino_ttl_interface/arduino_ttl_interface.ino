/*
 * Arduino TTL Interface for Basler Camera Application
 *
 * This sketch allows the Python application to read digital inputs
 * for gate and sync TTL signals.
 *
 * Protocol:
 * - CONFIG,<pin>,INPUT - Configure pin as input
 * - READ,<pin> - Read digital pin state (returns 0 or 1)
 *
 * Upload this to your Arduino before using the application.
 */

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // Parse command
    int firstComma = command.indexOf(',');
    int secondComma = command.indexOf(',', firstComma + 1);

    if (firstComma > 0) {
      String cmd = command.substring(0, firstComma);
      String pin_str = command.substring(firstComma + 1, secondComma > 0 ? secondComma : command.length());
      int pin = pin_str.toInt();

      if (cmd == "CONFIG") {
        String mode = command.substring(secondComma + 1);
        if (mode == "INPUT") {
          pinMode(pin, INPUT);
          Serial.println("OK");
        } else if (mode == "INPUT_PULLUP") {
          pinMode(pin, INPUT_PULLUP);
          Serial.println("OK");
        }
      } else if (cmd == "READ") {
        int value = digitalRead(pin);
        Serial.println(value);
      }
    }
  }
}

