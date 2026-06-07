#pragma once

// Copy this file to wifi_config.h before flashing.
// Do not commit wifi_config.h because it contains local Wi-Fi credentials.
static const char* WIFI_SSID = "YOUR_WIFI_NAME";
static const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Laptop/backend IP must be reachable from the ESP32 network.
static const char* BACKEND_URL = "http://YOUR_LAPTOP_IP:8000/analyze";

// Hardware pins for the current glasses wiring.
// Note: on the current Freenove camera pin map, GPIO 6 is also VSYNC. Keep this
// wiring only if your hardware has already been verified with this pin layout.
#define ULTRASONIC_TRIG_GPIO 21
#define ULTRASONIC_ECHO_GPIO 48
#define BUZZER_LEFT_GPIO 1
#define BUZZER_RIGHT_GPIO 2

// Most simple active buzzers are active HIGH. Set to 0 if your module is active LOW.
#define BUZZER_ACTIVE_HIGH 1

// HC-SR04 echo is often 5V. Use a voltage divider/level shifter into ESP32 GPIO.
#define ULTRASONIC_MAX_DISTANCE_M 4.5f
