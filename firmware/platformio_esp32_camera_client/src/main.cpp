#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "mbedtls/base64.h"

#include "camera_pins.h"
#include "wifi_config.h"

static const uint32_t FRAME_INTERVAL_MS = 900;
static const uint32_t HTTP_TIMEOUT_MS = 3000;
static const uint32_t WIFI_RETRY_INTERVAL_MS = 5000;
static const int JPEG_QUALITY = 24;
static const float FALLBACK_DISTANCE_M = 5.0f;

unsigned long lastFrameMs = 0;
unsigned long lastWifiAttemptMs = 0;
bool wifiWasConnected = false;

enum HapticMode {
  HAPTIC_CLEAR,
  HAPTIC_LEFT,
  HAPTIC_RIGHT,
  HAPTIC_STOP,
  HAPTIC_DANGER
};

HapticMode hapticMode = HAPTIC_CLEAR;
unsigned long hapticLastToggleMs = 0;
bool hapticPulseOn = false;

String base64Encode(const uint8_t* data, size_t len) {
  size_t encodedLen = 0;
  int rc = mbedtls_base64_encode(nullptr, 0, &encodedLen, data, len);
  if (rc != MBEDTLS_ERR_BASE64_BUFFER_TOO_SMALL || encodedLen == 0) {
    return "";
  }

  unsigned char* encoded = (unsigned char*)malloc(encodedLen + 1);
  if (!encoded) {
    Serial.println("base64 malloc failed");
    return "";
  }

  rc = mbedtls_base64_encode(encoded, encodedLen, &encodedLen, data, len);
  if (rc != 0) {
    Serial.printf("base64 encode failed: %d\n", rc);
    free(encoded);
    return "";
  }

  encoded[encodedLen] = '\0';
  String out = String((char*)encoded);
  free(encoded);
  return out;
}

String jsonStringValue(const String& json, const char* key) {
  String needle = "\"" + String(key) + "\"";
  int keyPos = json.indexOf(needle);
  if (keyPos < 0) {
    return "";
  }
  int colon = json.indexOf(':', keyPos + needle.length());
  int firstQuote = json.indexOf('"', colon + 1);
  int secondQuote = json.indexOf('"', firstQuote + 1);
  if (colon < 0 || firstQuote < 0 || secondQuote < 0) {
    return "";
  }
  return json.substring(firstQuote + 1, secondQuote);
}

void writeBuzzer(int pin, bool on) {
  digitalWrite(pin, on == (BUZZER_ACTIVE_HIGH == 1) ? HIGH : LOW);
}

void setBuzzers(bool leftOn, bool rightOn) {
  writeBuzzer(BUZZER_LEFT_GPIO, leftOn);
  writeBuzzer(BUZZER_RIGHT_GPIO, rightOn);
}

void initHardwareLayer() {
  pinMode(ULTRASONIC_TRIG_GPIO, OUTPUT);
  pinMode(ULTRASONIC_ECHO_GPIO, INPUT);
  digitalWrite(ULTRASONIC_TRIG_GPIO, LOW);

  pinMode(BUZZER_LEFT_GPIO, OUTPUT);
  pinMode(BUZZER_RIGHT_GPIO, OUTPUT);
  setBuzzers(false, false);
}

float readUltrasonicOnceM() {
  digitalWrite(ULTRASONIC_TRIG_GPIO, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_GPIO, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_GPIO, LOW);

  unsigned long timeoutUs = (unsigned long)((ULTRASONIC_MAX_DISTANCE_M * 2.0f / 343.0f) * 1000000.0f);
  unsigned long durationUs = pulseIn(ULTRASONIC_ECHO_GPIO, HIGH, timeoutUs);
  if (durationUs == 0) {
    return FALLBACK_DISTANCE_M;
  }

  float meters = (durationUs * 0.000343f) / 2.0f;
  if (meters <= 0.02f || meters > ULTRASONIC_MAX_DISTANCE_M) {
    return FALLBACK_DISTANCE_M;
  }
  return meters;
}

float readUltrasonicM() {
  float a = readUltrasonicOnceM();
  delay(25);
  float b = readUltrasonicOnceM();
  delay(25);
  float c = readUltrasonicOnceM();

  if (a > b) { float t = a; a = b; b = t; }
  if (b > c) { float t = b; b = c; c = t; }
  if (a > b) { float t = a; a = b; b = t; }
  return b;
}

HapticMode modeFromCommand(const String& command) {
  if (command == "DANGER") {
    return HAPTIC_DANGER;
  }
  if (command == "STOP") {
    return HAPTIC_STOP;
  }
  // Backend LEFT/RIGHT are movement directions. The haptic motor should mark
  // the obstacle side, so movement LEFT means right-side obstacle and vice versa.
  if (command == "LEFT") {
    return HAPTIC_RIGHT;
  }
  if (command == "RIGHT") {
    return HAPTIC_LEFT;
  }
  return HAPTIC_CLEAR;
}

void setHapticMode(HapticMode nextMode) {
  if (hapticMode == nextMode) {
    return;
  }
  hapticMode = nextMode;
  hapticLastToggleMs = 0;
  hapticPulseOn = false;
  setBuzzers(false, false);
}

void updateHaptics() {
  unsigned long now = millis();
  uint32_t periodMs = 0;
  uint32_t activeMs = 0;

  switch (hapticMode) {
    case HAPTIC_DANGER:
      periodMs = 180;
      activeMs = 110;
      break;
    case HAPTIC_STOP:
      periodMs = 420;
      activeMs = 160;
      break;
    case HAPTIC_LEFT:
    case HAPTIC_RIGHT:
      periodMs = 520;
      activeMs = 150;
      break;
    case HAPTIC_CLEAR:
    default:
      setBuzzers(false, false);
      return;
  }

  if (hapticLastToggleMs == 0 || now - hapticLastToggleMs >= (hapticPulseOn ? activeMs : periodMs - activeMs)) {
    hapticLastToggleMs = now;
    hapticPulseOn = !hapticPulseOn;
  }

  bool left = false;
  bool right = false;
  if (hapticPulseOn) {
    left = hapticMode == HAPTIC_LEFT || hapticMode == HAPTIC_STOP || hapticMode == HAPTIC_DANGER;
    right = hapticMode == HAPTIC_RIGHT || hapticMode == HAPTIC_STOP || hapticMode == HAPTIC_DANGER;
  }
  setBuzzers(left, right);
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, config.frame_size);
    sensor->set_quality(sensor, config.jpeg_quality);
  }

  return true;
}

void beginWiFiConnect() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.disconnect(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWifiAttemptMs = millis();
  wifiWasConnected = false;
  Serial.println("WiFi connect started");
}

bool ensureWiFiConnected() {
  wl_status_t status = WiFi.status();
  if (status == WL_CONNECTED) {
    if (!wifiWasConnected) {
      wifiWasConnected = true;
      Serial.print("WiFi connected, ESP IP: ");
      Serial.println(WiFi.localIP());
    }
    return true;
  }

  if (wifiWasConnected) {
    Serial.println("WiFi disconnected");
    wifiWasConnected = false;
  }

  if (lastWifiAttemptMs == 0 || millis() - lastWifiAttemptMs >= WIFI_RETRY_INTERVAL_MS) {
    beginWiFiConnect();
  }
  return false;
}

void sendFrame() {
  if (!ensureWiFiConnected()) {
    Serial.println("Skip frame: WiFi not connected");
    return;
  }

  float distanceM = readUltrasonicM();
  Serial.printf("Ultrasonic %.2fm\n", distanceM);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.printf("Captured %u bytes\n", (unsigned)fb->len);
  String imageBase64 = base64Encode(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (imageBase64.length() == 0) {
    Serial.println("Base64 failed");
    return;
  }

  String body;
  body.reserve(imageBase64.length() + 120);
  body = "{\"distance\":";
  body += String(distanceM, 2);
  body += ",\"speak\":true,\"image\":\"";
  body += imageBase64;
  body += "\"}";

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  String response = http.getString();
  String error = http.errorToString(code);
  http.end();

  Serial.printf("HTTP %d\n", code);
  if (code == 200) {
    String command = jsonStringValue(response, "command");
    if (command.length() == 0) {
      command = "STOP";
    }
    setHapticMode(modeFromCommand(command));

    Serial.print("command=");
    Serial.print(command);
    Serial.print(" obstacle=");
    Serial.print(jsonStringValue(response, "obstacle"));
    Serial.print(" message=");
    Serial.println(jsonStringValue(response, "message"));
  } else {
    Serial.print("HTTP error: ");
    Serial.println(error);
    Serial.println(response);
    setHapticMode(HAPTIC_STOP);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("rafachmo ESP32 camera client");
  Serial.print("Camera model: ");
  Serial.println(CAMERA_MODEL_NAME);
  Serial.print("Backend: ");
  Serial.println(BACKEND_URL);
  Serial.printf("HC-SR04 trig=%d echo=%d max=%.1fm\n",
                ULTRASONIC_TRIG_GPIO,
                ULTRASONIC_ECHO_GPIO,
                ULTRASONIC_MAX_DISTANCE_M);
  Serial.printf("Buzzers left=%d right=%d active_%s\n",
                BUZZER_LEFT_GPIO,
                BUZZER_RIGHT_GPIO,
                BUZZER_ACTIVE_HIGH == 1 ? "HIGH" : "LOW");

  initHardwareLayer();

  if (!initCamera()) {
    Serial.println("Stop: camera init failed. Check selected env/pin map.");
    return;
  }
  Serial.println("Camera ready");

  beginWiFiConnect();
}

void handleSerialTestCommands() {
  if (!Serial.available()) {
    return;
  }

  char c = Serial.read();
  if (c == 'l' || c == 'L') {
    Serial.println("TEST left buzzer");
    setHapticMode(HAPTIC_LEFT);
  } else if (c == 'r' || c == 'R') {
    Serial.println("TEST right buzzer");
    setHapticMode(HAPTIC_RIGHT);
  } else if (c == 's' || c == 'S') {
    Serial.println("TEST both buzzers");
    setHapticMode(HAPTIC_STOP);
  } else if (c == 'c' || c == 'C') {
    Serial.println("TEST clear buzzers");
    setHapticMode(HAPTIC_CLEAR);
  } else if (c == 'u' || c == 'U') {
    Serial.printf("TEST ultrasonic %.2fm\n", readUltrasonicM());
  }
}

void loop() {
  ensureWiFiConnected();
  handleSerialTestCommands();
  if (millis() - lastFrameMs >= FRAME_INTERVAL_MS) {
    lastFrameMs = millis();
    sendFrame();
  }
  updateHaptics();
  delay(10);
}
