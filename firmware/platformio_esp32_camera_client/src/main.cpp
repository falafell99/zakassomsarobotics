#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "mbedtls/base64.h"

#include "camera_pins.h"
#include "wifi_config.h"

static const uint32_t FRAME_INTERVAL_MS = 1200;
static const uint32_t HTTP_TIMEOUT_MS = 9000;
static const int JPEG_QUALITY = 18;

unsigned long lastFrameMs = 0;

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
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 16;
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

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, ESP IP: ");
  Serial.println(WiFi.localIP());
}

void sendFrame() {
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
  body.reserve(imageBase64.length() + 80);
  body = "{\"distance\":5.0,\"speak\":true,\"image\":\"";
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
    Serial.print("command=");
    Serial.print(jsonStringValue(response, "command"));
    Serial.print(" obstacle=");
    Serial.print(jsonStringValue(response, "obstacle"));
    Serial.print(" message=");
    Serial.println(jsonStringValue(response, "message"));
  } else {
    Serial.print("HTTP error: ");
    Serial.println(error);
    Serial.println(response);
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

  if (!initCamera()) {
    Serial.println("Stop: camera init failed. Check selected env/pin map.");
    return;
  }
  Serial.println("Camera ready");

  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (millis() - lastFrameMs >= FRAME_INTERVAL_MS) {
    lastFrameMs = millis();
    sendFrame();
  }
  delay(10);
}
