#include "esp_camera.h"
#include "esp_http_server.h"
#include <Arduino.h>
#include <WiFi.h>

// --- PRIVATE HOTSPOT CONFIGURATION ---
const char *ap_ssid = "ESP32-S3-Camera"; // 👈 Your new private Wi-Fi name
const char *ap_password = "formguest";   // 👈 The password to connect to it

// --- Smart Glasses Hardware Pins ---
#define LEFT_MOTOR_PIN 38
#define RIGHT_MOTOR_PIN 39
#define LEFT_BUZZER_PIN 40
#define RIGHT_BUZZER_PIN 41
#define MODE_SWITCH_PIN 42 // LOW = Vibro, HIGH = Speaker
#define DEPTH_TRIG_PIN 14  // Reserved for upcoming depth sensor
#define DEPTH_ECHO_PIN 21  // Reserved for upcoming depth sensor

unsigned long action_end_time = 0;

// --- ESP32-S3 Camera Pin Map ---
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

httpd_handle_t stream_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char *part_buf[64];

  res = httpd_resp_set_type(
      req, "multipart/x-mixed-replace;boundary=123456789000000000000987654321");
  if (res != ESP_OK)
    return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("❌ Camera capture failed");
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    if (res == ESP_OK) {
      size_t hlen =
          snprintf((char *)part_buf, 64,
                   "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                   _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(
          req, "\r\n--123456789000000000000987654321\r\n", 36);
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {.uri = "/",
                            .method = HTTP_GET,
                            .handler = stream_handler,
                            .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("🌐 Local Server Running!");
  }
}

static esp_err_t action_handler(httpd_req_t *req) {
  char buf[64];
  size_t buf_len = httpd_req_get_url_query_len(req) + 1;
  if (buf_len > 1) {
    if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
      char cmd[32];
      if (httpd_query_key_value(buf, "cmd", cmd, sizeof(cmd)) == ESP_OK) {
        bool use_speaker = digitalRead(MODE_SWITCH_PIN) == HIGH;

        // Turn everything off first
        digitalWrite(LEFT_MOTOR_PIN, LOW);
        digitalWrite(RIGHT_MOTOR_PIN, LOW);
        digitalWrite(LEFT_BUZZER_PIN, LOW);
        digitalWrite(RIGHT_BUZZER_PIN, LOW);

        if (strcmp(cmd, "LEFT") == 0) {
          digitalWrite(use_speaker ? LEFT_BUZZER_PIN : LEFT_MOTOR_PIN, HIGH);
          action_end_time = millis() + 400; // Pulse for 400ms
        } else if (strcmp(cmd, "RIGHT") == 0) {
          digitalWrite(use_speaker ? RIGHT_BUZZER_PIN : RIGHT_MOTOR_PIN, HIGH);
          action_end_time = millis() + 400; // Pulse for 400ms
        } else if (strcmp(cmd, "DANGER") == 0) {
          digitalWrite(use_speaker ? LEFT_BUZZER_PIN : LEFT_MOTOR_PIN, HIGH);
          digitalWrite(use_speaker ? RIGHT_BUZZER_PIN : RIGHT_MOTOR_PIN, HIGH);
          action_end_time = millis() + 800; // Pulse for 800ms
        } else if (strcmp(cmd, "STOP") == 0 || strcmp(cmd, "CLEAR") == 0) {
          // Just keep off
          action_end_time = 0;
        }

        const char *resp = "Action triggered";
        httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
        return ESP_OK;
      }
    }
  }
  const char *resp = "Invalid request";
  httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
  return ESP_FAIL;
}

void registerActionEndpoint() {
  httpd_uri_t action_uri = {.uri = "/action",
                            .method = HTTP_GET,
                            .handler = action_handler,
                            .user_ctx = NULL};
  httpd_register_uri_handler(stream_httpd, &action_uri);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize pins
  pinMode(LEFT_MOTOR_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_PIN, OUTPUT);
  pinMode(LEFT_BUZZER_PIN, OUTPUT);
  pinMode(RIGHT_BUZZER_PIN, OUTPUT);
  pinMode(MODE_SWITCH_PIN, INPUT_PULLUP);
  pinMode(DEPTH_TRIG_PIN, OUTPUT);
  pinMode(DEPTH_ECHO_PIN, INPUT);

  // Ensure outputs are off
  digitalWrite(LEFT_MOTOR_PIN, LOW);
  digitalWrite(RIGHT_MOTOR_PIN, LOW);
  digitalWrite(LEFT_BUZZER_PIN, LOW);
  digitalWrite(RIGHT_BUZZER_PIN, LOW);
  digitalWrite(DEPTH_TRIG_PIN, LOW);

  Serial.println("\n--- Starting Isolated AP Mode Camera Test ---");

  // Camera settings optimized for internal DRAM
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

  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.jpeg_quality =
      12;              // Optimized for better AI clarity while keeping speed
  config.fb_count = 2; // Increased buffer count for smoother streaming

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera Init Failed: 0x%x\n", err);
    return;
  }
  Serial.println("📷 Camera Hardware Ready!");

  // Start broadcasting our own Wi-Fi network
  WiFi.softAP(ap_ssid, ap_password);

  Serial.println("📶 Wi-Fi Hotspot Started!");
  Serial.print("👉 Connect your Mac to Wi-Fi network: ");
  Serial.println(ap_ssid);
  Serial.print("🔗 Then open this link in Chrome: http://");
  Serial.println(WiFi.softAPIP()); // This will show http://192.168.4.1

  startCameraServer();
  registerActionEndpoint();
}

void loop() {
  if (action_end_time > 0 && millis() > action_end_time) {
    digitalWrite(LEFT_MOTOR_PIN, LOW);
    digitalWrite(RIGHT_MOTOR_PIN, LOW);
    digitalWrite(LEFT_BUZZER_PIN, LOW);
    digitalWrite(RIGHT_BUZZER_PIN, LOW);
    action_end_time = 0;
  }
  delay(10);
}