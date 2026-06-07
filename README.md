# Smart Glasses for Visally-Impaired people.

MVP backend for blind-assistance smart glasses.

Architecture:

```text
ESP32-CAM / sensor client
  -> sends JPEG frame + ultrasonic distance
  -> POST /analyze

Laptop backend
  -> object detection
  -> distance/depth estimation
  -> navigation decision
  -> text instruction
  -> TTS / speaker output

Speaker output
  -> laptop speakers, Bluetooth speaker, or future speaker endpoint
```

The ESP32 is intentionally a thin client. It should capture frames, read simple
sensors, and send data to the laptop. AI runs on the laptop.

## Current ESP32 Setup

The connected ESP32 currently appears on macOS as:

```bash
/dev/cu.usbmodem1101
```

The ESP32 only captures camera frames and sends them to the laptop backend. The
laptop speaks navigation output aloud using `speaker.py`.

## Run

```bash
cd /Users/ax1le/Downloads/zakassomsarobotics-lastbranch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend uses local YOLO (`ultralytics`) for object detection. On first run it
downloads `yolo11n.pt`.

Distance is estimated from bbox height, object class, and how low the object is
in the frame. With ESP32 ultrasonic distance, the backend fuses sensor distance
with vision distance for center-path obstacles.

Navigation uses:

- walking-path corridor overlap
- left / center / right risk scores
- object class relevance
- bbox size and confidence
- temporal smoothing for distance
- command stability frames to reduce flicker

Office-aware labels include: `desk`, `office chair`, `monitor`, `laptop`,
`keyboard`, `mouse`, `phone`, `book`, `cup`, `bottle`, `printer`,
`whiteboard`, `filing cabinet`, `trash can`, and `plant`. Small office items are
kept as context unless they are close or inside the walking corridor.

Home-aware labels include: `wall`, `door`, `doorway`, `stairs`, `sofa`, `bed`,
`cabinet`, `wardrobe`, `drawer`, `bookshelf`, `nightstand`, `coffee table`,
`side table`, `floor lamp`, `rug`, `storage box`, `laundry basket`, `backpack`,
and `suitcase`.

Walls are hard for generic YOLO. For the MVP, the backend adds a center-path
`wall` fallback when the ultrasonic sensor reports a close obstacle but vision
does not find a better object.

Structural labels include `wall`, `door`, `doorway`, `stairs`, `step`,
`railing`, and `handrail`. Stairs/steps in the walking corridor produce a
specific stop instruction. Railings/handrails are treated as context unless they
block the center path.

Small non-critical items such as `keyboard`, `mouse`, `phone`, `book`, `paper`,
`notebook`, `cable`, and `charger` stay low priority unless they are close or
inside the walking corridor.

Calibration knobs in `.env`:

```bash
CAMERA_VERTICAL_FOV_DEG=55
DISTANCE_SCALE=1.0
DISTANCE_SMOOTHING_ALPHA=0.45
```

If distance reads too far, lower `DISTANCE_SCALE` such as `0.8`. If it reads too
close, raise it such as `1.2`.

## ESP32 Request

```json
{
  "distance": 1.2,
  "image": "base64-jpeg"
}
```

## Backend Response

```json
{
  "command": "RIGHT",
  "message": "Chair 1.2 meters ahead, go right",
  "obstacle": "chair",
  "distance": 1.2,
  "detections": [
    {
      "label": "chair",
      "confidence": 0.83,
      "distance": 1.2,
      "position": "center",
      "bbox": [120, 180, 500, 700],
      "action": "RIGHT",
      "source": "yolo"
    }
  ],
  "speak": true,
  "reason": "left/center risk is higher; right side is safer",
  "zones": {"left": 3.2, "center": 0.6, "right": 0.1}
}
```

`bbox` is normalized to `0..1000`: `[x1, y1, x2, y2]`.

## Files

- `main.py` - FastAPI entrypoint
- `detector.py` - local object detection
- `navigator.py` - command/action decision logic
- `speaker.py` - TTS and speaker output
- `schemas.py` - request/response models
- `image_utils.py` - base64 image helpers
- `webcam_debug.py` - laptop-camera test client
- `scripts/speak_output.py` - speak one command/message aloud
- `training/` - home/office fine-tuning dataset config and notes
- `scripts/collect_frames.py` - collect annotation images
- `scripts/train_home_office.py` - fine-tune YOLO on annotated home/office data
- `firmware/platformio_esp32_camera_client/` - ESP32 camera client

## Webcam Debug Controls

- `Q` / `Esc` - quit
- `+` - increase simulated ultrasonic distance
- `-` - decrease simulated ultrasonic distance
- `0` - reset simulated distance to `5.0m`

## ESP32 Camera Test

Laptop webcam debug uses `webcam_debug.py`. ESP32 camera testing uses a separate
PlatformIO project:

```bash
firmware/platformio_esp32_camera_client
```

Edit `src/wifi_config.h` before flashing:

```cpp
static const char* WIFI_SSID = "YOUR_WIFI";
static const char* WIFI_PASSWORD = "YOUR_PASSWORD";
static const char* BACKEND_URL = "http://10.9.99.2:8000/analyze";
```

If `src/wifi_config.h` does not exist yet:

```bash
cp firmware/platformio_esp32_camera_client/src/wifi_config.example.h \
   firmware/platformio_esp32_camera_client/src/wifi_config.h
```

`wifi_config.h` is intentionally ignored by git because it contains local Wi-Fi
credentials.

Available PlatformIO environments:

- `freenove_esp32_s3_wroom`
- `seeed_xiao_esp32s3_sense`

The ESP32 sends camera JPEG frames to the laptop backend and prints the backend
`command`, `obstacle`, and `message` in Serial Monitor.
The ESP32 request sets `"speak": true`, so non-clear navigation commands are
spoken aloud on the laptop backend.

## ESP32 HC-SR04 and Buzzers

The ESP32 firmware now reads HC-SR04 before every camera upload and sends the
real sensor distance as `distance` in `POST /analyze`. The backend uses this as
the primary close-range distance signal and fuses it with YOLO bbox distance.

Configure hardware pins in:

```bash
firmware/platformio_esp32_camera_client/src/wifi_config.h
```

Start from the example:

```cpp
#define ULTRASONIC_TRIG_GPIO 1
#define ULTRASONIC_ECHO_GPIO 2
#define BUZZER_LEFT_GPIO 41
#define BUZZER_RIGHT_GPIO 42
#define BUZZER_ACTIVE_HIGH 1
#define ULTRASONIC_MAX_DISTANCE_M 4.5f
```

Do not use camera pins from `camera_pins.h`. On the Freenove ESP32-S3 camera
setup, GPIO `6` and `7` are camera pins, so they cannot be used for buzzers while
the camera is enabled.

HC-SR04 echo is commonly 5V. ESP32 GPIO is 3.3V, so use a voltage divider or
level shifter on `ECHO`.

Buzzer patterns:

- `CLEAR` - both off
- `LEFT` - left buzzer pulses
- `RIGHT` - right buzzer pulses
- `STOP` - both buzzers slow pulse
- `DANGER` - both buzzers fast pulse

## Speak Output Test

To test laptop TTS without camera/backend:

```bash
python3 scripts/speak_output.py "Chair one meter ahead, go left"
```

If `ELEVENLABS_API_KEY` is configured, `speaker.py` uses ElevenLabs. Otherwise it
falls back to macOS `say`.

Live camera and ESP32 requests set `"speak": true`, so the backend speaks
non-clear navigation commands. Repeated messages are throttled by:

```bash
TTS_REPEAT_SECONDS=5.0
TTS_DANGER_REPEAT_SECONDS=1.5
```

## GitHub CI/CD

This project includes GitHub Actions in `.github/workflows/ci.yml`.

On every push and pull request it:

- installs Python dependencies
- compiles all Python files
- builds ESP32 firmware for both PlatformIO environments
- uploads built firmware binaries as GitHub Actions artifacts

First-time GitHub publish:

```bash
cd /Users/ax1le/Downloads/zakassomsarobotics-lastbranch
git init
git branch -M lastbranch
git add .
git commit -m "Initial smart glasses MVP"
gh auth login
gh repo create zakassomsarobotics --private --source=. --remote=origin --push
```

If the GitHub repo already exists:

```bash
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin lastbranch
```
