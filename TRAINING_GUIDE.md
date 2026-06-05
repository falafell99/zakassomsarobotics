# Rafachmo Model Training Guide

This guide is for training a home/office YOLO model for the blind-assistance MVP.

## Goal

The model should work well in presentation environments: rooms, hallways, desks,
office corners, stairs, doors, walls, and small floor obstacles. It does not need
perfect real-world generalization yet. It needs stable, believable detections for
the demo.

## Current Pipeline

Laptop/backend runs the AI. ESP32 only sends camera frames to:

```text
POST /analyze
```

Detection is handled by YOLO in `detector.py`. Navigation logic is handled by
`navigator.py`.

Current class list is in:

```text
training/home_office_dataset.yaml
```

Training script:

```text
scripts/train_home_office.py
```

## How The System Works

The system is split into two parts:

```text
ESP32 camera client
  -> captures JPEG frame
  -> sends base64 image to laptop backend

Laptop backend
  -> decodes image
  -> runs YOLO object detection
  -> estimates rough distance/depth
  -> decides navigation command
  -> returns text/action response
```

ESP32 does not run AI. ESP32 is only a camera/network client. This is important
because ESP32 is too weak for YOLO-level object detection in this project. The
laptop does the heavy model inference.

Request from ESP32 to backend:

```json
{
  "distance": 5.0,
  "speak": true,
  "image": "base64-jpeg"
}
```

Backend response:

```json
{
  "command": "STOP",
  "message": "Chair 1.2 meters ahead, stop",
  "obstacle": "chair",
  "distance": 1.2,
  "detections": [
    {
      "label": "chair",
      "confidence": 0.83,
      "distance": 1.2,
      "position": "center",
      "bbox": [120, 180, 500, 700],
      "action": "STOP"
    }
  ]
}
```

## Main Libraries

Python backend:

```text
fastapi              HTTP API: /health, /analyze, /speak
uvicorn              runs the FastAPI server
ultralytics          YOLO model loading, training, and inference
opencv-python        camera/debug preview and BGR image handling
pillow               base64 JPEG decode into image objects
numpy                image arrays
pydantic             request/response schemas
python-dotenv        loads .env config
elevenlabs           optional text-to-speech output
requests             debug client HTTP calls
```

ESP32 firmware:

```text
WiFi.h               connects ESP32 to Wi-Fi
HTTPClient.h         sends POST /analyze requests
esp_camera.h         captures JPEG frames from ESP32 camera
mbedtls/base64.h     base64 encodes JPEG bytes for JSON
```

Training:

```text
ultralytics.YOLO     fine-tunes YOLO model on home/office dataset
YOLO format labels   class_id x_center y_center width height
```

## Important Files

Backend:

```text
main.py              FastAPI entrypoint, /analyze pipeline
detector.py          loads YOLO and converts model boxes to Detection objects
navigator.py         distance estimation, risk scoring, LEFT/RIGHT/STOP logic
schemas.py           Pydantic request/response models
image_utils.py       base64 image decode and bbox normalization
config.py            model path, confidence, image size, tuning constants
speaker.py           optional TTS queue
scripts/speak_output.py             quick laptop TTS test
webcam_debug.py      Mac/laptop camera debug UI
```

Training:

```text
training/home_office_dataset.yaml    class list and dataset paths
scripts/train_home_office.py         YOLO fine-tuning script
scripts/collect_frames.py            laptop camera frame collector
scripts/export_model.py              model export helper
```

ESP32:

```text
firmware/platformio_esp32_camera_client
```

This PlatformIO project captures ESP32 camera frames and sends them to the
laptop backend. The current backend URL is configured in:

```text
firmware/platformio_esp32_camera_client/src/wifi_config.h
```

## Detection And Decision Flow

1. ESP32 captures a JPEG frame.
2. ESP32 base64-encodes the JPEG.
3. ESP32 sends JSON to `POST /analyze`.
4. `main.py` decodes the image with `image_utils.py`.
5. Image is converted to OpenCV BGR format.
6. `detector.py` runs YOLO using `ultralytics`.
7. YOLO outputs boxes, labels, and confidence.
8. Boxes are normalized to `0..1000` coordinates.
9. `navigator.py` estimates distance from bbox size and bottom position.
10. `navigator.py` scores risk by class, distance, and center corridor overlap.
11. `navigator.py` chooses command: `CLEAR`, `STOP`, `LEFT`, `RIGHT`, `DANGER`.
12. Backend returns command, message, obstacle, distance, and detections.

The model only detects objects. The model does not decide where to go. Movement
logic is rule-based in `navigator.py`.

## How Distance Works

Distance is not true stereo depth. It is an approximation based on:

```text
object class
bbox height
bbox bottom position in image
optional ultrasonic sensor distance
DISTANCE_SCALE from config/.env
```

Example: a large chair box near the bottom of the image is probably close. A
small chair box higher in the image is probably farther away.

This is why calibration is important. For better distance, collect examples at
known distances: 0.5m, 1m, 1.5m, 2m, 3m.

## Model Details

Current base model:

```text
yolo11n.pt
```

This is a small YOLO model from Ultralytics. It is fast enough for laptop MVP
testing. Fine-tuning creates a new `.pt` file, usually:

```text
runs/detect/home_office_yolo/weights/best.pt
```

The backend loads the model path from `config.py` / `.env`:

```bash
YOLO_MODEL=runs/detect/home_office_yolo/weights/best.pt
```

YOLO predicts:

```text
class label
confidence
bounding box
```

The project then adds:

```text
estimated distance
left/center/right position
risk score
navigation action
spoken/debug message
```

## Classes To Prioritize

High priority obstacle classes:

```text
person
wall
door
doorway
stairs
staircase
step
railing
handrail
chair
office_chair
desk
table
coffee_table
side_table
sofa
bed
cabinet
wardrobe
bookshelf
box
storage_box
backpack
suitcase
rug
cable
```

Office context classes:

```text
monitor
laptop
keyboard
mouse
phone
printer
whiteboard
trash_can
plant
paper
notebook
charger
```

Small objects like `keyboard`, `mouse`, `paper`, `notebook`, `phone`, and `cup`
are mostly context. Do not over-optimize the navigation command around them
unless they are on the floor or directly in the walking path.

## Dataset Collection

Collect real frames from the same type of camera used in the demo. If ESP32
camera is used in the final demo, collect some images from ESP32 too, because
ESP32 camera quality, lens, exposure, and blur are different from Mac webcam.

Recommended dataset size for MVP:

```text
minimum: 500-800 labeled images
good:    1500-3000 labeled images
demo:    300-600 very targeted images can already help
```

Collect these situations:

- hallway with wall ahead
- wall on left/right side
- closed door, open door, doorway
- stairs from top, bottom, side, and straight ahead
- one or two visible steps
- handrail/railing on side
- office chair in the walking path
- desk/table edge in front
- boxes and bags on the floor
- rug/cable/charger on the floor
- monitor/laptop/keyboard/mouse on desk
- shelves/cabinets/wardrobes
- strong light, low light, backlight
- close, medium, and far object distances
- blurry frames while walking

For a project presentation, also collect images in the exact room where the demo
will happen.

## Annotation Rules

Use CVAT, Label Studio, Roboflow, or LabelImg. Export in YOLO format.

Expected folder structure:

```text
datasets/home_office/images/train/*.jpg
datasets/home_office/labels/train/*.txt
datasets/home_office/images/val/*.jpg
datasets/home_office/labels/val/*.txt
```

Annotation rules:

- Box only visible object area.
- Label walls as `wall` only when a wall/surface blocks or bounds the route.
- Label open passages as `doorway`.
- Label closed/open physical doors as `door`.
- Label a full staircase as `stairs` or `staircase`.
- Label individual visible steps as `step`.
- Label side rails as `railing` or `handrail`.
- Label floor cables as `cable`, not generic object.
- Label boxes on floor as `box` or `storage_box`.
- Do not label every tiny desk item if it does not matter for navigation.
- Include negative images with no obstacles or no important objects.

Keep class names exactly matching `training/home_office_dataset.yaml`.

## Train

From project root:

```bash
cd /Users/ax1le/порно/rafachmo
python3 scripts/train_home_office.py \
  --data training/home_office_dataset.yaml \
  --base yolo11n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 8
```

On Apple Silicon, try:

```bash
python3 scripts/train_home_office.py \
  --data training/home_office_dataset.yaml \
  --base yolo11n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 8 \
  --device mps
```

If memory is a problem, lower batch:

```bash
--batch 4
```

Best weights will be here:

```text
runs/detect/home_office_yolo/weights/best.pt
```

## Use Trained Weights

Create or update `.env`:

```bash
YOLO_MODEL=runs/detect/home_office_yolo/weights/best.pt
DETECTION_CONFIDENCE=0.30
YOLO_IMGSZ=640
```

Then restart backend:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Check:

```bash
curl http://127.0.0.1:8000/health
```

## Validate

Validation should not only check mAP. For this project, check behavior:

- Does it detect stairs early enough?
- Does it detect wall/doorway reliably?
- Does it avoid treating keyboard/mouse as walking obstacles?
- Does it detect chair/desk/box in the path?
- Are boxes stable between frames?
- Are distances believable enough for demo?
- Does command output make sense: `STOP`, `LEFT`, `RIGHT`, `CLEAR`, `DANGER`?

Run laptop camera debug:

```bash
python3 webcam_debug.py
```

Run ESP camera client and watch Serial Monitor:

```text
Captured ... bytes
HTTP 200
command=...
obstacle=...
message=...
```

## What To Improve In Code

1. Save ESP frames for training

Add a backend option to save incoming ESP images plus response metadata. This
will create the exact dataset the model needs.

Suggested folder:

```text
datasets/esp_captures/raw
```

2. Add debug image endpoint

Create an endpoint that returns the latest analyzed frame with boxes drawn on it.
This makes ESP camera testing much easier than only reading Serial.

3. Better distance calibration

Current distance is approximate. Improve with a calibration table:

```text
class + bbox height + bottom position -> measured distance
```

Collect 0.5m, 1m, 1.5m, 2m, 3m examples for chair, wall, door, person, stairs.

4. Separate context from obstacle

Keep improving `navigator.py` so desk objects are scene context, while floor
objects and center-path objects affect movement.

5. Smooth detections by tracking

Add simple object tracking across frames using IoU. This reduces flicker and
prevents commands from changing too fast.

6. Add scenario mode for demo

For presentation, add a `SCENE_PROFILE=demo_home_office` mode with slightly more
stable, conservative behavior:

```text
stairs -> STOP
wall close -> STOP/turn
chair/desk/box center -> avoid
small desk items -> ignore
```

7. Export smaller model for speed

After training:

```bash
python3 scripts/export_model.py --weights runs/detect/home_office_yolo/weights/best.pt
```

Use `yolo11n` or `yolo11s` depending on laptop speed. For MVP, speed and stable
demo behavior matter more than maximum accuracy.

## Team Workflow

Recommended split:

```text
person A: collect ESP/laptop frames in target rooms
person B: annotate dataset
person C: train model and compare weights
person D: test commands and tune navigator.py
```

Every trained model should be tested with the same 20-30 demo scenes and logged:

```text
scene name
expected command
actual command
bad/missing detections
distance quality
notes
```

Do not merge a model only because mAP is higher. Merge it if demo behavior is
better and navigation commands are more stable.
