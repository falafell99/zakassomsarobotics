# ZINEDINE ZIDANE  S. V. O.

- Where's Rahhfaell?
- Oh, he's in the shower. 
- AND WITHOUT ME?????

# Smart Glasses - Blind Assistance System

This project is a complete hardware and software stack for smart glasses designed to assist blind and visually impaired individuals with navigation. It combines an ESP32-S3 microcontroller equipped with a camera, haptic motors, and buzzers, with a powerful Python backend that uses Depth AI and Computer Vision to analyze the surroundings.

## How It Works

The system operates in three main layers:
1. **Hardware (ESP32-S3)**: The ESP32 captures live video and hosts it as an MJPEG stream over its own Wi-Fi Access Point (`ESP32-S3-Camera`). It also listens for HTTP requests to trigger physical feedback (buzzers or vibration motors) to guide the user left, right, or stop them from walking into obstacles.
2. **Local Vision (Python Client)**: The `test_webcam.py` script runs on a connected PC/laptop. It consumes the ESP32's camera stream and runs **Depth Anything V2** (monocular depth estimation) and **OpenCV HOG** (person detection) locally. It calculates the proximity of obstacles in real-time and sends directional commands back to the ESP32.
3. **AI Scene Understanding (FastAPI Backend)**: For complex scenes, the system occasionally captures a frame and sends it to the `backend/main.py` FastAPI server, which queries a Vision AI (via OpenRouter/Gemini) to generate a detailed scene description and determine safety.

## Hardware Setup
The ESP32-S3 uses the following pins for feedback:
- **Left Motor:** GPIO 38
- **Right Motor:** GPIO 39
- **Left Buzzer:** GPIO 40
- **Right Buzzer:** GPIO 41
- **Mode Switch:** GPIO 42 (Toggle between Vibro Mode and Speaker Mode)
- **Depth Sensor (Planned):** TRIG=GPIO 14, ECHO=GPIO 21

## How to Run

### 1. Flash the ESP32 Firmware
1. Open the `zakaz-glasses` directory in PlatformIO (VS Code).
2. Build and upload the code to your ESP32-S3.
3. The ESP32 will boot and broadcast a Wi-Fi network named `ESP32-S3-Camera` (Password: `formguest`).

### 2. Start the Backend Server (Optional, for advanced AI)
To enable the Gemini Vision AI and text-to-speech features, run the backend server:
```bash
uv sync
cd backend
python main.py
```
*(Ensure you have your `.env` file set up with `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY`)*

### 3. Run the Client Navigation System
1. Connect your PC's Wi-Fi to the `ESP32-S3-Camera` network.
2. Run the main client script:
```bash
cd tests_and_clients
python test_webcam.py
```
3. A window will open showing the live feed, depth map, and detected obstacles. As obstacles appear, the script will automatically send commands back to the ESP32, triggering the motors or buzzers on your glasses!

### Test Images

The repository includes a set of test images you can use to experiment with the vision pipeline without a live camera.

- **Location:** `tests_and_clients/test-imgs/`
- Example images:
  - `1.jpg` – a sample scene.
  - `2.jpg` – another sample.
  - `3.jpg` – another sample.

You can run the client with a static image using:

```bash
cd tests_and_clients
python test_webcam.py --image test-imgs/1.jpg
```

*(Make sure the script supports the `--image` argument; otherwise, modify `test_webcam.py` to load the image instead of the camera stream.)*

The included `test_image_base64.txt` contains the base64 representation of the sample image, which can be used for API testing.


**Controls inside the Client:**
- `D` = Toggle Depth Map Overlay
- `H` = Toggle HOG Person Detection Boxes
- `S` = Force an AI API call
- `T` = Toggle Text-to-Speech (TTS)
- `Q` / `ESC` = Quit
