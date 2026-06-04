"""
test_direct.py — Smart Blind-Assistance Glasses — Simple Direct API Test
=========================================================================
Minimal fallback script: captures one frame from the webcam, sends it
directly to the /analyze endpoint, and prints the result.

No threading, no HUD, no keyboard interaction.
Use this to quickly verify the AI pipeline is working end-to-end.

Usage:
    python test_direct.py
    python test_direct.py --distance 0.8
    python test_direct.py --camera 1
"""

import cv2
import base64
import requests
import json
import argparse
import sys
import os
import time

# ─── Configuration ────────────────────────────────────────────────────────────

SERVER_URL   = "http://localhost:8000"
CAMERA_INDEX = 0
JPEG_QUALITY = 85
SEND_WIDTH   = 1280
SEND_HEIGHT  = 720
DISTANCE     = 1.0    # default fake distance in meters
DEBUG_SAVE   = True   # save debug_frame.jpg so you can see what was sent

# ─── CLI arguments ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Direct single-frame API test")
parser.add_argument("--distance", type=float, default=DISTANCE,
                    help=f"Fake distance in meters (default: {DISTANCE})")
parser.add_argument("--camera",   type=int,   default=CAMERA_INDEX,
                    help=f"Camera index (default: {CAMERA_INDEX})")
parser.add_argument("--url",      type=str,   default=SERVER_URL,
                    help=f"Server URL (default: {SERVER_URL})")
args = parser.parse_args()

SERVER_URL   = args.url
CAMERA_INDEX = args.camera
DISTANCE     = args.distance

# ─── Step 1: Check server health ─────────────────────────────────────────────

print(f"\n{'='*55}")
print(" test_direct.py — Single-frame API test")
print(f"{'='*55}")
print(f"  Server:   {SERVER_URL}")
print(f"  Distance: {DISTANCE:.2f}m")
print(f"  Camera:   {CAMERA_INDEX}")
print(f"  Quality:  {JPEG_QUALITY}")
print(f"  Size:     {SEND_WIDTH}×{SEND_HEIGHT}")
print(f"{'='*55}\n")

print("[1] Checking server health…")
try:
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    if r.status_code == 200:
        print(f"    ✅ Server OK  ({r.json()})")
    else:
        print(f"    ⚠️  HTTP {r.status_code}: {r.text}")
except Exception as e:
    print(f"    ❌ Server NOT REACHABLE: {e}")
    print("    → Start the server with:  uvicorn main:app --reload --port 8000")
    sys.exit(1)

# ─── Step 2: Capture one frame ────────────────────────────────────────────────

print(f"\n[2] Opening camera {CAMERA_INDEX}…")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  SEND_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SEND_HEIGHT)

if not cap.isOpened():
    print(f"    ❌ Could not open camera index {CAMERA_INDEX}")
    sys.exit(1)

# Warm up: discard first few frames (cameras often start dark)
for _ in range(5):
    cap.read()
    time.sleep(0.05)

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("    ❌ Failed to capture frame from camera")
    sys.exit(1)

actual_h, actual_w = frame.shape[:2]
print(f"    ✅ Captured frame  {actual_w}×{actual_h}")

# ─── Step 3: Encode frame ────────────────────────────────────────────────────

print(f"\n[3] Encoding frame (JPEG quality={JPEG_QUALITY}, size={SEND_WIDTH}×{SEND_HEIGHT})…")
resized = cv2.resize(frame, (SEND_WIDTH, SEND_HEIGHT), interpolation=cv2.INTER_AREA)

encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
success, buffer = cv2.imencode(".jpg", resized, encode_params)
if not success:
    print("    ❌ cv2.imencode failed")
    sys.exit(1)

b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
size_kb = len(b64) * 3 / 4 / 1024
print(f"    ✅ Encoded. Base64 length: {len(b64)} chars  (~{size_kb:.1f} KB)")

if DEBUG_SAVE:
    cv2.imwrite("debug_frame.jpg", resized)
    print(f"    📷 Saved debug_frame.jpg — this is exactly what the AI will see")

# ─── Step 4: Send to API ─────────────────────────────────────────────────────

print(f"\n[4] POST {SERVER_URL}/analyze  distance={DISTANCE:.2f}m …")
payload = {"distance": DISTANCE, "image": b64}

t0 = time.time()
try:
    resp = requests.post(f"{SERVER_URL}/analyze", json=payload, timeout=60)
    elapsed = time.time() - t0
    print(f"    Response time: {elapsed:.2f}s")
    print(f"    HTTP status:   {resp.status_code}")

    print(f"\n[5] Raw response body:")
    print(f"    {resp.text}\n")

    if resp.status_code == 200:
        data = resp.json()
        print("=" * 55)
        print(f"  COMMAND:   {data.get('command',  '???')}")
        print(f"  MESSAGE:   {data.get('message',  '???')}")
        print(f"  OBSTACLE:  {data.get('obstacle', '???')}")
        print("=" * 55)
    else:
        print(f"    ❌ Server returned HTTP {resp.status_code}")

except requests.exceptions.Timeout:
    print("    ❌ Request timed out (> 60s) — model may be overloaded")
    sys.exit(1)
except Exception as e:
    print(f"    ❌ Request error: {e}")
    sys.exit(1)

print("\nDone.\n")
