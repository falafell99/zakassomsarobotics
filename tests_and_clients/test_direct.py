"""
Direct analyzer test script for smart blind-assistance glasses.
Bypasses the server entirely - directly calls analyzer.py with webcam frames.
Useful for testing AI response without running the FastAPI server.
"""

import cv2
import time
import datetime
import io
import asyncio
import base64
import sys
import os
from PIL import Image

# Add backend directory to path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
)

# Import project modules
from image_utils import process_image 
from analyzer import analyze_image_with_vision

# Configuration
DISTANCE = 0.8  # Fake distance for testing (within AI analysis zone: 0.3m - 1.5m)
SEND_INTERVAL = 3  # Seconds between analyzer calls
WINDOW_NAME = "Smart Glasses - Direct Test (No Server)"

# Overlay text settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_COLOR = (0, 255, 255)  # Yellow - distinct from server test (white)
FONT_THICKNESS = 2


def cv2_frame_to_base64(cv2_frame, quality=70):
    """
    Convert an OpenCV frame (numpy array) to a base64 JPEG string.

    Args:
        cv2_frame: OpenCV image (BGR numpy array)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded JPEG string (no data URL prefix)
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Encode to JPEG base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_string


async def analyze_frame(base64_image, distance):
    """
    Call the analyzer directly (same as the server does).

    Args:
        base64_image: Base64 encoded JPEG string
        distance: Distance in meters

    Returns:
        Dictionary with keys: direction, message, obstacle
    """
    # Process image using project's image_utils (resize + compress)
    processed_b64 = process_image(
        base64_image, max_width=800, max_height=600, quality=70
    )

    # Call analyzer directly (this is an async function)
    result = await analyze_image_with_vision(processed_b64, distance)
    return result


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(
            "ERROR: Could not open laptop webcam (index 0). Check camera permissions."
        )
        return

    last_call_time = 0
    current_command = ""
    current_message = ""
    current_obstacle = ""
    current_direction = ""

    print("=" * 60)
    print("Direct Analyzer Test (No Server Needed)")
    print(f"Fake distance: {DISTANCE}m (within AI analysis zone)")
    print(f"Analyzer call interval: {SEND_INTERVAL}s")
    print("Press Q to quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Could not read frame from webcam, retrying...")
                time.sleep(0.1)
                continue

            current_time = time.time()

            # Call analyzer directly every SEND_INTERVAL seconds
            if current_time - last_call_time >= SEND_INTERVAL:
                try:
                    # Convert cv2 frame to base64 JPEG string
                    image_b64 = cv2_frame_to_base64(frame)

                    # Call analyzer asynchronously
                    result = asyncio.run(analyze_frame(image_b64, DISTANCE))

                    # Extract fields from response
                    current_direction = result.get("direction", "STOP")
                    current_message = result.get("message", "")
                    current_obstacle = result.get("obstacle", "unknown")

                    # Build command string for overlay
                    current_command = current_direction

                    # Print raw response with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] Direct analyzer call:")
                    print(f"  Distance sent: {DISTANCE}m")
                    print(f"  Raw response: {result}")
                    print(f"  Direction: {current_direction}")
                    print(f"  Message: {current_message}")
                    print(f"  Obstacle: {current_obstacle}")

                except Exception as e:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] ANALYZER ERROR: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    current_command = "ERROR"
                    current_message = ""
                    current_obstacle = ""

                last_call_time = current_time

            # Overlay command on live feed
            cv2.putText(
                frame,
                f"Dir: {current_command}",
                (10, 50),
                FONT,
                FONT_SCALE,
                FONT_COLOR,
                FONT_THICKNESS,
            )

            # Show live feed
            cv2.imshow(WINDOW_NAME, frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDirect test stopped.")


if __name__ == "__main__":
    main()
