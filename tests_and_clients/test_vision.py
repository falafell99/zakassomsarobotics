"""
Test script to verify computer vision is working.
Sends a real image to the /analyze endpoint.
"""

import requests
import base64
from PIL import Image
import io
import time

# Create a simple test image with a "chair" drawn on it
def create_test_image():
    """Create a test image to simulate what ESP32 might send."""
    # Create a 640x480 image
    img = Image.new('RGB', (640, 480), color='white')

    # Draw some simple shapes to simulate an obstacle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw a "chair" shape on the left side
    draw.rectangle([100, 200, 200, 400], fill='brown', outline='black')  # Seat
    draw.rectangle([120, 100, 180, 200], fill='brown', outline='black')  # Backrest
    draw.rectangle([100, 400, 130, 450], fill='black')  # Leg 1
    draw.rectangle([170, 400, 200, 450], fill='black')  # Leg 2

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_analyze_endpoint():
    """Test the /analyze endpoint with a real image."""

    print("=" * 60)
    print("COMPUTER VISION TEST")
    print("=" * 60)

    # Test 1: Clear path (distance > 1.5m)
    print("\nTest 1: Distance 2.0m (should return CLEAR, no AI call)")
    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "distance": 2.0,
            "image": "dummy_image_that_wont_be_used"
        }
    )
    print(f"Response: {response.json()}")

    time.sleep(3)  # Wait for cooldown

    # Test 2: Warning zone (should call AI vision)
    print("\nTest 2: Distance 1.2m with test image (should call AI vision)")
    test_image = create_test_image()
    print(f"Test image size: {len(test_image)} chars (base64)")

    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "distance": 1.2,
            "image": test_image
        }
    )
    print(f"Response: {response.json()}")

    time.sleep(3)  # Wait for cooldown

    # Test 3: Danger zone
    print("\nTest 3: Distance 0.2m (should return DANGER immediately)")
    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "distance": 0.2,
            "image": "dummy"
        }
    )
    print(f"Response: {response.json()}")

    print("\n" + "=" * 60)
    print("Test complete! Check server logs for AI vision calls.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_analyze_endpoint()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server!")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\nERROR: {e}")
