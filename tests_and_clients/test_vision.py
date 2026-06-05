"""
Test script to verify computer vision is working.
Iterates over all test images in the test-imgs/ directory and sends them to the /analyze endpoint.
"""

import os
import requests
import base64
import time


def get_base64_from_file(filepath):
    """Read a file and convert it to base64."""
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_analyze_endpoint():
    """Test the /analyze endpoint with real images from test-imgs/."""

    print("=" * 60)
    print("COMPUTER VISION TEST - USING REAL IMAGES")
    print("=" * 60)

    # Test 1: Clear path (distance > 1.5m)
    print("\nTest 1: Distance 2.0m (should return CLEAR, no AI call)")
    response = requests.post(
        "http://localhost:8000/analyze",
        json={"distance": 2.0, "image": "dummy_image_that_wont_be_used"},
    )
    print(f"Response: {response.json()}")

    time.sleep(3)  # Wait for cooldown

    # Test 2: Iterate over test images
    test_imgs_dir = os.path.join(os.path.dirname(__file__), "test-imgs")

    if os.path.exists(test_imgs_dir):
        image_files = sorted(
            [
                f
                for f in os.listdir(test_imgs_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )

        if not image_files:
            print(f"\nNo images found in {test_imgs_dir}!")

        for img_name in image_files:
            print(f"\nTesting image: {img_name} (Distance 1.2m -> calls AI vision)")
            img_path = os.path.join(test_imgs_dir, img_name)

            try:
                base64_img = get_base64_from_file(img_path)

                response = requests.post(
                    "http://localhost:8000/analyze",
                    json={"distance": 1.2, "image": base64_img},
                )
                print(f"Response: {response.json()}")

                print("Waiting 4 seconds for API cooldown...")
                time.sleep(4)
            except Exception as e:
                print(f"Error testing {img_name}: {e}")
    else:
        print(f"\nDirectory {test_imgs_dir} not found!")

    # Test 3: Danger zone
    print("\nTest 3: Distance 0.2m (should return DANGER immediately)")
    response = requests.post(
        "http://localhost:8000/analyze", json={"distance": 0.2, "image": "dummy"}
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
        print("Make sure the server is running: python backend/main.py")
    except Exception as e:
        print(f"\nERROR: {e}")
