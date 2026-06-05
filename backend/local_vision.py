"""
Local Computer Vision module using OpenCV + YOLO.
Provides offline object detection for blind-assistance glasses.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import base64
import io
from PIL import Image
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for YOLO model
yolo_model = None
opencv_available = False
ultralytics_available = False

# Try to import OpenCV and YOLO
try:
    import cv2
    opencv_available = True
    logger.info("OpenCV imported successfully")
except ImportError:
    logger.warning("OpenCV not installed. Install with: pip install opencv-python")

try:
    from ultralytics import YOLO
    ultralytics_available = True
    logger.info("Ultralytics (YOLO) imported successfully")
except ImportError:
    logger.warning("Ultralytics not installed. Install with: pip install ultralytics")


def load_yolo_model(model_name: str = "yolov8n.pt"):
    """
    Load YOLO model for object detection.

    Args:
        model_name: YOLO model to use (yolov8n.pt = nano, fastest)

    Returns:
        YOLO model instance or None
    """
    global yolo_model

    if not ultralytics_available:
        logger.error("Ultralytics not available. Cannot load YOLO model.")
        return None

    try:
        # Determine correct path for model in models/ directory
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", model_name))
        
        logger.info(f"Loading YOLO model: {model_path}")
        yolo_model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return yolo_model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None


def detect_objects_yolo(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Detect objects in image using YOLO.

    Args:
        image: PIL Image

    Returns:
        List of detected objects with keys: name, confidence, bbox (x1, y1, x2, y2)
    """
    global yolo_model

    if yolo_model is None:
        logger.warning("YOLO model not loaded. Attempting to load...")
        load_yolo_model()

    if yolo_model is None:
        logger.error("YOLO model not available")
        return []

    try:
        # Convert PIL Image to numpy array (RGB → BGR for OpenCV)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run YOLO detection
        results = yolo_model(img_array, verbose=False)

        detected_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]

                    detected_objects.append({
                        "name": class_name,
                        "confidence": float(confidence),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center_x": int((x1 + x2) / 2),
                        "center_y": int((y1 + y2) / 2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    })

        logger.info(f"YOLO detected {len(detected_objects)} objects")
        return detected_objects

    except Exception as e:
        logger.error(f"Error in YOLO detection: {e}")
        return []


def analyze_obstacle_position(
    detected_objects: List[Dict[str, Any]],
    image_width: int,
    distance: float
) -> Dict[str, Any]:
    """
    Analyze detected objects and determine navigation command.

    Args:
        detected_objects: List of detected objects from YOLO
        image_width: Width of the image in pixels
        distance: Distance from ultrasonic sensor in meters

    Returns:
        Dictionary with direction, message, obstacle
    """
    if not detected_objects:
        return {
            "direction": "CLEAR",
            "message": "Path is clear",
            "obstacle": "none"
        }

    # Sort by confidence (highest first)
    detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

    # Get the most confident detection
    main_obstacle = detected_objects[0]
    obstacle_name = main_obstacle["name"]
    center_x = main_obstacle["center_x"]
    bbox_width = main_obstacle["width"]

    # Calculate position (left/right/center)
    image_center = image_width / 2
    left_boundary = image_width * 0.33
    right_boundary = image_width * 0.66

    # Determine obstacle position
    obstacle_center = center_x

    # Check if obstacle is too close
    if distance < 0.3:
        return {
            "direction": "DANGER",
            "message": f"Danger! {obstacle_name} too close",
            "obstacle": obstacle_name
        }

    # Determine direction based on obstacle position
    if obstacle_center < left_boundary:
        # Obstacle on left → go RIGHT
        direction = "RIGHT"
        message = f"Turn right, {obstacle_name} on left"
    elif obstacle_center > right_boundary:
        # Obstacle on right → go LEFT
        direction = "LEFT"
        message = f"Turn left, {obstacle_name} on right"
    else:
        # Obstacle in center
        # Check if it's large (blocking entire path)
        if bbox_width > image_width * 0.5:
            direction = "STOP"
            message = f"Stop, {obstacle_name} ahead"
        else:
            # Could go around - suggest larger gap
            direction = "STOP"
            message = f"Stop, {obstacle_name} ahead"

    return {
        "direction": direction,
        "message": message,
        "obstacle": obstacle_name
    }


async def analyze_image_local(
    base64_image: str,
    distance: float
) -> Dict[str, Any]:
    """
    Analyze image using local YOLO + OpenCV.

    Args:
        base64_image: Base64 encoded JPEG image
        distance: Distance from ultrasonic sensor

    Returns:
        Dictionary with direction, message, obstacle
    """
    if not opencv_available or not ultralytics_available:
        logger.error("OpenCV or YOLO not available")
        return {
            "direction": "STOP",
            "message": "Vision system unavailable",
            "obstacle": "unknown"
        }

    try:
        logger.info("Starting local vision analysis with YOLO...")

        # Decode base64 to PIL Image
        from image_utils import decode_base64_image
        image = decode_base64_image(base64_image)
        image_width = image.width

        logger.info(f"Image decoded. Size: {image.size}")

        # Detect objects with YOLO
        detected_objects = detect_objects_yolo(image)

        if not detected_objects:
            logger.info("No objects detected")
            return {
                "direction": "CLEAR",
                "message": "Path is clear",
                "obstacle": "none"
            }

        # Log detected objects
        logger.info("Detected objects:")
        for obj in detected_objects:
            logger.info(f"  - {obj['name']} (confidence: {obj['confidence']:.2f})")

        # Analyze and determine navigation command
        result = analyze_obstacle_position(detected_objects, image_width, distance)

        logger.info(f"Local vision result: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in local vision analysis: {e}")
        return {
            "direction": "STOP",
            "message": "Vision analysis error",
            "obstacle": "unknown"
        }


def get_yolo_status() -> Dict[str, Any]:
    """
    Get status of YOLO and OpenCV.

    Returns:
        Dictionary with status information
    """
    return {
        "opencv_available": opencv_available,
        "ultralytics_available": ultralytics_available,
        "yolo_model_loaded": yolo_model is not None,
        "yolo_model_name": yolo_model.model_name if yolo_model else None
    }
