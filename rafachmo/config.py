import os
from dotenv import load_dotenv

load_dotenv()

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "20"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")

CAMERA_VERTICAL_FOV_DEG = float(os.getenv("CAMERA_VERTICAL_FOV_DEG", "55"))
DISTANCE_SCALE = float(os.getenv("DISTANCE_SCALE", "1.0"))
DISTANCE_SMOOTHING_ALPHA = float(os.getenv("DISTANCE_SMOOTHING_ALPHA", "0.45"))
SENSOR_VALID_MAX_M = float(os.getenv("SENSOR_VALID_MAX_M", "4.5"))
COMMAND_STABILITY_FRAMES = int(os.getenv("COMMAND_STABILITY_FRAMES", "2"))
CENTER_CORRIDOR_LEFT = int(os.getenv("CENTER_CORRIDOR_LEFT", "330"))
CENTER_CORRIDOR_RIGHT = int(os.getenv("CENTER_CORRIDOR_RIGHT", "670"))
MIN_BOX_AREA = float(os.getenv("MIN_BOX_AREA", "0.002"))
SENSOR_WALL_THRESHOLD_M = float(os.getenv("SENSOR_WALL_THRESHOLD_M", "1.6"))
SCENE_PROFILE = os.getenv("SCENE_PROFILE", "home_office")

SPEAK_ENABLED = os.getenv("SPEAK_ENABLED", "true").lower() == "true"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2")

COMMANDS = {"CLEAR", "STOP", "LEFT", "RIGHT", "DANGER"}
