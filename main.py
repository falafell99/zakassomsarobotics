import logging
import time
from datetime import datetime

from fastapi import FastAPI

import config
from detector import detector
from image_utils import decode_base64_image, pil_to_bgr
from navigator import enrich_detections, choose_primary
from schemas import AnalyzeRequest, AnalyzeResponse
from speaker import speak

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
distance_history: dict[tuple[str, str], float] = {}
command_state = {
    "stable_command": "CLEAR",
    "candidate_command": "CLEAR",
    "candidate_count": 0,
    "last_message": "Path clear",
    "last_obstacle": "none",
    "last_distance": 5.0,
    "last_reason": "",
}

app = FastAPI(
    title="Rafachmo Smart Glasses Backend",
    description="Laptop-side AI/navigation backend for ESP32-CAM smart glasses MVP",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "detector": "yolo" if detector.ready else "fallback",
        "timestamp": datetime.now().isoformat(),
    }


def smooth_distances(detections):
    global distance_history
    current_keys = set()
    for det in detections:
        key = (det.label, det.position)
        current_keys.add(key)
        previous = distance_history.get(key)
        if previous is not None:
            alpha = config.DISTANCE_SMOOTHING_ALPHA
            det.distance = round(alpha * det.distance + (1.0 - alpha) * previous, 2)
        distance_history[key] = det.distance

    for key in list(distance_history.keys()):
        if key not in current_keys:
            del distance_history[key]
    return detections


def add_sensor_wall_fallback(detections, sensor_distance: float):
    if sensor_distance <= 0 or sensor_distance > config.SENSOR_WALL_THRESHOLD_M:
        return detections

    has_center_obstacle = any(
        det.position == "center" and det.distance <= config.SENSOR_WALL_THRESHOLD_M
        for det in detections
    )
    if has_center_obstacle:
        return detections

    from schemas import Detection

    detections.append(
        Detection(
            label="wall",
            raw_label="sensor obstacle",
            category="structure",
            confidence=0.65,
            distance=round(sensor_distance, 2),
            position="center",
            bbox=[config.CENTER_CORRIDOR_LEFT, 250, config.CENTER_CORRIDOR_RIGHT, 1000],
            action="STOP",
            source="ultrasonic",
            priority=2.5,
            risk=2.5,
        )
    )
    return detections


def stabilize_decision(command: str, message: str, obstacle: str, distance: float, reason: str):
    if command == "DANGER":
        command_state.update(
            stable_command=command,
            candidate_command=command,
            candidate_count=config.COMMAND_STABILITY_FRAMES,
            last_message=message,
            last_obstacle=obstacle,
            last_distance=distance,
            last_reason=reason,
        )
        return command, message, obstacle, distance, reason

    stable = command_state["stable_command"]
    candidate = command_state["candidate_command"]

    if command == stable:
        command_state.update(
            candidate_command=command,
            candidate_count=0,
            last_message=message,
            last_obstacle=obstacle,
            last_distance=distance,
            last_reason=reason,
        )
        return command, message, obstacle, distance, reason

    if command == candidate:
        command_state["candidate_count"] += 1
    else:
        command_state["candidate_command"] = command
        command_state["candidate_count"] = 1

    if command_state["candidate_count"] >= config.COMMAND_STABILITY_FRAMES:
        command_state.update(
            stable_command=command,
            last_message=message,
            last_obstacle=obstacle,
            last_distance=distance,
            last_reason=reason,
        )
        return command, message, obstacle, distance, reason

    return (
        stable,
        command_state["last_message"],
        command_state["last_obstacle"],
        command_state["last_distance"],
        f"holding {stable} until {command} is stable",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    started = time.perf_counter()
    timestamp = datetime.now().isoformat()

    try:
        image = decode_base64_image(request.image)
        frame_bgr = pil_to_bgr(image)
        detections = detector.detect(frame_bgr)
    except Exception as exc:
        logger.error("Image/detection failed: %s", exc)
        detections = []

    detections = enrich_detections(detections, request.distance)
    detections = add_sensor_wall_fallback(detections, request.distance)
    detections = smooth_distances(detections)
    command, message, obstacle, distance, reason, zones = choose_primary(detections, request.distance)
    command, message, obstacle, distance, reason = stabilize_decision(
        command, message, obstacle, distance, reason
    )
    processing_ms = round((time.perf_counter() - started) * 1000, 1)

    should_speak = False
    if request.speak and config.SPEAK_ENABLED and command != "CLEAR":
        should_speak = speak(message)

    return AnalyzeResponse(
        command=command,
        message=message,
        obstacle=obstacle,
        distance=distance,
        detections=detections[:8],
        speak=should_speak,
        processing_ms=processing_ms,
        detector="yolo" if detector.ready else "fallback",
        reason=reason,
        zones=zones,
        timestamp=timestamp,
    )


@app.post("/speak")
async def speak_endpoint(payload: dict):
    text = str(payload.get("text", "")).strip()
    queued = speak(text) if text else False
    return {"queued": queued}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.SERVER_PORT)
