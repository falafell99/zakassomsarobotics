import base64
import threading
import time

import cv2
import requests


SERVER_URL = "http://127.0.0.1:8000/analyze"
CAMERA_INDEX = 0
SEND_INTERVAL_SEC = 0.8
JPEG_QUALITY = 75


def draw_response(frame, response: dict, sensor_distance: float, pending: bool):
    h, w = frame.shape[:2]
    command = str(response.get("command", "IDLE"))
    message = str(response.get("message", ""))
    detections = response.get("detections", [])

    color = {
        "CLEAR": (40, 200, 40),
        "STOP": (0, 120, 255),
        "DANGER": (0, 0, 255),
        "LEFT": (0, 220, 255),
        "RIGHT": (0, 220, 255),
    }.get(command, (180, 180, 180))

    cv2.rectangle(frame, (0, 0), (w, 112), (15, 15, 15), -1)
    cv2.putText(frame, command, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3)
    cv2.putText(frame, message[:90], (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1)
    meta = (
        f"sensor={sensor_distance:.1f}m  "
        f"latency={response.get('processing_ms', 0):.0f}ms  "
        f"detector={response.get('detector', '?')}  "
        f"{'sending' if pending else 'ready'}"
    )
    cv2.putText(frame, meta, (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
    reason = str(response.get("reason", ""))[:95]
    if reason:
        cv2.rectangle(frame, (0, h - 34), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, reason, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

    zones = response.get("zones", {}) or {}
    zone_text = f"L {zones.get('left', 0):.1f}  C {zones.get('center', 0):.1f}  R {zones.get('right', 0):.1f}"
    cv2.putText(frame, zone_text, (w - 230, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 220, 255), 1)

    cv2.line(frame, (w // 3, 112), (w // 3, h), (70, 70, 70), 1)
    cv2.line(frame, (2 * w // 3, 112), (2 * w // 3, h), (70, 70, 70), 1)

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        label = det.get("label", "object")
        raw = det.get("raw_label")
        dist = float(det.get("distance", 0))
        action = det.get("action", "")
        det_color = {
            "CLEAR": (40, 200, 40),
            "STOP": (0, 120, 255),
            "DANGER": (0, 0, 255),
            "LEFT": (0, 220, 255),
            "RIGHT": (0, 220, 255),
        }.get(action, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), det_color, 2)
        tag_label = f"{label}({raw})" if raw and raw != label else label
        tag = f"{tag_label} {dist:.1f}m {action}"
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 230, y1), det_color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def encode_frame(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    last_send = 0.0
    latest = {"command": "IDLE", "message": "Waiting for backend"}
    pending = False
    lock = threading.Lock()
    sensor_distance = 5.0

    def send_frame(frame_to_send, sensor_value):
        nonlocal latest, pending
        try:
            payload = {
                "distance": sensor_value,
                "speak": False,
                "image": encode_frame(frame_to_send),
            }
            resp = requests.post(SERVER_URL, json=payload, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            data = {"command": "ERROR", "message": str(exc)[:100]}
        with lock:
            latest = data
            pending = False

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        now = time.time()
        if not pending and now - last_send >= SEND_INTERVAL_SEC:
            last_send = now
            pending = True
            small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            threading.Thread(
                target=send_frame,
                args=(small, sensor_distance),
                daemon=True,
            ).start()

        with lock:
            response = dict(latest)
            is_pending = pending

        draw_response(frame, response, sensor_distance, is_pending)
        cv2.imshow("rafachmo debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("+"), ord("=")):
            sensor_distance = min(8.0, sensor_distance + 0.25)
        elif key in (ord("-"), ord("_")):
            sensor_distance = max(0.2, sensor_distance - 0.25)
        elif key == ord("0"):
            sensor_distance = 5.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
