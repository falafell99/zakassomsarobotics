"""
test_webcam.py — Smart Blind-Assistance Glasses
================================================
DETECTION (no downloads required, works immediately):
  1. cv2.HOGDescriptor   → detects PEOPLE (built into OpenCV, 100% reliable)
  2. Large-contour scan  → detects large OBSTACLES (chairs, walls, tables, boxes)
  3. ONNX upgrade path   → if you place yolov8n.onnx here, 80-class detection
                           activates automatically (see HOW_TO_GET_ONNX below)

DISTANCE ESTIMATION — real physics, not faking:
  Uses the pinhole camera model:
    distance_m = (focal_length_px × real_object_height_m) / bbox_height_px

  focal_length_px ≈ frame_width / (2 × tan(FOV/2))
  For a typical webcam (70° FOV, 1280px wide): focal_length ≈ 900 px

  Example — person (1.7m tall):
    bbox = 400px → distance = 900 × 1.7 / 400 = 3.8 m  ✓
    bbox = 700px → distance = 900 × 1.7 / 700 = 2.2 m  ✓
    bbox fills frame (720px) → distance ≈ 2.1 m  ✓

HOW_TO_GET_ONNX (optional upgrade):
  The ultralytics/assets GitHub repo is private so direct download is blocked.
  To get yolov8n.onnx, run this one-time command in a fresh Python environment:
    pip install "numpy<2" ultralytics
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
  Then place yolov8n.onnx in the same folder as this script.

KEYBOARD (click the OpenCV window first):
  S → force AI server call now
  T → toggle TTS
  Q / ESC → quit
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SERVER_URL       = "http://localhost:8000"
CAMERA_INDEX     = 0
USE_AI_SERVER    = True      # set False for pure-local mode
AI_COOLDOWN_SEC  = 5.0
JPEG_QUALITY     = 82
TTS_ENABLED      = False

# Webcam field-of-view in degrees (typical laptop webcam = 65-80°)
# Increase if distances seem too small; decrease if too large
WEBCAM_FOV_DEG   = 70.0

# HOG detection settings
HOG_WIN_STRIDE   = (8, 8)     # smaller = more sensitive but slower
HOG_SCALE        = 1.05
HOG_HIT_THRESH   = 0.0        # lower = more detections (try -0.3 for more hits)

# Contour obstacle detection settings
CONTOUR_MIN_FRAC = 0.04        # ignore contours smaller than 4% of frame
CONTOUR_MAX_FRAC = 0.85        # ignore contours covering whole frame (background)

# ── Real-world heights for distance estimation (metres) ──────────────────────
# These make distance accurate for each class via the pinhole camera model.
REAL_HEIGHTS = {
    "person":       1.70,
    "chair":        0.95,
    "couch":        0.85,
    "dining table": 0.75,
    "table":        0.75,
    "car":          1.50,
    "bus":          3.20,
    "truck":        2.80,
    "bicycle":      1.10,
    "motorcycle":   1.20,
    "dog":          0.55,
    "cat":          0.30,
    "bottle":       0.28,
    "backpack":     0.55,
    "suitcase":     0.70,
    "obstacle":     1.00,   # unknown large contour — assume ~1m
}

# Only these labels trigger navigation warnings
AVOID_CLASSES = set(REAL_HEIGHTS.keys())

# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import cv2
import time
import base64
import threading
import requests
import sys
import os
import math
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION BACKEND — HOG + Contours (always available)
#                    + ONNX via onnxruntime (if model file present)
# ═══════════════════════════════════════════════════════════════════════════════

# ── HOG people detector (zero download needed) ────────────────────────────────
print("[INIT] Loading HOG person detector…")
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("[INIT] HOG person detector ready  ✅")

# ── ONNX upgrade (optional) ───────────────────────────────────────────────────
_ort = None
_ort_input_name = None
ONNX_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
ONNX_AVAILABLE = False

if os.path.exists(ONNX_PATH) and os.path.getsize(ONNX_PATH) > 1_000_000:
    try:
        import onnxruntime as ort
        _ort = ort.InferenceSession(ONNX_PATH,
                                    providers=["CPUExecutionProvider"])
        _ort_input_name = _ort.get_inputs()[0].name
        ONNX_AVAILABLE = True
        print(f"[INIT] YOLOv8n ONNX loaded via onnxruntime  ✅  (80-class mode)")
    except Exception as e:
        print(f"[INIT] onnxruntime failed: {e} — using HOG mode")
else:
    print(f"[INIT] yolov8n.onnx not found — using HOG + contour mode")

# COCO 80 class names (same order as YOLOv8 ONNX output)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  DISTANCE ESTIMATION — pinhole camera model
# ═══════════════════════════════════════════════════════════════════════════════
def compute_focal_length(frame_w: int) -> float:
    """Compute focal length in pixels from webcam FOV and frame width."""
    fov_rad = math.radians(WEBCAM_FOV_DEG)
    return frame_w / (2.0 * math.tan(fov_rad / 2.0))


def estimate_distance(label: str, box, frame_shape) -> float:
    """
    Pinhole camera model: distance = (focal_length × real_height) / box_height_px
    Accurate for known-size objects. Clips to [0.2, 6.0] metres.
    """
    x1, y1, x2, y2 = box
    box_h = max(y2 - y1, 1)
    frame_h, frame_w = frame_shape[:2]
    focal_px = compute_focal_length(frame_w)
    real_h = REAL_HEIGHTS.get(label, 1.0)
    dist = (focal_px * real_h) / box_h
    return round(float(np.clip(dist, 0.2, 6.0)), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_with_hog(frame) -> list:
    """
    Detect people using HOG descriptor.
    Returns list of detection dicts.
    """
    # Downscale for speed; we scale boxes back after
    small_w = 640
    h, w = frame.shape[:2]
    small_h = int(h * small_w / w)
    small = cv2.resize(frame, (small_w, small_h))

    boxes_raw, weights = _hog.detectMultiScale(
        small,
        winStride=HOG_WIN_STRIDE,
        padding=(8, 8),
        scale=HOG_SCALE,
        hitThreshold=HOG_HIT_THRESH,
        useMeanshiftGrouping=True,
    )

    sx = w / small_w
    sy = h / small_h
    results = []

    for i, (x, y, bw, bh) in enumerate(boxes_raw):
        x1 = int(x * sx)
        y1 = int(y * sy)
        x2 = int((x + bw) * sx)
        y2 = int((y + bh) * sy)
        conf = float(np.squeeze(weights[i])) / 10.0 if i < len(weights) else 0.6
        conf = float(np.clip(conf, 0.3, 1.0))
        dist = estimate_distance("person", (x1, y1, x2, y2), frame.shape)
        results.append({
            "label":    "person",
            "conf":     conf,
            "box":      (x1, y1, x2, y2),
            "distance": dist,
            "is_avoid": True,
            "source":   "HOG",
        })

    return results


def detect_large_obstacles(frame) -> list:
    """
    Find large blobs using edge + contour detection.
    Catches chairs, walls, boxes, furniture — anything big in the way.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (9, 9), 0)
    edges  = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        frac = area / frame_area
        if frac < CONTOUR_MIN_FRAC or frac > CONTOUR_MAX_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + bw, y + bh

        # Aspect ratio filter: skip very wide thin lines (walls at distance)
        aspect = bw / max(bh, 1)
        if aspect > 8 or aspect < 0.1:
            continue

        # Skip boxes in top 20% of frame (usually background/ceiling)
        if y2 < h * 0.2:
            continue

        dist = estimate_distance("obstacle", (x1, y1, x2, y2), frame.shape)

        # Only include if closer than 3m (small contours are background noise)
        if dist > 3.0:
            continue

        results.append({
            "label":    "obstacle",
            "conf":     min(frac * 5, 0.9),
            "box":      (x1, y1, x2, y2),
            "distance": dist,
            "is_avoid": True,
            "source":   "contour",
        })

    return results


def detect_with_onnx(frame) -> list:
    """
    YOLOv8n ONNX inference via onnxruntime.
    Only active when yolov8n.onnx is present and valid.
    """
    if not ONNX_AVAILABLE:
        return []

    h, w = frame.shape[:2]
    S = 640

    # Preprocess: resize to 640×640, normalize 0-1, CHW, add batch dim
    resized = cv2.resize(frame, (S, S))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp     = rgb.astype(np.float32) / 255.0
    inp     = np.transpose(inp, (2, 0, 1))[np.newaxis]  # (1,3,640,640)

    raw   = _ort.run(None, {_ort_input_name: inp})[0]   # (1,84,8400)
    preds = raw[0].T                                     # (8400,84)

    sx, sy = w / S, h / S
    boxes_list, confs_list, cls_list = [], [], []

    for pred in preds:
        cls_scores = pred[4:]
        cls_id     = int(np.argmax(cls_scores))
        conf       = float(cls_scores[cls_id])
        if conf < 0.35:
            continue
        cx, cy, bw, bh = pred[:4]
        x1 = int((cx - bw / 2) * sx);  y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx);  y2 = int((cy + bh / 2) * sy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_list.append([x1, y1, x2 - x1, y2 - y1])
        confs_list.append(conf)
        cls_list.append(cls_id)

    if not boxes_list:
        return []

    indices = cv2.dnn.NMSBoxes(boxes_list, confs_list, 0.35, 0.45)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, (list, tuple)) and indices:
        indices = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in indices]

    results = []
    for i in indices:
        bx, by, bw, bh = boxes_list[i]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        label    = COCO_CLASSES[cls_list[i]] if cls_list[i] < len(COCO_CLASSES) else "unknown"
        is_avoid = label in AVOID_CLASSES
        dist     = estimate_distance(label, (x1, y1, x2, y2), frame.shape)
        results.append({
            "label":    label,
            "conf":     confs_list[i],
            "box":      (x1, y1, x2, y2),
            "distance": dist,
            "is_avoid": is_avoid,
            "source":   "ONNX",
        })

    return results


def run_detection(frame) -> list:
    """
    Main detection entry point.
    Uses ONNX if available (80 classes + distance), otherwise HOG + contours.
    Deduplicates overlapping boxes from different detectors.
    """
    if ONNX_AVAILABLE:
        return detect_with_onnx(frame)

    people   = detect_with_hog(frame)
    contours = detect_large_obstacles(frame)

    # Suppress contour boxes that heavily overlap with HOG person boxes
    merged = list(people)
    for c in contours:
        cx1, cy1, cx2, cy2 = c["box"]
        overlaps_person = False
        for p in people:
            px1, py1, px2, py2 = p["box"]
            iou_x = min(cx2, px2) - max(cx1, px1)
            iou_y = min(cy2, py2) - max(cy1, py1)
            if iou_x > 0 and iou_y > 0:
                c_area = (cx2 - cx1) * (cy2 - cy1)
                p_area = (px2 - px1) * (py2 - py1)
                inter  = iou_x * iou_y
                if inter / min(c_area, p_area) > 0.4:
                    overlaps_person = True
                    break
        if not overlaps_person:
            merged.append(c)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION (instant, local, no API)
# ═══════════════════════════════════════════════════════════════════════════════
def navigate(detections: list, frame_w: int):
    """
    Decide navigation command from detections.
    Returns (command, obstacle_name, distance_m, message).
    """
    avoid = [d for d in detections if d["is_avoid"]]
    if not avoid:
        return "CLEAR", "none", 5.0, "Path is clear"

    avoid.sort(key=lambda d: d["distance"])
    c    = avoid[0]
    dist = c["distance"]
    obj  = c["label"]
    x1, y1, x2, y2 = c["box"]
    cx   = (x1 + x2) / 2.0

    if dist < 0.4:
        return "DANGER", obj, dist, f"Danger! {obj} at {dist:.1f}m!"
    if dist < 0.8:
        if cx < frame_w * 0.38:
            return "RIGHT", obj, dist, f"{obj} left at {dist:.1f}m — go right"
        if cx > frame_w * 0.62:
            return "LEFT", obj, dist, f"{obj} right at {dist:.1f}m — go left"
        return "STOP", obj, dist, f"Stop — {obj} ahead at {dist:.1f}m"
    if dist < 1.5:
        if cx < frame_w * 0.38:
            return "RIGHT", obj, dist, f"Caution, {obj} left at {dist:.1f}m"
        if cx > frame_w * 0.62:
            return "LEFT", obj, dist, f"Caution, {obj} right at {dist:.1f}m"
        return "STOP", obj, dist, f"Slow — {obj} ahead at {dist:.1f}m"

    return "CLEAR", obj, dist, f"{obj} detected, {dist:.1f}m away"


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOURS
# ═══════════════════════════════════════════════════════════════════════════════
COL = {
    "CLEAR":  (55,  215, 55),
    "STOP":   (30,  30,  230),
    "DANGER": (0,   0,   255),
    "LEFT":   (0,   215, 255),
    "RIGHT":  (0,   215, 255),
    "IDLE":   (160, 160, 160),
    "ERROR":  (0,   0,   200),
    "white":  (255, 255, 255),
    "dim":    (110, 110, 110),
    "panel":  (20,  20,  20),
}

def dist_color(dist: float):
    """Return BGR color based on estimated distance."""
    if dist < 0.4:  return (0,   0,   255)   # red   — danger
    if dist < 0.8:  return (0,   80,  255)   # orange-red
    if dist < 1.5:  return (0,   200, 255)   # yellow
    if dist < 3.0:  return (55,  215, 55)    # green
    return                  (100, 100, 100)  # grey   — far


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def draw_boxes(frame, detections: list):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Sort: draw far first so close ones render on top
    dets_sorted = sorted(detections, key=lambda d: d["distance"], reverse=True)

    for d in dets_sorted:
        x1, y1, x2, y2 = d["box"]
        dist   = d["distance"]
        label  = d["label"]
        conf   = d["conf"]
        avoid  = d["is_avoid"]
        source = d.get("source", "")
        color  = dist_color(dist)
        thick  = 3 if dist < 0.8 else 2

        # ── Semi-transparent fill for very close objects ──────────────────────
        if dist < 0.8:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)

        # ── Box ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # ── Corner accents ────────────────────────────────────────────────────
        L = 16
        lw = thick + 1
        for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*L, py), color, lw)
            cv2.line(frame, (px, py), (px, py + dy*L), color, lw)

        # ── Distance arc on close objects ─────────────────────────────────────
        if dist < 1.5:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = max(18, min(45, (x2-x1)//5))
            cv2.ellipse(frame, (cx, cy), (r, r), 0, -120, 120, color, 2)

        # ── Label badge ───────────────────────────────────────────────────────
        tag  = f"{'⚠ ' if avoid else ''}{label}  {dist:.1f}m  {conf:.0%}"
        fs   = 0.50
        ft   = 1
        (tw, th), bl = cv2.getTextSize(tag, font, fs, ft)
        pad  = 4
        bx1  = x1
        by1  = max(0, y1 - th - pad*2 - bl)
        bx2  = x1 + tw + pad*2
        by2  = max(th + pad, y1)

        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx1, by1), (bx2, by2), color, -1)
        cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)

        cv2.putText(frame, tag, (x1+pad, by2-pad-bl),
                    font, fs, COL["panel"], ft+1, cv2.LINE_AA)
        cv2.putText(frame, tag, (x1+pad, by2-pad-bl),
                    font, fs, (255, 255, 255), ft, cv2.LINE_AA)


def draw_distance_ruler(frame, detections, h, w):
    """Draw a mini ruler on the right edge showing distances of all objects."""
    if not detections:
        return

    ruler_x = w - 36
    ruler_top = 130
    ruler_bot = h - 80
    ruler_h = ruler_bot - ruler_top

    # Background
    ov = frame.copy()
    cv2.rectangle(ov, (ruler_x - 4, ruler_top), (w - 2, ruler_bot), COL["panel"], -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Tick marks at 0.5, 1, 1.5, 2, 3m
    max_dist = 5.0
    for tick_m in [0.5, 1.0, 1.5, 2.0, 3.0]:
        y_tick = ruler_bot - int((tick_m / max_dist) * ruler_h)
        cv2.line(frame, (ruler_x - 6, y_tick), (ruler_x + 6, y_tick), COL["dim"], 1)
        cv2.putText(frame, f"{tick_m:.0f}m" if tick_m < 2.5 else "",
                    (ruler_x - 26, y_tick + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COL["dim"], 1)

    # Object dots on ruler
    for d in detections:
        if not d["is_avoid"]:
            continue
        dist = min(d["distance"], max_dist)
        y_dot = ruler_bot - int((dist / max_dist) * ruler_h)
        color = dist_color(d["distance"])
        cv2.circle(frame, (ruler_x, y_dot), 5, color, -1)
        cv2.circle(frame, (ruler_x, y_dot), 5, COL["white"], 1)


def draw_hud(frame, w, h, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX

    with _lock:
        cmd      = state["command"]
        obstacle = state["obstacle"]
        ai_msg   = state["ai_message"]
        status   = state["status"]
        fps      = state["fps"]
        busy     = state["api_busy"]
        dist     = state["worst_dist"]
        last_api = state["last_api_time"]
        mode_str = state["mode"]

    cool = max(0.0, AI_COOLDOWN_SEC - (time.time() - last_api))
    cmd_col = COL.get(cmd, COL["IDLE"])

    def panel(x1, y1, x2, y2, alpha=0.62):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), COL["panel"], -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def txt(text, x, y, color, scale=0.56, thick=1):
        cv2.putText(frame, text, (x+1, y+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
        cv2.putText(frame, text, (x,   y  ), font, scale, color,   thick,   cv2.LINE_AA)

    # ── TOP-LEFT: command + distance ─────────────────────────────────────────
    panel(0, 0, 340, 125)
    cv2.putText(frame, cmd, (12, 76), font, 2.1, (0,0,0), 8, cv2.LINE_AA)
    cv2.putText(frame, cmd, (12, 76), font, 2.1, cmd_col, 3, cv2.LINE_AA)
    dist_str = f"{dist:.1f} m" if dist < 5.0 else "clear"
    txt(f"~{dist_str}  |  {obstacle}", 12, 108, COL["white"], 0.50)

    # ── TOP-RIGHT: FPS + mode + status ───────────────────────────────────────
    panel(w - 260, 0, w, 100)
    txt(f"FPS  {fps:5.1f}", w - 248, 28, COL["white"], 0.60)
    txt(f"Mode: {mode_str}", w - 248, 54, COL["dim"], 0.44)
    s_col = COL["DANGER"] if status == "ERROR" else (COL["LEFT"] if busy else COL["dim"])
    txt(status, w - 248, 78, s_col, 0.42)

    # ── AI cooldown bar ───────────────────────────────────────────────────────
    if USE_AI_SERVER and not busy and 0.2 < cool < AI_COOLDOWN_SEC:
        bw_fill = int(cool / AI_COOLDOWN_SEC * 220)
        cx = w // 2
        panel(cx - 118, 6, cx + 118, 28, alpha=0.72)
        cv2.rectangle(frame, (cx - 110, 10), (cx - 110 + bw_fill, 24), COL["LEFT"], -1)
        txt(f"AI in {cool:.1f}s", cx - 36, 23, COL["panel"], 0.36)

    if busy:
        txt("[ SENDING TO AI ]", w//2 - 68, 50, COL["LEFT"], 0.52)

    # ── BOTTOM bar ───────────────────────────────────────────────────────────
    panel(0, h - 64, w, h)
    avoid_n = sum(1 for d in detections if d["is_avoid"])
    all_n   = len(detections)
    txt(f"Detected: {avoid_n} hazard / {all_n} total", 12, h - 38,
        COL["LEFT"] if avoid_n > 0 else COL["dim"], 0.50)
    ai_short = (ai_msg[:100] + "…") if len(ai_msg) > 100 else ai_msg
    txt(f"AI: {ai_short}", 12, h - 12, COL["white"], 0.50)
    txt("[S] AI  [Q] quit", w - 180, h - 12, COL["dim"], 0.36)

    # ── Danger border (pulsing) ───────────────────────────────────────────────
    if cmd in ("DANGER", "STOP"):
        alpha = 0.45 + 0.45 * abs(math.sin(time.time() * 5))
        border_col = COL["DANGER"] if cmd == "DANGER" else COL["STOP"]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w-1, h-1), border_col, 6)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    # ── Direction arrow ───────────────────────────────────────────────────────
    if cmd in ("LEFT", "RIGHT"):
        ax, ay = w // 2, h // 2
        size = 70
        if cmd == "LEFT":
            pts = np.array([[ax, ay-24],[ax-size, ay],[ax, ay+24],
                            [ax, ay+10],[ax+34, ay+10],[ax+34, ay-10],[ax, ay-10]], np.int32)
        else:
            pts = np.array([[ax, ay-24],[ax+size, ay],[ax, ay+24],
                            [ax, ay+10],[ax-34, ay+10],[ax-34, ay-10],[ax, ay-10]], np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], COL[cmd])
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.polylines(frame, [pts], True, COL["white"], 1, cv2.LINE_AA)

    draw_distance_ruler(frame, detections, h, w)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()
state = {
    "command":       "IDLE",
    "obstacle":      "—",
    "ai_message":    "Starting…",
    "status":        "INIT",
    "fps":           0.0,
    "api_busy":      False,
    "last_api_time": 0.0,
    "force_api":     False,
    "worst_dist":    5.0,
    "mode":          "ONNX" if ONNX_AVAILABLE else "HOG+contour",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TTS
# ═══════════════════════════════════════════════════════════════════════════════
_el_client = None
if TTS_ENABLED:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play as _el_play
        _key = os.getenv("ELEVENLABS_API_KEY", "")
        if _key:
            _el_client = ElevenLabs(api_key=_key)
    except Exception:
        pass

def speak(text_to_say: str):
    if not TTS_ENABLED:
        print(f"[TTS] {text_to_say}")
        return
    def _run():
        try:
            if _el_client:
                audio = _el_client.text_to_speech.convert(
                    text=text_to_say, voice_id="Rachel",
                    model_id="eleven_monolingual_v1")
                _el_play(audio)
            else:
                print(f"[TTS] {text_to_say}")
        except Exception as e:
            print(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  AI API CALL
# ═══════════════════════════════════════════════════════════════════════════════
def call_api(frame_to_send, distance: float):
    with _lock:
        state["api_busy"] = True
        state["status"]   = "SENDING"
    try:
        ok, buf = cv2.imencode(".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        with _lock:
            state["status"] = "WAITING"
        resp = requests.post(f"{SERVER_URL}/analyze",
                             json={"distance": distance, "image": b64}, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        d = resp.json()
        with _lock:
            state["command"]       = d.get("command",  state["command"])
            state["ai_message"]    = f"[AI] {d.get('message', '?')}"
            state["obstacle"]      = d.get("obstacle", state["obstacle"])
            state["status"]        = "IDLE"
            state["last_api_time"] = time.time()
        speak(d.get("message", ""))
    except Exception as e:
        print(f"[API] {e}")
        with _lock:
            state["ai_message"]    = f"API: {str(e)[:60]}"
            state["status"]        = "ERROR"
            state["last_api_time"] = time.time()
    finally:
        with _lock:
            state["api_busy"] = False


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP INFO
# ═══════════════════════════════════════════════════════════════════════════════
def startup_check(cap):
    print("\n" + "═" * 66)
    print("  SMART GLASSES  |  Object Detection + Distance Estimation")
    print("═" * 66)
    mode = "YOLOv8n ONNX (80 classes)" if ONNX_AVAILABLE else "HOG (people) + Contour (obstacles)"
    print(f"  Detection:  {mode}")
    print(f"  Distance:   Pinhole camera model  (FOV={WEBCAM_FOV_DEG}°)")
    print(f"              formula: dist = focal_length × real_height / box_height")
    if USE_AI_SERVER:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            print(f"  AI server:  {SERVER_URL}  HTTP {r.status_code}  ✅")
        except:
            print(f"  AI server:  NOT RUNNING  (YOLO-only mode)")
    if cap.isOpened():
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fl = compute_focal_length(aw)
        print(f"  Camera:     #{CAMERA_INDEX}  {aw}×{ah}  focal≈{fl:.0f}px")
    print(f"\n  BOX COLOURS:")
    print(f"    🔴 Red    < 0.4m  DANGER")
    print(f"    🟠 Orange 0.4–0.8m  STOP zone")
    print(f"    🟡 Yellow 0.8–1.5m  Caution")
    print(f"    🟢 Green  1.5–3m  OK")
    print(f"    ⚫ Grey   > 3m  Far / background")
    print()
    if not ONNX_AVAILABLE:
        print("  ℹ  For 80-class YOLO detection, get yolov8n.onnx:")
        print("     pip install 'numpy<2' ultralytics")
        print("     python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')\"")
        print("     mv yolov8n.onnx /path/to/blind_glasses_server/")
    print("═" * 66 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    startup_check(cap)

    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        sys.exit(1)

    WIN = "Smart Glasses  |  Q to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    fps_count = 0
    fps_t     = time.time()
    frame_n   = 0
    last_dets = []
    prev_cmd  = ""
    prev_msg  = ""

    # HOG is slow — run it every 3rd frame; contours every 2nd
    DETECT_N_HOG     = 3 if not ONNX_AVAILABLE else 1
    DETECT_N_CONTOUR = 2

    with _lock:
        state["status"]     = "IDLE"
        state["ai_message"] = "Running — move closer to an object"
        state["command"]    = "CLEAR"

    print("[MAIN] Running. S=force AI  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        frame_n += 1

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_count += 1
        dt = time.time() - fps_t
        if dt >= 0.5:
            with _lock:
                state["fps"] = fps_count / dt
            fps_count = 0
            fps_t = time.time()

        # ── Detection (throttled for CPU performance) ─────────────────────────
        if frame_n % DETECT_N_HOG == 0:
            last_dets = run_detection(frame)

        # ── Local navigation ──────────────────────────────────────────────────
        cmd, obj, dist, msg = navigate(last_dets, w)

        with _lock:
            state["worst_dist"] = dist
            state["worst_obj"]  = obj
            api_age = time.time() - state["last_api_time"]
            if api_age > 4.0 or state["command"] in ("IDLE", "CLEAR", "ERROR"):
                state["command"]  = cmd
                state["obstacle"] = obj
                if api_age > 8.0:
                    state["ai_message"] = f"[local] {msg}"

        # ── TTS on change ─────────────────────────────────────────────────────
        if cmd != prev_cmd or (cmd not in ("CLEAR", "IDLE") and msg != prev_msg):
            if cmd not in ("CLEAR", "IDLE"):
                speak(msg)
            prev_cmd = cmd
            prev_msg = msg

        # ── AI server call ────────────────────────────────────────────────────
        with _lock:
            busy     = state["api_busy"]
            last_api = state["last_api_time"]
            force    = state["force_api"]
            if force:
                state["force_api"] = False

        cooldown_ok = (time.time() - last_api) >= AI_COOLDOWN_SEC
        trigger_ai  = cmd in ("STOP", "DANGER", "LEFT", "RIGHT")
        if USE_AI_SERVER and not busy and (force or (cooldown_ok and trigger_ai)):
            send = cv2.resize(frame, (640, 480))
            threading.Thread(target=call_api, args=(send, dist), daemon=True).start()

        # ── Render ────────────────────────────────────────────────────────────
        display = frame.copy()
        draw_boxes(display, last_dets)
        draw_hud(display, w, h, last_dets)
        cv2.imshow(WIN, display)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            with _lock:
                state["force_api"] = True
            print("[KEY] Force AI send")

    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
