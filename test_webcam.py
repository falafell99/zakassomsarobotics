"""
test_webcam.py — Smart Blind-Assistance Glasses
================================================
Object detection via  cv2.dnn  +  YOLOv8n ONNX
  → NO PyTorch, NO ultralytics, NO numpy version conflicts
  → Works with any Python 3.8+ / numpy version / platform

PIPELINE:
  Camera (30 fps)
    → cv2.dnn YOLOv8 ONNX (local, every frame, zero API lag)
    → Filter: only AVOID_CLASSES matter
    → Estimate distance from bounding-box size
    → Local navigation decision (instant)
    → (optional) AI server for richer message — background thread

BOX COLOURS:
  🟢 Green  → far object (> 2 m, CLEAR)
  🟡 Yellow → medium (warning zone)
  🔴 Red    → close (< 0.5 m, STOP / DANGER)
  ⚫ Grey   → not in avoid list (shown but ignored)

KEYS (click the window first):
  S → force AI server call now
  T → toggle TTS on/off
  D → toggle debug_frame.jpg saving
  Q / ESC → quit

FIRST RUN: yolov8n.onnx (~12 MB) is downloaded automatically.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SERVER_URL       = "http://localhost:8000"
CAMERA_INDEX     = 0
USE_AI_SERVER    = True    # False = pure local YOLO, no API calls at all
AI_COOLDOWN_SEC  = 5.0     # seconds between AI API requests
JPEG_QUALITY     = 82
TTS_ENABLED      = False   # set True if ELEVENLABS_API_KEY is configured
DEBUG_SAVE_FRAME = True    # save debug_frame.jpg on each AI send

# YOLOv8n ONNX — auto-downloaded on first run (~12 MB)
ONNX_MODEL_PATH = "yolov8n.onnx"
ONNX_DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
)
YOLO_INPUT_SIZE  = 640      # YOLOv8 expects 640×640
YOLO_CONF_THRESH = 0.45     # minimum detection confidence
YOLO_NMS_THRESH  = 0.45     # NMS overlap threshold
DETECT_EVERY_N   = 1        # run detection every N frames (1=every, 2=every other)

# Objects that trigger navigation warnings.  Everything else → grey info box.
AVOID_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "couch", "dining table", "bench",
    "dog", "cat", "horse", "cow", "elephant", "bear",
    "backpack", "suitcase", "handbag",
    "traffic light", "stop sign", "fire hydrant",
    "refrigerator", "oven", "sink",
    "bottle", "cup", "bowl", "vase",
}

# Distance zones (fraction of frame area covered by bounding box)
BOX_FRAC_CLEAR = 0.03    # < 3%  → far  (> 2 m)  CLEAR
BOX_FRAC_WARN  = 0.12    # 3-12% → medium (0.5-2 m)  WARNING
BOX_FRAC_STOP  = 0.30    # 12-30%→ close (< 0.5 m)  STOP
                          # > 30% → very close        DANGER

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
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import cv2
import time
import base64
import threading
import requests
import sys
import os
import urllib.request
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX MODEL DOWNLOAD + LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def _download_model():
    """Download yolov8n.onnx if not already present."""
    if os.path.exists(ONNX_MODEL_PATH):
        size_mb = os.path.getsize(ONNX_MODEL_PATH) / 1_000_000
        print(f"[INIT] Found {ONNX_MODEL_PATH} ({size_mb:.1f} MB)")
        return True

    print(f"[INIT] Downloading YOLOv8n ONNX (~12 MB) …")
    print(f"       {ONNX_DOWNLOAD_URL}")
    try:
        def _progress(count, block, total):
            pct = min(count * block / total * 100, 100)
            print(f"\r       {pct:.0f}%", end="", flush=True)
        urllib.request.urlretrieve(ONNX_DOWNLOAD_URL, ONNX_MODEL_PATH, _progress)
        print()
        print(f"[INIT] Downloaded {ONNX_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[INIT] Download FAILED: {e}")
        print("       Manual download:")
        print(f"         curl -L '{ONNX_DOWNLOAD_URL}' -o {ONNX_MODEL_PATH}")
        return False


def _load_model():
    """Load YOLOv8n ONNX via cv2.dnn. Returns net or None."""
    if not _download_model():
        return None
    try:
        net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        # Use CPU backend (works everywhere)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[INIT] YOLOv8n ONNX loaded via cv2.dnn  ✅")
        return net
    except Exception as e:
        print(f"[INIT] Failed to load ONNX model: {e}")
        return None


print("[INIT] Loading YOLOv8n ONNX model…")
_net = _load_model()
DETECTION_AVAILABLE = _net is not None

# ═══════════════════════════════════════════════════════════════════════════════
#  TTS (optional, ElevenLabs)
# ═══════════════════════════════════════════════════════════════════════════════
_el_client = None
if TTS_ENABLED:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play as _el_play
        _key = os.getenv("ELEVENLABS_API_KEY", "")
        if _key:
            _el_client = ElevenLabs(api_key=_key)
            print("[INIT] ElevenLabs TTS ready")
        else:
            print("[INIT] No ELEVENLABS_API_KEY — TTS will print to console")
    except Exception as e:
        print(f"[INIT] TTS disabled: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOURS  (BGR)
# ═══════════════════════════════════════════════════════════════════════════════
COL = {
    "CLEAR":      (55,  215, 55),
    "STOP":       (25,  25,  230),
    "DANGER":     (0,   0,   255),
    "LEFT":       (0,   215, 255),
    "RIGHT":      (0,   215, 255),
    "IDLE":       (160, 160, 160),
    "ERROR":      (0,   0,   200),
    "box_danger": (0,   0,   255),
    "box_stop":   (0,   80,  255),
    "box_warn":   (0,   210, 255),
    "box_clear":  (55,  215, 55),
    "box_info":   (110, 110, 110),
    "white":      (255, 255, 255),
    "dim":        (110, 110, 110),
    "panel":      (20,  20,  20),
}

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
    "worst_dist":    2.5,
    "worst_obj":     "—",
    "tts_on":        TTS_ENABLED,
    "debug_save":    DEBUG_SAVE_FRAME,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO INFERENCE  (cv2.dnn, no PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════
def run_detection(frame: np.ndarray) -> list:
    """
    Run YOLOv8n ONNX on `frame` using cv2.dnn.
    Returns list of detection dicts (see below).
    Each dict: {label, conf, box:(x1,y1,x2,y2), area_frac, distance, is_avoid, tier}
    """
    if not DETECTION_AVAILABLE:
        return []

    h, w = frame.shape[:2]
    frame_area = w * h
    S = YOLO_INPUT_SIZE

    # ── Preprocess ────────────────────────────────────────────────────────────
    # blobFromImage: resize to 640×640, normalize 0-1, swap BGR→RGB
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (S, S),
                                  swapRB=True, crop=False)
    _net.setInput(blob)

    # ── Inference ─────────────────────────────────────────────────────────────
    # YOLOv8 ONNX output shape: (1, 84, 8400)
    # 84 = 4 box coords + 80 class scores
    raw = _net.forward()         # (1, 84, 8400)
    preds = raw[0].T             # (8400, 84)

    # ── Parse detections ──────────────────────────────────────────────────────
    # Scale from 640×640 back to original frame size
    sx = w / S
    sy = h / S

    raw_boxes, raw_confs, raw_cls = [], [], []

    for pred in preds:
        cx, cy, bw, bh = pred[:4]
        class_scores = pred[4:]            # shape (80,)
        cls_id = int(np.argmax(class_scores))
        conf   = float(class_scores[cls_id])

        if conf < YOLO_CONF_THRESH:
            continue

        # Convert cx,cy,bw,bh (640-space) → x1,y1,x2,y2 (frame-space)
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)

        # Clamp to frame
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
        raw_confs.append(conf)
        raw_cls.append(cls_id)

    # ── NMS ───────────────────────────────────────────────────────────────────
    detections = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confs,
                                   YOLO_CONF_THRESH, YOLO_NMS_THRESH)
        # OpenCV returns different shapes depending on version
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, (list, tuple)) and indices:
            indices = [i[0] if isinstance(i, (list, np.ndarray)) else i
                       for i in indices]

        for i in indices:
            x, y, bw, bh = raw_boxes[i]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            cls_id = raw_cls[i]
            conf   = raw_confs[i]
            label  = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"

            box_area  = (x2 - x1) * (y2 - y1)
            area_frac = box_area / frame_area
            distance  = _area_to_distance(area_frac)
            is_avoid  = label in AVOID_CLASSES
            tier      = _get_tier(is_avoid, area_frac)

            detections.append({
                "label":     label,
                "conf":      conf,
                "box":       (x1, y1, x2, y2),
                "area_frac": area_frac,
                "distance":  distance,
                "is_avoid":  is_avoid,
                "tier":      tier,
            })

    return detections


def _area_to_distance(area_frac: float) -> float:
    """Convert bounding-box fraction of frame → estimated distance in metres."""
    if area_frac < BOX_FRAC_CLEAR:  return 2.5
    if area_frac < BOX_FRAC_WARN:   return 1.2
    if area_frac < BOX_FRAC_STOP:   return 0.5
    return 0.15


def _get_tier(is_avoid: bool, area_frac: float) -> str:
    if not is_avoid:                   return "info"
    if area_frac > BOX_FRAC_STOP:     return "danger"
    if area_frac > BOX_FRAC_WARN:     return "stop"
    if area_frac > BOX_FRAC_CLEAR:    return "warn"
    return "clear"


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL NAVIGATION  (instant, no API needed)
# ═══════════════════════════════════════════════════════════════════════════════
def navigate(detections: list, frame_w: int):
    """
    Decide navigation command from YOLO detections.
    Returns (command, obstacle_name, distance_m, message).
    """
    avoid = [d for d in detections if d["is_avoid"]]
    if not avoid:
        return "CLEAR", "none", 2.5, "Path is clear"

    # Closest avoid-object
    avoid.sort(key=lambda d: d["distance"])
    c    = avoid[0]
    dist = c["distance"]
    obj  = c["label"]
    tier = c["tier"]
    x1, y1, x2, y2 = c["box"]
    cx = (x1 + x2) / 2

    if tier == "danger":
        return "DANGER", obj, dist, f"Danger! {obj} very close!"

    if tier in ("stop", "warn"):
        left_bias  = cx < frame_w * 0.38
        right_bias = cx > frame_w * 0.62
        if left_bias:
            return "RIGHT", obj, dist, f"{obj} on left — go right"
        if right_bias:
            return "LEFT",  obj, dist, f"{obj} on right — go left"
        return "STOP",  obj, dist, f"Stop — {obj} ahead"

    return "CLEAR", obj, dist, f"{obj} ahead, path clear"


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW BOUNDING BOXES
# ═══════════════════════════════════════════════════════════════════════════════
_TIER_COL = {
    "danger": COL["box_danger"],
    "stop":   COL["box_stop"],
    "warn":   COL["box_warn"],
    "clear":  COL["box_clear"],
    "info":   COL["box_info"],
}
_TIER_THICK = {"danger": 3, "stop": 2, "warn": 2, "clear": 1, "info": 1}


def draw_boxes(frame, detections: list):
    """Draw styled bounding boxes on `frame` in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        tier   = d["tier"]
        color  = _TIER_COL[tier]
        thick  = _TIER_THICK[tier]
        label  = d["label"]
        conf   = d["conf"]
        dist   = d["distance"]
        avoid  = d["is_avoid"]

        # ── Filled semi-transparent overlay for danger objects ────────────────
        if tier == "danger":
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # ── Box ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # ── Corner accent lines (professional look) ───────────────────────────
        L = 14
        lw = thick + 1
        for (px, py, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * L, py), color, lw)
            cv2.line(frame, (px, py), (px, py + dy * L), color, lw)

        # ── Label badge ───────────────────────────────────────────────────────
        badge = f"{'⚠ ' if avoid else ''}{label}  {conf:.0%}  ~{dist:.1f}m"
        fscale = 0.50
        fthick = 1
        (tw, th), bl = cv2.getTextSize(badge, font, fscale, fthick)
        pad = 4
        bx1, by1 = x1,           max(0, y1 - th - pad * 2 - bl)
        bx2, by2 = x1 + tw + pad * 2, max(th + pad, y1)

        # Badge background
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx1, by1), (bx2, by2), color, -1)
        cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)

        # Badge text
        cv2.putText(frame, badge, (x1 + pad, by2 - pad - bl),
                    font, fscale, COL["panel"], fthick + 1, cv2.LINE_AA)
        cv2.putText(frame, badge, (x1 + pad, by2 - pad - bl),
                    font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)

        # ── Pulsing ring for danger objects ───────────────────────────────────
        if tier == "danger":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = min(50, max(20, (x2 - x1) // 4))
            t  = time.time()
            # Animated radius
            r2 = r + int(6 * abs(np.sin(t * 6)))
            cv2.circle(frame, (cx, cy), r2, color, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════
def draw_hud(frame, w, h, detections):
    """Draw command panel, FPS, AI status, and direction arrow on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    with _lock:
        cmd      = state["command"]
        obstacle = state["obstacle"]
        ai_msg   = state["ai_message"]
        status   = state["status"]
        fps      = state["fps"]
        busy     = state["api_busy"]
        dist     = state["worst_dist"]
        tts_on   = state["tts_on"]
        last_api = state["last_api_time"]

    cool = max(0.0, AI_COOLDOWN_SEC - (time.time() - last_api))
    cmd_col = COL.get(cmd, COL["IDLE"])

    def panel(x1, y1, x2, y2, alpha=0.62):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), COL["panel"], -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    def txt(text, x, y, color, scale=0.58, thick=1):
        cv2.putText(frame, text, (x+1, y+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
        cv2.putText(frame, text, (x,   y),   font, scale, color,   thick,   cv2.LINE_AA)

    # ── TOP-LEFT: command word + distance ─────────────────────────────────────
    panel(0, 0, 320, 120)
    cv2.putText(frame, cmd, (12, 74), font, 2.1, (0,0,0), 8, cv2.LINE_AA)
    cv2.putText(frame, cmd, (12, 74), font, 2.1, cmd_col, 3, cv2.LINE_AA)
    txt(f"~{dist:.1f} m   {obstacle}", 12, 106, COL["white"], 0.50)

    # ── TOP-RIGHT: FPS + status ───────────────────────────────────────────────
    panel(w - 250, 0, w, 96)
    txt(f"FPS  {fps:5.1f}", w - 238, 30, COL["white"], 0.60)
    s_col = COL["DANGER"] if status == "ERROR" else (COL["LEFT"] if busy else COL["dim"])
    txt(status, w - 238, 58, s_col, 0.48)
    tts_str = "TTS ON  [T]" if tts_on else "TTS OFF [T]"
    txt(tts_str, w - 238, 82, COL["dim"], 0.38)

    # ── CENTRE-TOP: AI cooldown bar ───────────────────────────────────────────
    if USE_AI_SERVER and not busy and cool > 0.2:
        bar_max = 220
        bw_fill = int(cool / AI_COOLDOWN_SEC * bar_max)
        cx = w // 2
        panel(cx - 120, 6, cx + 120, 30, alpha=0.72)
        cv2.rectangle(frame, (cx - 112, 10), (cx - 112 + bw_fill, 26), COL["LEFT"], -1)
        txt(f"AI in {cool:.1f}s", cx - 38, 24, COL["panel"], 0.38)

    if busy:
        txt("[ SENDING TO AI ]", w//2 - 70, 52, COL["LEFT"], 0.52)

    # ── BOTTOM BAR: detected count + AI message ───────────────────────────────
    panel(0, h - 64, w, h)
    avoid_count = sum(1 for d in detections if d["is_avoid"])
    total_count = len(detections)
    txt(f"Objects: {avoid_count} hazard / {total_count} total", 12, h - 40,
        COL["LEFT"] if avoid_count > 0 else COL["dim"], 0.50)
    ai_short = (ai_msg[:105] + "…") if len(ai_msg) > 105 else ai_msg
    txt(f"AI: {ai_short}", 12, h - 14, COL["white"], 0.52)

    # ── Key hints strip ───────────────────────────────────────────────────────
    txt("[S] force AI  [D] debug  [Q] quit", w - 238, h - 14, COL["dim"], 0.36)

    # ── Danger border ─────────────────────────────────────────────────────────
    if cmd in ("DANGER", "STOP"):
        t = time.time()
        alpha = 0.5 + 0.5 * abs(np.sin(t * 4))  # pulsing
        border_col = COL["DANGER"] if cmd == "DANGER" else COL["STOP"]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w-1, h-1), border_col, 6)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    # ── Direction arrow (LEFT / RIGHT) ────────────────────────────────────────
    if cmd in ("LEFT", "RIGHT"):
        ax, ay = w // 2, h // 2
        size = 65
        if cmd == "LEFT":
            pts = np.array([
                [ax,        ay - 22], [ax - size, ay],     [ax,        ay + 22],
                [ax,        ay + 9],  [ax + 32,   ay + 9], [ax + 32,   ay - 9],
                [ax,        ay - 9],
            ], np.int32)
        else:
            pts = np.array([
                [ax,        ay - 22], [ax + size, ay],     [ax,        ay + 22],
                [ax,        ay + 9],  [ax - 32,   ay + 9], [ax - 32,   ay - 9],
                [ax,        ay - 9],
            ], np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], COL[cmd])
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.polylines(frame, [pts], True, COL["white"], 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  TTS
# ═══════════════════════════════════════════════════════════════════════════════
def speak(text_to_say: str):
    with _lock:
        tts_on = state["tts_on"]
    if not tts_on:
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
            print(f"[TTS FAIL] {e}")
    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  AI API CALL  (background thread, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════
def call_api(frame_to_send: np.ndarray, distance: float):
    """POST frame to FastAPI /analyze. Runs in daemon thread."""
    with _lock:
        state["api_busy"] = True
        state["status"]   = "SENDING"

    try:
        ok, buf = cv2.imencode(".jpg", frame_to_send,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        kb  = len(b64) * 3 / 4 / 1024

        with _lock:
            dbg = state["debug_save"]
        if dbg:
            cv2.imwrite("debug_frame.jpg", frame_to_send)
            print(f"[API] Saved debug_frame.jpg")

        print(f"[API] POST /analyze  dist={distance:.2f}m  {kb:.0f} KB …")
        with _lock:
            state["status"] = "WAITING"

        resp = requests.post(
            f"{SERVER_URL}/analyze",
            json={"distance": distance, "image": b64},
            timeout=45,
        )
        print(f"[API] HTTP {resp.status_code}: {resp.text[:200]}")

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
        print(f"[API] ERROR: {e}")
        with _lock:
            state["ai_message"]    = f"API error: {str(e)[:60]}"
            state["status"]        = "ERROR"
            state["last_api_time"] = time.time()
    finally:
        with _lock:
            state["api_busy"] = False


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def startup_check(cap):
    print("\n" + "═" * 64)
    print("  SMART GLASSES  |  YOLOv8 ONNX Object Detection Test")
    print("═" * 64)
    print(f"  {'✅' if DETECTION_AVAILABLE else '❌'} YOLOv8n ONNX   (cv2.dnn, no PyTorch needed)")
    if USE_AI_SERVER:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            print(f"  ✅ API server    {SERVER_URL}  HTTP {r.status_code}")
        except Exception as e:
            print(f"  ⚠  API server   NOT RUNNING — YOLO-only mode active")
            print(f"              ({e})")
    else:
        print(f"  ℹ  API server   DISABLED (USE_AI_SERVER=False)")
    if cap.isOpened():
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  ✅ Camera #{CAMERA_INDEX}     {aw}×{ah}")
    else:
        print(f"  ❌ Camera FAILED (index={CAMERA_INDEX})")
    print(f"\n  Avoid classes:  {len(AVOID_CLASSES)} categories")
    print(f"  Confidence:     {YOLO_CONF_THRESH:.0%}")
    print(f"  AI cooldown:    {AI_COOLDOWN_SEC}s")
    print(f"\n  BOX COLOUR GUIDE:")
    print(f"    🟢 Green  — avoid-object, far (> 2 m)")
    print(f"    🟡 Yellow — avoid-object, warning zone (0.5–2 m)")
    print(f"    🔴 Red    — avoid-object, close (< 0.5 m)  STOP/DANGER")
    print(f"    ⚫ Grey   — NOT in avoid list (info only)")
    print("═" * 64 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always fresh frames

    startup_check(cap)

    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        sys.exit(1)

    WIN = "Smart Glasses — YOLOv8 Detection  |  Q to quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    fps_count     = 0
    fps_t         = time.time()
    frame_n       = 0
    last_dets     = []
    prev_cmd      = ""
    prev_msg      = ""

    with _lock:
        state["status"]     = "IDLE"
        state["ai_message"] = "YOLO running — detecting objects…" if DETECTION_AVAILABLE \
                              else "No model — check yolov8n.onnx"
        state["command"]    = "CLEAR"

    print("[MAIN] Camera loop started.\n")

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

        # ── YOLO detection ────────────────────────────────────────────────────
        if frame_n % DETECT_EVERY_N == 0:
            last_dets = run_detection(frame)

            cmd, obj, dist, msg = navigate(last_dets, w)

            with _lock:
                state["worst_dist"] = dist
                state["worst_obj"]  = obj

                # Update command from local YOLO
                # (but don't override a recent AI server result)
                api_age = time.time() - state["last_api_time"]
                if api_age > 4.0 or state["command"] in ("IDLE", "CLEAR", "ERROR"):
                    state["command"]  = cmd
                    state["obstacle"] = obj
                    if api_age > 8.0:
                        state["ai_message"] = f"[YOLO] {msg}"

            # ── TTS on command change ─────────────────────────────────────────
            if cmd != prev_cmd or (cmd not in ("CLEAR", "IDLE") and msg != prev_msg):
                if cmd not in ("CLEAR", "IDLE"):
                    speak(msg)
                prev_cmd = cmd
                prev_msg = msg

            # ── Trigger AI API call when object in danger zone ────────────────
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
                threading.Thread(
                    target=call_api, args=(send, dist), daemon=True
                ).start()

        # ── Render ────────────────────────────────────────────────────────────
        display = frame.copy()
        draw_boxes(display, last_dets)
        draw_hud(display, w, h, last_dets)
        cv2.imshow(WIN, display)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            with _lock:
                state["force_api"] = True
            print("[KEY] Force AI send queued")
        elif key in (ord('t'), ord('T')):
            with _lock:
                state["tts_on"] = not state["tts_on"]
                on = state["tts_on"]
            print(f"[KEY] TTS {'ON' if on else 'OFF'}")
        elif key in (ord('d'), ord('D')):
            with _lock:
                state["debug_save"] = not state["debug_save"]
                on = state["debug_save"]
            print(f"[KEY] Debug save {'ON' if on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
