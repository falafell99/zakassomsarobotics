"""
test_webcam.py — Smart Blind-Assistance Glasses  v3.0
======================================================
DETECTION STACK (three layers):
  1. Depth Anything V2 (monocular depth AI, 94 MB ONNX)
     → Real depth map every frame in background thread
     → Accurate close-range obstacle detection without bounding boxes
     → Works on walls, chairs, people, cars — ANYTHING

  2. OpenCV HOG person detector (built-in, no download)
     → Reliable human detection with distance via pinhole camera model

  3. Google Gemini Flash (scene understanding AI)
     → Full scene description every 3 seconds
     → Names objects, estimates distances, gives navigation instructions

DISTANCE METHODS:
  - Depth Anything: relative depth map → normalized to 0–1 (1=closest)
  - Pinhole camera: dist = focal_length × real_height / box_height_pixels
  - AI (Gemini): vision model estimates distance from visual cues

CONTROLS (click OpenCV window first):
  D → toggle depth heatmap overlay
  S → force AI call now
  H → toggle HOG person detection overlay
  T → toggle TTS on/off
  Q / ESC → quit
""" 

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SERVER_URL = "http://localhost:8000"
CAMERA_INDEX = "http://192.168.4.1/"  # Connect to ESP32 Access Point
ESP32_URL = "http://192.168.4.1"  # Base URL for hardware commands
AI_INTERVAL_SEC = 3.0  # call AI every N seconds (always, not just on obstacle)
JPEG_QUALITY = 82
WEBCAM_FOV_DEG = 70.0  # typical laptop webcam
DEPTH_SIZE = 308  # px — trade-off: 196=fast, 308=accurate, 518=best
SHOW_DEPTH = True  # D key toggles this
SHOW_HOG = True  # H key toggles this

# ── TTS settings ─────────────────────────────────────────────────────────────
TTS_ENABLED = True  # T key toggles at runtime
TTS_USE_ELEVENLABS = True  # False = macOS 'say' (free, instant)
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — clear, calm voice
ELEVENLABS_MODEL = "eleven_turbo_v2"  # fastest ElevenLabs model
TTS_COOLDOWN_SEC = 2.5  # min seconds between spoken messages

# Pinhole camera: real-world heights for distance estimation (metres)
REAL_HEIGHTS = {
    "person": 1.70,
    "chair": 0.95,
    "couch": 0.85,
    "dining table": 0.75,
    "table": 0.75,
    "car": 1.50,
    "bus": 3.20,
    "truck": 2.80,
    "bicycle": 1.10,
    "motorcycle": 1.20,
    "dog": 0.55,
    "cat": 0.30,
    "bottle": 0.28,
    "backpack": 0.55,
    "suitcase": 0.70,
    "obstacle": 1.00,
}
AVOID_CLASSES = set(REAL_HEIGHTS.keys())

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import cv2
import time
import base64
import threading
import requests
import sys
import os

# Add backend directory to path for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
)

import math
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import onnxruntime as ort

    _ORT_OK = True
except ImportError:
    _ORT_OK = False
    print("[WARN] onnxruntime not installed — depth AI disabled")

import subprocess
import queue as _queue

# ══════════════════════════════════════════════════════════════════════════════
#  DEPTH ANYTHING V2  (monocular depth AI)
# ══════════════════════════════════════════════════════════════════════════════
_DEPTH_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "models", "depth_anything_v2_small.onnx"
    )
)
_depth_session = None
DEPTH_AI_OK = False

# ImageNet normalization constants for Depth Anything preprocessing
_DA_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DA_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if _ORT_OK and os.path.exists(_DEPTH_PATH) and os.path.getsize(_DEPTH_PATH) > 1_000_000:
    try:
        print("[INIT] Loading Depth Anything V2…")
        _depth_session = ort.InferenceSession(
            _DEPTH_PATH, providers=["CPUExecutionProvider"]
        )
        DEPTH_AI_OK = True
        print(
            f"[INIT] Depth Anything V2 loaded  ✅  (running at {DEPTH_SIZE}×{DEPTH_SIZE})"
        )
    except Exception as e:
        print(f"[INIT] Depth Anything load failed: {e}")
else:
    print("[INIT] depth_anything_v2_small.onnx not found — depth AI disabled")

# Shared depth state
_depth_lock = threading.Lock()
_depth_map_raw = None  # (H, W) float32 — raw disparity from model
_depth_map_01 = None  # (H, W) float32 — normalized 0–1 (1 = closest)
_depth_running = False


def _infer_depth(frame_bgr: np.ndarray):
    """Run Depth Anything V2 on a frame. Stores result in shared state."""
    global _depth_running, _depth_map_raw, _depth_map_01

    if not DEPTH_AI_OK:
        return

    _depth_running = True
    try:
        S = DEPTH_SIZE
        resized = cv2.resize(frame_bgr, (S, S))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization
        inp_norm = (rgb - _DA_MEAN) / _DA_STD  # (S, S, 3)
        inp_chw = np.transpose(inp_norm, (2, 0, 1))[np.newaxis]  # (1, 3, S, S)

        raw_out = _depth_session.run(None, {"pixel_values": inp_chw})[0][0]  # (S, S)

        # Normalize to 0–1 per frame: 1 = closest, 0 = farthest
        d_min = raw_out.min()
        d_max = raw_out.max()
        if d_max > d_min:
            norm = (raw_out - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(raw_out)

        with _depth_lock:
            _depth_map_raw = raw_out
            _depth_map_01 = norm
    except Exception as e:
        print(f"[DEPTH] Error: {e}")
    finally:
        _depth_running = False


def get_depth_snapshot():
    """Thread-safe copy of current depth map (0–1, 1=closest)."""
    with _depth_lock:
        return _depth_map_01.copy() if _depth_map_01 is not None else None


# ══════════════════════════════════════════════════════════════════════════════
#  DEPTH-BASED NAVIGATION (no bounding boxes needed)
# ══════════════════════════════════════════════════════════════════════════════
def depth_navigate(depth_01: np.ndarray) -> tuple:
    """
    Pure depth-map navigation. Analyses left / center / right strips.
    Returns (command, approx_dist_m, msg, zone_depths) where
    zone_depths = (left_d, center_d, right_d) normalized 0–1.
    """
    if depth_01 is None:
        return "IDLE", 5.0, "Depth AI loading…", (0, 0, 0)

    h, w = depth_01.shape

    # Focus on lower 55% of frame (ground-level obstacles)
    lo = depth_01[int(h * 0.45) :]

    # Split into thirds
    l_strip = lo[:, : w // 3]
    c_strip = lo[:, w // 3 : 2 * w // 3]
    r_strip = lo[:, 2 * w // 3 :]

    # 85th percentile in each zone (robust to noise)
    dl = float(np.percentile(l_strip, 85))
    dc = float(np.percentile(c_strip, 85))
    dr = float(np.percentile(r_strip, 85))

    # Convert relative disparity to approximate meters
    # When disparity ≈ 1.0 → very close (~0.3m)
    # When disparity ≈ 0.0 → far (>5m)
    # Calibration: dist ≈ 0.5 / (disp + 0.1)  (tune WEBCAM_FOV_DEG if off)
    def disp_to_m(d):
        return round(min(5.0, max(0.2, 0.5 / (d + 0.08))), 2)

    dist_l = disp_to_m(dl)
    dist_c = disp_to_m(dc)
    dist_r = disp_to_m(dr)
    min_dist = min(dist_l, dist_c, dist_r)

    # Decision tree — slightly more sensitive thresholds
    if dc > 0.75:
        return "DANGER", dist_c, f"Very close obstacle! {dist_c:.1f}m", (dl, dc, dr)

    if dc > 0.52:
        if dl < dr - 0.08:
            return "LEFT", dist_c, f"Obstacle {dist_c:.1f}m — go left", (dl, dc, dr)
        if dr < dl - 0.08:
            return "RIGHT", dist_c, f"Obstacle {dist_c:.1f}m — go right", (dl, dc, dr)
        return "STOP", dist_c, f"Stop — blocked at {dist_c:.1f}m", (dl, dc, dr)

    if dc > 0.34:
        if dl < dr - 0.07:
            return "LEFT", dist_c, f"Caution {dist_c:.1f}m — lean left", (dl, dc, dr)
        if dr < dl - 0.07:
            return "RIGHT", dist_c, f"Caution {dist_c:.1f}m — lean right", (dl, dc, dr)
        return "STOP", dist_c, f"Slow — {dist_c:.1f}m ahead", (dl, dc, dr)

    return "CLEAR", max(dist_c, 2.0), "Path clear", (dl, dc, dr)


# ══════════════════════════════════════════════════════════════════════════════
#  HOG PERSON DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
print("[INIT] Loading HOG person detector…")
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("[INIT] HOG ready  ✅")


def compute_focal_px(frame_w: int) -> float:
    return frame_w / (2.0 * math.tan(math.radians(WEBCAM_FOV_DEG) / 2.0))


def estimate_dist_pinhole(label: str, box, frame_shape) -> float:
    x1, y1, x2, y2 = box
    bh = max(y2 - y1, 1)
    focal = compute_focal_px(frame_shape[1])
    real_h = REAL_HEIGHTS.get(label, 1.0)
    return round(float(np.clip((focal * real_h) / bh, 0.2, 6.0)), 2)


def detect_hog(frame) -> list:
    """Returns list of {label, conf, box, distance, source}."""
    sw = 640
    h, w = frame.shape[:2]
    sh = int(h * sw / w)
    sm = cv2.resize(frame, (sw, sh))

    boxes, weights = _hog.detectMultiScale(
        sm,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
        hitThreshold=0.0,
        useMeanshiftGrouping=True,
    )

    sx, sy = w / sw, h / sh
    results = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        x1 = int(x * sx)
        y1 = int(y * sy)
        x2 = int((x + bw) * sx)
        y2 = int((y + bh) * sy)
        conf = float(np.clip(np.squeeze(weights[i]) / 10.0, 0.3, 1.0))
        dist = estimate_dist_pinhole("person", (x1, y1, x2, y2), frame.shape)
        # Refine with depth map if available
        dm = get_depth_snapshot()
        if dm is not None:
            dm_h, dm_w = dm.shape
            # Sample depth at box center
            bx = int((x1 + x2) / 2 * dm_w / w)
            by = int((y1 + y2) / 2 * dm_h / h)
            bx = np.clip(bx, 0, dm_w - 1)
            by = np.clip(by, 0, dm_h - 1)
            rel_depth = float(dm[by, bx])
            if rel_depth > 0.05:
                # Weight pinhole 70% + depth 30%
                depth_dist = round(min(5.0, 0.5 / (rel_depth + 0.08)), 2)
                dist = round(dist * 0.70 + depth_dist * 0.30, 2)
        results.append(
            {
                "label": "person",
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "distance": dist,
                "source": "HOG",
            }
        )
    return results


def detect_depth_blobs(frame_shape, top_pct: float = 28.0) -> list:
    """
    Find close objects: pixels in the TOP top_pct% of depth values (closest).
    Uses adaptive percentile threshold so it works regardless of scene scale.
    Returns bounding boxes around any compact close region.
    """
    dm = get_depth_snapshot()
    if dm is None:
        return []

    h, w = frame_shape[:2]
    frame_area = h * w
    dm_full = cv2.resize(dm, (w, h))

    # ✅ FIXED: adaptive threshold — find pixels in top 28% closest
    # (percentile 72 = only the closest 28% of the scene passes)
    disp_thr = float(np.percentile(dm_full, 100 - top_pct))
    disp_thr = max(disp_thr, 0.30)  # never threshold below 0.30 (avoid noise)

    mask = (dm_full >= disp_thr).astype(np.uint8) * 255

    # Morphological cleanup
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    results = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        frac = area / frame_area
        if frac < 0.025 or frac > 0.75:  # skip tiny specks and huge flat background
            continue
        bw_box = stats[i, cv2.CC_STAT_WIDTH]
        bh_box = stats[i, cv2.CC_STAT_HEIGHT]
        if bh_box < 40 or bw_box / max(bh_box, 1) > 6:  # skip thin horizontal strips
            continue
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        x2, y2 = x1 + bw_box, y1 + bh_box

        avg_disp = float(np.mean(dm_full[labels == i]))
        # avg_disp is already 0-1 (normalized per frame)
        # Map: 1.0 disparity → ~0.3m  |  0.3 disparity → ~3m
        dist_m = round(max(0.3, min(5.0, 0.3 + (1.0 - avg_disp) * 4.5)), 2)
        conf = min(0.95, frac * 6)

        results.append(
            {
                "label": "obstacle",
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "distance": dist_m,
                "source": "depth",
            }
        )

    results.sort(key=lambda d: d["distance"])
    return results[:6]


def detect_edges_fallback(frame) -> list:
    """
    Fallback detector using Canny edges + contours.
    Works when depth model is unavailable or warming up.
    Detects large solid objects by their edges — not lighting-dependent.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Dilate edges to fill objects, then find contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        frac = area / frame_area
        if frac < 0.04 or frac > 0.80:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < 60:
            continue
        # Estimate distance by vertical position: lower = closer
        # Object bottom at 90% of frame height → ~0.5m; at 50% → ~2m
        bottom_frac = (y + bh) / h
        dist_m = round(max(0.3, min(4.0, 0.5 + (1.0 - bottom_frac) * 4.0)), 2)
        results.append(
            {
                "label": "obstacle",
                "conf": min(0.7, frac * 4),
                "box": (x, y, x + bw, y + bh),
                "distance": dist_m,
                "source": "edge",
            }
        )

    results.sort(key=lambda d: d["distance"])
    return results[:4]


# ══════════════════════════════════════════════════════════════════════════════
#  COLOURS / HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CMD_COLORS = {
    "CLEAR": (55, 215, 55),
    "STOP": (30, 30, 230),
    "DANGER": (0, 0, 255),
    "LEFT": (0, 215, 255),
    "RIGHT": (0, 215, 255),
    "IDLE": (140, 140, 140),
    "ERROR": (0, 0, 200),
}


def dist_color(d: float):
    if d < 0.4:
        return (0, 0, 255)
    if d < 0.8:
        return (0, 80, 255)
    if d < 1.5:
        return (0, 210, 255)
    if d < 3.0:
        return (55, 215, 55)
    return (110, 110, 110)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_depth_overlay(frame, alpha=0.32):
    """Draw depth heatmap as transparent overlay (red=close, blue=far)."""
    dm = get_depth_snapshot()
    if dm is None:
        return
    h, w = frame.shape[:2]
    dm_r = cv2.resize(dm, (w, h))
    dm_u8 = (dm_r * 255).astype(np.uint8)
    # COLORMAP_TURBO: dark purple = far, yellow-red = close
    colored = cv2.applyColorMap(dm_u8, cv2.COLORMAP_TURBO)
    cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(
        frame, "DEPTH", (w - 72, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
    )


def draw_zone_bars(frame, zones, h, w):
    """Draw left/center/right depth bar indicators at bottom."""
    dl, dc, dr = zones
    bh = 30
    bw = w // 5
    by = h - 70
    pad = 12
    for val, label, bx in [
        (dl, "L", pad),
        (dc, "C", (w - bw) // 2),
        (dr, "R", w - bw - pad),
    ]:
        fill = int(val * bw)
        col = dist_color(round(min(5.0, 0.5 / (val + 0.08)), 2))
        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), col, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(
            frame,
            label,
            (bx + 4, by + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def draw_detections(frame, detections: list):
    """
    Draw colored bounding boxes for ALL detections (HOG people + depth blobs).
    Color encodes distance: red=danger, orange=warning, yellow=caution, green=safe.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw far objects first so close ones render on top
    for d in sorted(detections, key=lambda x: x["distance"], reverse=True):
        x1, y1, x2, y2 = d["box"]
        dist = d["distance"]
        label = d["label"]
        color = dist_color(dist)
        thick = 3 if dist < 0.8 else 2

        # Semi-transparent fill for very close objects
        if dist < 0.7:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.14, frame, 0.86, 0, frame)

        # Box + corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        L = 16
        for px, py, dx, dy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (px, py), (px + dx * L, py), color, thick + 1)
            cv2.line(frame, (px, py), (px, py + dy * L), color, thick + 1)

        # Label badge with distance
        tag = f"{'⚠ ' if dist < 1.0 else ''}{label}  {dist:.1f} m"
        (tw, th), bl = cv2.getTextSize(tag, font, 0.50, 1)
        by1 = max(0, y1 - th - 10)
        ov2 = frame.copy()
        cv2.rectangle(ov2, (x1, by1), (x1 + tw + 10, y1), color, -1)
        cv2.addWeighted(ov2, 0.80, frame, 0.20, 0, frame)
        cv2.putText(frame, tag, (x1 + 5, y1 - 4), font, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame, tag, (x1 + 5, y1 - 4), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )


def draw_direction_arrow(frame, cmd, w, h):
    if cmd not in ("LEFT", "RIGHT"):
        return
    ax, ay = w // 2, h // 2
    size = 72
    color = CMD_COLORS[cmd]
    if cmd == "LEFT":
        pts = np.array(
            [
                [ax, ay - 26],
                [ax - size, ay],
                [ax, ay + 26],
                [ax, ay + 12],
                [ax + 36, ay + 12],
                [ax + 36, ay - 12],
                [ax, ay - 12],
            ],
            np.int32,
        )
    else:
        pts = np.array(
            [
                [ax, ay - 26],
                [ax + size, ay],
                [ax, ay + 26],
                [ax, ay + 12],
                [ax - 36, ay + 12],
                [ax - 36, ay - 12],
                [ax, ay - 12],
            ],
            np.int32,
        )
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], color)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_danger_border(frame, cmd, h, w):
    if cmd not in ("DANGER", "STOP"):
        return
    alpha = 0.40 + 0.40 * abs(math.sin(time.time() * 5))
    col = CMD_COLORS.get(cmd, (0, 0, 255))
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w - 1, h - 1), col, 8)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(
    frame,
    w,
    h,
    cmd,
    dist,
    obj,
    scene,
    ai_msg,
    status,
    fps,
    busy,
    cool,
    zones,
    hog_n,
    depth_ok,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cmd_col = CMD_COLORS.get(cmd, CMD_COLORS["IDLE"])

    def panel(x1, y1, x2, y2, a=0.65):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(ov, a, frame, 1 - a, 0, frame)

    def txt(text, x, y, color, scale=0.56, thick=1):
        cv2.putText(
            frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
        cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    # ── TOP-LEFT: command + distance ─────────────────────────────────────────
    panel(0, 0, 370, 130)
    cv2.putText(frame, cmd, (12, 82), font, 2.2, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, cmd, (12, 82), font, 2.2, cmd_col, 3, cv2.LINE_AA)
    dist_lbl = f"~{dist:.1f} m" if dist < 4.9 else "clear"
    txt(f"{dist_lbl}  |  {obj[:30]}", 12, 114, (220, 220, 220), 0.50)

    # ── TOP-RIGHT: FPS + status ───────────────────────────────────────────────
    panel(w - 280, 0, w, 110)
    txt(f"FPS  {fps:5.1f}", w - 268, 30, (220, 220, 220), 0.60)
    txt(f"Depth AI: {'✅' if depth_ok else '—'}", w - 268, 56, (180, 180, 180), 0.44)
    txt(f"Objects: {hog_n} detected", w - 268, 80, (180, 180, 180), 0.44)
    s_col = (
        (0, 0, 240)
        if status == "ERROR"
        else ((0, 200, 255) if busy else (120, 120, 120))
    )
    txt(status, w - 268, 104, s_col, 0.42)

    # ── AI cooldown bar ───────────────────────────────────────────────────────
    if 0.2 < cool < AI_INTERVAL_SEC:
        cw = int(cool / AI_INTERVAL_SEC * 240)
        cx = w // 2
        panel(cx - 128, 6, cx + 128, 30, a=0.72)
        cv2.rectangle(frame, (cx - 120, 10), (cx - 120 + cw, 26), (0, 200, 255), -1)
        txt(f"AI in {cool:.1f}s", cx - 38, 25, (15, 15, 15), 0.38)
    if busy:
        txt("[ SENDING TO AI ]", w // 2 - 72, 54, (0, 200, 255), 0.52)

    # ── Scene description (AI) ───────────────────────────────────────────────
    if scene:
        panel(0, h - 110, w, h - 72)
        scene_short = (scene[:95] + "…") if len(scene) > 95 else scene
        txt(f"📷 {scene_short}", 8, h - 84, (200, 255, 200), 0.44)

    # ── BOTTOM bar: AI message ────────────────────────────────────────────────
    panel(0, h - 72, w, h)
    ai_short = (ai_msg[:95] + "…") if len(ai_msg) > 95 else ai_msg
    txt(f"AI: {ai_short}", 8, h - 46, (255, 255, 255), 0.52)
    txt("[D] depth  [H] HOG  [S] AI  [Q] quit", 8, h - 16, (110, 110, 110), 0.38)


# ══════════════════════════════════════════════════════════════════════════════
#  TTS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
_tts_queue = _queue.Queue(maxsize=2)  # drop old messages if full
_last_spoken_t = 0.0
_last_spoken_txt = ""


def _tts_worker():
    """Background thread: pull messages from queue and speak them."""
    global _last_spoken_t, _last_spoken_txt

    _el_client = None
    if TTS_USE_ELEVENLABS:
        try:
            from elevenlabs.client import ElevenLabs

            _key = os.getenv("ELEVENLABS_API_KEY", "")
            if _key:
                _el_client = ElevenLabs(api_key=_key)
                print("[TTS] ElevenLabs client ready  ✅")
            else:
                print("[TTS] ELEVENLABS_API_KEY not set — falling back to macOS say")
        except Exception as e:
            print(f"[TTS] ElevenLabs init failed ({e}) — falling back to macOS say")

    while True:
        try:
            text = _tts_queue.get(timeout=1)
        except _queue.Empty:
            continue
        if text is None:  # sentinel to stop thread
            break
        try:
            if _el_client:
                # ElevenLabs streaming
                from elevenlabs import stream as el_stream

                audio = _el_client.text_to_speech.convert(
                    text=text,
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=ELEVENLABS_MODEL,
                    output_format="mp3_22050_32",
                )
                el_stream(audio)
            else:
                # macOS built-in TTS (free, instant)
                subprocess.run(
                    ["say", "-r", "175", text], capture_output=True, timeout=10
                )
        except Exception as e:
            print(f"[TTS] speak error: {e}")
            try:
                subprocess.run(["say", text], capture_output=True, timeout=8)
            except Exception:
                pass
        _tts_queue.task_done()


# Start TTS worker thread
_tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="TTS")
_tts_thread.start()


def speak(text: str, force: bool = False):
    """
    Queue a spoken message. Drops duplicates and respects cooldown.
    Set force=True to bypass cooldown (e.g. for DANGER alerts).
    """
    global _last_spoken_t, _last_spoken_txt
    if not TTS_ENABLED:
        return
    now = time.time()
    # Skip if same message was just spoken
    if text == _last_spoken_txt and not force:
        return
    # Skip if too recent (unless forced)
    if not force and (now - _last_spoken_t) < TTS_COOLDOWN_SEC:
        return
    _last_spoken_t = now
    _last_spoken_txt = text
    # Non-blocking: drop if queue is full (previous message still playing)
    try:
        _tts_queue.put_nowait(text)
    except _queue.Full:
        pass  # skip this message rather than block


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()
state = dict(
    command="IDLE",
    obstacle="—",
    distance=5.0,
    ai_message="Starting…",
    scene="",
    status="INIT",
    fps=0.0,
    api_busy=False,
    last_api_time=0.0,
    force_api=False,
)


# ══════════════════════════════════════════════════════════════════════════════
#  AI SERVER CALL (background thread)
# ══════════════════════════════════════════════════════════════════════════════
def call_api(frame_to_send: np.ndarray, dist_hint: float):
    with _lock:
        state["api_busy"] = True
        state["status"] = "SENDING"
    try:
        ok, buf = cv2.imencode(
            ".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not ok:
            raise RuntimeError("imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode()

        with _lock:
            state["status"] = "WAITING"

        resp = requests.post(
            f"{SERVER_URL}/analyze",
            json={"distance": dist_hint, "image": b64},
            timeout=45,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        d = resp.json()
        cmd = str(d.get("direction", d.get("command", "STOP"))).upper().strip()
        obj = str(d.get("obstacle", "unknown")).strip()
        msg = str(d.get("message", "?")).strip()
        scn = str(d.get("scene", "")).strip()
        try:
            ai_dist = float(d.get("distance", dist_hint))
        except (ValueError, TypeError):
            ai_dist = dist_hint

        if cmd not in {"CLEAR", "STOP", "LEFT", "RIGHT", "DANGER"}:
            cmd = "STOP"

        with _lock:
            state["command"] = cmd
            state["obstacle"] = obj
            state["distance"] = ai_dist
            state["ai_message"] = msg
            state["scene"] = scn
            state["status"] = "IDLE"
            state["last_api_time"] = time.time()
        print(f"[AI] ✅  cmd={cmd}  obj={obj}  dist={ai_dist}m  msg={msg}")
        print(f"[AI] 📷  scene: {scn}")
        # ── Speak the AI guidance out loud ────────────────────────────────────
        if msg:
            force_speak = cmd in ("DANGER", "STOP")
            speak(msg, force=force_speak)
    except Exception as e:
        print(f"[API] Error: {e}")
        with _lock:
            state["ai_message"] = f"AI error: {str(e)[:60]}"
            state["status"] = "ERROR"
            state["last_api_time"] = time.time()
    finally:
        with _lock:
            state["api_busy"] = False


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════
def startup_print(cap):
    print("\n" + "═" * 64)
    print("  SMART GLASSES  v3.0  |  3-layer detection")
    print("═" * 64)
    print(
        f"  Depth Anything V2: {'✅ loaded  (' + str(DEPTH_SIZE) + 'px)' if DEPTH_AI_OK else '❌ not found'}"
    )
    print(f"  HOG person detect: ✅ ready")
    print(f"  Gemini Flash AI:   {'✅ ' + SERVER_URL if True else '—'}")
    print()
    if cap.isOpened():
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fl = compute_focal_px(fw)
        print(f"  Camera #{CAMERA_INDEX}: {fw}×{fh}  focal≈{fl:.0f}px")
    print()
    print("  HOW DISTANCE WORKS:")
    print("  • Depth AI: measures relative depth of every pixel in scene")
    print("  • Pinhole: dist = focal × real_height / box_height_pixels")
    print("  • AI: Gemini estimates distance from visual cues in image")
    print("  • Combined: weighted average for best accuracy")
    print()
    tts_mode = "ElevenLabs" if TTS_USE_ELEVENLABS else "macOS say (free)"
    print(
        f"  TTS voice:         {'✅ ' + tts_mode if TTS_ENABLED else '❌ disabled (T to enable)'}"
    )
    print("  CONTROLS:  D=depth overlay  H=HOG boxes  S=force AI  T=TTS  Q=quit")
    print("═" * 64 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global SHOW_DEPTH, SHOW_HOG

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    startup_print(cap)

    if not cap.isOpened():
        print("[FATAL] Cannot open camera.")
        sys.exit(1)

    WIN = "Smart Glasses v3  |  D=depth  H=hog  S=AI  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    fps_count = 0
    fps_t = time.time()
    frame_n = 0
    hog_dets = []
    depth_dets = []
    all_dets = []
    last_cmd = ""
    depth_zones = (0.0, 0.0, 0.0)

    # Throttle: run HOG every N frames (slow), depth every M frames
    HOG_EVERY = 5  # HOG at ~6fps when main loop is 30fps
    DEPTH_EVERY = 3  # depth thread trigger (model itself throttles via flag)
    prev_cmd = ""  # track command changes for TTS
    prev_esp_cmd = ""  # track command changes for ESP32 hardware feedback
    last_esp_time = 0.0

    with _lock:
        state["status"] = "IDLE"
        state["command"] = "CLEAR"
        state["ai_message"] = "Initializing…"

    speak("Smart glasses ready. Scanning your surroundings.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        frame_n += 1

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_count += 1
        dt = time.time() - fps_t
        if dt >= 0.5:
            with _lock:
                state["fps"] = fps_count / dt
            fps_count = 0
            fps_t = time.time()

        # ── Depth estimation (background thread) ─────────────────────────────
        if DEPTH_AI_OK and frame_n % DEPTH_EVERY == 0 and not _depth_running:
            t = threading.Thread(target=_infer_depth, args=(frame.copy(),), daemon=True)
            t.start()

        # ── Depth-based navigation (runs every frame, uses latest depth map) ──
        dm = get_depth_snapshot()
        depth_cmd, depth_dist, depth_msg, depth_zones = depth_navigate(dm)

        # ── HOG person detection (every HOG_EVERY frames) ──────────────────
        if frame_n % HOG_EVERY == 0:
            hog_dets = detect_hog(frame)

        # ── Depth-blob detection + edge fallback ──────────────────────────
        if frame_n % DEPTH_EVERY == 0:
            if DEPTH_AI_OK and dm is not None:
                depth_dets = detect_depth_blobs(frame.shape)
            else:
                # Edge fallback: always draws boxes even without depth AI
                depth_dets = detect_edges_fallback(frame)

        # ── Merge: prefer HOG label for people, depth-blobs for everything else
        # Suppress depth blobs that heavily overlap with HOG people boxes
        merged_depth = []
        for db in depth_dets:
            dx1, dy1, dx2, dy2 = db["box"]
            overlaps = False
            for hp in hog_dets:
                px1, py1, px2, py2 = hp["box"]
                ix = max(0, min(dx2, px2) - max(dx1, px1))
                iy = max(0, min(dy2, py2) - max(dy1, py1))
                if ix * iy / max((dx2 - dx1) * (dy2 - dy1), 1) > 0.4:
                    overlaps = True
                    break
            if not overlaps:
                merged_depth.append(db)
        all_dets = hog_dets + merged_depth

        # ── Combined navigation decision ──────────────────────────────────────
        best_person_dist = min((d["distance"] for d in hog_dets), default=5.0)
        best_blob_dist = min((d["distance"] for d in depth_dets), default=5.0)
        overall_best = min(best_person_dist, best_blob_dist)

        if best_person_dist <= best_blob_dist and best_person_dist < 2.0:
            final_cmd = "STOP" if best_person_dist < 0.8 else depth_cmd
            final_dist = best_person_dist
            final_obj = "person"
        elif best_blob_dist < 2.0:
            final_cmd = depth_cmd
            final_dist = best_blob_dist
            final_obj = "obstacle"
        else:
            final_cmd = depth_cmd
            final_dist = depth_dist
            final_obj = "obstacle"

        # Feed local navigation into state (only when AI isn't fresher)
        with _lock:
            api_age = time.time() - state["last_api_time"]
            if api_age > 4.0:
                state["command"] = final_cmd
                state["obstacle"] = final_obj
                state["distance"] = final_dist
                if api_age > 8.0:
                    state["ai_message"] = f"[local] {depth_msg}"

        # ── Speak on LOCAL navigation change (when AI hasn't updated recently) ─
        with _lock:
            cur_cmd = state["command"]
            cur_msg = state["ai_message"]
            api_age_now = time.time() - state["last_api_time"]
        if api_age_now > 4.0 and cur_cmd != prev_cmd:
            # Speak the depth message for local navigation changes
            if cur_cmd == "DANGER":
                speak(depth_msg, force=True)
            elif cur_cmd in ("STOP", "LEFT", "RIGHT"):
                speak(depth_msg)
            elif cur_cmd == "CLEAR" and prev_cmd in ("STOP", "DANGER", "LEFT", "RIGHT"):
                speak("Path is clear. Continue forward.")
            prev_cmd = cur_cmd

        # ── AI call (every AI_INTERVAL_SEC always) ───────────────────────────
        with _lock:
            busy = state["api_busy"]
            last_api = state["last_api_time"]
            force = state["force_api"]
            if force:
                state["force_api"] = False

        cooldown = (time.time() - last_api) >= AI_INTERVAL_SEC
        if not busy and (force or cooldown):
            send = cv2.resize(frame, (640, 480))
            dist_hint = min(final_dist, best_person_dist)
            threading.Thread(
                target=call_api, args=(send, dist_hint), daemon=True
            ).start()

        # ── Snapshot display state ────────────────────────────────────────────
        with _lock:
            cmd = state["command"]
            dist = state["distance"]
            obj = state["obstacle"]
            scn = state["scene"]
            ai_msg = state["ai_message"]
            status = state["status"]
            fps = state["fps"]
            busy = state["api_busy"]
            last_t = state["last_api_time"]

        cool = max(0.0, AI_INTERVAL_SEC - (time.time() - last_t))

        # ── Send Command to ESP32 Hardware ────────────────────────────────────
        now = time.time()
        if cmd != prev_esp_cmd or (
            cmd not in ("CLEAR", "IDLE") and now - last_esp_time > 0.8
        ):
            prev_esp_cmd = cmd
            last_esp_time = now
            try:
                # Fire and forget request to ESP32 to trigger buzzer/motor
                threading.Thread(
                    target=lambda c: requests.get(
                        f"{ESP32_URL}/action?cmd={c}", timeout=0.5
                    ),
                    args=(cmd,),
                    daemon=True,
                ).start()
            except Exception:
                pass

        # ── Render ────────────────────────────────────────────────────────────
        display = frame.copy()

        if SHOW_DEPTH and DEPTH_AI_OK:
            draw_depth_overlay(display)

        draw_detections(display, all_dets)

        draw_zone_bars(display, depth_zones, h, w)
        draw_direction_arrow(display, cmd, w, h)
        draw_danger_border(display, cmd, h, w)
        draw_hud(
            display,
            w,
            h,
            cmd,
            dist,
            obj,
            scn,
            ai_msg,
            status,
            fps,
            busy,
            cool,
            depth_zones,
            len(hog_dets),
            DEPTH_AI_OK,
        )

        cv2.imshow(WIN, display)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("d"), ord("D")):
            SHOW_DEPTH = not SHOW_DEPTH
            print(f"[KEY] Depth overlay: {'ON' if SHOW_DEPTH else 'OFF'}")
        elif key in (ord("h"), ord("H")):
            SHOW_HOG = not SHOW_HOG
            print(f"[KEY] HOG boxes: {'ON' if SHOW_HOG else 'OFF'}")
        elif key in (ord("s"), ord("S")):
            with _lock:
                state["force_api"] = True
            print("[KEY] Force AI call")
        elif key in (ord("t"), ord("T")):
            TTS_ENABLED = not TTS_ENABLED
            status_txt = "ON" if TTS_ENABLED else "OFF"
            print(f"[KEY] TTS: {status_txt}")
            if TTS_ENABLED:
                speak("Voice guidance enabled.", force=True)

    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Stopped.")


if __name__ == "__main__":
    main()
