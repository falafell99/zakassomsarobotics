import math

import config
from schemas import Detection

PATH_CLOSE_M = 1.8
DANGER_M = 0.45
STOP_M = 0.9


# COCO names plus common indoor/outdoor obstacle aliases.
REAL_HEIGHTS_M = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "airplane": 5.00,
    "bus": 3.20,
    "train": 3.60,
    "truck": 2.80,
    "boat": 2.00,
    "traffic light": 2.20,
    "fire hydrant": 0.75,
    "stop sign": 2.10,
    "parking meter": 1.30,
    "bench": 0.90,
    "desk": 0.75,
    "office chair": 0.95,
    "filing cabinet": 1.20,
    "printer": 0.45,
    "monitor": 0.55,
    "phone": 0.15,
    "whiteboard": 1.20,
    "trash can": 0.65,
    "laundry basket": 0.55,
    "box": 0.45,
    "storage box": 0.45,
    "moving box": 0.55,
    "shoe rack": 0.65,
    "bookshelf": 1.80,
    "wardrobe": 1.90,
    "cabinet": 1.40,
    "drawer": 0.75,
    "nightstand": 0.55,
    "coffee table": 0.45,
    "side table": 0.55,
    "floor lamp": 1.60,
    "lamp": 0.55,
    "rug": 0.03,
    "doorway": 2.00,
    "railing": 1.00,
    "handrail": 1.00,
    "banister": 1.00,
    "staircase": 1.00,
    "steps": 0.25,
    "step": 0.18,
    "window": 1.20,
    "pet": 0.45,
    "cat": 0.30,
    "dog": 0.55,
    "horse": 1.60,
    "sheep": 0.90,
    "cow": 1.45,
    "elephant": 3.00,
    "bear": 1.40,
    "zebra": 1.35,
    "giraffe": 4.50,
    "backpack": 0.55,
    "umbrella": 0.90,
    "handbag": 0.35,
    "suitcase": 0.70,
    "sports ball": 0.22,
    "chair": 0.95,
    "couch": 0.85,
    "sofa": 0.85,
    "potted plant": 0.90,
    "plant": 0.90,
    "bed": 0.80,
    "dining table": 0.75,
    "toilet": 0.75,
    "tv": 0.70,
    "laptop": 0.25,
    "keyboard": 0.05,
    "mouse": 0.04,
    "cable": 0.02,
    "charger": 0.04,
    "paper": 0.01,
    "notebook": 0.03,
    "cell phone": 0.15,
    "cup": 0.12,
    "bottle": 0.28,
    "microwave": 0.35,
    "oven": 0.85,
    "sink": 0.85,
    "refrigerator": 1.70,
    "book": 0.25,
    "vase": 0.35,
    "door": 2.00,
    "wall": 2.00,
    "stairs": 1.00,
    "railing": 1.00,
}

WALKING_RELEVANT = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "box", "pet", "cat", "dog", "horse", "sheep", "cow", "backpack", "umbrella",
    "handbag", "suitcase", "sports ball", "chair", "office chair", "sofa",
    "couch", "desk", "filing cabinet", "trash can", "plant", "potted plant",
    "bed", "dining table", "toilet", "monitor", "tv", "microwave", "oven",
    "sink", "refrigerator", "vase", "door", "doorway", "wall", "stairs",
    "railing", "handrail", "banister", "staircase", "steps", "step",
    "laundry basket", "storage box", "moving box", "shoe rack", "bookshelf",
    "wardrobe", "cabinet", "drawer", "nightstand", "coffee table",
    "side table", "floor lamp", "lamp", "rug", "window",
}

LOW_PRIORITY = {
    "book", "cell phone", "phone", "remote", "keyboard", "mouse", "cup", "bottle",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "orange",
    "sandwich", "pizza", "cake", "pen", "pencil", "paper", "notebook",
    "charger", "cable",
}

OFFICE_CONTEXT = {
    "desk", "monitor", "laptop", "keyboard", "mouse", "phone", "book", "cup",
    "bottle", "printer", "whiteboard", "filing cabinet", "office chair",
    "trash can", "plant",
    "whiteboard", "printer",
}

HOME_CONTEXT = {
    "sofa", "bed", "coffee table", "side table", "nightstand", "wardrobe",
    "cabinet", "drawer", "bookshelf", "shoe rack", "laundry basket",
    "storage box", "moving box", "floor lamp", "lamp", "rug", "door",
    "doorway", "wall", "window",
}

STRUCTURE_CRITICAL = {
    "wall", "stairs", "staircase", "steps", "step", "door", "doorway",
}

STRUCTURE_CONTEXT = {
    "railing", "handrail", "banister", "window",
}


def position_from_bbox(bbox: list[int] | None) -> str:
    if not bbox:
        return "center"
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    if center_x < 360:
        return "left"
    if center_x > 640:
        return "right"
    return "center"


def bbox_area(bbox: list[int] | None) -> float:
    if not bbox:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1) / 1_000_000


def horizontal_overlap(bbox: list[int] | None, left: int, right: int) -> float:
    if not bbox:
        return 0.0
    x1, _, x2, _ = bbox
    overlap = max(0, min(x2, right) - max(x1, left))
    width = max(1, x2 - x1)
    return overlap / width


def bbox_center_x(bbox: list[int] | None) -> float:
    if not bbox:
        return 500.0
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2.0


def bbox_width(bbox: list[int] | None) -> int:
    if not bbox:
        return 0
    x1, _, x2, _ = bbox
    return max(0, x2 - x1)


def center_corridor_overlap(bbox: list[int] | None) -> float:
    return horizontal_overlap(bbox, config.CENTER_CORRIDOR_LEFT, config.CENTER_CORRIDOR_RIGHT)


def is_low_priority_context(det: Detection) -> bool:
    """Small tabletop items should describe the scene, not steer the user."""
    if det.label not in LOW_PRIORITY:
        return False
    area = bbox_area(det.bbox)
    corridor = center_corridor_overlap(det.bbox)
    return det.distance > STOP_M and (area < 0.12 or corridor < 0.55)


def estimate_distance_from_bbox(label: str, bbox: list[int] | None) -> float:
    """Estimate distance from bbox height using pinhole + bottom-position cues."""
    if not bbox:
        return 5.0

    _, y1, _, y2 = bbox
    label = label.lower()
    box_h = max(1, y2 - y1) / 1000.0
    bottom = max(0.0, min(1.0, y2 / 1000.0))
    real_h = REAL_HEIGHTS_M.get(label, 1.0)

    fov_rad = math.radians(config.CAMERA_VERTICAL_FOV_DEG)
    focal_norm = 1.0 / (2.0 * math.tan(fov_rad / 2.0))
    pinhole_m = (real_h * focal_norm) / box_h

    # Ground-contact cue: objects whose lower edge is near the image bottom are
    # usually close even when the class height estimate is imperfect.
    bottom_m = 0.45 + (1.0 - bottom) * 5.0

    if label == "person":
        distance = 0.82 * pinhole_m + 0.18 * bottom_m
    elif label == "rug":
        distance = max(0.6, bottom_m)
    elif label in {"cable", "charger", "paper", "notebook"}:
        distance = max(0.35, bottom_m)
    elif label in {"chair", "office chair", "bench", "couch", "sofa", "desk", "dining table", "table", "plant", "potted plant", "coffee table", "side table", "nightstand"}:
        distance = 0.58 * pinhole_m + 0.42 * bottom_m
    elif label in {"wall", "door", "doorway", "wardrobe", "bookshelf", "cabinet", "whiteboard", "window", "railing", "handrail", "banister"}:
        distance = 0.72 * pinhole_m + 0.28 * bottom_m
    elif label in {"stairs", "staircase", "steps", "step"}:
        distance = 0.42 * pinhole_m + 0.58 * bottom_m
    elif label in OFFICE_CONTEXT:
        distance = 0.50 * pinhole_m + 0.50 * bottom_m
    else:
        distance = 0.68 * pinhole_m + 0.32 * bottom_m

    distance *= config.DISTANCE_SCALE
    return round(max(0.25, min(8.0, distance)), 2)


def fuse_sensor_distance(det: Detection, sensor_distance: float) -> float:
    if sensor_distance <= 0 or sensor_distance > config.SENSOR_VALID_MAX_M:
        return det.distance
    if det.position == "center":
        return round(0.55 * sensor_distance + 0.45 * det.distance, 2)
    if bbox_area(det.bbox) > 0.18:
        return round(0.30 * sensor_distance + 0.70 * det.distance, 2)
    return det.distance


def side_clearances(detections: list[Detection]) -> dict[str, float]:
    clearances = {"left": 5.0, "center": 5.0, "right": 5.0}
    for det in detections:
        if det.position in clearances:
            clearances[det.position] = min(clearances[det.position], det.distance)
    return clearances


def zone_risks(detections: list[Detection]) -> dict[str, float]:
    zones = {"left": 0.0, "center": 0.0, "right": 0.0}
    spans = {
        "left": (0, config.CENTER_CORRIDOR_LEFT),
        "center": (config.CENTER_CORRIDOR_LEFT, config.CENTER_CORRIDOR_RIGHT),
        "right": (config.CENTER_CORRIDOR_RIGHT, 1000),
    }
    for det in detections:
        for zone, (left, right) in spans.items():
            zones[zone] += det.risk * horizontal_overlap(det.bbox, left, right)
    return {zone: round(score, 3) for zone, score in zones.items()}


def action_for(det: Detection, clearances: dict[str, float], risks: dict[str, float]) -> str:
    if det.label in {"stairs", "staircase", "steps", "step"}:
        return "STOP" if det.distance <= 2.4 and center_corridor_overlap(det.bbox) >= 0.25 else "CLEAR"

    if det.label in {"railing", "handrail", "banister"}:
        if det.position == "center" and det.distance <= STOP_M:
            return "STOP"
        return "CLEAR"

    if det.label == "rug":
        return "STOP" if det.distance <= STOP_M and center_corridor_overlap(det.bbox) >= 0.35 else "CLEAR"

    if det.label == "wall":
        corridor_overlap = center_corridor_overlap(det.bbox)
        if corridor_overlap >= 0.35 and det.distance <= 2.4:
            if det.distance <= STOP_M or bbox_width(det.bbox) >= 760:
                return "STOP"
            return "RIGHT" if bbox_center_x(det.bbox) < 520 else "LEFT"
        return "CLEAR"

    if det.label in LOW_PRIORITY:
        return "STOP" if det.distance <= STOP_M and center_corridor_overlap(det.bbox) >= 0.55 else "CLEAR"

    if det.label in OFFICE_CONTEXT and det.distance > PATH_CLOSE_M and center_corridor_overlap(det.bbox) < 0.45:
        return "CLEAR"
    if det.label in HOME_CONTEXT and det.label not in {"wall", "door", "doorway", "stairs"} and det.distance > PATH_CLOSE_M and center_corridor_overlap(det.bbox) < 0.35:
        return "CLEAR"

    if det.distance <= DANGER_M:
        return "DANGER"

    corridor_overlap = center_corridor_overlap(det.bbox)
    blocks_path = det.position == "center" or corridor_overlap >= 0.35

    if blocks_path:
        if det.distance <= STOP_M:
            return "STOP"
        if det.distance <= PATH_CLOSE_M:
            left_clear = clearances["left"]
            right_clear = clearances["right"]
            left_risk = risks.get("left", 0.0)
            right_risk = risks.get("right", 0.0)
            if left_clear >= 1.4 and left_risk + 0.25 < right_risk:
                return "LEFT"
            if right_clear >= 1.4 and right_risk + 0.25 < left_risk:
                return "RIGHT"
            if left_clear >= 1.4 and left_clear > right_clear + 0.25:
                return "LEFT"
            if right_clear >= 1.4 and right_clear > left_clear + 0.25:
                return "RIGHT"
            if left_clear >= 1.4 and right_clear >= 1.4:
                return "RIGHT" if bbox_center_x(det.bbox) < 520 else "LEFT"
            return "STOP"
        return "CLEAR"

    if det.distance <= STOP_M:
        return "STOP"
    if det.distance <= PATH_CLOSE_M:
        return "RIGHT" if det.position == "left" else "LEFT"
    return "CLEAR"


def priority_for(det: Detection) -> float:
    label = det.label.lower()
    area = bbox_area(det.bbox)
    corridor = center_corridor_overlap(det.bbox)
    if label in LOW_PRIORITY and det.distance > 1.0 and area < 0.08:
        return round(0.15 + min(0.5, det.confidence * 0.5), 3)
    relevance = 1.0 if label in WALKING_RELEVANT else 0.45
    if label in OFFICE_CONTEXT or label in HOME_CONTEXT or label in STRUCTURE_CONTEXT:
        relevance = max(relevance, 0.65)
    if label in STRUCTURE_CRITICAL:
        relevance = 1.25
    if label in {"stairs", "staircase", "steps", "step"}:
        relevance = 1.45
    center_bonus = {"center": 0.45, "left": 0.18, "right": 0.18}.get(det.position, 0.0)
    corridor_bonus = corridor * 0.55
    close_score = max(0.0, 2.7 - det.distance)
    area_score = min(0.7, area * 3.0)
    confidence_score = min(0.4, det.confidence * 0.4)
    return round(relevance + center_bonus + corridor_bonus + close_score + area_score + confidence_score, 3)


def enrich_detections(detections: list[Detection], sensor_distance: float = 5.0) -> list[Detection]:
    prepared = []
    for det in detections:
        det.label = det.label.lower().strip()
        if bbox_area(det.bbox) < config.MIN_BOX_AREA and det.label not in {"person", "dog", "cat", "phone", "mouse"}:
            continue
        det.position = position_from_bbox(det.bbox)
        det.distance = estimate_distance_from_bbox(det.label, det.bbox)
        det.distance = fuse_sensor_distance(det, sensor_distance)
        prepared.append(det)

    clearances = side_clearances(prepared)
    for det in prepared:
        det.risk = priority_for(det)
    risks = zone_risks(prepared)
    enriched = []
    for det in prepared:
        det.action = action_for(det, clearances, risks)
        det.priority = det.risk
        if det.priority > 0.0:
            enriched.append(det)

    enriched.sort(key=lambda item: (-item.priority, item.distance))
    return enriched


def decision_reason(command: str, obstacle: str, distance: float, zones: dict[str, float]) -> str:
    if command == "CLEAR":
        return "no close walking-path obstacles"
    if command == "DANGER":
        return f"{obstacle} is extremely close"
    if command == "STOP":
        if obstacle in {"stairs", "staircase", "steps", "step"}:
            return f"{obstacle} detected in walking path at {distance:.1f}m"
        if obstacle == "wall":
            return f"wall or solid surface ahead at {distance:.1f}m"
        return f"{obstacle} blocks the center path at {distance:.1f}m"
    if command == "LEFT":
        return f"right/center risk is higher; left side is safer"
    if command == "RIGHT":
        return f"left/center risk is higher; right side is safer"
    return f"zone risks: {zones}"


def choose_primary(detections: list[Detection], sensor_distance: float) -> tuple[str, str, str, float, str, dict[str, float]]:
    zones = zone_risks(detections)
    if 0 < sensor_distance <= DANGER_M:
        return (
            "DANGER",
            "Obstacle very close, stop now",
            "too close",
            sensor_distance,
            "ultrasonic sensor reports immediate obstacle",
            zones,
        )

    relevant = []
    for det in detections:
        if is_low_priority_context(det):
            continue
        blocks_center_path = center_corridor_overlap(det.bbox) >= 0.45
        if det.label in WALKING_RELEVANT or det.priority >= 1.2:
            relevant.append(det)
        elif det.distance <= PATH_CLOSE_M and blocks_center_path:
            relevant.append(det)

    if not relevant:
        if 0 < sensor_distance <= STOP_M:
            return (
                "STOP",
                f"Obstacle {sensor_distance:.1f} meters ahead",
                "obstacle",
                sensor_distance,
                "ultrasonic sensor reports close obstacle",
                zones,
            )
        return "CLEAR", "Path clear", "none", sensor_distance, decision_reason("CLEAR", "none", sensor_distance, zones), zones

    primary = sorted(relevant, key=lambda item: (-item.priority, item.distance))[0]
    command = primary.action
    label = primary.label
    distance = primary.distance

    if command == "DANGER":
        message = f"{label} very close, stop now"
    elif label in {"stairs", "staircase", "steps", "step"} and command == "STOP":
        message = f"Stairs {distance:.1f} meters ahead, stop"
    elif label in {"railing", "handrail", "banister"}:
        message = f"Railing {primary.position}, keep walking carefully"
    elif label == "wall" and command == "STOP":
        message = f"Wall {distance:.1f} meters ahead, stop"
    elif command == "STOP":
        message = f"{label} {distance:.1f} meters ahead, stop"
    elif command == "LEFT":
        message = f"{label} {distance:.1f} meters ahead, go left"
    elif command == "RIGHT":
        message = f"{label} {distance:.1f} meters ahead, go right"
    else:
        message = "Path clear"

    return command, message, label, round(distance, 2), decision_reason(command, label, distance, zones), zones
