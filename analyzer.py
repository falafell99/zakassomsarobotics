"""
OpenRouter Vision Analyzer module for smart blind-assistance glasses.
Handles communication with OpenRouter API for image analysis.

MODEL NOTES:
  - meta-llama/llama-3.2-11b-vision-instruct:free  → free but unreliable JSON,
    often returns empty, rate-limited. Use only for quick tests.
  - google/gemini-flash-1.5                        → fast, reliable, good vision.
  - google/gemini-pro-1.5                          → best quality, slower.
  - openai/gpt-4o-mini                             → reliable JSON, fast, cheap.
  - anthropic/claude-3-haiku                       → very reliable JSON output.

  Recommendation: use "google/gemini-flash-1.5" or "openai/gpt-4o-mini"
  for reliable object detection. Free Llama is not suitable for production use.
"""

import os
import re
import logging
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ── OpenRouter client ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client  = None

if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info("OpenRouter client initialized")
    except Exception as e:
        logger.error(f"OpenRouter init failed: {e}")
else:
    logger.warning("OPENROUTER_API_KEY not set")


# ── Model selection ───────────────────────────────────────────────────────────
# Change this to switch models without touching anything else.
# Reliable choices (in order of recommendation):
#   "google/gemini-flash-1.5"      ← recommended: fast + reliable JSON
#   "openai/gpt-4o-mini"           ← very reliable JSON
#   "anthropic/claude-3-haiku"     ← reliable JSON
#   "meta-llama/llama-3.2-11b-vision-instruct:free"  ← free but unreliable
DEFAULT_MODEL = "google/gemini-flash-1.5"


# ── System prompt ─────────────────────────────────────────────────────────────
# IMPORTANT: explicit distance thresholds prevent the model from guessing.
SYSTEM_PROMPT = """You are a navigation assistant for a visually impaired person wearing smart glasses with a camera and ultrasonic distance sensor.

Your job: analyze the camera image and the distance reading, then output EXACTLY one JSON object with no extra text, no markdown, no explanation.

REQUIRED OUTPUT FORMAT (copy exactly, fill in the values):
{"direction": "STOP", "message": "short message", "obstacle": "object name"}

DISTANCE RULES (apply first, before looking at the image):
- distance > 1.5m  → direction MUST be "CLEAR", obstacle "none", message "Path is clear"
- distance < 0.3m  → direction MUST be "DANGER", message "Stop! Very close!"
- 0.3m to 1.5m     → analyze the image and choose STOP / LEFT / RIGHT

DIRECTION RULES (only when distance is 0.3m–1.5m):
- Obstacle fills most of the frame or blocks the path → "STOP"
- Obstacle is more on the LEFT side of frame → "RIGHT" (steer right to avoid it)
- Obstacle is more on the RIGHT side of frame → "LEFT" (steer left to avoid it)
- Path looks clear in the image despite sensor reading → "STOP" (be conservative)

MESSAGE RULES:
- Maximum 8 words
- Must say what the obstacle is and what to do
- Examples: "Turn right, chair ahead" / "Stop, person in path" / "Go left, table on right"

OBSTACLE FIELD:
- Name the specific object you see (chair, person, door, wall, table, car, etc.)
- If nothing identifiable, write "obstacle"

CRITICAL: output ONLY the JSON object. Nothing before it. Nothing after it."""


# ── Helper: extract JSON from whatever the model returns ─────────────────────
def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Try multiple strategies to get a valid JSON dict from `raw`.
    Returns the parsed dict, or raises ValueError if all fail.
    """
    raw = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json ... ```
    m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: ``` ... ```
    m = re.search(r'```\s*(.*?)\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 4: first { ... } block (handles extra prose before/after)
    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in: {raw[:300]}")


# ── Main function ─────────────────────────────────────────────────────────────
async def analyze_image_with_vision(
    base64_image: str,
    distance: float,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Analyze an image using OpenRouter vision model.

    Args:
        base64_image : raw base64 JPEG string (no data:// prefix)
        distance     : ultrasonic distance reading in meters
        model        : OpenRouter model identifier

    Returns:
        {"direction": str, "message": str, "obstacle": str}
    """
    _safe_default = {"direction": "STOP", "message": "Obstacle ahead, stop", "obstacle": "unknown"}

    if not openrouter_client:
        logger.error("OpenRouter client not initialized")
        return _safe_default

    # Fast-path: pure distance decision (no vision call needed)
    if distance > 1.5:
        logger.info(f"dist={distance:.2f}m > 1.5m → CLEAR (skipped AI)")
        return {"direction": "CLEAR", "message": "Path is clear", "obstacle": "none"}
    if distance < 0.3:
        logger.info(f"dist={distance:.2f}m < 0.3m → DANGER (skipped AI)")
        return {"direction": "DANGER", "message": "Danger! Stop now!", "obstacle": "too close"}

    try:
        logger.info(f"Calling {model}  dist={distance:.2f}m")

        image_url = f"data:image/jpeg;base64,{base64_image}"

        # NOTE: response_format=json_object is intentionally NOT used here.
        # Many OpenRouter vision models ignore or fail on that parameter.
        # We parse JSON manually with fallbacks instead.
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Distance sensor: {distance:.2f} meters. "
                                "Analyze the image. Output ONLY the JSON object."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            max_tokens=150,    # JSON is short; limit prevents rambling
            temperature=0.1,   # very low for deterministic JSON
        )

        content = response.choices[0].message.content

        # ── Always print raw response for debugging ───────────────────────────
        print("\n" + "─" * 55)
        print(f"[ANALYZER] model  : {model}")
        print(f"[ANALYZER] dist   : {distance:.2f}m")
        print(f"[ANALYZER] raw    : {content}")
        print("─" * 55 + "\n")

        if not content or not content.strip():
            logger.error("Model returned empty response!")
            print("[ANALYZER] ⚠  Empty response from model!")
            return _safe_default

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            result = _extract_json(content)
        except ValueError as parse_err:
            logger.error(f"JSON extraction failed: {parse_err}")
            print(f"[ANALYZER] ❌ Could not extract JSON: {parse_err}")
            return _safe_default

        direction = result.get("direction", "STOP").upper().strip()
        message   = result.get("message",   "Obstacle ahead").strip()
        obstacle  = result.get("obstacle",  "unknown").strip()

        # Validate direction
        valid = {"LEFT", "RIGHT", "STOP", "CLEAR", "DANGER"}
        if direction not in valid:
            logger.warning(f"Invalid direction '{direction}' → STOP")
            direction = "STOP"

        # Truncate message to 8 words
        words = message.split()
        if len(words) > 8:
            message = " ".join(words[:8])

        final = {"direction": direction, "message": message, "obstacle": obstacle}
        logger.info(f"Result: {final}")
        print(f"[ANALYZER] ✅ Parsed: {final}")
        return final

    except Exception as e:
        logger.error(f"OpenRouter call failed: {e}")
        print(f"[ANALYZER] ❌ API error: {e}")
        return _safe_default


# ── Connection test ───────────────────────────────────────────────────────────
async def test_openrouter_connection() -> bool:
    if not openrouter_client:
        return False
    try:
        response = openrouter_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Reply with the word OK only."}],
            max_tokens=5,
        )
        reply = response.choices[0].message.content or ""
        logger.info(f"Connection test reply: '{reply}'")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
