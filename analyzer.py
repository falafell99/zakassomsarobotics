"""
OpenRouter Vision Analyzer — Smart Blind-Assistance Glasses
============================================================
Upgraded to use Google Gemini Flash for full SCENE UNDERSTANDING,
not just obstacle detection. Runs every 3 seconds always.
"""

import os
import re
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client  = None

if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    except Exception as e:
        logger.error(f"OpenRouter init failed: {e}")

DEFAULT_MODEL = "google/gemini-flash-1.5"

# ── System prompt — scene UNDERSTANDING, not just obstacle ping ───────────────
SYSTEM_PROMPT = """You are the AI vision system for smart glasses worn by a COMPLETELY BLIND person who is actively walking.

They cannot see anything at all. Your guidance is their only way to navigate safely.

OUTPUT: ONLY this JSON object. No other text. No markdown. No explanation.
{
  "direction": "CLEAR or STOP or LEFT or RIGHT or DANGER",
  "obstacle": "specific object name (e.g. chair, person, wall, table, door, car)",
  "distance": "estimated distance as a number in meters (e.g. 1.5)",
  "message": "spoken instruction, max 8 words, specific and urgent",
  "scene": "one sentence describing the full environment"
}

HOW TO ESTIMATE DISTANCE FROM THE IMAGE:
- Average adult: 1.7m tall. If they fill 50% of frame height → ~1.7m. 20% → ~4m. 80% → ~1m.
- Chair: 0.9m tall. Table: 0.75m. Car: 1.5m tall.
- If object fills bottom half of frame → under 1 meter — VERY CLOSE
- If object is small and far in background → more than 3 meters
- Use shadows, perspective lines, and relative sizes for depth cues

DIRECTION RULES (apply strictly):
- Center path clearly blocked → STOP
- Close obstacle on LEFT third of frame, right side clear → RIGHT
- Close obstacle on RIGHT third of frame, left side clear → LEFT  
- Object within 0.5m or filling >60% of frame → DANGER
- No obstacles within 2m → CLEAR

OBSTACLE: Name the SPECIFIC closest object blocking the path.
- Good: "wooden chair", "glass door", "parked car", "person in black jacket"
- Bad: "object", "thing", "something"

DISTANCE: Your best estimate in meters as a decimal number.
- Be specific: 0.8, 1.5, 2.3 — not "close" or "far"

MESSAGE: Spoken aloud to the blind person. Max 8 words. Be specific and urgent.
- Good: "Chair 1 meter ahead, step right"  
- Good: "Person blocking path, stop now"
- Good: "Clear path, continue forward safely"
- Bad: "Obstacle detected" (too vague)
- Bad: "Be careful" (no actionable guidance)

SCENE: Full one-sentence description of the environment for context.
Examples:
- "Indoor office hallway with chairs on the left and a glass wall ahead"
- "Outdoor sidewalk with a parked car on the right and clear path ahead"
- "Kitchen with a table in the center and person standing near the counter"
"""


# ── JSON extraction with fallbacks ────────────────────────────────────────────
def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ```
    for pat in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # First { ... } block
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in: {raw[:300]}")


# ── Main analysis function ────────────────────────────────────────────────────
async def analyze_image_with_vision(
    base64_image: str,
    distance: float = 2.0,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Send image to Gemini for scene understanding and navigation guidance.
    Now returns scene description in addition to navigation command.
    """
    _default = {
        "direction": "STOP",
        "obstacle":  "unknown",
        "distance":  str(round(distance, 1)),
        "message":   "Cannot analyze — stop to be safe",
        "scene":     "Scene analysis unavailable",
    }

    if not openrouter_client:
        return _default

    try:
        image_url = f"data:image/jpeg;base64,{base64_image}"

        response = openrouter_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this image. "
                                f"Ultrasonic sensor distance: {distance:.2f}m (use as reference if available, otherwise estimate from image). "
                                f"Output ONLY the JSON object."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                    ]
                }
            ],
            max_tokens=220,
            temperature=0.1,
        )

        content = response.choices[0].message.content or ""

        print(f"\n{'─'*50}")
        print(f"[AI] model: {model}")
        print(f"[AI] raw:   {content}")
        print(f"{'─'*50}\n")

        if not content.strip():
            print("[AI] Empty response!")
            return _default

        parsed = _extract_json(content)

        direction = str(parsed.get("direction", "STOP")).upper().strip()
        obstacle  = str(parsed.get("obstacle",  "unknown")).strip()
        dist_str  = str(parsed.get("distance",  str(round(distance, 1)))).strip()
        message   = str(parsed.get("message",   "Obstacle ahead")).strip()
        scene     = str(parsed.get("scene",     "")).strip()

        if direction not in {"CLEAR", "STOP", "LEFT", "RIGHT", "DANGER"}:
            direction = "STOP"

        # Truncate message to 8 words
        words = message.split()
        if len(words) > 8:
            message = " ".join(words[:8])

        # Parse distance string to float
        try:
            dist_float = float(re.search(r'\d+\.?\d*', dist_str).group())
        except Exception:
            dist_float = distance

        result = {
            "direction": direction,
            "obstacle":  obstacle,
            "distance":  dist_float,
            "message":   message,
            "scene":     scene,
        }
        print(f"[AI] ✅ {result}")
        return result

    except Exception as e:
        print(f"[AI] ❌ Error: {e}")
        return _default


async def test_openrouter_connection() -> bool:
    if not openrouter_client:
        return False
    try:
        r = openrouter_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Reply OK only."}],
            max_tokens=5,
        )
        return bool(r.choices[0].message.content)
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
