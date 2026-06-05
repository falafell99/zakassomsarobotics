"""
OpenAI Vision Analyzer — Smart Blind-Assistance Glasses
==============================================================
Uses the official OpenAI Python SDK.
Requires OPENAI_API_KEY in your .env file.
Model used: gpt-4o-mini (Extremely fast, cheap, and vision-capable)
"""

import os
import re
import json
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── OpenAI client setup ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_openai_client = None

if OPENAI_API_KEY:
    try:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        print("[AI] ✅ OpenAI Vision (GPT-4o) ready")
    except ImportError:
        print("[AI] ❌ OpenAI SDK not installed. Run: pip install openai")
    except Exception as e:
        print(f"[AI] ❌ OpenAI init failed: {e}")
else:
    print("[AI] ⚠️  OPENAI_API_KEY not set in .env")
    print("[AI]    Add to .env:  OPENAI_API_KEY=sk-proj-...")

OPENAI_MODEL = "gpt-4o-mini"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the AI vision system for smart glasses worn by a COMPLETELY BLIND person who is actively walking.

They cannot see anything at all. Your guidance is their only way to navigate safely.

OUTPUT: ONLY this JSON object. No other text. No markdown. No explanation.
{
  "direction": "CLEAR or STOP or LEFT or RIGHT or DANGER",
  "obstacle": "specific object name (e.g. chair, person, wall, table, door, car)",
  "distance": 1.5,
  "message": "spoken instruction, max 8 words, specific and urgent",
  "scene": "one sentence describing the full environment"
}

HOW TO ESTIMATE DISTANCE FROM THE IMAGE:
- Average adult: 1.7m tall. If they fill 50% of frame → ~1.7m. 20% → ~4m. 80% → ~1m.
- Chair: 0.9m tall. Table: 0.75m. Car: 1.5m tall.
- Object fills bottom half of frame → under 1 meter — VERY CLOSE
- Object small/far in background → more than 3 meters
- Use shadows, perspective lines, and relative sizes for depth cues

DIRECTION RULES (apply strictly):
- Center path clearly blocked → STOP
- Close obstacle on LEFT third of frame, right side clear → RIGHT
- Close obstacle on RIGHT third of frame, left side clear → LEFT
- Object within 0.5m or filling >60% of frame → DANGER
- No obstacles within 2m → CLEAR

OBSTACLE: Name the SPECIFIC closest object blocking the path.
DISTANCE: Your best estimate in meters as a decimal number (e.g. 1.5).
MESSAGE: Spoken aloud. Max 8 words. Specific and urgent.
SCENE: One sentence describing the full environment."""


# ── JSON extraction with fallbacks ────────────────────────────────────────────
def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in: {raw[:300]}")


# ── Main analysis function ─────────────────────────────────────────────────────
async def analyze_image_with_vision(
    base64_image: str,
    distance: float = 2.0,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    Send image to OpenAI for scene understanding and navigation.
    """
    _default = {
        "direction": "STOP",
        "obstacle": "unknown",
        "distance": round(distance, 1),
        "message": "Cannot analyze — proceed with caution",
        "scene": "Scene analysis unavailable",
    }

    if not _openai_client:
        print("[AI] No OpenAI client — add OPENAI_API_KEY to .env")
        return _default

    try:
        # Build payload according to OpenAI Vision spec
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Sensor distance hint: {distance:.1f}m. Output ONLY the JSON object."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]

        response = await _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=250,
            temperature=0.1
        )

        content = response.choices[0].message.content or ""

        print(f"\n{'─'*50}")
        print(f"[AI] model: {model}")
        print(f"[AI] raw:   {content[:200]}")
        print(f"{'─'*50}\n")

        if not content.strip():
            return _default

        parsed = _extract_json(content)
        direction = str(parsed.get("direction", "STOP")).upper().strip()
        obstacle = str(parsed.get("obstacle", "unknown")).strip()
        dist_val = parsed.get("distance", distance)
        message = str(parsed.get("message", "Obstacle ahead")).strip()
        scene = str(parsed.get("scene", "")).strip()

        if direction not in {"CLEAR", "STOP", "LEFT", "RIGHT", "DANGER"}:
            direction = "STOP"

        words = message.split()
        if len(words) > 8:
            message = " ".join(words[:8])

        try:
            dist_float = (
                float(dist_val)
                if isinstance(dist_val, (int, float))
                else float(re.search(r"\d+\.?\d*", str(dist_val)).group())
            )
        except Exception:
            dist_float = distance

        result = {
            "direction": direction,
            "obstacle": obstacle,
            "distance": dist_float,
            "message": message,
            "scene": scene,
        }
        print(f"[AI] ✅ {result}")
        return result

    except Exception as e:
        err = str(e)
        print(f"[AI] ❌ Error: {err}")
        if "insufficient_quota" in err.lower() or "429" in err:
            print("[AI]    Rate limit or insufficient API credits.")
        elif "api_key" in err.lower() or "401" in err:
            print("[AI]    Bad API key — check OPENAI_API_KEY in .env")
        return _default


async def test_openrouter_connection() -> bool:
    """Tests OpenAI connection (function name kept for backwards compatibility in main.py)."""
    if not _openai_client:
        return False
    try:
        r = await _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply OK only."}],
            max_tokens=10
        )
        return bool(r.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI test failed: {e}")
        return False
