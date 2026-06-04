"""
FastAPI backend server for smart blind-assistance glasses.
Receives distance + image data from ESP32, analyzes with AI, returns navigation commands.
"""

import os
import sys
import time
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import custom modules
from analyzer import analyze_image_with_vision, test_openrouter_connection
from tts import generate_and_play_audio
from image_utils import process_image, get_image_size_kb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Glasses Backend",
    description="Backend server for blind-assistance smart glasses",
    version="1.0.0"
)

# Configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
COOLDOWN_SECONDS = 2.0

# Cooldown tracking
last_ai_call_time: float = 0.0
cooldown_lock = asyncio.Lock()


# Request/Response models
class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""
    distance: float
    image: str  # base64 encoded JPEG


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""
    command: str
    message: str
    obstacle: Optional[str] = "unknown"
    timestamp: Optional[str] = None


def should_call_ai(distance: float) -> tuple[bool, str]:
    """
    Determine if AI should be called based on distance.

    Args:
        distance: Distance in meters from ultrasonic sensor

    Returns:
        Tuple of (should_call, command_if_not)
        - should_call: True if AI should be called
        - command_if_not: Command to return if not calling AI
    """
    if distance > 1.5:
        return False, "CLEAR"
    elif distance < 0.3:
        return False, "DANGER"
    else:
        # 0.3 <= distance <= 1.5: call AI
        return True, ""


async def is_in_cooldown() -> bool:
    """
    Check if we're currently in cooldown period.

    Returns:
        True if in cooldown, False otherwise
    """
    global last_ai_call_time
    async with cooldown_lock:
        elapsed = time.time() - last_ai_call_time
        return elapsed < COOLDOWN_SECONDS


async def update_cooldown():
    """Update the last AI call time to now."""
    global last_ai_call_time
    async with cooldown_lock:
        last_ai_call_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    logger.info("=" * 60)
    logger.info("Smart Glasses Backend Server Starting...")
    logger.info("=" * 60)

    # Check environment variables
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    if not openrouter_key:
        logger.warning("OPENROUTER_API_KEY not set! AI analysis will not work.")
    else:
        logger.info("OPENROUTER_API_KEY is set")

    if not elevenlabs_key:
        logger.warning("ELEVENLABS_API_KEY not set! TTS will not work.")
    else:
        logger.info("ELEVENLABS_API_KEY is set")

    # Test OpenRouter connection
    if openrouter_key:
        try:
            test_result = await test_openrouter_connection()
            if test_result:
                logger.info("OpenRouter API connection: OK")
            else:
                logger.warning("OpenRouter API connection: FAILED")
        except Exception as e:
            logger.warning(f"OpenRouter API connection test error: {e}")

    logger.info(f"Server will run on port {SERVER_PORT}")
    logger.info(f"Cooldown period: {COOLDOWN_SECONDS} seconds")
    logger.info("=" * 60)
    logger.info("Server started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown."""
    logger.info("Server shutting down...")


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "online",
        "service": "Smart Glasses Backend",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main analysis endpoint.
    Receives distance + image from ESP32, returns navigation command.

    Flow:
    1. Check distance - decide if AI needed
    2. If AI needed, check cooldown
    3. Process image (resize, compress)
    4. Call OpenRouter vision API
    5. Generate TTS audio
    6. Return command to ESP32
    """
    timestamp = datetime.now().isoformat()
    distance = request.distance
    base64_image = request.image

    logger.info("=" * 60)
    logger.info(f"Received /analyze request")
    logger.info(f"Distance: {distance:.2f}m")
    logger.info(f"Image data length: {len(base64_image)} chars")

    # Step 1: Distance-based logic
    should_call, command_if_not = should_call_ai(distance)

    # Case: Distance > 1.5m → CLEAR
    if not should_call and command_if_not == "CLEAR":
        logger.info(f"Distance {distance:.2f}m > 1.5m → CLEAR (skipping AI)")
        return AnalyzeResponse(
            command="CLEAR",
            message="Path is clear",
            obstacle="none",
            timestamp=timestamp
        )

    # Case: Distance < 0.3m → DANGER (ignore cooldown)
    if not should_call and command_if_not == "DANGER":
        logger.info(f"Distance {distance:.2f}m < 0.3m → DANGER (emergency, skipping AI)")
        # Play danger audio
        try:
            await generate_and_play_audio("Danger! Stop immediately!")
        except Exception as e:
            logger.error(f"Error playing danger audio: {e}")

        return AnalyzeResponse(
            command="DANGER",
            message="Danger! Stop immediately!",
            obstacle="too close",
            timestamp=timestamp
        )

    # Case: 0.3m <= distance <= 1.5m → Call AI
    logger.info(f"Distance {distance:.2f}m in analysis zone → Calling AI")

    # Step 2: Check cooldown (only for AI calls)
    in_cooldown = await is_in_cooldown()
    if in_cooldown:
        elapsed = time.time() - last_ai_call_time
        remaining = COOLDOWN_SECONDS - elapsed
        logger.info(f"In cooldown period. {remaining:.1f}s remaining. Returning STOP.")
        return AnalyzeResponse(
            command="STOP",
            message="Please wait",
            obstacle="unknown",
            timestamp=timestamp
        )

    # Step 3: Process image
    try:
        logger.info("Processing image (decode → resize → re-encode)")
        processed_image = process_image(base64_image, max_width=1280, max_height=720, quality=85)
        image_size_kb = get_image_size_kb(processed_image)
        logger.info(f"Image processed. Size: {image_size_kb:.1f}KB")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return AnalyzeResponse(
            command="STOP",
            message="Image processing error",
            obstacle="unknown",
            timestamp=timestamp
        )

    # Step 4: Call OpenRouter vision API
    try:
        logger.info("Calling OpenRouter vision API...")
        analysis_result = await analyze_image_with_vision(processed_image, distance)

        direction = analysis_result.get("direction", "STOP")
        message = analysis_result.get("message", "Obstacle ahead")
        obstacle = analysis_result.get("obstacle", "unknown")

        logger.info(f"AI Analysis Result:")
        logger.info(f"  Direction: {direction}")
        logger.info(f"  Message: {message}")
        logger.info(f"  Obstacle: {obstacle}")

    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        # Safe default
        direction = "STOP"
        message = "Obstacle ahead, stop"
        obstacle = "unknown"

    # Update cooldown
    await update_cooldown()

    # Step 5: Generate and play TTS
    try:
        logger.info(f"Generating TTS for message: '{message}'")
        await generate_and_play_audio(message)
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")

    # Step 6: Return response to ESP32
    logger.info(f"Sending response to ESP32: command={direction}, message={message}")
    logger.info("=" * 60)

    return AnalyzeResponse(
        command=direction,
        message=message,
        obstacle=obstacle,
        timestamp=timestamp
    )


@app.get("/status")
async def get_status():
    """
    Get server status including cooldown info.
    """
    global last_ai_call_time
    elapsed = time.time() - last_ai_call_time
    in_cooldown = elapsed < COOLDOWN_SECONDS

    return {
        "cooldown_active": in_cooldown,
        "cooldown_remaining": max(0, COOLDOWN_SECONDS - elapsed) if in_cooldown else 0,
        "last_ai_call_seconds_ago": elapsed if last_ai_call_time > 0 else None,
        "server_time": time.time()
    }


@app.post("/reset-cooldown")
async def reset_cooldown():
    """
    Reset the cooldown timer (for testing).
    """
    global last_ai_call_time
    async with cooldown_lock:
        last_ai_call_time = 0
    logger.info("Cooldown timer reset")
    return {"status": "cooldown reset"}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
