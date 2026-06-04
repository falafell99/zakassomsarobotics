"""
OpenRouter Vision Analyzer module for smart blind-assistance glasses.
Handles communication with OpenRouter API for image analysis.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None

if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info("OpenRouter client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter client: {e}")
        openrouter_client = None
else:
    logger.warning("OPENROUTER_API_KEY not set in environment variables")


# System prompt for the vision model
SYSTEM_PROMPT = """You are a navigation assistant for a blind person wearing smart glasses.
Analyze the image and the distance sensor reading.
Identify the main obstacle.
Respond ONLY in this exact JSON format (no markdown, no extra text):
{
    "direction": "LEFT" or "RIGHT" or "STOP" or "CLEAR",
    "message": "one short sentence max 8 words telling where to go",
    "obstacle": "name of the obstacle detected"
}
Rules:
- If obstacle is on the left side of frame → direction RIGHT (go around right)
- If obstacle is on the right side → direction LEFT
- If obstacle fills center and no way around → STOP
- Message must be simple, direct, actionable. Example: 'Turn right, chair ahead'"""


async def analyze_image_with_vision(
    base64_image: str,
    distance: float,
    model: str = "meta-llama/llama-3.2-11b-vision-instruct"
) -> Dict[str, Any]:
    """
    Analyze an image using OpenRouter's vision model.

    Args:
        base64_image: Base64 encoded JPEG image string
        distance: Distance from ultrasonic sensor in meters
        model: Vision model to use (must support vision)

    Returns:
        Dictionary with keys: direction, message, obstacle
        Default (on error): {"direction": "STOP", "message": "Obstacle ahead, stop", "obstacle": "unknown"}
    """
    if not openrouter_client:
        logger.error("OpenRouter client not initialized. Cannot analyze image.")
        return {
            "direction": "STOP",
            "message": "System error, stop",
            "obstacle": "unknown"
        }

    try:
        logger.info(f"Calling OpenRouter API with distance: {distance}m, model: {model}")

        # Prepare the message with image
        # Note: OpenAI SDK expects image_url with data:image/jpeg;base64 prefix
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Create the completion request
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
                            "text": f"Distance sensor reading: {distance:.2f} meters. Analyze the image and provide navigation guidance."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.3,  # Lower temperature for more consistent JSON output
            response_format={"type": "json_object"}  # Request JSON mode if supported
        )

        # Extract the response content
        content = response.choices[0].message.content
        logger.info(f"OpenRouter raw response: {content}")

        # Parse JSON response
        try:
            # Try to parse the JSON directly
            result = json.loads(content)

            # Validate required fields
            direction = result.get("direction", "STOP")
            message = result.get("message", "Obstacle ahead, stop")
            obstacle = result.get("obstacle", "unknown")

            # Validate direction value
            valid_directions = ["LEFT", "RIGHT", "STOP", "CLEAR"]
            if direction not in valid_directions:
                logger.warning(f"Invalid direction '{direction}', defaulting to STOP")
                direction = "STOP"

            # Truncate message if too long
            words = message.split()
            if len(words) > 8:
                message = " ".join(words[:8])

            final_result = {
                "direction": direction,
                "message": message,
                "obstacle": obstacle
            }

            logger.info(f"Parsed result: {final_result}")
            return final_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw content: {content}")

            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                try:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)

                    direction = result.get("direction", "STOP")
                    message = result.get("message", "Obstacle ahead, stop")
                    obstacle = result.get("obstacle", "unknown")

                    return {
                        "direction": direction,
                        "message": message,
                        "obstacle": obstacle
                    }
                except Exception as e2:
                    logger.error(f"Failed to extract JSON from markdown: {e2}")

            # Return safe default
            return {
                "direction": "STOP",
                "message": "Obstacle ahead, stop",
                "obstacle": "unknown"
            }

    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return {
            "direction": "STOP",
            "message": "System error, stop",
            "obstacle": "unknown"
        }


async def test_openrouter_connection() -> bool:
    """
    Test the OpenRouter API connection.

    Returns:
        True if connection is successful, False otherwise
    """
    if not openrouter_client:
        return False

    try:
        # Simple test with a text-only request
        response = openrouter_client.chat.completions.create(
            model="meta-llama/llama-3.2-11b-vision-instruct",
            messages=[
                {"role": "user", "content": "Say 'API connection successful'"}
            ],
            max_tokens=10
        )
        logger.info("OpenRouter API connection test successful")
        return True
    except Exception as e:
        logger.error(f"OpenRouter API connection test failed: {e}")
        return False
