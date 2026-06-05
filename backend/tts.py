"""
ElevenLabs TTS (Text-to-Speech) module for smart blind-assistance glasses.
Handles audio generation and playback using ElevenLabs SDK.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize ElevenLabs client
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
elevenlabs_client = None

if ELEVENLABS_API_KEY:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play as elevenlabs_play

        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("ElevenLabs client initialized successfully")
    except ImportError:
        logger.error("ElevenLabs SDK not installed. Run: pip install elevenlabs")
        elevenlabs_client = None
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs client: {e}")
        elevenlabs_client = None
else:
    logger.warning("ELEVENLABS_API_KEY not set in environment variables")


async def generate_and_play_audio(text: str, voice_name: str = "Rachel") -> bool:
    """
    Generate audio from text using ElevenLabs TTS and play it immediately.

    Args:
        text: Text to convert to speech
        voice_name: Name of the ElevenLabs voice to use

    Returns:
        True if audio was generated and played successfully, False otherwise
    """
    if not elevenlabs_client:
        logger.error("ElevenLabs client not initialized. Cannot generate audio.")
        return False

    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return False

    try:
        logger.info(f"Generating TTS audio for: '{text}'")

        # Generate audio using ElevenLabs
        audio = elevenlabs_client.text_to_speech.convert(
            text=text, voice_id=voice_name, model_id="eleven_monolingual_v1"
        )

        logger.info("Audio generated successfully, playing now...")

        # Play audio directly (non-blocking)
        elevenlabs_play(audio)

        logger.info("Audio playback completed")
        return True

    except Exception as e:
        logger.error(f"Error generating or playing audio: {e}")
        return False


async def generate_audio_only(text: str, voice_name: str = "Rachel") -> Optional[bytes]:
    """
    Generate audio from text and return the audio bytes without playing.

    Args:
        text: Text to convert to speech
        voice_name: Name of the ElevenLabs voice to use

    Returns:
        Audio bytes if successful, None otherwise
    """
    if not elevenlabs_client:
        logger.error("ElevenLabs client not initialized. Cannot generate audio.")
        return None

    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return None

    try:
        logger.info(f"Generating TTS audio for: '{text}'")

        # Generate audio using ElevenLabs
        audio = elevenlabs_client.text_to_speech.convert(
            text=text, voice_id=voice_name, model_id="eleven_monolingual_v1"
        )

        logger.info("Audio generated successfully")
        return audio

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None


def list_available_voices() -> list:
    """
    List available ElevenLabs voices.

    Returns:
        List of voice objects with id, name, and other metadata
    """
    if not elevenlabs_client:
        logger.error("ElevenLabs client not initialized.")
        return []

    try:
        voices = elevenlabs_client.voices.get_all()
        return voices.voices
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []
