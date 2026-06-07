import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def short_error(exc: Exception) -> str:
    text = str(exc)
    status_match = re.search(r"status_code:\s*(\d+)", text)
    message_match = re.search(r"'message':\s*'([^']+)'", text)
    if status_match and message_match:
        return f"HTTP {status_match.group(1)}: {message_match.group(1)}"
    if message_match:
        return message_match.group(1)
    return text[:240]


def main():
    if not config.ELEVENLABS_API_KEY:
        raise SystemExit("ELEVENLABS_API_KEY is missing in .env")

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    try:
        voices = client.voices.get_all().voices
    except Exception as exc:
        raise SystemExit(f"Could not list voices: {short_error(exc)}") from exc
    if not voices:
        raise SystemExit("No voices returned by ElevenLabs")

    print("Available ElevenLabs voices for this API key:")
    for voice in voices:
        name = getattr(voice, "name", "unknown")
        voice_id = getattr(voice, "voice_id", "")
        category = getattr(voice, "category", "")
        print(f"- {name}: {voice_id} {f'({category})' if category else ''}")


if __name__ == "__main__":
    main()
