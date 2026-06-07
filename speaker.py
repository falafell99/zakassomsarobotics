import logging
import os
import re
import subprocess
import tempfile
import threading
import queue
import config

logger = logging.getLogger(__name__)

_queue: queue.Queue[str | None] = queue.Queue(maxsize=3)
_elevenlabs_client = None
_elevenlabs_disabled = False


def _init_elevenlabs():
    global _elevenlabs_client, _elevenlabs_disabled
    if _elevenlabs_disabled or not config.ELEVENLABS_API_KEY:
        return None
    if _elevenlabs_client is not None:
        return _elevenlabs_client
    try:
        from elevenlabs.client import ElevenLabs
        _elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        return _elevenlabs_client
    except Exception as exc:
        logger.warning("ElevenLabs unavailable, using system TTS: %s", exc)
        _elevenlabs_disabled = True
        return None


def _say(text: str):
    subprocess.run(["say", "-r", "175", text], capture_output=True, timeout=12)


def _audio_to_bytes(audio) -> bytes:
    if isinstance(audio, bytes):
        return audio
    return b"".join(chunk for chunk in audio if chunk)


def _play_mp3(audio_bytes: bytes, text: str):
    if not audio_bytes:
        raise RuntimeError("ElevenLabs returned empty audio")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            temp_path = tmp.name
            tmp.write(audio_bytes)

        # afplay is more reliable than streaming playback for short navigation
        # prompts; it waits until the whole file is played.
        timeout = max(8, min(30, len(text.split()) * 1.2 + 4))
        subprocess.run(["afplay", temp_path], check=True, capture_output=True, timeout=timeout)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _short_error(exc: Exception) -> str:
    text = str(exc)
    status_match = re.search(r"status_code:\s*(\d+)", text)
    message_match = re.search(r"'message':\s*'([^']+)'", text)
    if status_match and message_match:
        return f"HTTP {status_match.group(1)}: {message_match.group(1)}"
    if message_match:
        return message_match.group(1)
    return text[:240]


def _worker():
    global _elevenlabs_disabled
    while True:
        text = _queue.get()
        if text is None:
            return
        try:
            client = _init_elevenlabs()
            if client:
                try:
                    audio = client.text_to_speech.convert(
                        text=text,
                        voice_id=config.ELEVENLABS_VOICE_ID,
                        model_id=config.ELEVENLABS_MODEL,
                        output_format="mp3_22050_32",
                    )
                    _play_mp3(_audio_to_bytes(audio), text)
                except Exception as exc:
                    logger.warning("ElevenLabs failed, using system TTS: %s", _short_error(exc))
                    _elevenlabs_disabled = True
                    _say(text)
            else:
                _say(text)
        except Exception as exc:
            logger.error("TTS failed: %s", exc)
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True, name="speaker").start()


def speak(text: str) -> bool:
    if not config.SPEAK_ENABLED or not text.strip():
        return False
    try:
        while True:
            try:
                _queue.get_nowait()
                _queue.task_done()
            except queue.Empty:
                break
        _queue.put_nowait(text)
        return True
    except queue.Full:
        return False


def speak_blocking(text: str) -> bool:
    queued = speak(text)
    if queued:
        _queue.join()
    return queued
