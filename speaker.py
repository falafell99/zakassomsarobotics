import logging
import subprocess
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
                    from elevenlabs import stream
                    audio = client.text_to_speech.convert(
                        text=text,
                        voice_id=config.ELEVENLABS_VOICE_ID,
                        model_id=config.ELEVENLABS_MODEL,
                        output_format="mp3_22050_32",
                    )
                    stream(audio)
                except Exception as exc:
                    logger.warning("ElevenLabs failed, using system TTS: %s", exc)
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
        _queue.put_nowait(text)
        return True
    except queue.Full:
        return False


def speak_blocking(text: str) -> bool:
    queued = speak(text)
    if queued:
        _queue.join()
    return queued
