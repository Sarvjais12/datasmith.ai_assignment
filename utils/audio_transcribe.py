"""
Utility: Audio Transcription
──────────────────────────────
Uses OpenAI Whisper (local) to convert audio files to text.

Supported formats: MP3, WAV, M4A, OGG, FLAC
"""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

_whisper_model = None  # Lazy-loaded (large download on first use)


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper  # openai-whisper
            logger.info("Loading Whisper 'base' model…")
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        except ImportError:
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            )
    return _whisper_model


def transcribe_audio(file_bytes: bytes, ext: str) -> str:
    """
    Transcribe audio bytes to text using Whisper.

    Args:
        file_bytes: Raw audio file bytes.
        ext:        File extension (mp3, wav, m4a, ogg, flac).

    Returns:
        Transcribed text string.
    """
    ext = ext.lower().lstrip(".")
    allowed = {"mp3", "wav", "m4a", "ogg", "flac"}
    if ext not in allowed:
        return f"⚠️ Unsupported audio format: .{ext}"

    model = _get_model()

    # Write to a named temp file (Whisper requires a file path)
    suffix = f".{ext}"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        logger.info("Transcribing %s (%d bytes)…", suffix, len(file_bytes))
        result = model.transcribe(tmp_path, fp16=False)
        text   = result.get("text", "").strip()

        if not text:
            return "No speech detected in the audio."

        logger.info("Transcription done: %d chars", len(text))
        return text

    except Exception as exc:
        logger.exception("Audio transcription failed")
        return f"⚠️ Transcription failed: {exc}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
