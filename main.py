"""Audio transcription using faster-whisper."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default whisper model (can be HuggingFace repo ID or local path)
DEFAULT_WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "ivrit-ai/whisper-large-v3-turbo-ct2"
)


@dataclass
class Word:
    """Represents a single word with timing information."""
    word: str
    start: float
    end: float
    probability: float | None = None
    speaker: str | None = None


@dataclass
class Segment:
    """Represents a transcription segment."""
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    extra_data: dict[str, Any] = field(default_factory=dict)

# Global model instance (loaded once at startup)
_model: WhisperModel = None


def load_model() -> WhisperModel:
    """
    Load the Whisper model.

    The model is loaded once and reused for all transcriptions.

    Returns:
        Loaded WhisperModel instance.
    """
    global _model

    if _model is not None:
        return _model   
    

    _model = WhisperModel(
        DEFAULT_WHISPER_MODEL,
        device="cuda",
        compute_type="int8_float16",
    )

    logger.info(f"Model '{DEFAULT_WHISPER_MODEL}' loaded successfully")
    return _model

def _format_timestamp(seconds: float) -> str:
    """
    Convert seconds to timestamp format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"    


def transcribe(audio_path: str) -> tuple[list[Segment], Any] | tuple[None, None]:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Tuple of (list of Segment objects, transcription info), or (None, None) on failure.
    """
    try:
        model = load_model()

        logger.info(f"Transcribing: {audio_path}")
        

        segments, info = model.transcribe(
            audio_path,
            language="he",
            beam_size=5,
            word_timestamps=True,
        )

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        logger.info(f"Audio duration: {info.duration:.2f}s")
        logger.info(f"Audio duration: {_format_timestamp(info.duration)}")

        # Collect segments for diarization
        all_segments: list[Segment] = []

        for segment in segments:
            logger.info(f"[{_format_timestamp(segment.start)} -> {_format_timestamp(segment.end)}] {segment.text}")
            # Build extra_data dictionary
            segment_extra_data = {
                "language": info.language,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
            }

            # Process words if available
            segment_words: list[Word] = []
            if hasattr(segment, "words") and segment.words:
                for word_data in segment.words:
                    word = Word(
                        word=word_data.word,
                        start=word_data.start,
                        end=word_data.end,
                        probability=getattr(word_data, "probability", None),
                    )
                    segment_words.append(word)

            # Create Segment object
            segment_obj = Segment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                words=segment_words,
                extra_data=segment_extra_data,
            )
            all_segments.append(segment_obj)

        logger.info(f"Transcribed {len(all_segments)} segments")
        return all_segments, info

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}", exc_info=True)
        return None, None


if __name__ == "__main__":
    transcribe("C:\\portal\\diarization\\abadyan.mp4")
