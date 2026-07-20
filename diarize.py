"""Speaker diarization using PyAnnote.audio pipeline.

SPEAKER IDENTIFICATION
    You can optionally provide reference voice samples for known speakers:
    
    speaker_refs = {
        "Alice": "path/to/alice_sample.wav",
        "Bob": ["path/to/bob1.wav", "path/to/bob2.wav"],  # multiple samples = better
    }
    segments = diarize(audio_path, transcription_segments, speaker_references=speaker_refs)
    # segments will have speaker="Alice", "Bob", or "SPEAKER_XX" for unknowns

    Requires accepting the embedding model license (one-time, free):
        https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore", message="TensorFloat-32.*has been disabled")

# Workaround for PyTorch 2.6+ weights_only=True default
# pyannote models need this disabled to load properly
# os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"


# Workaround for PyTorch 2.6+ weights_only=True default
# pyannote models need this disabled to load properly
# Must FORCE override (not setdefault) because lightning_fabric passes weights_only=True explicitly
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # FORCE override, not setdefault
    return _original_torch_load(*args, **kwargs)
_patched_torch_load._diarize_patched = True
torch.load = _patched_torch_load

# _original_torch_load = torch.load
# def _patched_torch_load(*args, **kwargs):
#     kwargs.setdefault("weights_only", False)
#     return _original_torch_load(*args, **kwargs)
# torch.load = _patched_torch_load

from pyannote.audio import Pipeline
from pyannote.audio import Model, Inference
from pyannote.core import Segment as PyannoteSegment

from main import Segment, Word

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# =============================================================================
# SPEAKER IDENTIFICATION — match detected speakers to known voice samples
# =============================================================================

class SpeakerIdentifier:
    """Identifies speakers by matching voice embeddings to reference samples."""
    
    EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
    
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device
        self._inference: Optional[Inference] = None
    
    @property
    def inference(self) -> Inference:
        """Lazy-load the embedding model."""
        if self._inference is None:
            token = os.getenv("HF_TOKEN")
            model = Model.from_pretrained(self.EMBEDDING_MODEL, token=token)
            self._inference = Inference(model, window="whole")
            self._inference.to(self.device)
            logger.info(f"Loaded speaker embedding model on {self.device}")
        return self._inference
    
    def _to_wav16k(self, src: str) -> str:
        """Transcode any audio to 16 kHz mono WAV via ffmpeg."""
        dst = tempfile.mktemp(suffix=".16k.wav")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", dst],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            raise RuntimeError("ffmpeg transcode failed: " 
                               + proc.stderr.decode("utf-8", "replace")[:300])
        return dst
    
    def extract_embedding(self, audio_path: str) -> npt.NDArray:
        """Extract a speaker embedding vector from an audio file.
        
        Args:
            audio_path: Path to audio file (any format ffmpeg supports).
        
        Returns:
            numpy array of shape (256,) — the speaker embedding.
        """
        wav = self._to_wav16k(audio_path)
        try:
            embedding = self.inference(wav)
            return np.array(embedding)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
    
    def extract_embedding_from_segment(self, audio_path: str, start: float, end: float) -> npt.NDArray:
        """Extract embedding from a specific time segment of an audio file.
        
        Args:
            audio_path: Path to the full audio file.
            start: Start time in seconds.
            end: End time in seconds.
        
        Returns:
            numpy array of shape (256,) — the speaker embedding for that segment.
        """
        wav = self._to_wav16k(audio_path)
        try:
            segment = PyannoteSegment(start, end)
            embedding = self.inference.crop(wav, segment)
            return np.array(embedding)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
    
    def build_speaker_profiles(
        self, 
        speaker_references: dict[str, str | list[str]]
    ) -> dict[str, npt.NDArray]:
        """Build embedding profiles for known speakers from reference audio samples.
        
        Args:
            speaker_references: Dict mapping speaker name to audio path(s).
                e.g. {"Alice": "alice.wav"} or {"Bob": ["bob1.wav", "bob2.wav"]}
        
        Returns:
            Dict mapping speaker name to averaged embedding vector.
        """
        profiles = {}
        for name, paths in speaker_references.items():
            if isinstance(paths, str):
                paths = [paths]
            
            embeddings = []
            for path in paths:
                if not os.path.exists(path):
                    logger.warning(f"Reference audio not found: {path}")
                    continue
                try:
                    emb = self.extract_embedding(path)
                    embeddings.append(emb)
                    logger.info(f"Loaded reference for '{name}' from {path}")
                except Exception as e:
                    logger.warning(f"Failed to extract embedding from {path}: {e}")
            
            if embeddings:
                # Average multiple samples for more robust profile
                profiles[name] = np.mean(embeddings, axis=0)
        
        return profiles
    
    @staticmethod
    def cosine_similarity(a: npt.NDArray, b: npt.NDArray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def identify_speakers(
        self,
        diarization_df: pd.DataFrame,
        audio_path: str,
        speaker_profiles: dict[str, npt.NDArray],
        similarity_threshold: float = 0.5,
    ) -> dict[str, str]:
        """Match detected speakers to known speaker profiles.
        
        For each unique speaker in the diarization output, extracts their embedding
        (using their longest turn) and compares against known profiles.
        
        Args:
            diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker'].
            audio_path: Path to the original audio file.
            speaker_profiles: Dict from build_speaker_profiles().
            similarity_threshold: Min cosine similarity to match (0.5 is reasonable).
        
        Returns:
            Dict mapping detected speaker labels to identified names.
        """
        if not speaker_profiles:
            return {}
        
        # Find the longest segment for each detected speaker
        speaker_segments = {}
        for _, row in diarization_df.iterrows():
            spk = row["speaker"]
            duration = row["end"] - row["start"]
            if spk not in speaker_segments or duration > speaker_segments[spk]["duration"]:
                speaker_segments[spk] = {
                    "start": row["start"],
                    "end": row["end"],
                    "duration": duration
                }
        
        # Extract embedding for each detected speaker (using their longest turn)
        detected_embeddings = {}
        wav = self._to_wav16k(audio_path)
        try:
            for spk, seg in speaker_segments.items():
                # Need at least 0.5s of audio for a reliable embedding
                if seg["duration"] < 0.5:
                    logger.warning(f"Speaker {spk} has no segment >= 0.5s, skipping ID")
                    continue
                try:
                    segment = PyannoteSegment(seg["start"], seg["end"])
                    emb = self.inference.crop(wav, segment)
                    detected_embeddings[spk] = np.array(emb)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for {spk}: {e}")
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
        
        # Match detected speakers to known profiles
        speaker_mapping = {}
        used_profiles = set()
        
        for detected_spk, detected_emb in detected_embeddings.items():
            best_match = None
            best_score = similarity_threshold
            
            for profile_name, profile_emb in speaker_profiles.items():
                if profile_name in used_profiles:
                    continue
                score = self.cosine_similarity(detected_emb, profile_emb)
                if score > best_score:
                    best_score = score
                    best_match = profile_name
            
            if best_match:
                speaker_mapping[detected_spk] = best_match
                used_profiles.add(best_match)
                logger.info(f"Matched {detected_spk} -> '{best_match}' (similarity={best_score:.3f})")
            else:
                logger.info(f"{detected_spk}: no match above threshold ({similarity_threshold})")
        
        return speaker_mapping


def load_audio(file: str, sr: int = SAMPLE_RATE) -> npt.NDArray:
    """
    Load an audio file as mono waveform, resampling as necessary.

    Args:
        file: Path to the audio file.
        sr: Target sample rate.

    Returns:
        NumPy array containing the audio waveform in float32.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to load audio via ffmpeg: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


class PyannoteDiarizationEngine:
    """Speaker diarization engine using PyAnnote.audio pipeline."""

    #DEFAULT_CHECKPOINT = "ivrit-ai/pyannote-speaker-diarization-3.1"
    DEFAULT_CHECKPOINT = "pyannote/speaker-diarization-3.1"


    def _match_speaker_to_interval(
        self,
        diarization_df: pd.DataFrame,
        start: float,
        end: float,
        fill_nearest: bool = False,
    ) -> Optional[str]:
        """
        Match the best speaker for a given time interval.

        Args:
            diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker'].
            start: Start time of the interval.
            end: End time of the interval.
            fill_nearest: If True, match speakers even when there's no direct time overlap.

        Returns:
            The speaker ID with the highest intersection, or None if no match found.
        """
        diarization_df["intersection"] = (
            np.minimum(diarization_df["end"], end) - 
            np.maximum(diarization_df["start"], start)
        )
        diarization_df["union"] = (
            np.maximum(diarization_df["end"], end) - 
            np.minimum(diarization_df["start"], start)
        )

        if not fill_nearest:
            tmp_df = diarization_df[diarization_df["intersection"] > 0]
        else:
            tmp_df = diarization_df

        speaker = None
        if len(tmp_df) > 0:
            speaker = (
                tmp_df.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )

        return speaker

    def _assign_speakers(
        self,
        diarization_df: pd.DataFrame,
        transcription_segments: list[Segment],
        fill_nearest: bool = False,
    ) -> list[Segment]:
        """
        Assign speakers to segments in the transcript.

        Args:
            diarization_df: Diarization dataframe with columns ['start', 'end', 'speaker'].
            transcription_segments: List of Segment objects to augment with speaker labels.
            fill_nearest: If True, assign speakers even when there's no direct time overlap.

        Returns:
            Updated transcription_segments with speaker assignments.
        """
        for seg in transcription_segments:
            speaker = self._match_speaker_to_interval(
                diarization_df, start=seg.start, end=seg.end, fill_nearest=fill_nearest
            )
            # Store speaker in extra_data
            seg.extra_data["speaker"] = speaker

            # Assign speaker to words
            for word in seg.words:
                if word.start is not None:
                    word_speaker = self._match_speaker_to_interval(
                        diarization_df, start=word.start, end=word.end, fill_nearest=fill_nearest
                    )
                    word.speaker = word_speaker

        return transcription_segments

    def diarize(
        self,
        audio: str | npt.NDArray,
        transcription_segments: list[Segment],
        *,
        device: str | torch.device | None = None,
        checkpoint_path: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_auth_token: Optional[str] = None,
        speaker_references: Optional[dict[str, str | list[str]]] = None,
        similarity_threshold: float = 0.5,
        verbose: bool = True,
    ) -> list[Segment]:
        """
        Perform speaker diarization using PyAnnote.audio pipeline.

        Args:
            audio: Path to audio file or NumPy array containing audio waveform.
            transcription_segments: List of transcription segments to assign speaker labels to.
            device: Device to run on ("cpu", "cuda", or torch.device).
            checkpoint_path: Model checkpoint path.
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum number of speakers to consider.
            max_speakers: Maximum number of speakers to consider.
            use_auth_token: Authentication token for model download.
            speaker_references: Optional dict mapping speaker names to reference audio paths.
                e.g. {"Alice": "alice.wav", "Bob": ["bob1.wav", "bob2.wav"]}
                When provided, detected speakers will be matched to these references.
            similarity_threshold: Cosine similarity threshold for speaker matching (0-1).
                Default 0.5 works well; raise to 0.6-0.7 for stricter matching.
            verbose: Whether to enable verbose logging.

        Returns:
            List of transcription segments with speaker labels assigned.
        """
        checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if verbose:
            logger.info(
                f"Diarizing with pyannote: checkpoint={checkpoint_path}, device={device}, "
                f"num_speakers={num_speakers}, min_speakers={min_speakers}, max_speakers={max_speakers}"
            )

        if isinstance(device, str):
            device = torch.device(device)

        # Keep original audio path for speaker identification
        audio_path = audio if isinstance(audio, str) else None
        
        if isinstance(audio, str):
            audio = load_audio(audio)

        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }

        # Use HF_TOKEN from environment if use_auth_token not provided
        token = use_auth_token or os.getenv("HF_TOKEN")

        # if token:
        #     from huggingface_hub import login
        #     login(token=token, add_to_git_credential=False)

        # logger.info("Loading diarization pipeline...")
        # diarization_pipeline = Pipeline.from_pretrained(checkpoint_path).to(device)

        # if token:
        #     # Set HF_TOKEN env var for huggingface_hub to pick up automatically
        #     os.environ["HF_TOKEN"] = token

        # logger.info("Loading diarization pipeline...")
        # diarization_pipeline = Pipeline.from_pretrained(checkpoint_path).to(device)

        logger.info("Loading diarization pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            checkpoint_path, use_auth_token=token
        ).to(device)

        logger.info("Running diarization...")
        diarization = diarization_pipeline(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        diarization_df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarization_df["start"] = diarization_df["segment"].apply(lambda x: x.start)
        diarization_df["end"] = diarization_df["segment"].apply(lambda x: x.end)

        if verbose:
            unique_speakers = diarization_df["speaker"].unique()
            logger.info(f"Diarization completed: found {len(unique_speakers)} speakers")

        # Speaker identification: match detected speakers to known voice samples
        if speaker_references and audio_path:
            logger.info("Running speaker identification...")
            identifier = SpeakerIdentifier(device=device)
            speaker_profiles = identifier.build_speaker_profiles(speaker_references)
            
            if speaker_profiles:
                speaker_mapping = identifier.identify_speakers(
                    diarization_df, 
                    audio_path,
                    speaker_profiles,
                    similarity_threshold=similarity_threshold
                )
                
                # Apply mapping to diarization_df
                if speaker_mapping:
                    diarization_df["speaker"] = diarization_df["speaker"].apply(
                        lambda s: speaker_mapping.get(s, s)
                    )
                    logger.info(f"Identified {len(speaker_mapping)} speakers: {speaker_mapping}")
        elif speaker_references and not audio_path:
            logger.warning("Speaker references provided but audio is numpy array; "
                          "pass audio path for speaker identification")

        diarized_segments = self._assign_speakers(diarization_df, transcription_segments)
        return diarized_segments


def diarize(
    audio_path: str,
    transcription_segments: list[Segment],
    *,
    device: str | None = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    speaker_references: Optional[dict[str, str | list[str]]] = None,
    similarity_threshold: float = 0.5,
) -> list[Segment]:
    """
    Convenience function to perform speaker diarization.

    Args:
        audio_path: Path to the audio file.
        transcription_segments: List of transcription segments from transcribe().
        device: Device to run on ("cpu" or "cuda").
        num_speakers: Exact number of speakers (if known).
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.
        speaker_references: Optional dict mapping speaker names to reference audio paths.
            e.g. {"Alice": "alice.wav", "Bob": ["bob1.wav", "bob2.wav"]}
            When provided, detected speakers will be matched to these references.
        similarity_threshold: Cosine similarity threshold for speaker matching (0-1).
            Default 0.5 works well; raise to 0.6-0.7 for stricter matching.

    Returns:
        List of segments with speaker labels assigned in extra_data["speaker"].
    """
    engine = PyannoteDiarizationEngine()
    return engine.diarize(
        audio_path,
        transcription_segments,
        device=device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        speaker_references=speaker_references,
        similarity_threshold=similarity_threshold,
    )


def _format_vtt_timestamp(seconds: float) -> str:
    """
    Convert seconds to VTT timestamp format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted VTT timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def segments_to_vtt(
    diarized_segments: list[Segment],
    output_path: Optional[str] = None,
) -> str:
    """
    Convert diarized segments to WebVTT format.

    Args:
        diarized_segments: List of segments with speaker labels in extra_data["speaker"].
        output_path: Optional path to save the VTT file. If None, only returns the VTT string.

    Returns:
        VTT content as a string.
    """
    vtt_lines = ["WEBVTT", ""]

    for i, segment in enumerate(diarized_segments, start=1):
        speaker = segment.extra_data.get("speaker", "Unknown")
        start_time = _format_vtt_timestamp(segment.start)
        end_time = _format_vtt_timestamp(segment.end)

        # Add cue
        #vtt_lines.append(str(i))
        vtt_lines.append(f"{start_time} --> {end_time}")
        vtt_lines.append(f"[{speaker}] {segment.text.strip()}")
        vtt_lines.append("")

    vtt_content = "\n".join(vtt_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)
        logger.info(f"VTT file saved to: {output_path}")

    return vtt_content


if __name__ == "__main__":
    from main import transcribe

    audio_file = "C:\\portal\\diarization\\recording-1784121028538.webm"
    audio_file = "C:\\portal\\diarization\\abadyan.mp4"
    
    # First transcribe
    segments, info = transcribe(audio_file)
    
    speaker_refs = {
        "rabbi": "C:\\portal\\diarization\\12345.mp3",
        # "Bob": ["samples/bob1.wav", "samples/bob2.wav"],  # multiple = more robust
    }
    if segments:
        # Then diarize
        diarized_segments = diarize(audio_file, segments, speaker_references=speaker_refs)
        
        # Print results
        for seg in diarized_segments:
            speaker = seg.extra_data.get("speaker", "Unknown")
            logger.info(f"[{speaker}] {seg.text}")
        
        # Export to VTT
        import os
        vtt_path = os.path.splitext(audio_file)[0] + ".vtt"
        segments_to_vtt(diarized_segments, vtt_path)
