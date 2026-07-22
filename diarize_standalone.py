#!/usr/bin/env python3
"""
Standalone speaker diarization with pyannote.audio 3.1 — a tested reference.

Distilled from a production service (the WAIP `diarize-svc` pod) that runs this exact
logic on an NVIDIA RTX PRO 6000 (Blackwell, sm_120) and is verified end-to-end
(a 90s Hebrew clip -> 3 speakers, device=cuda, ~3.5s). Use it as a starting point to
add speaker diarization ("who spoke when") to your own project.

WHAT IT DOES
    audio file  -->  list of speaker turns:  [{"start": float, "end": float, "speaker": str}, ...]

QUICK START
  1) Install torch. For a Blackwell / RTX 50-series (sm_120) GPU you MUST use a cu128
     build — older cu124/cu126 wheels have NO sm_120 kernels and pyannote will error or
     silently fall back to CPU:
        pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
     On older GPUs (Ampere/Ada, sm_80..sm_89) or CPU-only, a normal `pip install torch` is fine.
  2) pip install "pyannote.audio==3.3.2"
  3) Accept the model licenses (one-time, free) while logged into Hugging Face:
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0
     Create a token ( https://huggingface.co/settings/tokens ) and export it:
        export HF_TOKEN=hf_xxx
  4) Install ffmpeg (used to normalize any audio to 16 kHz mono WAV):
        # ubuntu/debian: sudo apt-get install -y ffmpeg
  5) Run:
        python diarize_standalone.py path/to/audio.(wav|mp3|m4a|...)  [num_speakers]

GOTCHAS WE HIT (all handled in the code below — these cost real debugging time)
  * torch >= 2.6 defaults torch.load(weights_only=True), which REJECTS pyannote 3.x
    checkpoints (they pickle ListConfig/TorchVersion globals). We force weights_only=False
    for this process. Without it you get an UnpicklingError on pipeline load. It's safe
    because the only checkpoints loaded are the official pyannote ones.
  * pyannote's soundfile/torchaudio backend only reads WAV/FLAC/OGG/MP3 cleanly; m4a/AAC/
    opus from phone recorders fail with "Format not recognised". We transcode to 16 kHz
    mono WAV via ffmpeg first -> format-agnostic input.
  * num_speakers (when you know it) is a FREE accuracy win — it stops the model over- or
    under-counting speakers. Pass it whenever the count is known.
  * embedding_batch_size defaults to 1, which underuses the GPU. Raising it (e.g. 32) is
    the main speed lever — that's how we get ~50x realtime on the GPU.
  * HF_HUB_OFFLINE: leave it UNSET for the first run (downloads + caches the models). Set
    HF_HUB_OFFLINE=1 afterwards for fully offline / air-gapped runs (models read from cache).

To attach speakers to an existing TRANSCRIPT, assign each transcript segment the speaker
whose turn overlaps it most — see `assign_speakers()` at the bottom.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _patch_torch_load() -> None:
    """Force torch.load(weights_only=False) — required for pyannote 3.x on torch >= 2.6."""
    import torch
    if getattr(torch.load, "_diarize_patched", False):
        return
    _orig = torch.load

    def _compat(*args, **kwargs):
        # FORCE-override: lightning_fabric passes weights_only=True explicitly, so
        # setdefault is not enough.
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    _compat._diarize_patched = True
    torch.load = _compat


def _to_wav16k(src: str) -> str:
    """Transcode any audio to 16 kHz mono WAV via ffmpeg. Returns the new path."""
    dst = src + ".16k.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", dst],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError("ffmpeg transcode failed: "
                           + proc.stderr.decode("utf-8", "replace")[:300])
    return dst


def diarize(audio_path: str,
            num_speakers: int | None = None,
            embedding_batch_size: int = 32,
            pipeline_id: str = "pyannote/speaker-diarization-3.1") -> list[dict]:
    """Diarize an audio file. Returns [{'start','end','speaker'}, ...] sorted by time.

    Model choice (we measured — see README): pyannote-3.1 on GPU beat our previous
    SpeechBrain-ECAPA-on-CPU on both speed (~50x vs ~1.7x realtime) and DER (0.418 vs
    0.442 on the reliable clip; 0.413 with a num_speakers hint), with no quality
    regression. `pyannote/speaker-diarization-community-1` has the best published DER and
    is the upgrade path, but it needs pyannote.audio>=4.x + torch 2.8 (it errors on the
    3.3.2 stack here) — switch pipeline_id to it once you're on those versions.
    """
    _patch_torch_load()
    import torch
    from pyannote.audio import Pipeline

    # token is needed on the first (online) run to fetch the gated models; once cached it
    # is ignored. Harmless if None when HF_HUB_OFFLINE=1 and models are already cached.
    token = os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained(pipeline_id, token=token)

    device = "cpu"
    if torch.cuda.is_available():
        try:
            pipe.to(torch.device("cuda"))
            device = "cuda"
        except Exception as e:  # noqa: BLE001 - report and degrade to CPU
            print(f"[warn] pipeline -> cuda failed ({str(e)[:120]}); using CPU",
                  file=sys.stderr)
    if embedding_batch_size:
        try:
            pipe.embedding_batch_size = int(embedding_batch_size)
        except Exception:  # noqa: BLE001 - older pyannote may not expose it
            pass

    wav = _to_wav16k(audio_path)
    try:
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        t0 = time.time()
        annotation = pipe(wav, **kwargs)
        turns = [
            {"start": float(seg.start), "end": float(seg.end), "speaker": str(spk)}
            for seg, _, spk in annotation.itertracks(yield_label=True)
        ]
        n_spk = len({t["speaker"] for t in turns})
        print(f"[ok] {len(turns)} turns, {n_spk} speakers, device={device}, "
              f"{time.time() - t0:.1f}s", file=sys.stderr)
        return turns
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Attach a speaker to each transcript segment by max time-overlap with the turns.

    `segments` = [{'start','end','text', ...}]; returns the same dicts with 'speaker' set.
    """
    out = []
    for s in segments:
        s_start, s_end = float(s["start"]), float(s["end"])
        best_ov, best_spk = 0.0, "SPEAKER_UNK"
        for t in turns:
            ov = max(0.0, min(s_end, t["end"]) - max(s_start, t["start"]))
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
        out.append({**s, "speaker": best_spk})
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python diarize_standalone.py <audio> [num_speakers]")
    nspk = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for turn in diarize(sys.argv[1], num_speakers=nspk):
        print(f"{turn['start']:8.2f} - {turn['end']:8.2f}  {turn['speaker']}")
