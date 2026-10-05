"""Thin wrapper around NVIDIA Nemotron 3 Diarization (Hugging Face Transformers backend)."""

import os
import subprocess
import threading
import time

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any ffmpeg-readable file to a 16 kHz mono float32 waveform."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"Could not decode audio: {proc.stderr.decode(errors='ignore').strip()}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio file contains no samples")
    return audio


class Diarizer:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "float32"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id, dtype=self.dtype).to(device)
        self.model.eval()
        # one GPU, one model: serialise requests
        self._lock = threading.Lock()

    def diarize(self, audio: np.ndarray, threshold: float = 0.5, min_duration: float = 0.0) -> dict:
        start = time.perf_counter()
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE).to(self.device, dtype=self.dtype)
        with self._lock, torch.inference_mode():
            logits = self.model(**inputs).logits.float()
        raw = self.processor.extract_speaker_dict(logits, inputs.attention_mask, threshold=threshold)[0]

        segments = [
            {"start": s["Start"], "end": s["End"], "speaker": f"speaker_{s['Speaker']}"}
            for s in raw
            if s["End"] - s["Start"] >= min_duration
        ]
        speakers = sorted({s["speaker"] for s in segments}, key=lambda x: int(x.split("_")[1]))
        return {
            "duration": round(len(audio) / SAMPLE_RATE, 2),
            "num_speakers": len(speakers),
            "speakers": speakers,
            "segments": segments,
            "processing_time": round(time.perf_counter() - start, 3),
        }


def to_rttm(result: dict, file_id: str = "audio") -> str:
    return "".join(
        f"SPEAKER {file_id} 1 {s['start']:.2f} {s['end'] - s['start']:.2f} <NA> <NA> {s['speaker']} <NA> <NA>\n"
        for s in result["segments"]
    )


_diarizer: Diarizer | None = None


def get_diarizer() -> Diarizer:
    global _diarizer
    if _diarizer is None:
        _diarizer = Diarizer(
            model_id=os.getenv("MODEL_ID", "nvidia/Nemotron-3-Diarization"),
            device=os.getenv("DEVICE", "auto"),
            dtype=os.getenv("DTYPE", "float32"),
        )
    return _diarizer
