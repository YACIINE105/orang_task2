
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))

import torch
from qwen_asr import Qwen3ASRModel

_model = None


def get_model():
    """Lazy-load: only builds the model once per process, reuses after."""
    global _model
    if _model is not None:
        return _model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=dtype,
        device_map=device,
    )
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file to text. Returns the recognized text only."""
    model = get_model()
    results = model.transcribe(audio=audio_path)
    return results[0].text


def transcribe_with_language(audio_path: str) -> dict:
    """Same as transcribe(), but also returns detected language."""
    model = get_model()
    results = model.transcribe(audio=audio_path)
    return {"language": results[0].language, "text": results[0].text}

