import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
from qwen_asr import Qwen3ASRModel


class QwenASRProvider:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-0.6B"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.hf_cache = self.project_root / "models" / "hf"
        
        # Check if local snapshot directory exists
        snapshot_dir = list(self.hf_cache.glob("hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/*"))
        if snapshot_dir:
            self.model_name = str(snapshot_dir[0])
        else:
            self.model_name = model_name

        self._model: Optional[Qwen3ASRModel] = None

    def get_model(self) -> Qwen3ASRModel:
        """Lazy-loads the ASR model on first invocation."""
        if self._model is not None:
            return self._model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device,
        )
        return self._model

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribe an audio file to text. Returns transcription text only."""
        model = self.get_model()
        results = model.transcribe(audio=str(audio_path))
        return results[0].text

    def transcribe_with_language(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribe an audio file and return detected language along with text."""
        model = self.get_model()
        results = model.transcribe(audio=str(audio_path))
        return {"language": results[0].language, "text": results[0].text}
    