import os
from pathlib import Path
from typing import Dict, Any, Union
import httpx


class QwenASRProvider:
    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        
        # Persistent HTTP connection pool for low latency
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self.client = httpx.Client(base_url=self.base_url, limits=limits, timeout=timeout)

    def _prepare_file_payload(self, audio: Union[str, Path, bytes]):
        """Convert a file path or raw wav bytes into a multipart file tuple."""
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            return ("audio.wav", open(audio_path, "rb"), "audio/wav")
        elif isinstance(audio, (bytes, bytearray)):
            return ("audio.wav", bytes(audio), "audio/wav")
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    def transcribe(self, audio: Union[str, Path, bytes]) -> str:
        """
        Transcribe a WAV audio file or raw WAV bytes to text.
        Returns the transcription text.
        """
        file_tuple = self._prepare_file_payload(audio)
        files = {"file": file_tuple}
        data = {
            "model": self.model_name,
            "response_format": "json",
        }

        try:
            response = self.client.post("/audio/transcriptions", files=files, data=data)
            response.raise_for_status()
            return response.json().get("text", "").strip()
        finally:
            # Safely close file descriptor if an opened file was passed
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()

    def transcribe_with_language(self, audio: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Transcribe an audio file/bytes and return structured data (language and text).
        """
        file_tuple = self._prepare_file_payload(audio)
        files = {"file": file_tuple}
        data = {
            "model": self.model_name,
            "response_format": "json",
        }

        try:
            response = self.client.post("/audio/transcriptions", files=files, data=data)
            response.raise_for_status()
            res_json = response.json()
            return {
                "language": res_json.get("language", "auto"),
                "text": res_json.get("text", "").strip(),
            }
        finally:
            if hasattr(file_tuple[1], "close"):
                file_tuple[1].close()

    def close(self):
        """Close connection pool."""
        self.client.close()
        
        