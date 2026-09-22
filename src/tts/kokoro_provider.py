import io
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, Any

import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files


class KokoroTTSProvider:
    REPO_ID = "oddadmix/Kokoro-7M-Distill"

    def __init__(self, repo_id: Optional[str] = None):
        self.repo_id = repo_id or self.REPO_ID
        self.patched_dir = Path(__file__).parent / "_kokoro_patched"
        self.patched_kokoro_pkg = self.patched_dir / "kokoro"
        self._model: Optional[Any] = None
        self._pipeline: Optional[Any] = None

    def _ensure_patched_kokoro(self) -> None:
        if (self.patched_kokoro_pkg / "__init__.py").exists():
            return

        patched_files = [
            "kokoro_patched/kokoro/__init__.py",
            "kokoro_patched/kokoro/__main__.py",
            "kokoro_patched/kokoro/custom_stft.py",
            "kokoro_patched/kokoro/istftnet.py",
            "kokoro_patched/kokoro/model.py",
            "kokoro_patched/kokoro/modules.py",
            "kokoro_patched/kokoro/pipeline.py",
        ]
        self.patched_kokoro_pkg.mkdir(parents=True, exist_ok=True)
        for f in patched_files:
            downloaded = hf_hub_download(self.repo_id, f)
            local_name = f.split("/")[-1]
            shutil.copyfile(downloaded, self.patched_kokoro_pkg / local_name)

    def _purge_cached_kokoro_module(self) -> None:
        for mod_name in list(sys.modules):
            if mod_name == "kokoro" or mod_name.startswith("kokoro."):
                del sys.modules[mod_name]

    def get_pipeline(self) -> Tuple[Any, Any]:
        if self._model is not None and self._pipeline is not None:
            return self._model, self._pipeline

        self._ensure_patched_kokoro()
        sys.path.insert(0, str(self.patched_dir.absolute()))
        self._purge_cached_kokoro_module()

        from kokoro import KModel, KPipeline

        files = list_repo_files(self.repo_id)
        weight_file = next(f for f in files if f.endswith((".pth", ".safetensors")))
        config_file_name = next(f for f in files if f.endswith("config.json"))

        model_path = hf_hub_download(self.repo_id, weight_file)
        config_path = hf_hub_download(self.repo_id, config_file_name)

        self._model = KModel(config=config_path, model=model_path)
        self._pipeline = KPipeline(lang_code="a", model=self._model)
        return self._model, self._pipeline

    def synthesize(self, text: str, voice: str = "af_heart", out_path: str = "output.wav") -> str:
        """Generates audio and writes directly to an output file."""
        _, pipeline = self.get_pipeline()
        generator = pipeline(text, voice=voice)
        for _, _, audio in generator:
            sf.write(out_path, audio, 24000)
            return out_path
        return ""

    def synthesize_stream_bytes(self, text: str, voice: str = "af_heart") -> bytes:
        """Generates raw audio in-memory for live FastAPI chunk streaming."""
        _, pipeline = self.get_pipeline()
        generator = pipeline(text, voice=voice)
        buffer = io.BytesIO()
        for _, _, audio in generator:
            sf.write(buffer, audio, 24000, format="RAW", subtype="PCM_16")
            return buffer.getvalue()
        return b""
    