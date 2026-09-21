import sys
import shutil
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "oddadmix/Kokoro-7M-Distill"

PATCHED_DIR = Path(__file__).parent / "_kokoro_patched"
PATCHED_KOKORO_PKG = PATCHED_DIR / "kokoro"

_model = None
_pipeline = None


def _ensure_patched_kokoro():
    if (PATCHED_KOKORO_PKG / "__init__.py").exists():
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
    PATCHED_KOKORO_PKG.mkdir(parents=True, exist_ok=True)
    for f in patched_files:
        downloaded = hf_hub_download(REPO_ID, f)
        local_name = f.split("/")[-1]
        shutil.copyfile(downloaded, PATCHED_KOKORO_PKG / local_name)


def _purge_cached_kokoro_module():
    for mod_name in list(sys.modules):
        if mod_name == "kokoro" or mod_name.startswith("kokoro."):
            del sys.modules[mod_name]


def get_model_and_pipeline():
    global _model, _pipeline
    if _model is not None:
        return _model, _pipeline

    _ensure_patched_kokoro()
    sys.path.insert(0, str(PATCHED_DIR.absolute()))
    _purge_cached_kokoro_module()

    from kokoro import KModel, KPipeline

    files = list_repo_files(REPO_ID)
    weight_file = next(f for f in files if f.endswith((".pth", ".safetensors")))
    config_file_name = next(f for f in files if f.endswith("config.json"))

    model_path = hf_hub_download(REPO_ID, weight_file)
    config_path = hf_hub_download(REPO_ID, config_file_name)

    _model = KModel(config=config_path, model=model_path)
    _pipeline = KPipeline(lang_code="a", model=_model)
    return _model, _pipeline


def synthesize(text: str, voice: str = "af_heart", out_path: str = "output.wav") -> str:
    _, pipeline = get_model_and_pipeline()
    generator = pipeline(text, voice=voice)
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(out_path, audio, 24000)
        return out_path
    
    