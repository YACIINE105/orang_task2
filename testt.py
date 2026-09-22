from pathlib import Path
from src.stt.qwen_asr_provider import QwenASRProvider

def main():
    wav_path = Path("test2.wav")
    if not wav_path.exists():
        raise FileNotFoundError("test.wav does not exist. Run the ffmpeg command first.")

    print(f"[STT] Sending {wav_path.name} to Qwen ASR on vLLM...")
    stt = QwenASRProvider()
    
    # 1. Plain transcription
    text = stt.transcribe(wav_path)
    print(f"\nExtracted Text: {text}")

    # 2. Transcription with language detection
    result = stt.transcribe_with_language(wav_path)
    print(f"Detected Language: {result['language']}")

if __name__ == "__main__":
    main()
    