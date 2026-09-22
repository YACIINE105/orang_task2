import time, torch
from src.stt.qwen_asr_provider import QwenASRProvider

print('=== Hardware Check ===')
print('PyTorch Version:', torch.__version__)
print('CUDA Available in PyTorch:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))

print('\n=== Starting ASR Benchmark on test.wav ===')
t_start = time.perf_counter()

# 1. Measure Model Load Time
t0 = time.perf_counter()
provider = QwenASRProvider()
model = provider.get_model()
t_load = time.perf_counter() - t0
print(f'[1/2] Model Load Time: {t_load:.2f} seconds')

# 2. Measure Transcription Inference Time
t1 = time.perf_counter()
transcription = provider.transcribe('test.wav')
t_infer = time.perf_counter() - t1
print(f'[2/2] ASR Inference Time: {t_infer:.2f} seconds')

t_total = time.perf_counter() - t_start
print(f'Total Elapsed Time: {t_total:.2f} seconds')
print('----------------------------------------')
print('Transcription Result:')
print(transcription)
print('----------------------------------------')

