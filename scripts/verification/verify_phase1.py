import sys
import torch
import torchaudio
import librosa
import soundfile
import pyworld
import yaml
import json

print("=== Phase 1: Environment & Repository Verification ===")
print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Torchaudio Version:", torchaudio.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
    mem_free, mem_total = torch.cuda.mem_get_info()
    print(f"GPU Memory: Free = {mem_free / (1024**3):.2f} GB / Total = {mem_total / (1024**3):.2f} GB")

print("Librosa Version:", librosa.__version__)
print("Soundfile Version:", soundfile.__version__)
print("PyWorld Version:", pyworld.__version__)

# Verify HiFi-GAN Vocoder Checkpoint
import os
hifigan_ckpt = "reference_fastspeech2/hifigan/generator_universal.pth.tar"
hifigan_cfg = "reference_fastspeech2/hifigan/config.json"
print("HiFi-GAN checkpoint exists:", os.path.exists(hifigan_ckpt), f"({os.path.getsize(hifigan_ckpt)/(1024**2):.1f} MB)")
print("HiFi-GAN config exists:", os.path.exists(hifigan_cfg))

# Test loading HiFi-GAN
sys.path.insert(0, "reference_fastspeech2")
import hifigan
with open(hifigan_cfg) as f:
    cfg = json.load(f)
cfg = hifigan.AttrDict(cfg)
generator = hifigan.Generator(cfg)
ckpt = torch.load(hifigan_ckpt, map_location="cpu")
generator.load_state_dict(ckpt["generator"])
generator.eval()
print("HiFi-GAN model loaded successfully!")
print("Phase 1 verification: PASS")

