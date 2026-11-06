"""
Phase 1 Verification Script
Checks all critical imports and GPU functionality.
Run: venv_fs2\\Scripts\\python.exe verify_env.py
"""
import sys
print(f"Python: {sys.version}")

errors = []

# ── PyTorch + CUDA ──
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n[PyTorch]")
    print(f"  Version      : {torch.__version__}")
    print(f"  CUDA avail   : {cuda_ok}")
    if cuda_ok:
        print(f"  CUDA version : {torch.version.cuda}")
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM         : {vram:.1f} GB")

        # Quick GPU compute test
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"  Compute test : PASSED (1024×1024 matmul on GPU) ✓")
    else:
        errors.append("CUDA not available — PyTorch may be CPU-only build")
except Exception as e:
    errors.append(f"torch: {e}")

# ── Audio libraries ──
try:
    import librosa; print(f"\n[librosa]  {librosa.__version__} ✓")
except Exception as e: errors.append(f"librosa: {e}")

try:
    import soundfile; print(f"[soundfile] {soundfile.__version__} ✓")
except Exception as e: errors.append(f"soundfile: {e}")

try:
    import pyworld as pw; print(f"[pyworld]  installed ✓")
except Exception as e: errors.append(f"pyworld: {e}")

try:
    import resampy; print(f"[resampy]  installed ✓")
except Exception as e: errors.append(f"resampy: {e}")

# ── NLP / Text ──
try:
    from g2p_en import G2p
    g2p = G2p()
    out = g2p("Hello world")
    print(f"\n[g2p-en]   G2P test: 'Hello world' → {out} ✓")
except Exception as e: errors.append(f"g2p-en: {e}")

try:
    import tgt; print(f"[tgt]      {tgt.__version__} ✓")
except Exception as e: errors.append(f"tgt: {e}")

try:
    import inflect; print(f"[inflect]  installed ✓")
except Exception as e: errors.append(f"inflect: {e}")

try:
    from unidecode import unidecode; print(f"[unidecode] installed ✓")
except Exception as e: errors.append(f"unidecode: {e}")

# ── ML / Viz ──
try:
    import sklearn; print(f"\n[sklearn]  {sklearn.__version__} ✓")
except Exception as e: errors.append(f"sklearn: {e}")

try:
    import matplotlib; print(f"[matplotlib] {matplotlib.__version__} ✓")
except Exception as e: errors.append(f"matplotlib: {e}")

try:
    import yaml; print(f"[PyYAML]   installed ✓")
except Exception as e: errors.append(f"PyYAML: {e}")

try:
    import tensorboard; print(f"[tensorboard] {tensorboard.__version__} ✓")
except Exception as e: errors.append(f"tensorboard: {e}")

try:
    import numpy as np; print(f"[numpy]    {np.__version__} ✓")
except Exception as e: errors.append(f"numpy: {e}")

try:
    import scipy; print(f"[scipy]    {scipy.__version__} ✓")
except Exception as e: errors.append(f"scipy: {e}")

# ── Reference implementation import test ──
try:
    import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_fastspeech2"))
    from model.fastspeech2 import FastSpeech2
    from transformer.Models import Encoder, Decoder
    from model.modules import VarianceAdaptor
    from model.loss import FastSpeech2Loss
    from model.optimizer import ScheduledOptim
    import hifigan
    print(f"\n[reference_fastspeech2] All core imports OK ✓")
except Exception as e: errors.append(f"reference_fastspeech2 imports: {e}")

# ── HiFi-GAN weights ──
import os
hifigan_dir = os.path.join(os.path.dirname(__file__), "reference_fastspeech2", "hifigan")
for ckpt in ["generator_LJSpeech.pth.tar", "generator_universal.pth.tar", "config.json"]:
    path = os.path.join(hifigan_dir, ckpt)
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024**2
        print(f"[HiFi-GAN] {ckpt} found ({mb:.1f} MB) ✓")
    else:
        errors.append(f"HiFi-GAN weight missing: {ckpt}")

# ── Summary ──
print("\n" + "="*55)
if errors:
    print(f"❌ {len(errors)} error(s) found:")
    for e in errors:
        print(f"   • {e}")
else:
    print("✅ All checks passed — environment ready for Phase 3!")
print("="*55)

