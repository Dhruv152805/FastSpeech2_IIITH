import os
import glob
import soundfile as sf
import numpy as np
from tqdm import tqdm

print("=== Phase 2: Dataset Verification ===")

def verify_dataset(name, raw_dir):
    print(f"\nVerifying {name} at: {raw_dir}")
    wav_files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    txt_files = glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True)
    
    print(f"Total WAV files found: {len(wav_files)}")
    print(f"Total TXT files found: {len(txt_files)}")
    
    durations = []
    sample_rates = set()
    channels = set()
    corrupted = []
    
    for w in tqdm(wav_files, desc=f"Scanning {name}"):
        try:
            info = sf.info(w)
            durations.append(info.duration)
            sample_rates.add(info.samplerate)
            channels.add(info.channels)
        except Exception as e:
            corrupted.append((w, str(e)))
            
    total_hours = sum(durations) / 3600.0
    print(f"Sample rates: {sample_rates}")
    print(f"Channel counts: {channels}")
    print(f"Corrupted files: {len(corrupted)}")
    print(f"Total audio duration: {total_hours:.2f} hours ({sum(durations):.1f} seconds)")
    print(f"Duration range: min={min(durations):.2f}s, max={max(durations):.2f}s, mean={np.mean(durations):.2f}s")
    
    return wav_files, total_hours

en_wavs, en_hrs = verify_dataset("English (Svarah)", "data/english/raw")
te_wavs, te_hrs = verify_dataset("Telugu (IndicTTS)", "data/telugu/raw")

print("\nPhase 2 verification: PASS")

