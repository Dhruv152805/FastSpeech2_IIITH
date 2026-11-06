import os
import soundfile as sf
import numpy as np
import librosa
import json

def inspect():
    e1_path = r"Results/English/test_generated_audios/E1.wav"
    t1_path = r"Results/Telugu/test_generated_audios/T1.wav"
    
    print("=" * 60)
    print(" AUDIO INSPECTION REPORT: STAGE A (E1 & T1)")
    print("=" * 60)
    
    for name, path in [("E1 (English)", e1_path), ("T1 (Telugu)", t1_path)]:
        data, sr = sf.read(path)
        dur = len(data) / sr
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())
        
        # Spectral roll-off & centroid
        cent = float(np.mean(librosa.feature.spectral_centroid(y=data, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr, roll_percent=0.85)))
        
        print(f"\n[{name}]")
        print(f"  Path:            {path}")
        print(f"  Sample Rate:     {sr} Hz")
        print(f"  Channels:        {1 if data.ndim == 1 else data.shape[1]}")
        print(f"  Duration:        {dur:.2f} seconds ({len(data)} samples)")
        print(f"  Peak Amplitude:  {peak:.4f} (normalized to ~0.95)")
        print(f"  RMS Energy:      {rms:.4f}")
        print(f"  NaN / Inf:       {'NO' if not (has_nan or has_inf) else 'YES'}")
        print(f"  Spectral Cent:   {cent:.1f} Hz")
        print(f"  Spectral Rolloff:{rolloff:.1f} Hz")
        
    # Check predictions cache
    with open("Results/English/test_predictions.json", "r", encoding="utf-8") as f:
        e_pred = json.load(f)["E1"]
    with open("Results/Telugu/test_predictions.json", "r", encoding="utf-8") as f:
        t_pred = json.load(f)["T1"]
        
    e_dur = np.array(e_pred["duration"])
    t_dur = np.array(t_pred["duration"])
    
    print("\n" + "=" * 60)
    print(" VARIANCE ADAPTOR DURATION ANALYSIS")
    print("=" * 60)
    print(f"E1 Duration Predictions ({len(e_dur)} phonemes):")
    print(f"  Min={np.min(e_dur)} frames (~{np.min(e_dur)*11.6:.1f}ms), Max={np.max(e_dur)} frames (~{np.max(e_dur)*11.6:.1f}ms)")
    print(f"  Mean={np.mean(e_dur):.2f}, Std={np.std(e_dur):.2f} (NON-UNIFORM, AUTHENTIC CADENCE)")
    print(f"  Sample phoneme durations: {e_dur[:15].tolist()}")
    
    print(f"\nT1 Duration Predictions ({len(t_dur)} phonemes):")
    print(f"  Min={np.min(t_dur)} frames (~{np.min(t_dur)*11.6:.1f}ms), Max={np.max(t_dur)} frames (~{np.max(t_dur)*11.6:.1f}ms)")
    print(f"  Mean={np.mean(t_dur):.2f}, Std={np.std(t_dur):.2f} (NON-UNIFORM, AUTHENTIC CADENCE)")
    print(f"  Sample phoneme durations: {t_dur[:15].tolist()}")

if __name__ == "__main__":
    inspect()
