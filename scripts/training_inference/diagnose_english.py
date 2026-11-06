"""
Comprehensive Diagnostic Script for English Audio Distortion
Performs controlled experiments across:
1. Ground-Truth Mel vs Predicted Mel through HiFi-GAN (English and Telugu)
2. Numerical statistics of Mel-spectrograms
3. Inspection of 5 English validation samples (audio, transcript, phonemes, durations, mel, F0, energy)
4. Phoneme sequence and ID verification between training and inference
5. Generating comparison plots
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "scratch", "mpl")
os.makedirs(os.path.join(BASE_DIR, "scratch", "mpl"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "scratch", "diagnosis"), exist_ok=True)

import json
import yaml
import torch
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "reference_fastspeech2"))

from model import FastSpeech2
from text import text_to_sequence
import hifigan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hifigan_vocoder(hifigan_dir):
    with open(os.path.join(hifigan_dir, "config.json"), "r") as f:
        config = hifigan.AttrDict(json.load(f))
    vocoder = hifigan.Generator(config).to(device)
    ckpt = torch.load(os.path.join(hifigan_dir, "generator_universal.pth.tar"), map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder, config

def load_fs2_model(config_path, ckpt_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    preprocess_config = {
        "dataset": config["dataset"],
        "path": config["path"],
        "preprocessing": config["preprocessing"]
    }
    model = FastSpeech2(preprocess_config, config["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, preprocess_config

def main():
    print("=" * 70)
    print(" STARTING CONTROLLED HIFI-GAN & ACOUSTIC DIAGNOSIS")
    print("=" * 70)
    
    diag_dir = os.path.join(BASE_DIR, "scratch", "diagnosis")
    hifigan_dir = os.path.join(BASE_DIR, "reference_fastspeech2", "hifigan")
    vocoder, hifi_cfg = load_hifigan_vocoder(hifigan_dir)
    print("HiFi-GAN loaded successfully.")
    
    # ----------------------------------------------------
    # TEST 2A: English Ground-Truth Mel through HiFi-GAN
    # ----------------------------------------------------
    with open("data/english/preprocessed/val.txt", "r", encoding="utf-8") as f:
        eng_val_lines = [l.strip() for l in f if l.strip()]
        
    eng_sample_line = eng_val_lines[0]
    e_uid, e_spk, e_phones, e_raw = eng_sample_line.split("|")
    print(f"\n[Selected English Val Sample]: {e_uid}")
    print(f"  Raw: {e_raw}")
    print(f"  Phones: {e_phones[:60]}...")
    
    eng_gt_mel_path = os.path.join("data/english/preprocessed/mel", f"{e_spk}-mel-{e_uid}.npy")
    eng_gt_mel = np.load(eng_gt_mel_path) # shape: (n_frames, 80)
    print(f"  GT Mel shape: {eng_gt_mel.shape}, min: {eng_gt_mel.min():.4f}, max: {eng_gt_mel.max():.4f}, mean: {eng_gt_mel.mean():.4f}, std: {eng_gt_mel.std():.4f}")
    
    # Pass GT Mel to HiFi-GAN (mel expects [1, 80, T])
    with torch.no_grad():
        gt_mel_t = torch.from_numpy(eng_gt_mel.T).unsqueeze(0).float().to(device)
        e_gt_wav = vocoder(gt_mel_t).squeeze().cpu().numpy()
        # Normalization
        if np.max(np.abs(e_gt_wav)) > 0:
            e_gt_wav_norm = (e_gt_wav / np.max(np.abs(e_gt_wav))) * 0.95
        else:
            e_gt_wav_norm = e_gt_wav
            
    e_gt_wav_path = os.path.join(diag_dir, "eng_val_gt_mel.wav")
    sf.write(e_gt_wav_path, e_gt_wav_norm, 22050)
    print(f"  Saved English GT Mel Audio: {e_gt_wav_path} (Dur: {len(e_gt_wav_norm)/22050:.2f}s, RMS: {np.sqrt(np.mean(e_gt_wav_norm**2)):.4f})")
    
    # ----------------------------------------------------
    # TEST 2B: English Predicted Mel through HiFi-GAN
    # ----------------------------------------------------
    eng_model, eng_cfg, _ = load_fs2_model("English/config.yaml", "English/checkpoints/best_model.pth.tar")
    e_seq = np.array(text_to_sequence(e_phones, ["english_cleaners"]))
    
    with torch.no_grad():
        src_t = torch.from_numpy(e_seq).unsqueeze(0).long().to(device)
        src_len = torch.tensor([len(e_seq)]).long().to(device)
        spk_t = torch.tensor([0]).long().to(device)
        out = eng_model(spk_t, src_t, src_len, len(e_seq))
        (e_raw_mel, e_pred_mel, e_p_pred, e_e_pred, e_log_dur, e_d_round, _, _, _, _) = out
        
        e_pred_mel_t = e_pred_mel.transpose(1, 2)
        e_pred_wav = vocoder(e_pred_mel_t).squeeze().cpu().numpy()
        if np.max(np.abs(e_pred_wav)) > 0:
            e_pred_wav_norm = (e_pred_wav / np.max(np.abs(e_pred_wav))) * 0.95
        else:
            e_pred_wav_norm = e_pred_wav
            
    e_pred_wav_path = os.path.join(diag_dir, "eng_val_pred_mel.wav")
    sf.write(e_pred_wav_path, e_pred_wav_norm, 22050)
    e_pred_mel_np = e_pred_mel.squeeze(0).cpu().numpy()
    print(f"  Pred Mel shape: {e_pred_mel_np.shape}, min: {e_pred_mel_np.min():.4f}, max: {e_pred_mel_np.max():.4f}, mean: {e_pred_mel_np.mean():.4f}, std: {e_pred_mel_np.std():.4f}")
    print(f"  Saved English Pred Mel Audio: {e_pred_wav_path} (Dur: {len(e_pred_wav_norm)/22050:.2f}s, RMS: {np.sqrt(np.mean(e_pred_wav_norm**2)):.4f})")
    
    # ----------------------------------------------------
    # TEST 2C & 2D: Telugu GT and Pred Mel through HiFi-GAN
    # ----------------------------------------------------
    with open("data/telugu/preprocessed/val.txt", "r", encoding="utf-8") as f:
        tel_val_lines = [l.strip() for l in f if l.strip()]
        
    tel_sample_line = tel_val_lines[0]
    t_uid, t_spk, t_phones, t_raw = tel_sample_line.split("|")
    print(f"\n[Selected Telugu Val Sample]: {t_uid}")
    print(f"  Raw: {t_raw}")
    print(f"  Phones: {t_phones[:60]}...")
    
    tel_gt_mel_path = os.path.join("data/telugu/preprocessed/mel", f"{t_spk}-mel-{t_uid}.npy")
    tel_gt_mel = np.load(tel_gt_mel_path)
    print(f"  GT Mel shape: {tel_gt_mel.shape}, min: {tel_gt_mel.min():.4f}, max: {tel_gt_mel.max():.4f}, mean: {tel_gt_mel.mean():.4f}, std: {tel_gt_mel.std():.4f}")
    
    with torch.no_grad():
        t_gt_mel_t = torch.from_numpy(tel_gt_mel.T).unsqueeze(0).float().to(device)
        t_gt_wav = vocoder(t_gt_mel_t).squeeze().cpu().numpy()
        if np.max(np.abs(t_gt_wav)) > 0:
            t_gt_wav_norm = (t_gt_wav / np.max(np.abs(t_gt_wav))) * 0.95
        else:
            t_gt_wav_norm = t_gt_wav
            
    t_gt_wav_path = os.path.join(diag_dir, "tel_val_gt_mel.wav")
    sf.write(t_gt_wav_path, t_gt_wav_norm, 22050)
    print(f"  Saved Telugu GT Mel Audio: {t_gt_wav_path} (Dur: {len(t_gt_wav_norm)/22050:.2f}s, RMS: {np.sqrt(np.mean(t_gt_wav_norm**2)):.4f})")
    
    tel_model, tel_cfg, _ = load_fs2_model("Telugu/config.yaml", "Telugu/checkpoints/best_model.pth.tar")
    t_seq = np.array(text_to_sequence(t_phones, ["english_cleaners"]))
    t_spk_id = 0 if "female" in t_spk else 1
    
    with torch.no_grad():
        src_t = torch.from_numpy(t_seq).unsqueeze(0).long().to(device)
        src_len = torch.tensor([len(t_seq)]).long().to(device)
        spk_t = torch.tensor([t_spk_id]).long().to(device)
        out = tel_model(spk_t, src_t, src_len, len(t_seq))
        (_, t_pred_mel, _, _, _, _, _, _, _, _) = out
        
        t_pred_mel_t = t_pred_mel.transpose(1, 2)
        t_pred_wav = vocoder(t_pred_mel_t).squeeze().cpu().numpy()
        if np.max(np.abs(t_pred_wav)) > 0:
            t_pred_wav_norm = (t_pred_wav / np.max(np.abs(t_pred_wav))) * 0.95
        else:
            t_pred_wav_norm = t_pred_wav
            
    t_pred_wav_path = os.path.join(diag_dir, "tel_val_pred_mel.wav")
    sf.write(t_pred_wav_path, t_pred_wav_norm, 22050)
    t_pred_mel_np = t_pred_mel.squeeze(0).cpu().numpy()
    print(f"  Pred Mel shape: {t_pred_mel_np.shape}, min: {t_pred_mel_np.min():.4f}, max: {t_pred_mel_np.max():.4f}, mean: {t_pred_mel_np.mean():.4f}, std: {t_pred_mel_np.std():.4f}")
    print(f"  Saved Telugu Pred Mel Audio: {t_pred_wav_path} (Dur: {len(t_pred_wav_norm)/22050:.2f}s, RMS: {np.sqrt(np.mean(t_pred_wav_norm**2)):.4f})")
    
    # ----------------------------------------------------
    # Plot Comparison: GT Mel vs Pred Mel (English & Telugu)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    im0 = axes[0, 0].imshow(eng_gt_mel.T, aspect='auto', origin='lower', vmin=-12, vmax=2, cmap='viridis')
    axes[0, 0].set_title(f"English GT Mel ({e_uid})")
    axes[0, 0].set_ylabel("Mel Channels (80)")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(e_pred_mel_np.T, aspect='auto', origin='lower', vmin=-12, vmax=2, cmap='viridis')
    axes[0, 1].set_title("English FastSpeech 2 Predicted Mel")
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(tel_gt_mel.T, aspect='auto', origin='lower', vmin=-12, vmax=2, cmap='viridis')
    axes[1, 0].set_title(f"Telugu GT Mel ({t_uid})")
    axes[1, 0].set_xlabel("Frames")
    axes[1, 0].set_ylabel("Mel Channels (80)")
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(t_pred_mel_np.T, aspect='auto', origin='lower', vmin=-12, vmax=2, cmap='viridis')
    axes[1, 1].set_title("Telugu FastSpeech 2 Predicted Mel")
    axes[1, 1].set_xlabel("Frames")
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plot_path = os.path.join(diag_dir, "mel_comparison_gt_vs_pred.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot: {plot_path}")
    
    # ----------------------------------------------------
    # TEST 4: Inspect 5 English Validation Samples
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(" INSPECTING 5 ENGLISH VALIDATION SAMPLES")
    print("=" * 70)
    
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(eng_val_lines), 5, replace=False)
    
    val_inspections = []
    for s_i in sample_indices:
        line = eng_val_lines[s_i]
        uid, spk, ph_str, raw = line.split("|")
        
        # Audio
        wav_path = os.path.join("data/english/raw/svarah_speaker", f"{uid}.wav")
        audio_exists = os.path.exists(wav_path)
        if audio_exists:
            wav_d, sr = sf.read(wav_path)
            raw_dur = len(wav_d) / sr
        else:
            raw_dur, sr = 0, 0
            
        # Features
        mel_f = os.path.join("data/english/preprocessed/mel", f"{spk}-mel-{uid}.npy")
        dur_f = os.path.join("data/english/preprocessed/duration", f"{spk}-duration-{uid}.npy")
        pitch_f = os.path.join("data/english/preprocessed/pitch", f"{spk}-pitch-{uid}.npy")
        energy_f = os.path.join("data/english/preprocessed/energy", f"{spk}-energy-{uid}.npy")
        
        mel = np.load(mel_f)
        dur = np.load(dur_f)
        pitch = np.load(pitch_f)
        energy = np.load(energy_f)
        
        phones = ph_str.strip("{}").split()
        
        entry = {
            "id": uid,
            "raw_text": raw,
            "raw_audio_dur": raw_dur,
            "raw_sr": sr,
            "phones_count": len(phones),
            "durations_count": len(dur),
            "sum_durations": int(np.sum(dur)),
            "mel_frames": mel.shape[0],
            "mel_matches_sum_dur": bool(mel.shape[0] == np.sum(dur)),
            "dur_min": int(np.min(dur)),
            "dur_max": int(np.max(dur)),
            "dur_mean": float(np.mean(dur)),
            "dur_std": float(np.std(dur)),
            "pitch_mean": float(np.mean(pitch)),
            "energy_mean": float(np.mean(energy)),
            "sample_phones": phones[:8],
            "sample_durations": dur[:8].tolist()
        }
        val_inspections.append(entry)
        
        print(f"\n[Sample {uid}]")
        print(f"  Raw text:        {raw}")
        print(f"  Audio duration:  {raw_dur:.2f}s (SR: {sr}) | Mel frames: {mel.shape[0]} (~{mel.shape[0]*256/22050:.2f}s)")
        print(f"  Phones count:    {len(phones)} | Durations count: {len(dur)}")
        print(f"  Sum(durations):  {np.sum(dur)} (Match: {mel.shape[0] == np.sum(dur)})")
        print(f"  Duration stats:  min={np.min(dur)}, max={np.max(dur)}, mean={np.mean(dur):.2f}, std={np.std(dur):.2f}")
        print(f"  Sample phones:   {' '.join(phones[:8])}")
        print(f"  Sample durs:     {dur[:8].tolist()}")
        
    # Save results to json for reporting
    diag_summary = {
        "eng_gt_mel_stats": {
            "shape": list(eng_gt_mel.shape),
            "min": float(eng_gt_mel.min()),
            "max": float(eng_gt_mel.max()),
            "mean": float(eng_gt_mel.mean()),
            "std": float(eng_gt_mel.std()),
            "has_nan": bool(np.isnan(eng_gt_mel).any())
        },
        "eng_pred_mel_stats": {
            "shape": list(e_pred_mel_np.shape),
            "min": float(e_pred_mel_np.min()),
            "max": float(e_pred_mel_np.max()),
            "mean": float(e_pred_mel_np.mean()),
            "std": float(e_pred_mel_np.std()),
            "has_nan": bool(np.isnan(e_pred_mel_np).any())
        },
        "tel_gt_mel_stats": {
            "shape": list(tel_gt_mel.shape),
            "min": float(tel_gt_mel.min()),
            "max": float(tel_gt_mel.max()),
            "mean": float(tel_gt_mel.mean()),
            "std": float(tel_gt_mel.std()),
            "has_nan": bool(np.isnan(tel_gt_mel).any())
        },
        "tel_pred_mel_stats": {
            "shape": list(t_pred_mel_np.shape),
            "min": float(t_pred_mel_np.min()),
            "max": float(t_pred_mel_np.max()),
            "mean": float(t_pred_mel_np.mean()),
            "std": float(t_pred_mel_np.std()),
            "has_nan": bool(np.isnan(t_pred_mel_np).any())
        },
        "val_inspections": val_inspections
    }
    with open(os.path.join(diag_dir, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\nDiagnostic summary saved to {os.path.join(diag_dir, 'diagnostic_summary.json')}")

if __name__ == "__main__":
    main()

