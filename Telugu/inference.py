"""
Telugu FastSpeech 2 Inference Script (Phase 6)
Synthesizes mandatory test sentences (T1, T2a, T2b, T3, T4)
and arbitrary input text using the trained Telugu model and HiFi-GAN vocoder.
"""
import os
import sys
import re
import json
import yaml
import torch
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "reference_fastspeech2"))

from model import FastSpeech2
from text import text_to_sequence
from test_telugu_g2p import telugu_to_phonemes, normalize_telugu_numbers
import hifigan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hifigan(hifigan_dir, device):
    config_file = os.path.join(hifigan_dir, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(os.path.join(hifigan_dir, "generator_universal.pth.tar"), map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder

def load_model(config_path, ckpt_path, device):
    with open(os.path.join(BASE_DIR, config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    preprocess_config = {
        "dataset": config["dataset"],
        "path": config["path"],
        "preprocessing": config["preprocessing"]
    }
    model_config = config["model"]
    
    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config

def synthesize_sentence(model, vocoder, text_seq, speaker_id=0, p_control=1.0, e_control=1.0, d_control=1.0):
    seq_len = np.array([len(text_seq)])
    text_tensor = torch.from_numpy(text_seq).long().unsqueeze(0).to(device)
    src_len = torch.from_numpy(seq_len).long().to(device)
    speakers = torch.tensor([speaker_id]).long().to(device)
    max_src_len = torch.max(src_len).item()
    
    with torch.no_grad():
        output = model(
            speakers,
            text_tensor,
            src_len,
            max_src_len,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control
        )
        (
            raw_mel,
            mel_postnet,
            pitch_pred,
            energy_pred,
            log_dur_pred,
            d_rounded,
            src_mask,
            mel_mask,
            src_len_out,
            mel_len,
        ) = output
        
        mel_postnet_T = mel_postnet.transpose(1, 2)
        wav = vocoder(mel_postnet_T)
        wav = wav.squeeze().cpu().numpy()
        
        # Audio normalization: scale to 0.95 peak amplitude
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = (wav / max_val) * 0.95
        
    duration_pred = torch.round(torch.exp(log_dur_pred) - 1).squeeze().cpu().numpy()
    pitch_pred = pitch_pred.squeeze().cpu().numpy()
    energy_pred = energy_pred.squeeze().cpu().numpy()
    mel_out = mel_postnet.squeeze().cpu().numpy()
    
    return wav, duration_pred, pitch_pred, energy_pred, mel_out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="all", help="Sentence key to synthesize (e.g. T1) or 'all'")
    parser.add_argument("--ckpt", type=str, default="best_model.pth.tar", help="Checkpoint filename in checkpoints/")
    args = parser.parse_args()
    
    model_ckpt = os.path.join(BASE_DIR, "Telugu", "checkpoints", args.ckpt)
    if not os.path.exists(model_ckpt):
        model_ckpt = os.path.join(BASE_DIR, "Telugu", "checkpoints", "best_model.pth.tar")
        
    print(f"Loading Telugu model from: {model_ckpt}")
    model, config = load_model("Telugu/config.yaml", model_ckpt, device)
    
    hifigan_dir = os.path.join(BASE_DIR, "reference_fastspeech2", "hifigan")
    print(f"Loading HiFi-GAN Vocoder from: {hifigan_dir}")
    vocoder = load_hifigan(hifigan_dir, device)
    
    test_sentences = {
        "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "T2b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    }
    
    out_dir = os.path.join(BASE_DIR, "Results", "Telugu", "test_generated_audios")
    os.makedirs(out_dir, exist_ok=True)
    
    predictions_cache = {}
    
    if args.sentence != "all":
        if args.sentence in test_sentences:
            test_sentences = {args.sentence: test_sentences[args.sentence]}
        else:
            print(f"Error: Sentence key {args.sentence} not found.")
            return

    for key, raw_text in test_sentences.items():
        norm_text = normalize_telugu_numbers(raw_text)
        phones = telugu_to_phonemes(raw_text)
        seq = np.array(text_to_sequence(phones, ["english_cleaners"]))
        
        print(f"\n--- Synthesizing {key} ---")
        print(f"Tokens: {len(seq)} | Phonemes: {phones[:60]}...")
        
        # Synthesize with primary speaker (speaker 0: female)
        wav, dur, pitch, energy, mel = synthesize_sentence(model, vocoder, seq, speaker_id=0)
        out_wav_path = os.path.join(out_dir, f"{key}.wav")
        sf.write(out_wav_path, wav, 22050)
        print(f"--> Saved audio to: {out_wav_path} (Duration: {len(wav)/22050:.2f}s)")
        
        predictions_cache[key] = {
            "raw": raw_text,
            "norm": norm_text,
            "phones": phones,
            "seq": seq.tolist(),
            "duration": dur.tolist(),
            "pitch": pitch.tolist(),
            "energy": energy.tolist()
        }
        
    cache_path = os.path.join(BASE_DIR, "Results", "Telugu", "test_predictions.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions_cache, f, indent=2, ensure_ascii=False)
    print(f"\nAll Telugu mandatory test sentences synthesized successfully! Saved in {out_dir}")

if __name__ == "__main__":
    main()

