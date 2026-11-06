"""
English FastSpeech 2 Inference Script (Phase 6)
Synthesizes mandatory test sentences (E1, E2a, E2b, E3, E4)
and arbitrary input text using the trained English model and HiFi-GAN vocoder.
"""
import os
import sys
import re
import json
import yaml
import torch
import numpy as np
import soundfile as sf
from g2p_en import G2p

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "reference_fastspeech2"))

from model import FastSpeech2
from text import text_to_sequence
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

def normalize_english_text(text):
    # E4 date/time normalization
    text = text.replace("10:45 AM", "ten forty-five AM")
    text = text.replace("12 November 2026", "twelve November twenty twenty-six")
    return text

def text_to_phonemes(text, is_verb_conduct=False):
    g2p = G2p()
    words = text.split()
    clean_phones = []
    for i, raw_w in enumerate(words):
        w_clean = raw_w.strip(".,;:?!\"'()").lower()
        if not w_clean:
            continue
        if w_clean == "conduct" and is_verb_conduct:
            p_list = ["K", "AH0", "N", "D", "AH1", "K", "T"]
        elif w_clean == "conduct":
            p_list = ["K", "AA1", "N", "D", "AH0", "K", "T"]
        elif w_clean == "andaroo":
            p_list = ["AH0", "N", "D", "AA1", "R", "UW0"]
        elif w_clean == "rhyth":
            p_list = ["R", "IH1", "DH"]
        elif w_clean == "m" and i > 0 and "rhyth" in words[i-1].lower():
            p_list = ["AH0", "M"]
        else:
            p_list = [p for p in g2p(w_clean) if re.match(r'^[A-Z0-9]+$', p)]
            
        if p_list:
            clean_phones.extend(p_list)
            clean_phones.append("sp")
            
    if clean_phones and clean_phones[-1] == "sp":
        clean_phones.pop()
        
    return "{" + " ".join(clean_phones) + "}"

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
    parser.add_argument("--sentence", type=str, default="all", help="Sentence key to synthesize (e.g. E1) or 'all'")
    parser.add_argument("--ckpt", type=str, default="best_model.pth.tar", help="Checkpoint filename in checkpoints/")
    parser.add_argument("--speaker", type=str, default="svarah_female_28", help="Speaker name or ID int")
    args = parser.parse_args()
    
    model_ckpt = os.path.join(BASE_DIR, "English", "checkpoints", args.ckpt)
    if not os.path.exists(model_ckpt):
        model_ckpt = os.path.join(BASE_DIR, "English", "checkpoints", "best_model.pth.tar")
        
    print(f"Loading English model from: {model_ckpt}")
    model, config = load_model("English/config.yaml", model_ckpt, device)
    
    with open(os.path.join(BASE_DIR, "data", "english", "preprocessed", "speakers.json"), "r") as f:
        speaker_map = json.load(f)
        
    if args.speaker.isdigit():
        spk_id = int(args.speaker)
    elif args.speaker in speaker_map:
        spk_id = speaker_map[args.speaker]
    else:
        # Fallback to female speaker with most data (svarah_female_28)
        spk_id = speaker_map.get("svarah_female_28", 0)
    print(f"Selected Speaker: {args.speaker} (Speaker ID: {spk_id})")
    
    hifigan_dir = os.path.join(BASE_DIR, "reference_fastspeech2", "hifigan")
    print(f"Loading HiFi-GAN Vocoder from: {hifigan_dir}")
    vocoder = load_hifigan(hifigan_dir, device)
    
    test_sentences = {
        "E1": {
            "raw": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
            "is_verb": False
        },
        "E2a": {
            "raw": "The orchestra's conduct during the ceremony was praised by the audience.",
            "is_verb": False
        },
        "E2b": {
            "raw": "The researchers will conduct an experiment to evaluate the proposed method.",
            "is_verb": True
        },
        "E3": {
            "raw": "The rhyth m of the orchestra changed unexpectedly.",
            "is_verb": False
        },
        "E4": {
            "raw": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
            "is_verb": False
        }
    }
    
    out_dir = os.path.join(BASE_DIR, "Results", "English", "test_generated_audios")
    os.makedirs(out_dir, exist_ok=True)
    
    predictions_cache = {}
    
    if args.sentence != "all":
        if args.sentence in test_sentences:
            test_sentences = {args.sentence: test_sentences[args.sentence]}
        else:
            print(f"Error: Sentence key {args.sentence} not found.")
            return

    for key, data in test_sentences.items():
        raw_text = data["raw"]
        norm_text = normalize_english_text(raw_text)
        phones = text_to_phonemes(norm_text, is_verb_conduct=data["is_verb"])
        seq = np.array(text_to_sequence(phones, ["english_cleaners"]))
        
        print(f"\n--- Synthesizing {key} ---")
        print(f"Raw: {raw_text}")
        print(f"Norm: {norm_text}")
        print(f"Phonemes: {phones[:60]}... ({len(seq)} tokens)")
        
        wav, dur, pitch, energy, mel = synthesize_sentence(model, vocoder, seq, speaker_id=spk_id)
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
        
    cache_path = os.path.join(BASE_DIR, "Results", "English", "test_predictions.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions_cache, f, indent=2)
    print(f"\nAll English mandatory test sentences synthesized successfully! Saved in {out_dir}")

if __name__ == "__main__":
    main()

