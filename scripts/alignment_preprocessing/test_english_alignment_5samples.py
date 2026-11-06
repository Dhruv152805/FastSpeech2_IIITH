import os
import sys
import re
import json
import torch
import torchaudio
import librosa
import numpy as np
import torchaudio.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "reference_fastspeech2"))

import audio as Audio
from g2p_en import G2p
from text.cleaners import english_cleaners

def test_english_alignment():
    print("=== Testing English Forced Alignment on 5 Samples ===")
    bundle = torchaudio.pipelines.MMS_FA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = bundle.get_model().to(device)
    dictionary = bundle.get_dict()
    
    g2p = G2p()
    sr = 22050
    hop_length = 256
    filter_length = 1024
    win_length = 1024
    n_mel_channels = 80
    
    stft = Audio.stft.TacotronSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        sampling_rate=sr,
        mel_fmin=0,
        mel_fmax=8000
    )
    
    with open('data/english/preprocessed/train_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)[:5]
        
    for idx, item in enumerate(manifest):
        uid = item['id']
        text = item['text']
        wav_path = item['wav']
        
        # 1. Clean and tokenize text into words
        cleaned = english_cleaners(text)
        raw_words = [w for w in re.split(r'\s+', re.sub(r'[^a-z\s]', '', cleaned.lower())) if w]
        
        # 2. Get phonemes per word via G2P
        word_data = [] # (word, phones, token_indices)
        all_tokens = []
        for w in raw_words:
            raw_p = [p for p in g2p(w) if re.match(r'^[A-Z0-9]+$', p)]
            if not raw_p:
                continue
            char_toks = [dictionary[c] for c in w if c in dictionary]
            if not char_toks:
                continue
            start_tok = len(all_tokens)
            all_tokens.extend(char_toks)
            word_data.append((w, raw_p, start_tok, len(all_tokens)))
            
        if not word_data:
            continue
            
        # 3. Load audio & MMS forward pass
        waveform, orig_sr = torchaudio.load(wav_path)
        wav_16k = torchaudio.functional.resample(waveform, orig_sr, bundle.sample_rate).to(device)
        with torch.inference_mode():
            emission, _ = model(wav_16k)
            
        tokens_t = torch.tensor([all_tokens], dtype=torch.int32, device=device)
        aligned_tokens, scores = F.forced_align(emission, tokens_t, blank=0)
        spans = F.merge_tokens(aligned_tokens[0], scores[0])
        
        # 4. Extract word time boundaries (seconds)
        mms_stride = 320 / 16000 # 0.02s
        words_timing = [] # (word, phones, start_sec, end_sec)
        for w, ps, s_idx, e_idx in word_data:
            s_sec = spans[s_idx].start * mms_stride
            e_sec = spans[e_idx - 1].end * mms_stride
            words_timing.append((w, ps, s_sec, e_sec))
            
        # Active speech window
        speech_start_sec = max(0.0, words_timing[0][2] - 0.04)
        speech_end_sec = words_timing[-1][3] + 0.04
        
        # 5. Build full phone sequence with pauses (sp)
        full_phones = []
        phone_boundaries = [speech_start_sec]
        
        # Phonetic relative duration weights:
        # Vowels: 1.4, Fricatives/nasals: 1.0, Stops: 0.65, Liquids/Glides: 0.85
        def phone_weight(p):
            if any(p.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
                return 1.4
            elif any(p.startswith(c) for c in ['P', 'T', 'K', 'B', 'D', 'G']):
                return 0.65
            elif any(p.startswith(c) for c in ['L', 'R', 'W', 'Y']):
                return 0.85
            else:
                return 1.0
                
        for w_idx, (w, ps, w_start, w_end) in enumerate(words_timing):
            # If pause before this word (except first)
            if w_idx > 0:
                prev_end = words_timing[w_idx - 1][3]
                pause_gap = w_start - prev_end
                if pause_gap > 0.04: # >= 40ms pause
                    full_phones.append('sp')
                    phone_boundaries.append(w_start)
                    
            # Allocate phonemes inside word proportionally according to intrinsic weights
            w_dur = max(0.05, w_end - w_start)
            weights = [phone_weight(p) for p in ps]
            tot_w = sum(weights)
            
            curr_t = w_start
            for p, wt in zip(ps, weights):
                p_dur = w_dur * (wt / tot_w)
                curr_t += p_dur
                full_phones.append(p)
                phone_boundaries.append(curr_t)
                
        phone_boundaries[-1] = speech_end_sec
        
        # 6. Load target audio, trim, extract mel
        wav_target, _ = librosa.load(wav_path, sr=sr, mono=True)
        start_samp = int(round(speech_start_sec * sr))
        end_samp = int(round(speech_end_sec * sr))
        wav_trimmed = wav_target[start_samp:end_samp].astype(np.float32)
        
        mel, energy = Audio.tools.get_mel_from_wav(wav_trimmed, stft)
        n_mel_frames = mel.shape[1]
        
        # 7. Convert boundaries to mel frame durations
        mel_frame_dur = hop_length / sr
        frame_boundaries = [int(round((b - speech_start_sec) / mel_frame_dur)) for b in phone_boundaries]
        frame_boundaries[0] = 0
        frame_boundaries[-1] = n_mel_frames
        
        durations = []
        for i in range(len(frame_boundaries) - 1):
            d = max(1, frame_boundaries[i+1] - frame_boundaries[i])
            durations.append(d)
            
        durations = np.array(durations, dtype=np.int32)
        diff = n_mel_frames - int(np.sum(durations))
        if diff != 0:
            max_idx = int(np.argmax(durations))
            durations[max_idx] = max(1, durations[max_idx] + diff)
            
        assert int(np.sum(durations)) == n_mel_frames, f"Sum mismatch: {np.sum(durations)} vs {n_mel_frames}"
        assert len(durations) == len(full_phones), f"Length mismatch: {len(durations)} vs {len(full_phones)}"
        
        print(f"\n[Sample {idx+1}] ID: {uid}")
        print(f"  Text: {text[:45]}...")
        print(f"  Phonemes ({len(full_phones)}): {' '.join(full_phones[:12])}...")
        print(f"  Mel frames: {n_mel_frames} | Sum(durations): {np.sum(durations)} (EXACT MATCH)")
        print(f"  Durations stats: min={np.min(durations)}, max={np.max(durations)}, mean={np.mean(durations):.2f}, std={np.std(durations):.2f}")
        print(f"  Sample durations: {durations[:12].tolist()}")
        print(f"  Active speech window: [{speech_start_sec:.3f}s, {speech_end_sec:.3f}s] out of {orig_sr and len(waveform[0])/orig_sr:.3f}s")
        print("  Status: VERIFIED PASS")

if __name__ == "__main__":
    test_english_alignment()

