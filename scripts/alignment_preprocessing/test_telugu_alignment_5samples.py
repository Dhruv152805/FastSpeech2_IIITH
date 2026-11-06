import os
import sys
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
from test_telugu_g2p import telugu_to_phonemes

def test_telugu_alignment():
    print("=== Testing Telugu Forced Alignment on 5 Samples ===")
    bundle = torchaudio.pipelines.MMS_FA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = bundle.get_model().to(device)
    dictionary = bundle.get_dict()
    
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
    
    with open('data/telugu/preprocessed/train_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)[:5]
        
    for idx, item in enumerate(manifest):
        uid = item['id']
        text = item['text']
        wav_path = item['wav']
        
        # 1. G2P conversion
        phone_str = telugu_to_phonemes(text)
        phones = phone_str.strip('{}').split()
        
        # 2. Map phones to MMS tokens
        tokens = []
        phone_to_tokens = [] # (phone, start_tok_idx, end_tok_idx)
        for p in phones:
            if p == 'sp':
                phone_to_tokens.append(('sp', None, None))
                continue
            p_toks = [dictionary[c] for c in p if c in dictionary]
            if not p_toks:
                phone_to_tokens.append((p, None, None))
            else:
                s_idx = len(tokens)
                tokens.extend(p_toks)
                phone_to_tokens.append((p, s_idx, len(tokens)))
                
        # 3. Load audio & MMS forward
        waveform, orig_sr = torchaudio.load(wav_path)
        wav_16k = torchaudio.functional.resample(waveform, orig_sr, bundle.sample_rate).to(device)
        with torch.inference_mode():
            emission, _ = model(wav_16k)
            
        tokens_t = torch.tensor([tokens], dtype=torch.int32, device=device)
        aligned_tokens, scores = F.forced_align(emission, tokens_t, blank=0)
        spans = F.merge_tokens(aligned_tokens[0], scores[0])
        
        # 4. Extract token timestamps (in seconds)
        mms_stride = 320 / 16000 # 0.02s
        phone_timestamps = [] # (phone, s_sec, e_sec)
        for p, s_idx, e_idx in phone_to_tokens:
            if s_idx is None:
                phone_timestamps.append((p, None, None))
            else:
                s_sec = spans[s_idx].start * mms_stride
                e_sec = spans[e_idx - 1].end * mms_stride
                phone_timestamps.append((p, s_sec, e_sec))
                
        # 5. Trim leading and trailing silences
        spoken_indices = [i for i, (p, s, e) in enumerate(phone_timestamps) if s is not None and p != 'sp']
        first_spoken = spoken_indices[0]
        last_spoken = spoken_indices[-1]
        
        speech_start_sec = max(0.0, phone_timestamps[first_spoken][1] - 0.04)
        speech_end_sec = phone_timestamps[last_spoken][2] + 0.04
        
        # Trim phones list to active speech
        trimmed_phones_meta = phone_timestamps[first_spoken : last_spoken + 1]
        trimmed_phones = [p for p, s, e in trimmed_phones_meta]
        
        # Assign continuous time boundaries for each phoneme
        # Each phoneme gets an interval [b_start, b_end]
        boundaries = [speech_start_sec]
        for i in range(len(trimmed_phones_meta)):
            p, s, e = trimmed_phones_meta[i]
            if p == 'sp':
                # pause between prev and next
                prev_e = trimmed_phones_meta[i-1][2] if i > 0 and trimmed_phones_meta[i-1][2] is not None else boundaries[-1]
                next_s = trimmed_phones_meta[i+1][1] if i+1 < len(trimmed_phones_meta) and trimmed_phones_meta[i+1][1] is not None else prev_e
                mid = (prev_e + next_s) / 2.0
                boundaries.append(mid)
            else:
                if i + 1 < len(trimmed_phones_meta):
                    next_s = trimmed_phones_meta[i+1][1]
                    if next_s is not None:
                        # transition between this phone and next
                        mid = (e + next_s) / 2.0
                    else:
                        mid = e
                else:
                    mid = speech_end_sec
                boundaries.append(max(boundaries[-1] + 0.01, mid))
                
        boundaries[-1] = speech_end_sec
        
        # 6. Load target audio and trim
        wav_target, _ = librosa.load(wav_path, sr=sr, mono=True)
        start_samp = int(round(speech_start_sec * sr))
        end_samp = int(round(speech_end_sec * sr))
        wav_trimmed = wav_target[start_samp:end_samp].astype(np.float32)
        
        mel, energy = Audio.tools.get_mel_from_wav(wav_trimmed, stft)
        n_mel_frames = mel.shape[1]
        
        # 7. Convert continuous boundaries to mel frame durations
        mel_frame_dur = hop_length / sr
        frame_boundaries = [int(round((b - speech_start_sec) / mel_frame_dur)) for b in boundaries]
        frame_boundaries[0] = 0
        frame_boundaries[-1] = n_mel_frames
        
        durations = []
        for i in range(len(frame_boundaries) - 1):
            d = max(1, frame_boundaries[i+1] - frame_boundaries[i])
            durations.append(d)
            
        durations = np.array(durations, dtype=np.int32)
        # Fix any rounding mismatch with n_mel_frames
        diff = n_mel_frames - int(np.sum(durations))
        if diff != 0:
            # Add or remove from largest duration
            max_idx = int(np.argmax(durations))
            durations[max_idx] = max(1, durations[max_idx] + diff)
            
        assert int(np.sum(durations)) == n_mel_frames, f"Sum mismatch: {np.sum(durations)} vs {n_mel_frames}"
        assert len(durations) == len(trimmed_phones), f"Length mismatch: {len(durations)} vs {len(trimmed_phones)}"
        
        print(f"\n[Sample {idx+1}] ID: {uid}")
        print(f"  Text: {text[:45]}...")
        print(f"  Phonemes ({len(trimmed_phones)}): {' '.join(trimmed_phones[:12])}...")
        print(f"  Mel frames: {n_mel_frames} | Sum(durations): {np.sum(durations)} (EXACT MATCH)")
        print(f"  Durations stats: min={np.min(durations)}, max={np.max(durations)}, mean={np.mean(durations):.2f}, std={np.std(durations):.2f}")
        print(f"  Sample durations: {durations[:12].tolist()}")
        print(f"  Active speech window: [{speech_start_sec:.3f}s, {speech_end_sec:.3f}s] out of {orig_sr and len(waveform[0])/orig_sr:.3f}s")
        print("  Status: VERIFIED PASS")

if __name__ == "__main__":
    test_telugu_alignment()
