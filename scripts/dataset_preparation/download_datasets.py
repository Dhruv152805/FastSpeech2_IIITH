"""
Phase 2 — Dataset Download Script
Downloads Svarah (English) and IndicTTS Telugu datasets from HuggingFace.
Run from: fastspeech2-iiith/
Activate venv first: .\\venv_fs2\\Scripts\\Activate.ps1
"""
import os
import sys
import json
import argparse
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_EN  = ROOT / "data" / "english" / "raw"
DATA_TE  = ROOT / "data" / "telugu"  / "raw"
DATA_EN.mkdir(parents=True, exist_ok=True)
DATA_TE.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# English: Svarah  (ai4bharat/Svarah)
# ─────────────────────────────────────────────
def download_svarah(hf_token: str = None):
    print("\n" + "="*60)
    print("Downloading Svarah (Indian English) from HuggingFace")
    print("="*60)
    from datasets import load_dataset

    ds = load_dataset(
        "ai4bharat/Svarah",
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"  Splits: {list(ds.keys())}")
    print(f"  Total examples: {sum(len(ds[s]) for s in ds)}")

    out_dir = DATA_EN
    manifest = []

    for split_name, split in ds.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Processing split: {split_name}  ({len(split)} examples)")

        for i, sample in enumerate(tqdm(split, desc=f"  Saving {split_name}")):
            audio_dict = sample["audio"]
            sr         = audio_dict["sampling_rate"]
            arr        = np.array(audio_dict["array"], dtype=np.float32)
            duration   = len(arr) / sr

            # filter: 1.5–15 seconds
            if not (1.5 <= duration <= 15.0):
                continue

            text = sample.get("text") or sample.get("transcript") or sample.get("sentence", "")
            if not text.strip():
                continue

            spk = sample.get("speaker_id") or sample.get("client_id") or f"spk_{i%117:04d}"
            utt_id = f"{split_name}_{i:06d}"
            wav_path = split_dir / f"{spk}_{utt_id}.wav"

            sf.write(str(wav_path), arr, sr)
            manifest.append({
                "id": utt_id,
                "speaker": str(spk),
                "text": text.strip(),
                "wav": str(wav_path),
                "duration": round(duration, 3),
                "sr": sr,
                "split": split_name,
            })

    manifest_path = out_dir / "svarah_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_dur = sum(m["duration"] for m in manifest) / 3600
    print(f"\n  Saved {len(manifest)} utterances → {total_dur:.2f} hours")
    print(f"  Manifest: {manifest_path}")
    return manifest


# ─────────────────────────────────────────────
# Telugu: IndicTTS  (SPRINGLab/IndicTTS_Telugu)
# ─────────────────────────────────────────────
def download_indictts_telugu(hf_token: str = None):
    print("\n" + "="*60)
    print("Downloading IndicTTS Telugu from HuggingFace")
    print("="*60)
    from datasets import load_dataset

    ds = load_dataset(
        "SPRINGLab/IndicTTS_Telugu",
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"  Splits: {list(ds.keys())}")
    print(f"  Total examples: {sum(len(ds[s]) for s in ds)}")

    out_dir = DATA_TE
    manifest = []
    gender_map = {0: "female", 1: "male"}

    for split_name, split in ds.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Processing split: {split_name}  ({len(split)} examples)")

        for i, sample in enumerate(tqdm(split, desc=f"  Saving {split_name}")):
            audio_dict = sample["audio"]
            sr         = audio_dict["sampling_rate"]
            arr        = np.array(audio_dict["array"], dtype=np.float32)
            duration   = len(arr) / sr

            if not (1.5 <= duration <= 15.0):
                continue

            text   = sample.get("text", "").strip()
            gender = gender_map.get(sample.get("gender"), "unknown")

            if not text:
                continue

            spk    = f"telugu_{gender}"
            utt_id = f"{split_name}_{i:06d}"
            wav_path = split_dir / f"{spk}_{utt_id}.wav"

            sf.write(str(wav_path), arr, sr)
            manifest.append({
                "id": utt_id,
                "speaker": spk,
                "gender": gender,
                "text": text,
                "wav": str(wav_path),
                "duration": round(duration, 3),
                "sr": sr,
                "split": split_name,
            })

    manifest_path = out_dir / "indictts_telugu_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_dur = sum(m["duration"] for m in manifest) / 3600
    male_dur  = sum(m["duration"] for m in manifest if m["gender"]=="male")   / 3600
    female_dur= sum(m["duration"] for m in manifest if m["gender"]=="female") / 3600
    print(f"\n  Saved {len(manifest)} utterances → {total_dur:.2f} hours")
    print(f"    Male:   {male_dur:.2f}h | Female: {female_dur:.2f}h")
    print(f"  Manifest: {manifest_path}")
    return manifest


# ─────────────────────────────────────────────
# Inspect & summarize downloaded data
# ─────────────────────────────────────────────
def summarize_manifest(manifest_path: Path, label: str):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    total_dur = sum(m["duration"] for m in manifest) / 3600
    speakers  = set(m["speaker"] for m in manifest)
    sr_set    = set(m["sr"] for m in manifest)
    durations = [m["duration"] for m in manifest]

    print(f"\n{'='*60}")
    print(f"Dataset Summary: {label}")
    print(f"{'='*60}")
    print(f"  Total utterances : {len(manifest)}")
    print(f"  Total duration   : {total_dur:.2f} hours")
    print(f"  Speakers         : {len(speakers)}  → {sorted(speakers)[:5]}{'...' if len(speakers)>5 else ''}")
    print(f"  Sample rates     : {sr_set}")
    print(f"  Duration range   : {min(durations):.1f}s – {max(durations):.1f}s")
    print(f"  Duration mean    : {np.mean(durations):.2f}s  std={np.std(durations):.2f}s")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for FastSpeech2 assignment")
    parser.add_argument("--lang", choices=["en", "te", "both"], default="both")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (required for gated datasets)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.lang in ("en", "both"):
        download_svarah(hf_token)

    if args.lang in ("te", "both"):
        download_indictts_telugu(hf_token)

    # Print summaries
    en_manifest = DATA_EN / "svarah_manifest.json"
    te_manifest = DATA_TE / "indictts_telugu_manifest.json"

    if en_manifest.exists():
        summarize_manifest(en_manifest, "Svarah (English)")
    if te_manifest.exists():
        summarize_manifest(te_manifest, "IndicTTS Telugu")

