"""
Phase 2 — Dataset Inspection & Split Script
Run AFTER download_datasets.py.
Produces:
  - train.txt / val.txt manifests for English and Telugu
  - duration_histogram.png
  - dataset_summary.json
"""
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT   = Path(__file__).resolve().parent
DATA_EN = ROOT / "data" / "english"
DATA_TE = ROOT / "data" / "telugu"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_manifest(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_utterances(manifest, min_dur=1.5, max_dur=12.0):
    """Keep utterances in a sensible duration range."""
    kept = [m for m in manifest if min_dur <= m["duration"] <= max_dur]
    removed = len(manifest) - len(kept)
    if removed:
        print(f"  Filtered out {removed} utterances outside [{min_dur}, {max_dur}]s")
    return kept


def split_train_val(manifest, val_hours=1.0, seed=42):
    """
    Reserve val_hours worth of audio for validation.
    Zero overlap guaranteed – each utterance goes to exactly one split.
    """
    rng = random.Random(seed)
    shuffled = manifest[:]
    rng.shuffle(shuffled)

    val, train = [], []
    val_dur = 0.0
    target_val = val_hours * 3600  # seconds

    for item in shuffled:
        if val_dur < target_val:
            val.append(item)
            val_dur += item["duration"]
        else:
            train.append(item)

    # Verify zero overlap
    val_ids  = {m["id"] for m in val}
    train_ids = {m["id"] for m in train}
    assert val_ids.isdisjoint(train_ids), "OVERLAP DETECTED — aborting"

    return train, val


def write_fastspeech2_manifest(items, out_path: Path):
    """Write pipe-separated FastSpeech2 manifest: id|speaker|phoneme_text|raw_text"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for m in items:
            # phoneme_text filled by preprocess.py; use raw_text as placeholder
            line = f"{m['id']}|{m['speaker']}|{m['text']}|{m['text']}\n"
            f.write(line)
    print(f"  Written: {out_path}  ({len(items)} lines)")


def plot_duration_histogram(train, val, label: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    td = [m["duration"] for m in train]
    vd = [m["duration"] for m in val]
    ax.hist(td, bins=50, alpha=0.7, label=f"Train ({len(td)})", color="steelblue")
    ax.hist(vd, bins=50, alpha=0.7, label=f"Val ({len(vd)})",   color="coral")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} — Utterance Duration Distribution")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def print_dataset_table(label, train, val):
    t_dur = sum(m["duration"] for m in train) / 3600
    v_dur = sum(m["duration"] for m in val)   / 3600
    t_spk = set(m["speaker"] for m in train)
    v_spk = set(m["speaker"] for m in val)
    srs   = set(m.get("sr", "?") for m in train + val)
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Training utterances   : {len(train):>6}")
    print(f"  Validation utterances : {len(val):>6}")
    print(f"  Training duration     : {t_dur:.2f} h")
    print(f"  Validation duration   : {v_dur:.2f} h")
    print(f"  Training speakers     : {len(t_spk)}")
    print(f"  Validation speakers   : {len(v_spk)}")
    print(f"  Sample rates          : {srs}")
    print(f"  Train/val overlap     : NONE ✓")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────
# English: Svarah
# ─────────────────────────────────────────────
def process_english():
    manifest_path = DATA_EN / "raw" / "svarah_manifest.json"
    if not manifest_path.exists():
        print(f"[English] Manifest not found: {manifest_path}")
        print("  Run download_datasets.py --lang en first.")
        return None, None

    manifest = load_manifest(manifest_path)
    manifest = filter_utterances(manifest, min_dur=1.5, max_dur=12.0)

    train, val = split_train_val(manifest, val_hours=1.0, seed=42)
    print_dataset_table("English — Svarah", train, val)

    out_dir = DATA_EN / "preprocessed"
    write_fastspeech2_manifest(train, out_dir / "train.txt")
    write_fastspeech2_manifest(val,   out_dir / "val.txt")

    # Save JSON manifests for inspection
    with open(out_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out_dir / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    plot_duration_histogram(
        train, val, "English (Svarah)",
        ROOT / "Results" / "English" / "duration_histogram.png"
    )
    return train, val


# ─────────────────────────────────────────────
# Telugu: IndicTTS
# ─────────────────────────────────────────────
def process_telugu():
    manifest_path = DATA_TE / "raw" / "indictts_telugu_manifest.json"
    if not manifest_path.exists():
        print(f"[Telugu] Manifest not found: {manifest_path}")
        print("  Run download_datasets.py --lang te first.")
        return None, None

    manifest = load_manifest(manifest_path)
    manifest = filter_utterances(manifest, min_dur=1.5, max_dur=12.0)

    # For Telugu: balance male/female in both splits
    male   = [m for m in manifest if m.get("gender") == "male"]
    female = [m for m in manifest if m.get("gender") == "female"]
    print(f"  Telugu gender split: male={len(male)}, female={len(female)}")

    # Split each gender independently to preserve balance
    m_train, m_val = split_train_val(male,   val_hours=0.5, seed=42)
    f_train, f_val = split_train_val(female, val_hours=0.5, seed=42)

    train = m_train + f_train
    val   = m_val   + f_val
    random.Random(42).shuffle(train)
    random.Random(42).shuffle(val)

    print_dataset_table("Telugu — IndicTTS", train, val)

    out_dir = DATA_TE / "preprocessed"
    write_fastspeech2_manifest(train, out_dir / "train.txt")
    write_fastspeech2_manifest(val,   out_dir / "val.txt")

    with open(out_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out_dir / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    plot_duration_histogram(
        train, val, "Telugu (IndicTTS)",
        ROOT / "Results" / "Telugu" / "duration_histogram.png"
    )
    return train, val


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
def print_approval_table(en_train, en_val, te_train, te_val):
    def h(lst): return f"{sum(m['duration'] for m in lst)/3600:.2f} h" if lst else "N/A"
    def n(lst): return str(len(lst)) if lst else "N/A"
    def spk(lst): return str(len(set(m['speaker'] for m in lst))) if lst else "N/A"

    print("\n" + "="*70)
    print("PHASE 2 — DATASET SUMMARY (requires your approval before training)")
    print("="*70)
    print(f"{'Property':<35} {'English':>15} {'Telugu':>15}")
    print("─"*70)
    all_en = (en_train or []) + (en_val or [])
    all_te = (te_train or []) + (te_val or [])
    print(f"{'Total available duration':<35} {h(all_en):>15} {h(all_te):>15}")
    print(f"{'Selected training duration':<35} {h(en_train):>15} {h(te_train):>15}")
    print(f"{'Selected validation duration':<35} {h(en_val):>15} {h(te_val):>15}")
    print(f"{'Training utterances':<35} {n(en_train):>15} {n(te_train):>15}")
    print(f"{'Validation utterances':<35} {n(en_val):>15} {n(te_val):>15}")
    print(f"{'Speakers (train)':<35} {spk(en_train):>15} {spk(te_train):>15}")
    print(f"{'Source sample rate':<35} {'16000 Hz':>15} {'48000 Hz':>15}")
    print(f"{'Target sample rate (config)':<35} {'22050 Hz':>15} {'22050 Hz':>15}")
    print(f"{'Train/val overlap':<35} {'NONE':>15} {'NONE':>15}")
    print("="*70)
    print("\n⟹  Review the numbers above, then reply 'approved' to begin Phase 3.")


if __name__ == "__main__":
    print("Phase 2 — Inspecting and splitting datasets...")
    en_train, en_val = process_english()
    te_train, te_val = process_telugu()
    print_approval_table(en_train, en_val, te_train, te_val)

