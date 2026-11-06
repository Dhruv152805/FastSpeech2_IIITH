import os
import io
import json
import random
import soundfile as sf
import pyarrow.parquet as pq

raw_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\english\raw"
index_path = os.path.join(raw_dir, "full_index.json")

with open(index_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Cutoff: 1.5s to 12.0s yields 7.03 hours of clean audio
valid_items = [item for item in all_data if 1.5 <= item["duration"] <= 12.0 and len(item["text"]) > 3]

# Deterministic shuffle
rng = random.Random(42)
rng.shuffle(valid_items)

val_target_sec = 1.00 * 3600
train_target_sec = 5.00 * 3600

def select_split(items, target_sec):
    selected = []
    tot = 0.0
    for it in items:
        if tot + it["duration"] <= target_sec:
            selected.append(it)
            tot += it["duration"]
        elif tot < target_sec:
            selected.append(it)
            tot += it["duration"]
            break
    return selected

val_all = select_split(valid_items, val_target_sec)
rem_items = [x for x in valid_items if x["index"] not in {y["index"] for y in val_all}]
train_all = select_split(rem_items, train_target_sec)

# Strict disjoint verification
train_indices = {x["index"] for x in train_all}
val_indices = {x["index"] for x in val_all}
assert train_indices.isdisjoint(val_indices), "Fatal: Overlap detected in English dataset!"

print("\n--- Final English 6-Hour Split Verification (5h Train / 1h Val) ---")
print(f"Train utterances: {len(train_all)}")
print(f"Train duration: {sum(x['duration'] for x in train_all)/3600:.2f} hours")
print(f"Val utterances: {len(val_all)}")
print(f"Val duration: {sum(x['duration'] for x in val_all)/3600:.2f} hours")
print("Train / Val overlap: 0 utterances (100% disjoint)")

# Extract audio to disk
parquet_cache = {}

def get_row(p_file, row_idx):
    if p_file not in parquet_cache:
        parquet_cache[p_file] = pq.read_table(p_file)
    table = parquet_cache[p_file]
    audio_col = table["audio_filepath"] if "audio_filepath" in table.column_names else table["audio"]
    raw_bytes = audio_col[row_idx]["bytes"].as_py()
    return raw_bytes

spk_dir = os.path.join(raw_dir, "svarah_speaker")
os.makedirs(spk_dir, exist_ok=True)

def save_subset(subset, split_name):
    saved_items = []
    for it in subset:
        base_id = f"svarah_{it['index']:05d}"
        wav_path = os.path.join(spk_dir, f"{base_id}.wav")
        lab_path = os.path.join(spk_dir, f"{base_id}.lab")
        
        raw_bytes = get_row(it["p_file"], it["row_idx"])
        with io.BytesIO(raw_bytes) as bio:
            data, sr = sf.read(bio)
            sf.write(wav_path, data, sr)
            
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(it["text"])
            
        saved_items.append({
            "id": base_id,
            "speaker": "svarah_speaker",
            "text": it["text"],
            "duration": it["duration"],
            "sr": sr,
            "wav": wav_path,
            "split": split_name
        })
    return saved_items

print("\nExtracting English audio clips...")
saved_train = save_subset(train_all, "train")
saved_val = save_subset(val_all, "val")

preproc_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\english\preprocessed"
os.makedirs(preproc_dir, exist_ok=True)

with open(os.path.join(preproc_dir, "train_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(saved_train, f, ensure_ascii=False, indent=2)

with open(os.path.join(preproc_dir, "val_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(saved_val, f, ensure_ascii=False, indent=2)

with open(os.path.join(preproc_dir, "train.txt"), "w", encoding="utf-8") as f:
    for m in saved_train:
        f.write(f"{m['id']}|{m['speaker']}|{m['text']}|{m['text']}\n")

with open(os.path.join(preproc_dir, "val.txt"), "w", encoding="utf-8") as f:
    for m in saved_val:
        f.write(f"{m['id']}|{m['speaker']}|{m['text']}|{m['text']}\n")

print("\nSaved English 5.0h train + 1.0h val manifests successfully!")

