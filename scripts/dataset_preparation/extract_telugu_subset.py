import os
import io
import json
import random
import soundfile as sf
import pyarrow.parquet as pq

raw_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\telugu\raw"
index_path = os.path.join(raw_dir, "full_index.json")

with open(index_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Cutoff: 1.5s to 14.0s yields 11.08 hours of clean speech
valid_items = [item for item in all_data if 1.5 <= item["duration"] <= 14.0 and len(item["text"]) > 3]

male_items = [x for x in valid_items if x["gender"] == "male"]
female_items = [x for x in valid_items if x["gender"] == "female"]

rng = random.Random(42)
rng.shuffle(male_items)
rng.shuffle(female_items)

# Val targets: 0.5h male + 0.5h female = 1.0h
# Train targets: 2.5h male + 2.5h female = 5.0h
val_target_sec = 0.50 * 3600
train_target_sec = 2.50 * 3600

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

val_male = select_split(male_items, val_target_sec)
rem_male = [x for x in male_items if x["index"] not in {y["index"] for y in val_male}]
train_male = select_split(rem_male, train_target_sec)

val_female = select_split(female_items, val_target_sec)
rem_female = [x for x in female_items if x["index"] not in {y["index"] for y in val_female}]
train_female = select_split(rem_female, train_target_sec)

train_all = train_male + train_female
val_all = val_male + val_female

rng.shuffle(train_all)
rng.shuffle(val_all)

# Disjointness check
train_indices = {x["index"] for x in train_all}
val_indices = {x["index"] for x in val_all}
assert train_indices.isdisjoint(val_indices), "Fatal: Overlap between train and val!"

print("\n--- Final 6-Hour Split Verification (5h Train / 1h Val) ---")
print(f"Train utterances: {len(train_all)} (Male: {len(train_male)}, Female: {len(train_female)})")
print(f"Train duration: {sum(x['duration'] for x in train_all)/3600:.2f} hours")
print(f"Val utterances: {len(val_all)} (Male: {len(val_male)}, Female: {len(val_female)})")
print(f"Val duration: {sum(x['duration'] for x in val_all)/3600:.2f} hours")
print("Train / Val overlap: 0 utterances (100% disjoint)")

# Extract to disk
parquet_cache = {}

def get_row(p_file, row_idx):
    if p_file not in parquet_cache:
        parquet_cache[p_file] = pq.read_table(p_file)
    table = parquet_cache[p_file]
    raw_bytes = table["audio"][row_idx]["bytes"].as_py()
    return raw_bytes

def save_subset(subset, split_name):
    saved_items = []
    for it in subset:
        spk = f"telugu_{it['gender']}"
        spk_dir = os.path.join(raw_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        
        base_id = f"{spk}_{it['index']:05d}"
        wav_path = os.path.join(spk_dir, f"{base_id}.wav")
        lab_path = os.path.join(spk_dir, f"{base_id}.lab")
        
        if not os.path.exists(wav_path):
            raw_bytes = get_row(it["p_file"], it["row_idx"])
            with io.BytesIO(raw_bytes) as bio:
                data, sr = sf.read(bio)
                sf.write(wav_path, data, sr)
                
        if not os.path.exists(lab_path):
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(it["text"])
            
        saved_items.append({
            "id": base_id,
            "speaker": spk,
            "text": it["text"],
            "duration": it["duration"],
            "sr": it["sr"],
            "wav": wav_path,
            "split": split_name
        })
    return saved_items

print("\nExtracting audio clips...")
saved_train = save_subset(train_all, "train")
saved_val = save_subset(val_all, "val")

preproc_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\telugu\preprocessed"
os.makedirs(preproc_dir, exist_ok=True)

with open(os.path.join(preproc_dir, "train_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(saved_train, f, ensure_ascii=False, indent=2)

with open(os.path.join(preproc_dir, "val_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(saved_val, f, ensure_ascii=False, indent=2)

# Write fastspeech manifest (id|speaker|text|raw_text)
with open(os.path.join(preproc_dir, "train.txt"), "w", encoding="utf-8") as f:
    for m in saved_train:
        f.write(f"{m['id']}|{m['speaker']}|{m['text']}|{m['text']}\n")

with open(os.path.join(preproc_dir, "val.txt"), "w", encoding="utf-8") as f:
    for m in saved_val:
        f.write(f"{m['id']}|{m['speaker']}|{m['text']}|{m['text']}\n")

print("\nSaved exact 5.0h train + 1.0h val manifests successfully!")
