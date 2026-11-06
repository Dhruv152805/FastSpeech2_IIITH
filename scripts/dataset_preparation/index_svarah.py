import os
import io
import json
import soundfile as sf
import pyarrow.parquet as pq

parquet_dir = r"C:\Users\dkhan\.cache\huggingface\hub\datasets--ai4bharat--Svarah\snapshots\ebbf7777fe771490696a3f7b007097606fa8c924\data"
out_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\english\raw"
os.makedirs(out_dir, exist_ok=True)

files = [os.path.join(parquet_dir, f"test-{i:05d}-of-00003.parquet") for i in range(3)]

all_manifest = []

for p_file in files:
    print(f"Reading {os.path.basename(p_file)}...")
    table = pq.read_table(p_file)
    audio_col = table["audio_filepath"] if "audio_filepath" in table.column_names else table["audio"]
    text_col = table["text"]
    dur_col = table["duration"]
    gender_col = table["gender"] if "gender" in table.column_names else None
    
    for i in range(len(table)):
        text = text_col[i].as_py().strip()
        dur = float(dur_col[i].as_py())
        gender = gender_col[i].as_py() if gender_col else "unknown"
        
        all_manifest.append({
            "index": len(all_manifest),
            "text": text,
            "gender": gender,
            "duration": dur,
            "p_file": p_file,
            "row_idx": i
        })

print(f"Total utterances indexed in Svarah: {len(all_manifest)}")
total_hours = sum(m["duration"] for m in all_manifest) / 3600
print(f"Total available Svarah duration: {total_hours:.2f} hours")

with open(os.path.join(out_dir, "full_index.json"), "w", encoding="utf-8") as f:
    json.dump(all_manifest, f, ensure_ascii=False, indent=2)

print("Svarah indexing complete!")

