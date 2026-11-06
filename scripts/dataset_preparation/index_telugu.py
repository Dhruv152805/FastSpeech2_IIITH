import os
import io
import json
import soundfile as sf
import pyarrow.parquet as pq

parquet_dir = r"C:\Users\dkhan\.cache\huggingface\hub\datasets--SPRINGLab--IndicTTS_Telugu\snapshots\7ab3c18491bef3f5d4e820c2701b5f78f54031d6\data"
out_dir = r"C:\Users\dkhan\Desktop\fastspeech2-iiith\data\telugu\raw"
os.makedirs(out_dir, exist_ok=True)

files = [os.path.join(parquet_dir, f"train-{i:05d}-of-00004.parquet") for i in range(4)]
gender_map = {0: "female", 1: "male"}

all_manifest = []

for p_file in files:
    print(f"Reading {os.path.basename(p_file)}...")
    table = pq.read_table(p_file)
    audio_col = table["audio"]
    text_col = table["text"]
    gender_col = table["gender"]
    
    for i in range(len(table)):
        text = text_col[i].as_py().strip()
        gender = gender_map.get(gender_col[i].as_py(), "female")
        raw_bytes = audio_col[i]["bytes"].as_py()
        
        with io.BytesIO(raw_bytes) as bio:
            with sf.SoundFile(bio) as sfile:
                sr = sfile.samplerate
                frames = sfile.frames
                dur = frames / sr
                
        all_manifest.append({
            "index": len(all_manifest),
            "text": text,
            "gender": gender,
            "duration": dur,
            "sr": sr,
            "p_file": p_file,
            "row_idx": i
        })

print(f"Total utterances indexed: {len(all_manifest)}")
total_hours = sum(m["duration"] for m in all_manifest) / 3600
print(f"Total available duration: {total_hours:.2f} hours")

male_items = [m for m in all_manifest if m["gender"] == "male"]
female_items = [m for m in all_manifest if m["gender"] == "female"]

m_hours = sum(m["duration"] for m in male_items) / 3600
f_hours = sum(m["duration"] for m in female_items) / 3600
print(f"Male utterances: {len(male_items)}, duration: {m_hours:.2f} h")
print(f"Female utterances: {len(female_items)}, duration: {f_hours:.2f} h")

with open(os.path.join(out_dir, "full_index.json"), "w", encoding="utf-8") as f:
    json.dump(all_manifest, f, ensure_ascii=False, indent=2)

print("Indexing complete!")

