"""
Phase 7: Variance Adaptor Visualizations
Generates all 30 required Variance Adaptor plots:
- 10 Utterances (5 English: E1, E2a, E2b, E3, E4; 5 Telugu: T1, T2a, T2b, T3, T4)
- 3 Predictor Types: Duration, Pitch (F0), Energy
Saved under Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "scratch", "mpl")

def plot_for_language(lang):
    json_path = os.path.join(BASE_DIR, "Results", lang, "test_predictions.json")
    if not os.path.exists(json_path):
        print(f"Predictions JSON not found for {lang}: {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dur_dir = os.path.join(BASE_DIR, "Results", lang, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(BASE_DIR, "Results", lang, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(BASE_DIR, "Results", lang, "variance_adaptor_plots", "energy")
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)
    
    for key, item in data.items():
        durations = np.array(item["duration"])
        pitches = np.array(item["pitch"])
        energies = np.array(item["energy"])
        raw = item["raw"]
        
        title_text = raw[:55] if lang == "English" else item.get("phones", "").strip("{}")[:50]
        
        # 1. Duration Plot
        plt.figure(figsize=(9, 4))
        plt.bar(range(len(durations)), durations, color="royalblue", alpha=0.85, width=0.6)
        plt.title(f"[{lang}] {key} - Predicted Phoneme Durations\n\"{title_text}...\"", fontsize=11)
        plt.xlabel("Phoneme Token Index")
        plt.ylabel("Frames (1 frame = 11.6ms)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(dur_dir, f"{key}_duration.png"), dpi=150)
        plt.close()
        
        # 2. Pitch Plot
        plt.figure(figsize=(9, 4))
        plt.plot(pitches, color="crimson", lw=2)
        plt.title(f"[{lang}] {key} - Predicted Pitch (F0) Contour\n\"{title_text}...\"", fontsize=11)
        plt.xlabel("Phoneme Token Index")
        plt.ylabel("Pitch (Normalized F0)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(pitch_dir, f"{key}_pitch.png"), dpi=150)
        plt.close()
        
        # 3. Energy Plot
        plt.figure(figsize=(9, 4))
        plt.plot(energies, color="forestgreen", lw=2)
        plt.title(f"[{lang}] {key} - Predicted Energy Contour\n\"{title_text}...\"", fontsize=11)
        plt.xlabel("Phoneme Token Index")
        plt.ylabel("Energy (Normalized dB)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(energy_dir, f"{key}_energy.png"), dpi=150)
        plt.close()
        
        print(f"Generated 3 plots for [{lang}] {key}")

def main():
    print("Generating Variance Adaptor Plots...")
    plot_for_language("English")
    plot_for_language("Telugu")
    print("Variance Adaptor plots generation complete!")

if __name__ == "__main__":
    main()

