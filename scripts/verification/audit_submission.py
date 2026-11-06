import os
import json
import soundfile as sf
from PIL import Image

def audit_submission():
    print("==================================================")
    print(" FINAL SUBMISSION & DELIVERABLE AUDIT ")
    print("==================================================")
    
    passed = True
    
    # 1. Checkpoints
    print("\n[1] Checkpoints Audit:")
    en_ckpts = os.listdir("English/checkpoints")
    te_ckpts = os.listdir("Telugu/checkpoints")
    print(f"--> English Checkpoints ({len(en_ckpts)} files): {sorted(en_ckpts)}")
    print(f"--> Telugu Checkpoints ({len(te_ckpts)} files): {sorted(te_ckpts)}")
    assert "best_model.pth.tar" in en_ckpts, "Missing English best_model.pth.tar!"
    assert "best_model.pth.tar" in te_ckpts, "Missing Telugu best_model.pth.tar!"
    assert "speakers.json" in en_ckpts, "Missing English speakers.json!"
    
    # 2. Audio Files
    print("\n[2] Audio Files Audit (10 Utterances):")
    en_audios = ["E1.wav", "E2a.wav", "E2b.wav", "E3.wav", "E4.wav"]
    te_audios = ["T1.wav", "T2a.wav", "T2b.wav", "T3.wav", "T4.wav"]
    
    for a in en_audios:
        p = os.path.join("Results", "English", "test_generated_audios", a)
        assert os.path.exists(p), f"Missing {p}"
        data, sr = sf.read(p)
        assert sr == 22050, f"Bad SR {sr} in {p}"
        assert not (np.isnan(data).any() or np.isinf(data).any()), f"NaN/Inf in {p}"
        print(f"  - [EN] {a}: {len(data)/sr:.2f}s | {os.path.getsize(p):,} bytes | Peak: {np.max(np.abs(data)):.4f}")
        
    for a in te_audios:
        p = os.path.join("Results", "Telugu", "test_generated_audios", a)
        assert os.path.exists(p), f"Missing {p}"
        data, sr = sf.read(p)
        assert sr == 22050, f"Bad SR {sr} in {p}"
        assert not (np.isnan(data).any() or np.isinf(data).any()), f"NaN/Inf in {p}"
        print(f"  - [TE] {a}: {len(data)/sr:.2f}s | {os.path.getsize(p):,} bytes | Peak: {np.max(np.abs(data)):.4f}")
        
    # 3. Variance Adaptor Plots
    print("\n[3] Variance Adaptor Plots Audit (30 Plots):")
    plot_count = 0
    for lang in ["English", "Telugu"]:
        base = os.path.join("Results", lang, "variance_adaptor_plots")
        for feat in ["duration", "pitch", "energy"]:
            feat_dir = os.path.join(base, feat)
            assert os.path.exists(feat_dir), f"Missing directory: {feat_dir}"
            files = [f for f in os.listdir(feat_dir) if f.endswith(".png")]
            assert len(files) == 5, f"Expected 5 plots in {feat_dir}, got {len(files)}"
            for f in files:
                p = os.path.join(feat_dir, f)
                with Image.open(p) as img:
                    img.verify()
                plot_count += 1
    print(f"--> All {plot_count} variance adaptor plots verified (100% valid PNG images)")
    
    # 4. Reports & Documentation
    print("\n[4] Reports & Documentation Audit:")
    pdf_path = os.path.join("Report", "Technical_Report.pdf")
    assert os.path.exists(pdf_path), f"Missing {pdf_path}"
    pdf_size = os.path.getsize(pdf_path)
    print(f"--> Technical_Report.pdf: {pdf_size:,} bytes")
    
    assert os.path.exists("README.md"), "Missing README.md"
    print(f"--> README.md: {os.path.getsize('README.md'):,} bytes")
    
    assert os.path.exists("requirements.txt"), "Missing requirements.txt"
    print(f"--> requirements.txt: {os.path.getsize('requirements.txt'):,} bytes")
    
    # 5. Backup verification
    print("\n[5] Single-Speaker Backup Audit:")
    backup_dir = "scratch/backup_english_single_speaker"
    assert os.path.exists(backup_dir), f"Missing backup directory: {backup_dir}"
    b_ckpts = os.listdir(os.path.join(backup_dir, "checkpoints"))
    print(f"--> Backup Checkpoints ({len(b_ckpts)} files preserved): {sorted(b_ckpts)[:3]}...")
    
    print("\n==================================================")
    print(" AUDIT RESULT: ALL DELIVERABLES 100% VERIFIED ")
    print("==================================================")

if __name__ == "__main__":
    import numpy as np
    audit_submission()

