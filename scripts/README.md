# Utility Scripts Documentation

This directory contains utility, verification, preprocessing, and plotting scripts categorized by their function in the FastSpeech 2 bilingual TTS pipeline.

All scripts should be executed from the project root (`fastspeech2-iiith/`) using the virtual environment:
```powershell
.\venv_fs2\Scripts\Activate.ps1
```

---

## Directory Organization

```
scripts/
├── dataset_preparation/          # Corpus acquisition, parsing, and subset extraction
│   ├── download_datasets.py      # Automated downloader for Svarah and IndicTTS datasets
│   ├── extract_svarah_subset.py  # Partitions 5.0h train / 1.0h val subsets for Indian English
│   ├── extract_telugu_subset.py  # Partitions 5.0h train / 1.0h val subsets for Telugu
│   ├── index_svarah.py           # Builds index and metadata manifests for Svarah
│   ├── index_telugu.py           # Builds index and metadata manifests for IndicTTS
│   └── inspect_datasets.py       # Audits sample rates, channel count, and durations
│
├── alignment_preprocessing/      # Forced alignment, G2P, and text cleaning validations
│   ├── test_english_alignment_5samples.py # MMS_FA alignment verification on 5 English samples
│   ├── test_telugu_alignment_5samples.py  # MMS_FA alignment verification on 5 Telugu samples
│   ├── test_cleaners.py          # Validates English text normalization cleaners
│   ├── test_symbols.py           # Verifies phoneme symbol mappings and ARPAbet dictionary
│   └── test_telugu_g2p.py        # Verifies Telugu akshara tokenization and geminate handling
│
├── training_inference/           # Acoustic model diagnostics and intermediate inspection
│   ├── diagnose_english.py       # Controlled test: ground-truth vs predicted mel through HiFi-GAN
│   └── inspect_e1_t1.py          # Direct waveform and acoustic feature inspector for E1 & T1
│
├── plotting/                     # Visualization generators
│   └── plot_variance_adaptor.py  # Generates all 30 duration, pitch (F0), and energy plots
│
└── verification/                 # End-to-end deliverables and environment testing
    ├── verify_env.py             # Verifies CUDA 12.4, PyTorch, PyWORLD, and Librosa
    ├── verify_phase1.py          # Phase 1 verification (Environment & GPU)
    ├── verify_phase2.py          # Phase 2 verification (Dataset Partitions & Disjointness)
    ├── verify_phase3_phase4.py   # Phase 3 & 4 verification (MMS_FA and Feature Extraction)
    └── audit_submission.py       # Comprehensive audit for all checkpoints, audio, and plots
```

---

## Common Usage Examples

### 1. Generating Variance Adaptor Visualizations (30 Plots)
```powershell
.\venv_fs2\Scripts\python.exe scripts/plotting/plot_variance_adaptor.py
```

### 2. Running Complete Submission Audit
```powershell
.\venv_fs2\Scripts\python.exe scripts/verification/audit_submission.py
```

### 3. Testing Forced Alignment on Indian English & Telugu
```powershell
.\venv_fs2\Scripts\python.exe scripts/alignment_preprocessing/test_english_alignment_5samples.py
.\venv_fs2\Scripts\python.exe scripts/alignment_preprocessing/test_telugu_alignment_5samples.py
```

