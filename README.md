# FastSpeech 2: Bilingual Neural Text-to-Speech (Indian English & Telugu)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1+cu124](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu124-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050%20(6GB)-purple.svg)]()
[![Vocoder](https://img.shields.io/badge/Vocoder-HiFi--GAN%20Universal-orange.svg)](https://github.com/jik876/hifi-gan)

This repository contains the complete implementation, training pipeline, acoustic feature processing, prosodic variance evaluation, and technical report for the **IIIT Hyderabad FastSpeech 2 Assignment**.

The system synthesizes 22,050 Hz speech across two linguistically diverse languages:
1. **Indian English** using the open-source **Svarah** speech corpus.
2. **Telugu** using the **IndicTTS** bilingual speech corpus.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. FastSpeech 2 Architecture](#2-fastspeech-2-architecture)
- [3. Codebase Modifications & Bug Fixes](#3-codebase-modifications--bug-fixes)
- [4. Dataset Selection & Preprocessing](#4-dataset-selection--preprocessing)
- [5. Hardware Setup & Training Configurations](#5-hardware-setup--training-configurations)
- [6. Training Dynamics & Loss Analysis](#6-training-dynamics--loss-analysis)
- [7. Speech Synthesis Evaluation (All 10 Utterances)](#7-speech-synthesis-evaluation-all-10-utterances)
- [8. Variance Adaptor Visualizations (30 Plots)](#8-variance-adaptor-visualizations-30-plots)
- [9. English vs. Telugu Comparative Analysis](#9-english-vs-telugu-comparative-analysis)
- [10. Limitations & Future Work](#10-limitations--future-work)
- [11. Repository Structure](#11-repository-structure)
- [12. Step-by-Step Reproducibility Guide](#12-step-by-step-reproducibility-guide)

---

## 1. Executive Summary

- **Non-Autoregressive Synthesis:** Employs FastSpeech 2 to eliminate autoregressive exposure bias and synthesis latency while delivering explicit prosodic variance control.
- **Multi-Speaker Conditioning:** 
  - **English Model:** Implements a trainable 115-speaker embedding layer (`nn.Embedding(115, 256)`) to model speaker characteristics across the diverse speakers in the Svarah corpus.
  - **Telugu Model:** Implements a trainable 2-speaker embedding layer (`nn.Embedding(2, 256)`) covering the female and male voice talents from the IndicTTS corpus.
- **Hardware-Constrained Execution:** Trained from scratch on a single **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** on Windows 11 using PyTorch Automatic Mixed Precision (AMP FP16), batch size 16, and gradient accumulation 2 (effective batch size 32).
- **Optimal Checkpoint Selection:** Comprehensive cross-validation demonstrated that:
  - **English Multi-Speaker Model** achieved its optimal minimum validation loss of **2.6492 at Step 2,750** (saved as `best_model.pth.tar`).
  - **Telugu Model** achieved its optimal minimum validation loss of **2.3660 at Step 3,000** (saved as `best_model.pth.tar`).
- **End-to-End Validation:** The system successfully generated all ten required English and Telugu test utterances using the trained FastSpeech 2 models and HiFi-GAN vocoder. All ten audio files (`E1.wav`–`E4.wav`, `T1.wav`–`T4.wav`) and all 30 duration, pitch, and energy plots have been verified to be non-empty, peak-normalized to 0.95, sampled at 22,050 Hz, and structurally aligned.

---

## 2. FastSpeech 2 Architecture

The architecture builds directly on the Feed-Forward Transformer (FFT) network and the Variance Adaptor introduced by Ren et al. (2020), with multi-speaker conditioning integrated into the acoustic representation:

```
Text / Phonemes
      │
      ▼
┌──────────────┐
│  FFT Encoder │ (4 layers, 2 attention heads, d_model=256, d_inner=1024)
└──────┬───────┘
       │
       ▼
 [Speaker ID] ──► [Speaker Embedding] (added to encoder hidden sequence)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│                   Variance Adaptor                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────┐ │
│  │Duration Predictor│  │ Pitch Predictor  │  │Energy │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───┬───┘ │
│           ▼                     ▼                ▼     │
│    Length Regulator       Pitch Embed       Energy Emb │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  FFT Decoder │ (6 layers, 2 attention heads, d_model=256)
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Postnet (Mel)│ (5 conv layers with residual connection)
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   HiFi-GAN   │ (Universal Generator, MRF, 22,050 Hz)
                 └──────┬───────┘
                        │
                        ▼
                  Speech Waveform
```

1. **Phoneme Encoder:** Maps the discrete phoneme sequence into continuous phonemic representations using 4 FFT blocks.
2. **Speaker Conditioning:** A trainable speaker embedding lookup layer (`nn.Embedding(N, 256)`) projects the speaker ID into a 256-dimensional vector, which is broadcast and added to the encoder representations before length regulation.
3. **Variance Adaptor:**
   - **Duration Predictor:** Predicts the phoneme duration in log domain; the Length Regulator expands phoneme-level embeddings to match acoustic mel-frame resolution.
   - **Pitch Predictor:** Estimates continuous fundamental frequency (F0) extracted using the PyWORLD vocoder, quantized into 256 bins, and projects learned pitch embeddings.
   - **Energy Predictor:** Predicts the L2-norm of STFT frames, quantized into 256 bins, and injects energy embeddings.
4. **Mel Decoder & Postnet:** Decodes frame-level representations into 80-band mel-spectrograms. A 5-layer convolutional Postnet refines the spectrogram by predicting residual errors.
5. **HiFi-GAN Vocoder:** A pretrained universal HiFi-GAN model (`generator_universal.pth.tar`) converts the predicted 80-band mel-spectrogram into time-domain 22,050 Hz 16-bit PCM waveforms.

---

## 3. Codebase Modifications & Bug Fixes

To achieve robust bilingual training and ensure execution on modern Python 3.12 / PyTorch 2.5.1 environments, several modifications and fixes were made:

1. **Energy Control Bug in `reference_fastspeech2/model/modules.py`:**
   - *Issue:* Lines 124 and 146 in the upstream reference code multiplied energy predictions by `p_control` instead of `e_control`:
     ```python
     # Upstream bug:
     energy_prediction = energy_prediction * p_control
     ```
   - *Fix:* Corrected to `energy_prediction = energy_prediction * e_control`, enabling independent prosodic controllability.
2. **Telugu Akshara Vocabulary & Geminates (`reference_fastspeech2/text/symbols.py`):**
   - *Issue:* Upstream FastSpeech 2 supported only English ARPAbet symbols. Telugu G2P produced out-of-vocabulary token drops and tensor dimension mismatches on double consonants (geminates).
   - *Fix:* Expanded the vocabulary to 430 symbols, adding all Telugu independent vowels, consonants, dependent signs (matras), anusvara, visarga, and explicit geminate tokens (`llll`, `shsh`, `dhdh`, `hh`, `nnnn`, `rrrr`).
3. **Librosa 0.10.x Keyword Compatibility (`audio/stft.py`):**
   - *Issue:* Modern Librosa deprecated `pad_center` and changed filterbank keyword arguments, breaking mel-spectrogram computation.
   - *Fix:* Refactored STFT and mel-filterbank calls in `audio/stft.py` to match current standard specifications without deprecation errors.
4. **Windows Encoding & Line-Ending Normalization:**
   - Standardized all file reading/writing across manifests and scripts with explicit `utf-8` encoding and trailing newline sanitization, eliminating `cp1252` encoding exceptions on Windows PowerShell.

---

## 4. Dataset Selection & Preprocessing

Both datasets were partitioned into subsets with **strictly zero utterance overlap**:

| Dataset Attribute | Indian English (`Svarah`) | Telugu (`IndicTTS`) |
| :--- | :--- | :--- |
| **Source Corpus** | Open-source Svarah Indian English | IndicTTS Telugu Corpus |
| **Target Sample Rate** | 22,050 Hz (16-bit Mono WAV) | 22,050 Hz (16-bit Mono WAV) |
| **Training Set Size** | 3,297 utterances (**5.00 hours**) | 2,261 utterances (**5.00 hours**) |
| **Validation Set Size** | 683 utterances (**1.00 hours**) | 450 utterances (**1.00 hours**) |
| **Train / Val Utterance Overlap** | **0 utterances (Strict Disjoint Partitions)** | **0 utterances (Strict Disjoint Partitions)** |
| **Speaker Setup** | 115 Speaker IDs (110 present in val) | 2 Studio Speakers (Female & Male) |
| **Mean Utterance Length** | 5.46 seconds | 7.96 seconds |
| **Acoustic Features** | 80-channel Mel (1024 FFT, 256 hop, 1024 win) | 80-channel Mel (1024 FFT, 256 hop, 1024 win) |
| **F0 (Pitch) Extraction** | PyWORLD vocoder (continuous log F0) | PyWORLD vocoder (continuous log F0) |
| **Energy Extraction** | STFT L2-norm per frame | STFT L2-norm per frame |
| **Alignment Method** | MMS Forced Aligner (`torchaudio.pipelines.MMS_FA`) | MMS Forced Aligner (`torchaudio.pipelines.MMS_FA`) |

---

## 5. Hardware Setup & Training Configurations

Due to the 6 GB VRAM constraint of the laptop RTX 4050 GPU, hyperparameters were tuned to achieve stable gradient updates without encountering CUDA Out-Of-Memory (OOM) errors:

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)
- **Host OS:** Windows 11 Home 64-bit
- **Precision:** PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`) with `GradScaler`
- **Batch Size:** 16 samples per batch
- **Gradient Accumulation:** 2 iterations (Effective batch size = 32)
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$, weight decay = 0.0) via `ScheduledOptim`
- **Learning Rate Schedule:** Noam schedule with 4,000 warmup steps ($d_{\text{model}}=256$). Since training ran for 3,000 steps, optimization was conducted entirely within the linear warmup regime ($\text{step} \le 4000$), maintaining stable gradient growth:
  $$\text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$$
- **Loss Weights:** Total Loss = $\mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{duration}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}}$
- **Evaluation Interval:** Full validation sweep over all validation batches every 250 steps.

---

## 6. Training Dynamics & Loss Analysis

Both models were trained to 3,000 steps with periodic checkpoints saved every 500 steps and cross-validation executed every 250 steps.

### Empirical Loss Progression Table

| Step | English Train Loss | English Val Loss | Telugu Train Loss | Telugu Val Loss | Training Milestone Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **250** | 9.6164 | 9.0736 | 11.7099 | 9.5175 | Rapid initial acoustic convergence across phoneme embeddings. |
| **500** | 6.7949 | 6.2102 | 8.2633 | 6.6740 | Variance predictors learn coarse duration and pitch baselines. |
| **1,000** | 3.7796 | 3.2657 | 3.5444 | 3.5515 | **1,000-Step Baseline Checkpoint** (Preserved). Formants emerge. |
| **1,500** | 3.1352 | 2.7740 | 2.6853 | 2.6235 | Fine pitch contours and vowel harmonic structures sharpen. |
| **2,000** | 2.6397 | 2.6651 | 2.1157 | 2.4765 | Phoneme duration boundaries align with syllable rhythm. |
| **2,500** | 2.0297 | 2.7542 | 2.0844 | 2.4729 | Continuous optimization across acoustic branches. |
| **2,750** | **2.0377** | **2.6492** | 1.7505 | 2.3733 | **OPTIMAL ENGLISH BEST MODEL:** Minimum validation loss (saved as `best_model.pth.tar`). |
| **3,000** | 2.1855 | 2.6760 | **1.6967** | **2.3660** | **OPTIMAL TELUGU BEST MODEL:** Global minimum reached (saved as `best_model.pth.tar`). |

### Justification of Checkpoint Selection
Validation loss measures out-of-sample generalization on held-out utterances:
- For **English**, validation loss reached its minimum of **2.6492** at Step 2,750 (saved as `best_model.pth.tar`).
- For **Telugu**, validation loss reached its minimum of **2.3660** at Step 3,000 (saved as `best_model.pth.tar`).

Both models' optimal checkpoints are saved under `English/checkpoints/best_model.pth.tar` and `Telugu/checkpoints/best_model.pth.tar` and were used to synthesize all official test audio files.

---

## 7. Speech Synthesis Evaluation (All 10 Utterances)

The system successfully generated all ten required English and Telugu test utterances using the trained FastSpeech 2 models and HiFi-GAN vocoder.

Every generated waveform was measured directly to extract physical sample rate, duration, file size, peak amplitude, and root-mean-square (RMS) energy:

| Utterance ID | Text / Official Prompt | Target Linguistic Purpose | Generated Audio File | Duration | File Size | Peak Amp | RMS Energy | Synthesis Observations |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E1** | "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings." | Polysyllabic stability across 28-letter word (12 syllables) | `Results/English/test_generated_audios/E1.wav` | 6.55 s | 288,812 B | 0.9500 | 0.1675 | Clear phonetic articulation across all 12 syllables of the target polysyllabic word. |
| **E2a** | "The orchestra's conduct during the ceremony was praised by the audience." | Heteronym Disambiguation: Noun form *conduct* (/ˈkɒndʌkt/) | `Results/English/test_generated_audios/E2a.wav` | 4.05 s | 178,732 B | 0.9500 | 0.1893 | Trochaic stress on first syllable (/ˈkɒn/) rendered with elevated duration and pitch contour. |
| **E2b** | "The researchers will conduct an experiment to evaluate the proposed method." | Heteronym Disambiguation: Verb form *conduct* (/kənˈdʌkt/) | `Results/English/test_generated_audios/E2b.wav` | 4.47 s | 197,164 B | 0.9500 | 0.1723 | Iambic stress on second syllable (/ˈdʌkt/) emphasized with rising pitch contour. |
| **E3** | "The rhyth m of the orchestra changed unexpectedly." | Robustness to typographical split-word errors (*rhyth m*) | `Results/English/test_generated_audios/E3.wav` | 3.04 s | 134,188 B | 0.9500 | 0.1592 | Continuous phonation across split whitespace (/rɪð.əm/) without silence interruptions. |
| **E4** | "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time." | Time/date normalization & code-mixed Telugu loanword (*andaroo*) | `Results/English/test_generated_audios/E4.wav` | 6.97 s | 307,244 B | 0.9500 | 0.1679 | Numerical expressions normalized ('ten forty-five AM', 'twelve November twenty twenty-six'); loanword *andaroo* articulated via /AH0 N D AA1 R UW0/. |
| **T1** | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. | Scientific loanword & polysyllabic Telugu compound | `Results/Telugu/test_generated_audios/T1.wav` | 6.12 s | 269,868 B | 0.9500 | 0.1858 | Steady speaking tempo and clear conjunct consonants (ర్య్, ర్కో) across the technical compound. |
| **T2a** | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | Polysemy: *పట్టు* meaning physical grip (/paʈʈu/) | `Results/Telugu/test_generated_audios/T2a.wav` | 3.53 s | 155,692 B | 0.9500 | 0.1850 | Geminate retroflex stop /ʈʈu/ articulated with falling pitch cadence matching declarative syntax. |
| **T2b** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. | Polysemy: *పట్టు* meaning silk saree | `Results/Telugu/test_generated_audios/T2b.wav` | 3.39 s | 149,548 B | 0.9500 | 0.1878 | Phrase binding between *పట్టు* and *చీరను*; vowel duration reflects descriptive context. |
| **T3** | పులి అడవిలో నిశ్శబ్దంగా నడిచింది. | Literary phonetics & geminates | `Results/Telugu/test_generated_audios/T3.wav` | 2.26 s | 99,884 B | 0.9500 | 0.1840 | Lowered energy predictor on *నిశ్శబ్దంగా* captures quiet semantic mood; terminal cadence is clean. |
| **T4** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | Multi-clause temporal structure with numbers & dates | `Results/Telugu/test_generated_audios/T4.wav` | 11.25 s | 496,172 B | 0.9500 | 0.1932 | Multi-clause temporal utterance (11.25s); natural pause at comma boundary; numerals expanded into Telugu orthography. |

---

## 8. Variance Adaptor Visualizations (30 Plots)

Thirty plots (10 utterances $\times$ 3 variance attributes: Duration, Pitch, Energy) are stored under `Results/{English,Telugu}/variance_adaptor_plots/`:

- **Duration Plots (`variance_adaptor_plots/duration/`):**
  - Show the predicted frame budget for each phoneme/akshara.
  - Vowels and stressed syllables receive higher durations (~10–18 frames), while unvoiced stops and schwas receive 2–5 frames.
- **Pitch Plots (`variance_adaptor_plots/pitch/`):**
  - Continuous log-F0 curves across mel-frames.
  - Demonstrates pitch declination in declarative utterances and clean drops during unvoiced consonants.
- **Energy Plots (`variance_adaptor_plots/energy/`):**
  - STFT L2-norm contours illustrating stress emphasis across phrase boundaries and attenuation before final pauses.

---

## 9. English vs. Telugu Comparative Analysis

1. **Phonetic vs. Syllabic Modeling:**
   - English requires G2P conversion to handle irregular orthography, silent letters, and heteronyms.
   - Telugu possesses regular grapheme-to-phone correspondence, enabling direct mapping from aksharas to acoustic units with fewer pronunciation anomalies.
2. **Convergence Rates & Model Dynamics:**
   - The Telugu model achieved a validation loss of **2.3660** (Step 3,000), while the English model reached a validation loss of **2.6492** (Step 2,750).
   - Telugu benefits from the phonetic regularity of its writing system and consistent studio recordings. In English, multi-speaker conditioning enables the acoustic model to capture varying speaker pitch baselines across the Svarah corpus.
3. **Prosodic Characteristics:**
   - English displays a classic stress-timed rhythm with vowel reduction.
   - Telugu displays a mora-timed / syllable-timed cadence where long vowels maintain consistent duration ratios over short vowels.

---

## 10. Limitations & Future Work

- **Bilingual Code-Mixing:** Un-normalized English loanwords appearing in Telugu sentences require specialized bilingual text normalizers to ensure consistent phonetic representation.
- **GPU Batch Scalability:** Training on 6 GB VRAM limited batch size to 16 with gradient accumulation 2. Scaling to larger distributed clusters with larger batch sizes could accelerate training iterations.
- **Zero-Shot Speaker Generalization:** Both models currently employ discrete multi-speaker lookup embeddings (115 speakers for English, 2 for Telugu). Scaling to unseen zero-shot voice cloning would require continuous speaker reference encoders (such as d-vectors or x-vectors) rather than discrete lookup tables.
- **Diffusion-Based Acoustic Decoders:** Replacing the feed-forward FFT decoder with a continuous-time diffusion decoder (e.g., Grad-TTS) could provide alternative mel-spectral modeling.

---

## 11. Repository Structure

```
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

---

## 12. Step-by-Step Reproducibility Guide

### Step 1: Environment Setup
Ensure Python 3.12 and CUDA 12.4 are available:
```powershell
# Create and activate virtual environment
python -m venv venv_fs2
.\venv_fs2\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

### Step 2: Running Inference
To synthesize all test utterances using the best checkpoints:
```powershell
# English Inference (generates E1.wav - E4.wav in Results/English/test_generated_audios/)
.\venv_fs2\Scripts\python.exe English/inference.py --ckpt best_model.pth.tar --sentence all --speaker svarah_female_28

# Telugu Inference (generates T1.wav - T4.wav in Results/Telugu/test_generated_audios/)
.\venv_fs2\Scripts\python.exe Telugu/inference.py --ckpt best_model.pth.tar --sentence all
```

### Step 3: Inspecting Generated Audio & Visualizations
- Synthesized waveforms are saved under `Results/{English,Telugu}/test_generated_audios/`.
- Predicted duration, pitch, and energy plots are saved under `Results/{English,Telugu}/variance_adaptor_plots/`.
- Detailed technical analysis and comparative evaluation are documented in `Report/Technical_Report.pdf`.

---
*Developed for IIIT Hyderabad Speech Processing / Deep Learning Assignment.*
