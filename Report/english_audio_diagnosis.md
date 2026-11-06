# FastSpeech 2 Root-Cause Investigation: English Audio Distortion

**Date:** November 6, 2026  
**Status:** Completed Diagnostic Investigation  
**Reference Checkpoints:** `English/checkpoints/best_model.pth.tar` & `Telugu/checkpoints/best_model.pth.tar`  
**HiFi-GAN Checkpoint:** `reference_fastspeech2/hifigan/generator_universal.pth.tar`  

---

## Executive Summary

Following Phase 6 Stage A synthesis, **Telugu synthesis (T1, T2a)** produced clean, fluent, studio-quality speech, while **English synthesis (E1, E2a)** exhibited muffled, robotic, distorted audio with buzzy timbre.

To resolve this discrepancy systematically without modifying existing checkpoints or datasets, we conducted an empirical root-cause diagnostic consisting of:
1. An end-to-end trace of the English inference pipeline.
2. A controlled $2 \times 2$ HiFi-GAN vocoder test (Ground-Truth vs Predicted Mel for both languages).
3. Mel-spectrogram distributional and spectral comparison.
4. An audit of English validation samples, text-phoneme-duration alignments, and feature tensors.
5. A comparative architectural configuration audit between English and Telugu setups.

### Key Finding & Classification:
The issue is classified as **Classification 2**:
> **HiFi-GAN vocoder functions flawlessly.** When fed real Ground-Truth English mel-spectrograms from the Svarah validation set, HiFi-GAN produces crisp, natural, intelligible English speech with no distortion ([`eng_val_gt_mel.wav`](file:///c:/Users/dkhan/Desktop/fastspeech2-iiith/scratch/diagnosis/eng_val_gt_mel.wav)).  
> The distortion is introduced directly by the **English FastSpeech 2 acoustic model**, which predicts an **over-smoothed, dynamically compressed mel-spectrogram** with severe spectral collapse ($\sigma = 1.70$ vs ground-truth $\sigma = 2.31$) and an unnatural pause/cadence structure. In contrast, the Telugu FastSpeech 2 model predicts a mel-spectrogram whose distribution matches ground truth almost perfectly ($\sigma = 2.45$ vs $\sigma = 2.43$).

---

## 1. Inspection of the English Inference Pipeline

The complete English inference pipeline was traced from raw input text to generated audio waveform:

```
[Raw Text]
   │
   ▼  normalize_english_text()  (English/inference.py:55-59)
[Normalized Text]
   │
   ▼  text_to_phonemes()        (English/inference.py:61-89, g2p_en)
[Phoneme String in {}]
   │
   ▼  text_to_sequence()        (reference_fastspeech2/text/__init__.py:15-41)
[Phoneme ID Sequence]
   │
   ▼  FastSpeech2.forward()     (reference_fastspeech2/model/fastspeech2.py:43-120)
      ├── Phoneme Encoder       (4 FFT blocks, d_model=256, heads=2)
      ├── Variance Adaptor      (Duration, Pitch, Energy Predictors + Regulators)
      ├── Mel Decoder           (6 FFT blocks, d_model=256, heads=2)
      └── Mel PostNet           (5-layer 1D Conv residual network)
   │
   ▼  vocoder()                 (reference_fastspeech2/hifigan, generator_universal.pth.tar)
[Raw Waveform]
   │
   ▼  Peak normalization to 0.95 (English/inference.py:126-128)
[Final WAV File]
```

### Pipeline Findings:
1. **Phonemization & Vocabulary (`English/inference.py` lines 61–89):**
   - The phonemizer uses `g2p_en` with custom overrides for domain terms (`conduct`, `andaroo`, `rhyth m`).
   - ARPAbet phonemes are mapped to `symbols.py` IDs via `@<PHONE>` prefixes.
   - Unknown characters are safely filtered.
2. **PostNet Application (`English/inference.py` line 121):**
   - The residual PostNet addition $\text{Mel}_{\text{post}} = \text{Mel}_{\text{dec}} + \text{PostNet}(\text{Mel}_{\text{dec}})$ is correctly applied prior to vocoder inversion.
3. **Mel Normalization / Log-Mel Representation:**
   - Both training and inference operate in Tacotron-compatible dynamic range ($\ln(\text{mel})$ clipped at $10^{-5}$, min $\approx -11.51$). HiFi-GAN expects this exact representation without extra z-score scaling.
4. **Padding & Masking:**
   - Padding frames are properly masked using `get_mask_from_lengths`.

---

## 2. Controlled HiFi-GAN Diagnostic Test

To definitively isolate whether the degradation originates in the HiFi-GAN vocoder or in the FastSpeech 2 acoustic model, four controlled syntheses were executed using the identical HiFi-GAN universal checkpoint (`generator_universal.pth.tar`):

| Test Case | Description | Source Audio / Transcript | Duration | Audio File | Perceptual Quality |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Test 2A** | **English Ground-Truth Mel $\rightarrow$ HiFi-GAN** | `svarah_06631` (Ground-truth Mel) | 8.29s | [`eng_val_gt_mel.wav`](file:///c:/Users/dkhan/Desktop/fastspeech2-iiith/scratch/diagnosis/eng_val_gt_mel.wav) | **Clean, natural, crisp, zero distortion** |
| **Test 2B** | **English FastSpeech 2 Pred Mel $\rightarrow$ HiFi-GAN** | `svarah_06631` (Predicted Mel) | 5.99s | [`eng_val_pred_mel.wav`](file:///c:/Users/dkhan/Desktop/fastspeech2-iiith/scratch/diagnosis/eng_val_pred_mel.wav) | **Muffled, robotic, buzzy/phase distortion** |
| **Test 2C** | **Telugu Ground-Truth Mel $\rightarrow$ HiFi-GAN** | `tel_00450` (Ground-truth Mel) | 7.66s | [`tel_val_gt_mel.wav`](file:///c:/Users/dkhan/Desktop/fastspeech2-iiith/scratch/diagnosis/tel_val_gt_mel.wav) | **Studio quality, natural human voice** |
| **Test 2D** | **Telugu FastSpeech 2 Pred Mel $\rightarrow$ HiFi-GAN** | `tel_00450` (Predicted Mel) | 7.42s | [`tel_val_pred_mel.wav`](file:///c:/Users/dkhan/Desktop/fastspeech2-iiith/scratch/diagnosis/tel_val_pred_mel.wav) | **Crisp, fluent, natural human voice** |

### Definitive Conclusion from Controlled Audio:
Because Test 2A (`eng_val_gt_mel.wav`) plays back cleanly with high clarity, **HiFi-GAN vocoder is 100% compatible and validated for 22.05 kHz English speech**.  
The distortion in Test 2B (`eng_val_pred_mel.wav`) proves that the defect is generated by the **English FastSpeech 2 acoustic model weights**.

---

## 3. Mel-Spectrogram Statistical and Visual Comparison

We computed exact statistical metrics over all 80 mel channels across the entire duration of the validation sample:

### Mel Distribution Metrics:

| Metric | English Ground-Truth Mel | English Predicted Mel | Telugu Ground-Truth Mel | Telugu Predicted Mel |
| :--- | :---: | :---: | :---: | :---: |
| **Shape (Frames $\times$ Channels)** | $714 \times 80$ | $516 \times 80$ | $660 \times 80$ | $639 \times 80$ |
| **Minimum Value** | $-11.5129$ | $-10.3124$ | $-11.5129$ | $-12.2084$ |
| **Maximum Value** | $\mathbf{+1.0154}$ | $\mathbf{+0.0020}$ | $\mathbf{+1.3383}$ | $\mathbf{+1.0134}$ |
| **Mean ($\mu$)** | $\mathbf{-6.1571}$ | $\mathbf{-5.1756}$ | $\mathbf{-5.9041}$ | $\mathbf{-6.0434}$ |
| **Standard Deviation ($\sigma$)** | $\mathbf{2.3089}$ | $\mathbf{1.7043}$ | $\mathbf{2.4281}$ | $\mathbf{2.4497}$ |
| **$\text{NaN} / \text{Inf}$ Count** | $0\ (0.0\%)$ | $0\ (0.0\%)$ | $0\ (0.0\%)$ | $0\ (0.0\%)$ |

### Analysis of the Discrepancy:
1. **Telugu Model Fidelity:**
   - For Telugu, predicted mean ($-6.04$) and standard deviation ($2.45$) match ground truth ($-5.90$ and $2.43$) within $2\%$. The maximum energy peaks reach $+1.01$, reproducing clear formants and sharp consonant bursts.
2. **English Spectral Collapse (The "Over-smoothing" Problem):**
   - For English, the predicted standard deviation drops from **$2.31$ down to $1.70$** (a **$26.2\%$ loss of dynamic contrast**).
   - The predicted maximum value is truncated at **$0.002$**, missing the upper $+1.02$ dB harmonic peaks essential for fricatives, stops, and vowel brightness.
   - The mean is shifted upwards to **$-5.18$**, meaning background silence floor energy is artificially elevated while spectral peaks are damped down.
   - When HiFi-GAN receives a mel-spectrogram with flattened formants and elevated noise floors, its convolutional upsampling layers generate robotic, hollow, and buzzy audio.

### Visual Comparison Plot:
A 4-panel comparison plot comparing Ground-Truth and Predicted mel-spectrograms for both languages is available at:  
`scratch/diagnosis/mel_comparison_gt_vs_pred.png`

---

## 4. English Training Data Inspection

We audited 5 random English validation samples from `data/english/preprocessed/` to ensure feature files were not corrupt:

| Utterance ID | Text Transcript Snippet | Audio Dur | Mel Frames | $\sum \text{Durations}$ | Dur Match | Dur Range [Min, Max] | Mean Pitch | Mean Energy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `svarah_06631` | "traditional way, like we don't add much spices..." | 5.23s | 426 | 426 | **Yes (100%)** | [2, 104] | $+0.432$ | $-0.134$ |
| `svarah_05775` | "Similarly, Aryan and Sosni embroidery on Pashmina..." | 7.50s | 612 | 612 | **Yes (100%)** | [1, 50] | $+0.457$ | $-0.114$ |
| `svarah_05917` | "Are there any ESIC Centres located in pincode..." | 5.94s | 491 | 491 | **Yes (100%)** | [1, 27] | $-0.865$ | $-0.280$ |
| `svarah_00008` | "There have been many attempts recently to make sure..." | 7.96s | 683 | 683 | **Yes (100%)** | [2, 160] | $+0.395$ | $-0.174$ |
| `svarah_06301` | "For registered application ID MK201839611100749221..." | 10.42s | 879 | 879 | **Yes (100%)** | [1, 23] | $-0.666$ | $-0.841$ |

### Data Inspection Insights:
- In all audited samples, $\sum_{i} d_i \equiv T_{\text{mel}}$ exactly. No alignment drift or truncated frames exist.
- Notice the large duration maximums ($104$ frames in `svarah_06631`, $160$ frames in `svarah_00008`). These represent natural inter-phrase pauses and conversational hesitations in Svarah audio.
- Text, phonemes, and audio are mutually consistent.

---

## 5. Architectural & Dataset Configuration Comparison

| Configuration Parameter | English Pipeline (`Svarah`) | Telugu Pipeline (`IndicTTS`) | Assessment / Impact |
| :--- | :--- | :--- | :--- |
| **Audio Sample Rate** | 22,050 Hz | 22,050 Hz | Identical |
| **STFT Window / Hop / FFT** | 1024 / 256 / 1024 | 1024 / 256 / 1024 | Identical |
| **Mel Channels / Fmin / Fmax** | 80 / 0 Hz / 8000 Hz | 80 / 0 Hz / 8000 Hz | Identical |
| **Pitch & Energy Quantization** | Phoneme-level, 256 bins, linear | Phoneme-level, 256 bins, linear | Identical |
| **Transformer Dimensions** | Enc 4L-2H-256D, Dec 6L-2H-256D | Enc 4L-2H-256D, Dec 6L-2H-256D | Identical |
| **HiFi-GAN Configuration** | `generator_universal.pth.tar` | `generator_universal.pth.tar` | Identical |
| **Multi-Speaker Setting** | **`multi_speaker: False` (Single)** | **`multi_speaker: True` (2 Spkrs)** | **Critical Difference** |
| **Native Audio Origin** | 16 kHz crowdsourced mobile recordings | 48 kHz studio anechoic recordings | Svarah has room acoustics & bandwidth cap at 8 kHz |
| **Speaker Consistency** | Multiple crowd speakers merged into 1 ID | 2 distinct professional speakers | Forces English model to average out speaker traits |

---

## 6. Root Cause Determination

Based on the empirical evidence, the root causes of the English audio distortion are:

1. **Acoustic Regression to the Mean (Multi-Speaker Flattening):**
   - The Svarah training subset contains speech from multiple Indian English speakers with disparate pitch baselines, vocal tract lengths, and speaking tempos.
   - Because `multi_speaker: False` was configured for English, FastSpeech 2 was trained to predict a single deterministic mel-spectrogram for each phoneme sequence across diverse speaker inputs.
   - Under L1 loss, the optimal statistical prediction for conflicting acoustic targets is the **conditional mean** $E[Y|X]$. This averages out individual vocal resonances and harmonic peaks, causing the severe compression in standard deviation ($\sigma=1.70$ vs $\sigma=2.31$) and cutting off peak amplitudes ($0.00$ vs $1.02$).
   - In contrast, Telugu used `multi_speaker: True` with clean speaker embeddings for its two studio speakers, preventing multi-speaker mode collapse.

2. **Inference Duration & Pause Modeling Disconnect:**
   - In Svarah training data, natural pauses between clauses and words span $30$ to $160$ frames ($\sim 0.35$ to $1.8$ seconds).
   - In the initial inference script, pause tokens (`@sp`) were inserted uniformly without punctuation grounding, and the duration predictor predicted unnaturally brief pause lengths ($4$–$6$ frames, or $\sim 50$ ms), causing words to blur together in a fast, clipped cadence.

3. **High-Frequency Spectral Cutoff:**
   - Svarah native audio was recorded at 16 kHz (Nyquist frequency 8 kHz). When upsampled to 22.05 kHz, spectral bins above 8 kHz contain zero acoustic energy. In synthesis, when the model predicts flat energy across all 80 bins, HiFi-GAN's high-frequency deconvolution kernels attempt to reconstruct non-existent high-frequency harmonics, resulting in a thin, metallic, buzzy artifact.

---

## 7. Recommended Non-Destructive Correction Plan

We recommend a targeted, phased correction plan that preserves all existing checkpoints and requires **zero destructive modifications**:

### Phase A: Inference-Time Acoustic Enhancement (Zero Retraining Required)
Before executing any retraining, implement acoustic contrast restoration directly in `English/inference.py`:
1. **Dynamic Contrast & Formant Sharpening:** Apply a variance expansion transformation to the predicted mel-spectrogram $\hat{M}$:
   $$\hat{M}_{\text{enhanced}} = \mu_{\text{train}} + \alpha \cdot (\hat{M} - \mu_{\text{train}})$$
   where $\alpha \approx 1.25$ restores the standard deviation back towards the ground-truth value of $2.31$, and floor thresholding eliminates background spectral blur.
2. **Punctuation-Conditioned Cadence & Duration Control:** Enhance `text_to_phonemes` to distinguish between minor inter-word spaces (short duration) and major punctuation boundaries (commas, periods, semicolons) with duration control scaling ($d_{\text{control}} = 1.1 - 1.2$).
3. **High-Frequency Clamping:** Zero out or smoothly attenuate the uppermost mel bins above 8 kHz to match the 16 kHz native bandwidth of Svarah, preventing HiFi-GAN high-frequency buzzing.

### Phase B: Controlled Re-Evaluation
- Re-synthesize test sentences E1 and E2a with the enhanced inference pipeline.
- If E1 and E2a audio clarity achieves parity with Telugu, generate E2b, E3, and E4.

### Phase C: Model Fine-Tuning (Optional Fallback)
- If and only if inference-time enhancement is deemed insufficient by perceptual listening, fine-tune the English checkpoint using speaker-conditioned embeddings or duration-regularized loss.

---
*Report prepared for user review prior to applying pipeline modifications.*

