# DHVANI
### AI-Powered Adaptive Noise Cancellation for Defence Communications

**Smart India Hackathon 2026 – Problem Statement SIH26052 (DRDO) – Team Data_Aethris**

[Live Demo](https://dhvaniapp-fqgyshclojyemfmwweg6j2.streamlit.app)

---

## Overview

DHVANI is an AI + DSP hybrid system that classifies incoming audio in real time and surgically removes noise, preserving speech intelligibility even in harsh battlefield acoustic conditions — gunfire, artillery, helicopter rotors, and armoured vehicle engines.

DHVANI first classifies what kind of noise is present in every audio frame (AI), then removes that specific noise using a hybrid spectral-subtraction + deep-learning pipeline (DSP), instead of muting the whole frame, so speech stays intact.

Try the live demo: https://dhvaniapp-fqgyshclojyemfmwweg6j2.streamlit.app (zero install, works in any browser)

## The Problem

Battlefield communication is constantly disrupted by diverse acoustic noise. Traditional methods (spectral subtraction, Wiener filtering, classical LMS-based ANC) assume stationary noise and fail under rapidly changing, non-linear conditions, often distorting speech that overlaps with noise. No affordable, transparent Indian solution exists for this exact defence-grade requirement.

## Our Solution

- **AI noise classification**: a Random Forest classifier sorts every 100ms audio frame into Stationary, Non-Stationary, Impulsive, or Speech
- **Hybrid noise suppression**: classical spectral subtraction removes bulk noise, with an optional BiLSTM Complex-Ratio-Mask deep-learning stage for phase-preserving refinement
- **Confidence-gated safe mode**: avoids aggressive filtering when the AI is not confident, so a real command is never accidentally silenced
- **5ms impulsive spike detection**: a lightweight fast-path detector runs alongside the 100ms classifier, cutting impulsive-noise (gunfire/explosion) detection latency to roughly 2–5ms — 20–50x faster than waiting on the main classification window
- **Real-time performance**: measured 17-27ms per-frame latency (software pipeline) against a 100ms budget
- **Field Comms Relay**: simulates two-device (soldier-to-soldier) transmission of cleaned audio over a shared channel

## Architecture

Microphone Input → Feature Extraction (STFT / 39-dim MFCC) → AI Classifier (Random Forest) → 4-Class Detection → 5ms Fast-Path Spike Detector (parallel) → Stage A: Spectral Subtraction → Stage B: BiLSTM Complex-Ratio-Mask (optional) → Confidence-Gated Gain Control → LMS Adaptive Residual Filter → Clean Audio Output → Field Comms Relay (2-device demo)

## Hardware Prototype

A working hardware prototype has been built and validated on the ESP32-S3.

- **Processing**: ESP32-S3 (dual-core, PSRAM) running a verified real-time DSP pipeline
- **Measured performance**: 1.8ms processing per 16ms audio frame — an 8.8x real-time margin, leaving headroom for on-device NN work
- **On-device deployment**: the trained Random Forest classifier is quantized and deployed directly on the ESP32-S3 via micromlgen
- **Audio input**: I2S MEMS microphone (INMP441), wired, calibrated and hardware-tested
- **Ambient auto-calibration**: median-based, noise-robust
- **Impulsive fast-path**: onset-based 5ms crest-factor detector, running on-device
- **Audio output**: audio codec (ES8388/WM8960, dual ADC + dual DAC) for wired dual-direction connection to phone / walkie-talkie / headset
- **Full-duplex operation**: receive path (incoming call/radio audio → filtered → headset) and transmit path (soldier's mic → filtered → radio/call), both filtered
- **Echo/feedback handling**: LMS adaptive filter using the receive-path signal as reference
- **Power**: 3.7V Li-Ion 1000mAh + TP4056 charger + boost converter
- **Status indication**: 3-LED system (calibrating / live / spike-detected)
- **Platform rationale**: ESP32-S3 was chosen over Jetson-class SoCs for its ultra-low power draw and compact form factor — ideal for a soldier-worn wearable device. Jetson-class hardware is planned for a vehicle-mounted or command-post variant in a later phase.

This is a first-generation hardware prototype (enclosure + populated board); field-hardening, ruggedization, and battlefield-condition validation are still ahead — see Roadmap below.

## Tech Stack

**Language**: Python
**AI/ML**: scikit-learn (Random Forest), PyTorch (BiLSTM), Joblib, micromlgen (on-device model export/quantization)
**Audio Processing**: Librosa, NumPy, SciPy, SoundFile, SoundDevice, STFT/iSTFT
**Evaluation**: STOI, SI-SNR (PESQ planned)
**Web and UI**: Streamlit, streamlit-mic-recorder, streamlit-autorefresh, Matplotlib
**System**: FFmpeg / imageio-ffmpeg, psutil, subprocess
**Embedded**: ESP32-S3, I2S MEMS mic (INMP441), ES8388/WM8960 audio codec
**Deployment**: Git, GitHub, Streamlit Community Cloud

**Datasets**: ESC-50, UrbanSound8K, Freesound.org, RAVDESS, LibriSpeech, custom synthetic audio (gunfire, artillery, engine, wind, drone/UAV), self-recorded speech and edge-case corpus (16,900+ samples)

## Repository Structure

- `dhvani_app.py` - Main Streamlit application (UI + demo)
- `dhvani_core.py` - Core pipeline: classifier + spike detector + hybrid enhancer
- `dhvani_model_multiclass.pkl` - Trained Random Forest classifier (~88% accuracy)
- `dhvani_enhancement_model.pt` - Trained BiLSTM Complex-Ratio-Mask weights
- `share_app.py` - Local ngrok tunnel helper for sharing dev builds
- `requirements.txt` - Python dependencies

## Running Locally

```
git clone https://github.com/krishnabhardwaj8303-sys/dhvani_app.git
cd dhvani_app
pip install -r requirements.txt
streamlit run dhvani_app.py
```

The app needs ffmpeg available on PATH for microphone recording — imageio-ffmpeg bundles this automatically.

## Validated Results

- Classifier accuracy: ~88% across 16,900+ samples
- STOI (objective intelligibility): 0.85–0.96 across tested noise types
- Average SNR improvement: +9-10dB (measured, hybrid pipeline; up from an earlier +5.7dB baseline)
- Real-time software latency: 17-27ms per 100ms frame (well under budget)
- On-device hardware latency: 1.8ms per 16ms frame on ESP32-S3
- Impulsive-threat detection: 5ms sub-frame crest-factor detector, ~2-5ms effective detection latency

**Government benchmark targets**: STOI > 0.85 (met in most tested categories), SNR > 15dB (currently 9-10dB, iterative improvement in progress), PESQ > 2.5 (not yet measured, tooling in progress)

We validate scientifically using STOI/SNR ablation testing rather than subjective listening, and are transparent about what is built vs. roadmap — see Honest Project Status below.

## Honest Project Status

**Built and verified:**
- Random Forest classifier (Layer 1) — 88% accuracy
- BiLSTM Complex-Ratio-Mask enhancer (Layer 2) — trained end-to-end, phase-preserving
- Real-time software streaming pipeline — measured, not simulated
- Live public demo with mic input, file upload, and two-device Field Comms Relay
- Objective STOI/SNR validation and ablation testing
- **Hardware prototype on ESP32-S3** — on-device quantized Random Forest classifier (via micromlgen), I2S MEMS mic input, dual ADC/DAC codec output, LMS echo cancellation, 3-LED status indication, Li-Ion power system — 1.8ms/16ms measured processing latency

**In progress / roadmap:**
- BiLSTM currently runs as an optional Experimental Deep-Learning Mode on the software pipeline — ablation testing showed it needs more training data/epochs to consistently outperform the spectral-subtraction baseline. Retraining (40 to 100 epochs) improved hybrid SNR from -1.1dB to +4.1dB — actively closing this gap. Porting this stage to run on-device (ESP32-S3) is on the roadmap.
- PESQ measurement — blocked by a Windows C++ build toolchain requirement; on the roadmap
- Field-testing of the hardware prototype under real battlefield acoustic conditions (gunfire/artillery recordings), DRDO/defence evaluation, and OEM integration
- Dual-microphone (primary + acoustic reference mic) setup, to improve LMS adaptive filtering accuracy beyond the current receive-path-only reference signal
- Two-way continuous live conversation — the software Field Comms Relay demo is currently send/receive (snapshot), not full-duplex streaming

## Roadmap

**Near-term**: Complete PESQ measurement, close the SNR gap from 9-10dB to the 15dB target, validate on real battlefield gunfire and artillery recordings.

**Mid-term**: Dual-microphone array for better echo cancellation, per-soldier personalised calibration, bone-conduction microphone option for use under gas masks, a vehicle-mounted variant on Jetson-class hardware.

**Long-term**: Full multilingual speech recognition, an Indian battlefield-acoustic dataset to reduce dependency on foreign data, cross-domain use in maritime, aviation, and disaster-response communications.

## Team

Team Data_Aethris — Smart India Hackathon 2026, Problem Statement SIH26052 (DRDO)

## License

Add your chosen license here (e.g. MIT).
