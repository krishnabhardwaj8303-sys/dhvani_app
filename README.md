# DHVANI

**AI-Powered Adaptive Noise Cancellation for Defence Communications**

Smart India Hackathon 2026 - Problem Statement SIH26052 (DRDO) - Team Dhvani

[Live Demo](https://dhvaniapp-fqgyshclojyemfmwweg6j2.streamlit.app)

## Overview

DHVANI is an AI + DSP hybrid system that classifies incoming audio in real time and surgically removes noise, preserving speech intelligibility even in harsh battlefield acoustic conditions - gunfire, artillery, helicopter rotors, and armoured vehicle engines.

DHVANI first classifies what kind of noise is present in every audio frame (AI), then removes that specific noise using a hybrid spectral-subtraction + deep-learning pipeline (DSP), instead of muting the whole frame, so speech stays intact.

Try the live demo: https://dhvaniapp-fqgyshclojyemfmwweg6j2.streamlit.app (zero install, works in any browser)

## The Problem

Battlefield communication is constantly disrupted by diverse acoustic noise. Traditional methods (spectral subtraction, Wiener filtering, classical LMS-based ANC) assume stationary noise and fail under rapidly changing, non-linear conditions, often distorting speech that overlaps with noise. No affordable, transparent Indian solution exists for this exact defence-grade requirement.

## Our Solution

- AI noise classification: a Random Forest classifier sorts every 100ms audio frame into Stationary, Non-Stationary, Impulsive, or Speech
- Hybrid noise suppression: classical spectral subtraction removes bulk noise, with an optional BiLSTM Complex-Ratio-Mask deep-learning stage for phase-preserving refinement
- Confidence-gated safe mode: avoids aggressive filtering when the AI is not confident, so a real command is never accidentally silenced
- 5ms impulsive spike detection: a lightweight fast-path detector catches gunfire/explosions 20-50x faster than the main classifier
- Real-time performance: measured 17-27ms per-frame latency against a 100ms budget
- Field Comms Relay: simulates two-device (soldier-to-soldier) transmission of cleaned audio over a shared channel

## Architecture

Microphone Input -> Feature Extraction (STFT / 39-dim MFCC) -> AI Classifier (Random Forest) -> 4-Class Detection -> 5ms Fast-Path Spike Detector (parallel) -> Stage A: Spectral Subtraction -> Stage B: BiLSTM Complex-Ratio-Mask (optional) -> Confidence-Gated Gain Control -> Clean Audio Output -> Field Comms Relay (2-device demo)

## Tech Stack

Language: Python
AI/ML: scikit-learn (Random Forest), PyTorch (BiLSTM), Joblib, micromlgen
Audio Processing: Librosa, NumPy, SciPy, SoundFile, SoundDevice, STFT/iSTFT
Evaluation: STOI, SI-SNR (PESQ planned)
Web and UI: Streamlit, streamlit-mic-recorder, streamlit-autorefresh, Matplotlib
System: FFmpeg / imageio-ffmpeg, psutil, subprocess
Deployment: Git, GitHub, Streamlit Community Cloud

Datasets: ESC-50, RAVDESS, LibriSpeech, self-recorded speech and edge-case corpus (16,900+ samples)

## Repository Structure

dhvani_app.py - Main Streamlit application (UI + demo)
dhvani_core.py - Core pipeline: classifier + spike detector + hybrid enhancer
dhvani_model_multiclass.pkl - Trained Random Forest classifier (~88% accuracy)
dhvani_enhancement_model.pt - Trained BiLSTM Complex-Ratio-Mask weights
share_app.py - Local ngrok tunnel helper for sharing dev builds
requirements.txt - Python dependencies

## Running Locally

git clone https://github.com/krishnabhardwaj8303-sys/dhvani_app.git
cd dhvani_app
pip install -r requirements.txt
streamlit run dhvani_app.py

The app needs ffmpeg available on PATH for microphone recording - imageio-ffmpeg bundles this automatically.

## Validated Results

Classifier accuracy: ~88% across 16,900+ samples
STOI (objective intelligibility): 0.85-0.96 across tested noise types
Average SNR improvement: +5.7dB (measured, hybrid pipeline)
Real-time latency: 17-27ms per 100ms frame (well under budget)
Impulsive-threat detection: 5ms sub-frame crest-factor detector

Government benchmark targets: STOI > 0.85 (met in most tested categories), PESQ > 2.5 (not yet measured, tooling in progress)

We validate scientifically using STOI/SNR ablation testing rather than subjective listening, and are transparent about what is built vs. roadmap - see Honest Project Status below.

## Honest Project Status

Built and verified:
- Random Forest classifier (Layer 1) - 88% accuracy
- BiLSTM Complex-Ratio-Mask enhancer (Layer 2) - trained end-to-end, phase-preserving
- Real-time streaming pipeline - measured, not simulated
- Live public demo with mic input, file upload, and two-device Field Comms Relay
- Objective STOI/SNR validation and ablation testing

In progress / roadmap:
- BiLSTM currently runs as an optional Experimental Deep-Learning Mode - ablation testing showed it needs more training data/epochs to consistently outperform the spectral-subtraction baseline. Retraining (40 to 100 epochs) improved hybrid SNR from -1.1dB to +4.1dB - actively closing this gap.
- PESQ measurement - blocked by a Windows C++ build toolchain requirement; on the roadmap
- Embedded hardware deployment (ESP32-S3) - currently laptop/cloud only
- Two-way continuous live conversation - the Field Comms Relay is currently send/receive (snapshot), not full-duplex streaming

## Team

Team Dhvani - Smart India Hackathon 2026, Problem Statement SIH26052 (DRDO)

## License

Add your chosen license here (e.g. MIT).
