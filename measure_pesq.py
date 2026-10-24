"""
measure_pesq.py (debug version)
"""
import os
import traceback
import numpy as np
import librosa
from pesq import pesq

from dhvani_core import DhvaniPipeline, SR

print("Loading pipeline...")
pipeline = DhvaniPipeline(
    classifier_path="dhvani_model_multiclass.pkl",
    enhancer_path="dhvani_enhancement_model.pt",
)
print("Pipeline loaded OK")


def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    sp = np.mean(clean ** 2) + 1e-10
    npow = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt((sp / (10 ** (snr_db / 10))) / npow)
    noisy = clean + noise * scale
    mx = np.max(np.abs(noisy)) + 1e-8
    if mx > 1.0:
        noisy, clean = noisy / mx, clean / mx
    return noisy.astype(np.float32), clean.astype(np.float32)


def find_first_file(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3"))]
    print(f"  Found {len(files)} files in {folder}")
    return os.path.join(folder, files[0])


test_cases = [
    ("C:/Users/krish/data/normal_speech", "C:/Users/krish/data/engine_noise", "STATIONARY (engine)"),
    ("C:/Users/krish/data/normal_speech", "C:/Users/krish/data/gunfire_blast", "IMPULSIVE (gunfire)"),
    ("C:/Users/krish/data/normal_speech", "C:/Users/krish/data/wind_noise", "NON-STATIONARY (wind)"),
]

print(f"\nNumber of test cases: {len(test_cases)}")
pesq_scores = []

for speech_folder, noise_folder, label in test_cases:
    print(f"\n--- Processing: {label} ---")
    try:
        speech_path = find_first_file(speech_folder)
        noise_path = find_first_file(noise_folder)
        print(f"  speech_path={speech_path}")
        print(f"  noise_path={noise_path}")

        clean, _ = librosa.load(speech_path, sr=SR, duration=5.0)
        clean = clean / (np.max(np.abs(clean)) + 1e-8) * 0.7
        noise, _ = librosa.load(noise_path, sr=SR, duration=5.0)
        noisy, clean = mix_at_snr(clean, noise, snr_db=5.0)
        print(f"  Mixed audio OK, len={len(noisy)}")

        audio_int16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
        result = pipeline.process(SR, audio_int16, use_lms=False, use_bilstm=False)
        cleaned = result["cleaned"]
        print(f"  Pipeline processed OK, cleaned len={len(cleaned)}")

        min_len = min(len(clean), len(cleaned))
        clean_seg = clean[:min_len].astype(np.float32)
        cleaned_seg = cleaned[:min_len].astype(np.float32)

        score_wb = pesq(SR, clean_seg, cleaned_seg, 'wb')
        pesq_scores.append(score_wb)
        status = "PASS" if score_wb > 2.5 else "NOT YET"
        print(f"  RESULT [{label}]  PESQ (wideband) = {score_wb:.2f}   target >2.5: {status}")

    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        traceback.print_exc()

if pesq_scores:
    avg = np.mean(pesq_scores)
    print("\n" + "=" * 65)
    print(f"AVERAGE PESQ across all tested noise types: {avg:.2f}")
    print(f"Government target (>2.5): {'PASS' if avg > 2.5 else 'NOT YET'}")
    print("=" * 65)
else:
    print("\nNo PESQ scores computed -- see errors above.")
