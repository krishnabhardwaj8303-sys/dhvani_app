"""
ablation_test.py
Isolates which stage helps vs hurts: NOISY vs SPECTRAL-SUBTRACTION-ONLY
vs SPECTRAL+BiLSTM (current hybrid). Same speech/noise/SNR as
validate_pipeline.py for a fair comparison.
"""
import os
import numpy as np
import librosa
import noisereduce as nr
from pystoi import stoi

from dhvani_core import DhvaniPipeline as PinakaPipeline, SR

pipeline = PinakaPipeline(
    classifier_path="dhvani_model_multiclass.pkl",
    enhancer_path="dhvani_enhancement_model.pt",
)


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


def calc_snr(clean, test):
    m = min(len(clean), len(test))
    clean, test = clean[:m], test[:m]
    noise = test - clean
    return 10 * np.log10((np.mean(clean**2)) / (np.mean(noise**2) + 1e-10))


def find_first_file(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3"))]
    return os.path.join(folder, files[0])


speech_path = find_first_file("data/normal_speech")
noise_path = find_first_file("data/engine_noise")

clean, _ = librosa.load(speech_path, sr=SR, duration=5.0)
clean = clean / (np.max(np.abs(clean)) + 1e-8) * 0.7
noise, _ = librosa.load(noise_path, sr=SR, duration=5.0)
noisy, clean = mix_at_snr(clean, noise, snr_db=5.0)

spectral_only = nr.reduce_noise(y=noisy, sr=SR, stationary=True, prop_decrease=0.6)

audio_int16 = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)
hybrid = pipeline.process(SR, audio_int16, use_lms=False)["cleaned"]

print("=" * 60)
print(f"ABLATION TEST  (speech: {os.path.basename(speech_path)}, noise: {os.path.basename(noise_path)})")
print("=" * 60)
for name, sig in [("NOISY (raw input)", noisy),
                   ("SPECTRAL SUBTRACTION ONLY", spectral_only),
                   ("HYBRID (spectral + BiLSTM)", hybrid)]:
    m = min(len(clean), len(sig))
    s = stoi(clean[:m], sig[:m], SR, extended=False)
    snr = calc_snr(clean[:m], sig[:m])
    print(f"{name:30s}  STOI={s:.3f}   SNR={snr:+.1f}dB")
print("=" * 60)
