"""
dhvani_core.py
Unified DHVANI pipeline: Layer1 classifier + 5ms spike detector +
hybrid Stage A (spectral subtraction) + Stage B (optional BiLSTM
Complex-Ratio-Mask) enhancer + confidence gating + gain smoothing.
"""
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
import os

SR = 16000
FRAME_LEN = 1600
SUBFRAME_LEN = 80
CREST_FACTOR_THRESHOLD = 6.0
CONFIDENCE_THRESHOLD = 0.45
MAX_GAIN_STEP = 0.15
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
MASK_SCALE = 1.0
CLASS_NAMES = {-1: "SAFE-MODE", 0: "STATIONARY", 1: "NON-STATIONARY", 2: "IMPULSIVE", 3: "SPEECH"}


class ComplexMaskNet(nn.Module):
    def __init__(self, freq_bins, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=freq_bins * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, freq_bins * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        mask = torch.tanh(self.fc(out)) * MASK_SCALE
        fb = mask.shape[-1] // 2
        return mask[..., :fb], mask[..., fb:]


def si_snr_loss(estimate, target, eps=1e-8):
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    s_target = (torch.sum(estimate * target, dim=-1, keepdim=True) * target) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    )
    e_noise = estimate - s_target
    ratio = (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return -(10 * torch.log10(ratio + eps)).mean()


def _stft(wave_t, window):
    return torch.stft(wave_t, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, return_complex=True)


def _istft(spec, window, length):
    return torch.istft(spec, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, length=length)


def enhance_with_crm(model, window, noisy_wave, device="cpu"):
    wave_t = torch.from_numpy(noisy_wave.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        noisy_stft = _stft(wave_t, window)
        real = noisy_stft.real.transpose(1, 2)
        imag = noisy_stft.imag.transpose(1, 2)
        norm = real.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        feat = torch.cat([real / norm, imag / norm], dim=-1)

        mask_real, mask_imag = model(feat)
        mask_real = mask_real.transpose(1, 2)
        mask_imag = mask_imag.transpose(1, 2)

        est_real = mask_real * noisy_stft.real - mask_imag * noisy_stft.imag
        est_imag = mask_real * noisy_stft.imag + mask_imag * noisy_stft.real
        est_stft = torch.complex(est_real, est_imag)

        enhanced = _istft(est_stft, window, length=wave_t.shape[-1])
    return enhanced.squeeze(0).cpu().numpy()


def extract_features(chunk, sr=SR):
    n_fft = min(2048, len(chunk))
    mfcc = librosa.feature.mfcc(y=chunk.astype(np.float32), sr=sr, n_mfcc=13, n_fft=n_fft)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    width = max(width, 3)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc, width=width)
    delta_mean = np.mean(delta, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean])


def detect_fast_impulsive_spike(frame, subframe_len=SUBFRAME_LEN, threshold=CREST_FACTOR_THRESHOLD):
    n_sub = len(frame) // subframe_len
    for j in range(n_sub):
        s = j * subframe_len
        sub = frame[s:s + subframe_len]
        peak = np.max(np.abs(sub))
        avg = np.mean(np.abs(sub)) + 1e-8
        if (peak / avg) > threshold and peak > 0.15:
            return True
    return False


def get_base_suppression(class_id, confidence):
    if class_id == 3:
        return 1.0
    elif class_id == 0:
        return max(1.0 - confidence, 0.05)
    elif class_id == 1:
        return max(1.0 - (confidence * 0.8), 0.15)
    elif class_id == 2:
        return max(1.0 - confidence, 0.02)
    return 1.0


def lms_residual_filter(primary, reference, mu=0.01, filter_len=32):
    n = len(primary)
    w = np.zeros(filter_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    ref_padded = np.concatenate([np.zeros(filter_len, dtype=np.float32), reference])
    for i in range(n):
        x = ref_padded[i:i + filter_len][::-1]
        y = np.dot(w, x)
        e = primary[i] - y
        out[i] = e
        w += 2 * mu * e * x
    return out


class DhvaniPipeline:
    def __init__(self, classifier_path="dhvani_model_multiclass.pkl",
                 enhancer_path="dhvani_enhancement_model.pt", device="cpu"):
        self.device = device
        self.classifier = joblib.load(classifier_path)
        self.freq_bins = N_FFT // 2 + 1
        self.window = torch.hann_window(WIN_LENGTH)
        self.enhancer = None
        self.enhancer_loaded = False
        if enhancer_path and os.path.exists(enhancer_path):
            try:
                self.enhancer = ComplexMaskNet(freq_bins=self.freq_bins).to(device)
                self.enhancer.load_state_dict(torch.load(enhancer_path, map_location=device))
                self.enhancer.eval()
                self.enhancer_loaded = True
            except Exception as e:
                print(f"[DHVANI] Could not load BiLSTM enhancer, falling back: {e}")

    def classify_frames(self, audio_float, sr=SR, frame_len=FRAME_LEN):
        num_frames = len(audio_float) // frame_len
        class_log, conf_log, spike_log = [], [], []
        class_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
        prev_gain = 1.0
        gains = np.ones(num_frames, dtype=np.float32)

        for i in range(num_frames):
            frame = audio_float[i * frame_len:(i + 1) * frame_len]
            has_spike = detect_fast_impulsive_spike(frame)
            feat = [extract_features(frame, sr)]
            proba = self.classifier.predict_proba(feat)[0]
            pred = int(np.argmax(proba))
            confidence = float(np.max(proba))

            if has_spike:
                pred_class, confidence = 2, max(confidence, 0.85)
            else:
                pred_class = pred

            if confidence < CONFIDENCE_THRESHOLD and not has_spike:
                pred_class, raw_gain = -1, 1.0
            else:
                raw_gain = get_base_suppression(pred_class, confidence)

            diff = raw_gain - prev_gain
            gain = prev_gain + MAX_GAIN_STEP * np.sign(diff) if abs(diff) > MAX_GAIN_STEP else raw_gain
            prev_gain = gain

            gains[i] = gain
            class_log.append(pred_class)
            conf_log.append(confidence)
            spike_log.append(has_spike)
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        return class_log, conf_log, spike_log, class_counts, gains, num_frames

    def enhance(self, audio_float, use_bilstm=False):
        try:
            import noisereduce as nr
            pre_cleaned = nr.reduce_noise(y=audio_float, sr=SR, stationary=True, prop_decrease=0.7)
        except Exception:
            pre_cleaned = audio_float

        if not use_bilstm or not self.enhancer_loaded:
            return pre_cleaned, False

        cleaned = enhance_with_crm(self.enhancer, self.window, pre_cleaned, device=self.device)
        min_len = min(len(cleaned), len(pre_cleaned))
        return cleaned[:min_len], True

    def process(self, sr, audio_int16, use_lms=False, use_bilstm=False):
        audio_float = audio_int16.astype(np.float32) / 32767.0
        class_log, conf_log, spike_log, class_counts, gains, num_frames = self.classify_frames(audio_float, sr)
        enhanced, used_bilstm = self.enhance(audio_float, use_bilstm=use_bilstm)

        gained = np.copy(enhanced)
        for i in range(num_frames):
            s, e = i * FRAME_LEN, (i + 1) * FRAME_LEN
            gained[s:e] = gained[s:e] * gains[i]

        if use_lms:
            gained = lms_residual_filter(gained, audio_float[:len(gained)])

        return {
            "cleaned": gained,
            "class_log": class_log,
            "conf_log": conf_log,
            "spike_log": spike_log,
            "class_counts": class_counts,
            "num_frames": num_frames,
            "used_bilstm": used_bilstm,
        }

