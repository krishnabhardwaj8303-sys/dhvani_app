"""
train_enhancement_model.py (v2 -- stabilized)
Trains the Layer-2 BiLSTM Complex-Ratio-Mask (CRM) enhancement network.
Fixes from v1: tighter mask range (prevents energy amplification/artifacts),
gradient clipping, LR scheduling, more epochs, saves the BEST checkpoint
(lowest loss) instead of just the last one.
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import librosa

SR = 16000
SEGMENT_SEC = 2.0
SEGMENT_LEN = int(SR * SEGMENT_SEC)
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
MASK_SCALE = 1.0  # was 1.8 -- too aggressive, was amplifying noise energy

CLEAN_FOLDERS = ["data/normal_speech", "data/shouting_warning", "data/whispered_speech", "data/radio_communication"]
NOISE_FOLDERS = [
    "data/engine_noise", "data/machinery_industrial", "data/tank_heavy_vehicle",
    "data/wind_noise", "data/crowd_chatter", "data/footsteps_movement",
    "data/rain_water", "data/aircraft_helicopter", "data/radio_static",
    "data/gunfire_blast", "data/explosion", "data/artillery_fire",
    "data/clapping_impact", "data/thunder_storm", "data/alarm_siren",
]


def list_audio_files(folders):
    files = []
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
            files.extend(glob.glob(os.path.join(folder, ext)))
    return files


CLEAN_FILES = list_audio_files(CLEAN_FOLDERS)
NOISE_FILES = list_audio_files(NOISE_FOLDERS)
print(f"Found {len(CLEAN_FILES)} clean speech files")
print(f"Found {len(NOISE_FILES)} noise files")
if len(CLEAN_FILES) == 0 or len(NOISE_FILES) == 0:
    raise SystemExit("Missing clean speech or noise files -- check data/ folders exist.")


class ComplexMaskNet(nn.Module):
    def __init__(self, freq_bins, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=freq_bins * 2, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True, bidirectional=True)
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
        torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = estimate - s_target
    ratio = (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return -(10 * torch.log10(ratio + eps)).mean()


def load_random_segment(file_path, target_len):
    try:
        y, sr = librosa.load(file_path, sr=SR)
    except Exception:
        return np.zeros(target_len, dtype=np.float32)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="wrap") if len(y) > 0 else np.zeros(target_len, dtype=np.float32)
    else:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start + target_len]
    return y.astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    signal_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    required_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(required_noise_power / noise_power)
    noisy = clean + noise * scale
    max_val = np.max(np.abs(noisy)) + 1e-8
    if max_val > 1.0:
        noisy = noisy / max_val
        clean = clean / max_val
    return noisy.astype(np.float32), clean.astype(np.float32)


class EnhancementDataset(torch.utils.data.Dataset):
    def __init__(self, steps_per_epoch=300):
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        clean_file = random.choice(CLEAN_FILES)
        noise_file = random.choice(NOISE_FILES)
        clean = load_random_segment(clean_file, SEGMENT_LEN)
        noise = load_random_segment(noise_file, SEGMENT_LEN)
        snr_db = random.uniform(-5, 15)
        noisy, clean = mix_at_snr(clean, noise, snr_db)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def process_batch(model, noisy, clean, window):
    noisy_stft = torch.stft(noisy, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, return_complex=True)
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

    return torch.istft(est_stft, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, length=clean.shape[-1])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  (mask scale: {MASK_SCALE})")

    freq_bins = N_FFT // 2 + 1
    model = ComplexMaskNet(freq_bins=freq_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    window = torch.hann_window(WIN_LENGTH).to(device)

    dataset = EnhancementDataset(steps_per_epoch=300)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    EPOCHS = 100
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            estimated = process_batch(model, noisy, clean, window)
            loss = si_snr_loss(estimated, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.3f} | Avg SI-SNR: {-avg_loss:.2f} dB | LR: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "pinaka_enhancement_model.pt")
            print(f"  -> New best model saved (SI-SNR {-avg_loss:.2f} dB)")

    print(f"\nTraining complete. Best SI-SNR achieved: {-best_loss:.2f} dB")
    print("Model saved as pinaka_enhancement_model.pt (best checkpoint)")


if __name__ == "__main__":
    main()
