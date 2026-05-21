"""simulate_mcu_features.py — host-side verification of the MCU pipeline.

Reproduces extract_features() from firmware/src/main.cpp in pure NumPy
(no librosa) using exactly the same building blocks (mel filterbank,
savgol kernels, log10 with ref=max, savgol edge handling). Compares the
result against the librosa-derived `model/expected_feature.bin` so we can
flag algorithm bugs before flashing hardware.

This is the same algorithm the MCU runs, just in float64 instead of float32 —
any mismatch here means a logic bug, not a precision issue.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SR, CLIP        = 16_000, 16_000
N_FFT, HOP      = 512, 256
N_MELS          = 40
N_BINS          = N_FFT // 2 + 1
N_FRAMES        = 1 + (CLIP - N_FFT) // HOP
DELTA_HALF      = 4
INPUT_SCALE     = 0.07230392098426819
INPUT_ZP        = -4

# ── Load the same building blocks the firmware uses ──────────────────────────
import librosa
from scipy.signal import savgol_coeffs

mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, htk=False).astype(np.float64)
# REVERSED — matches what the header gives the MCU
d1_k   = savgol_coeffs(2*DELTA_HALF+1, polyorder=1, deriv=1)[::-1].astype(np.float64)
d2_k   = savgol_coeffs(2*DELTA_HALF+1, polyorder=2, deriv=2)[::-1].astype(np.float64)

norm    = np.load(ROOT / "model" / "norm_stats.npz")
mean    = np.squeeze(norm["mean"]).astype(np.float64)
std     = np.squeeze(norm["std"]).astype(np.float64)

# Hann window — same formula the firmware uses
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1))


def mcu_extract(samples_i16: np.ndarray) -> np.ndarray:
    """Pure-numpy reimplementation of extract_features()."""
    x = samples_i16.astype(np.float64) / 32768.0

    # (1) Per-frame STFT → mel energy
    raw_mel = np.zeros((N_MELS, N_FRAMES), dtype=np.float64)
    for t in range(N_FRAMES):
        frame  = x[t*HOP : t*HOP + N_FFT] * hann
        spec   = np.fft.fft(frame, n=N_FFT)
        power  = np.abs(spec[:N_BINS]) ** 2
        raw_mel[:, t] = mel_fb @ power

    # (2) librosa power_to_db(ref=np.max, amin=1e-10, top_db=80)
    amin, top_db = 1e-10, 80.0
    max_e = max(amin, raw_mel.max())
    ref_db = 10.0 * np.log10(max_e)
    log_mel = 10.0 * np.log10(np.maximum(raw_mel, amin)) - ref_db
    log_mel = np.maximum(log_mel, log_mel.max() - top_db)

    # (3) + (4) Savgol deltas along time. polyorder == deriv ⇒ boundary
    # frames inherit nearest interior value.
    def savgol_t(kern, src):
        dst = np.zeros_like(src)
        for t in range(DELTA_HALF, N_FRAMES - DELTA_HALF):
            dst[:, t] = np.sum(kern[None, :] * src[:, t-DELTA_HALF : t+DELTA_HALF+1], axis=1)
        for t in range(DELTA_HALF):
            dst[:, t] = dst[:, DELTA_HALF]
        for t in range(N_FRAMES - DELTA_HALF, N_FRAMES):
            dst[:, t] = dst[:, N_FRAMES - DELTA_HALF - 1]
        return dst

    delta1 = savgol_t(d1_k, log_mel)
    delta2 = savgol_t(d2_k, log_mel)

    # (5)-(8) Stack → z-score → quantize
    feat = np.concatenate([log_mel, delta1, delta2], axis=0)
    feat = (feat - mean[:, None]) / std[:, None]
    q = np.clip(np.rint(feat / INPUT_SCALE + INPUT_ZP), -128, 127).astype(np.int8)
    return q  # (120, 61)


def main():
    audio = np.frombuffer((ROOT / "model" / "test_audio.bin").read_bytes(), dtype=np.int16)
    expected = np.frombuffer((ROOT / "model" / "expected_feature.bin").read_bytes(), dtype=np.int8)
    expected = expected.reshape(120, N_FRAMES)

    mcu = mcu_extract(audio)

    diff = mcu.astype(int) - expected.astype(int)
    abs_diff = np.abs(diff)
    print(f"MCU-simulated vs librosa-reference (both flat int8):")
    print(f"  mean |Δ| = {abs_diff.mean():.4f}")
    print(f"  max  |Δ| = {abs_diff.max()}")
    print(f"  |Δ| > 1  : {int((abs_diff > 1).sum())} / {abs_diff.size}")
    print(f"  |Δ| > 2  : {int((abs_diff > 2).sum())} / {abs_diff.size}")

    if abs_diff.max() <= 2:
        print(f"\n✓ PASS — pipeline algorithm matches librosa within ±2")
        return 0
    print(f"\n✗ FAIL — algorithm drifts from librosa beyond tolerance")
    # Where do the diffs concentrate?
    by_channel = abs_diff.max(axis=1)
    bad = np.where(by_channel > 2)[0]
    print(f"  channels with max |Δ| > 2: {bad.tolist()}")
    return 1


if __name__ == "__main__":
    import sys; sys.exit(main())
