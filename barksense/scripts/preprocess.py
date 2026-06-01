"""
preprocess.py — Prepare BarkSense audio clips for training.

Reads:
  data/bark/dog_{bark,growl,grunt}_{train,test}/*.wav   (pre-split)
  data/ambient_{train,test}/**/*.wav  (optional, preferred form)
  data/ambient/**/*.wav               (optional fallback, split 80/20 by seed)
  data/human/**/*.wav                 (optional, split 80/20 by seed;
                                        silence-aware windowing; folded into
                                        ambient — human is not a model class)

Pipeline:
  - resample to 16 kHz mono
  - slice into non-overlapping 1 s segments (drop trailing < 1 s)
  - peak-normalize each segment to -3 dBFS
  - extract log-mel (40 bins) + delta + delta-delta → 120 × T features
    (n_fft=512, hop=256, n_mels=40, center=False)

Classes: bark, growl, grunt, ambient (4). Human speech is folded into ambient
so that people talking reads as "not a dog event" rather than a false bark.

Train balancing (val/test untouched):
  - ambient downsampled to ~TARGET_PER_CLASS_TRAIN
  - growl/grunt/human augmented up to ~TARGET_PER_CLASS_TRAIN
    (human is augmented under a sentinel label, then remapped to ambient)
    Augment recipe per copy: ±2 semi pitch shift, ±6 dB gain jitter,
    additive white noise at 20 dB SNR. Augmented clips skip dBFS
    renormalization so gain jitter survives into MFCC.

Output:
  model/train.npz, model/val.npz, model/test.npz   (features, labels)

Val: 10% of original train, stratified by class — sampled *before*
balancing so it stays a clean snapshot of the source distribution.
"""

from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

SR           = 16_000
CLIP_SAMPLES = 16_000
N_FFT        = 512
HOP_LENGTH   = 256
N_MELS       = 40
DELTA_WIDTH  = 9                       # frames for delta filter
FEAT_DIM     = N_MELS * 3              # log-mel + delta + delta-delta = 120
TARGET_DBFS  = -3.0
VAL_FRACTION = 0.10
SEED         = 42

TARGET_PER_CLASS_TRAIN = 150           # for ambient (down) and growl/grunt (up)
AUG_PITCH_SEMITONES    = 2.0           # uniform in ±this
AUG_GAIN_DB            = 6.0           # uniform in ±this
AUG_NOISE_SNR_DB       = 20.0          # signal-to-noise ratio of added white noise
TRIM_TOP_DB            = 30.0          # silence threshold for dog clips (librosa.trim)

LABELS = {"bark": 0, "growl": 1, "grunt": 2, "ambient": 3}
INV_LABELS = {v: k for k, v in LABELS.items()}

# Human speech is NOT a model class. To stop people talking from triggering a
# dog event, human voice is folded into ambient: the model learns "human = not a
# dog event". Human clips are carried through collection + balancing under this
# temporary sentinel label (so they get augmented to a healthy count instead of
# being randomly downsampled away with ambient), then remapped to ambient just
# before features are written.
HUMAN_TMP_LABEL = 4

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "model"

DOG_PARENT = DATA_DIR / "bark"


# ── Audio helpers ────────────────────────────────────────────────────────────

def normalize_dbfs(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    return y * (10 ** (target_dbfs / 20.0) / peak)


def slice_clips(y: np.ndarray) -> list[np.ndarray]:
    n = len(y) // CLIP_SAMPLES
    return [y[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES] for i in range(n)]


def trim_and_window(y: np.ndarray) -> list[np.ndarray]:
    """Trim leading/trailing silence, then return 1-sec windows.

    If trimmed audio is shorter than 1 sec, center-pad to 1 sec so the
    vocalization fills the window rather than getting buried in silence.
    """
    y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y_trim) < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - len(y_trim)
        left = pad // 2
        return [np.pad(y_trim, (left, pad - left))]
    return slice_clips(y_trim)


def extract_features(y: np.ndarray) -> np.ndarray:
    """Log-mel (40) stacked with delta + delta-delta → (120, T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, center=False,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    d1 = librosa.feature.delta(logmel, width=DELTA_WIDTH, order=1)
    d2 = librosa.feature.delta(logmel, width=DELTA_WIDTH, order=2)
    return np.concatenate([logmel, d1, d2], axis=0).astype(np.float32)


def augment(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pitch shift + gain jitter + white noise. No dBFS renorm."""
    n_steps = rng.uniform(-AUG_PITCH_SEMITONES, AUG_PITCH_SEMITONES)
    seg = librosa.effects.pitch_shift(seg, sr=SR, n_steps=n_steps)
    seg = seg * (10 ** (rng.uniform(-AUG_GAIN_DB, AUG_GAIN_DB) / 20.0))
    rms = np.sqrt(np.mean(seg ** 2))
    if rms > 1e-9:
        noise_rms = rms / (10 ** (AUG_NOISE_SNR_DB / 20.0))
        seg = seg + rng.standard_normal(len(seg)).astype(seg.dtype) * noise_rms
    return np.clip(seg, -1.0, 1.0).astype(np.float32)


# ── Segment collection (raw, pre-feature) ────────────────────────────────────

def load_segments(d: Path, label: int, trim: bool = False) -> list[tuple[np.ndarray, int]]:
    """Return normalized 1-second segments paired with `label`.

    trim=True (dog classes): strip silence, center-pad if shorter than 1s.
    trim=False (ambient): straight sliding-window slicing.
    """
    out: list[tuple[np.ndarray, int]] = []
    if not d.exists():
        return out
    windower = trim_and_window if trim else slice_clips
    for wav in sorted(d.rglob("*.wav")):
        try:
            y, _ = librosa.load(str(wav), sr=SR, mono=True)
        except Exception as exc:
            print(f"  [WARN] {wav.name}: {exc}")
            continue
        for seg in windower(y):
            out.append((normalize_dbfs(seg, TARGET_DBFS), label))
    return out


def load_segments_voiced(d: Path, label: int) -> list[tuple[np.ndarray, int]]:
    """Speech segments: split each file on silence, window only the voiced parts.

    Recordings of people talking are mostly gaps; straight slicing would label
    those silent windows as the class and teach the model that silence == speech
    (colliding with ambient). So we split on silence (top_db=TRIM_TOP_DB), then:
      - voiced region >= 1 s  → non-overlapping 1 s windows (drop trailing < 1 s)
      - 0.5 s <= region < 1 s → center-pad to 1 s
      - region < 0.5 s        → dropped (too short to carry a vocalization)
    """
    out: list[tuple[np.ndarray, int]] = []
    if not d.exists():
        return out
    half = CLIP_SAMPLES // 2
    for wav in sorted(d.rglob("*.wav")):
        try:
            y, _ = librosa.load(str(wav), sr=SR, mono=True)
        except Exception as exc:
            print(f"  [WARN] {wav.name}: {exc}")
            continue
        for s, e in librosa.effects.split(y, top_db=TRIM_TOP_DB):
            seg = y[s:e]
            if len(seg) >= CLIP_SAMPLES:
                windows = slice_clips(seg)
            elif len(seg) >= half:
                pad = CLIP_SAMPLES - len(seg)
                left = pad // 2
                windows = [np.pad(seg, (left, pad - left))]
            else:
                continue
            for w in windows:
                out.append((normalize_dbfs(w, TARGET_DBFS), label))
    return out


def collect_split(split: str) -> list[tuple[np.ndarray, int]]:
    segs: list[tuple[np.ndarray, int]] = []

    for name in ("bark", "growl", "grunt"):
        d = DOG_PARENT / f"dog_{name}_{split}"
        if not d.exists():
            print(f"  [WARN] {d} missing — skipping")
            continue
        segs.extend(load_segments(d, LABELS[name], trim=True))

    amb_split = DATA_DIR / f"ambient_{split}"
    amb_flat  = DATA_DIR / "ambient"
    if amb_split.exists() and any(amb_split.rglob("*.wav")):
        segs.extend(load_segments(amb_split, LABELS["ambient"], trim=False))
    elif amb_flat.exists() and any(amb_flat.rglob("*.wav")):
        all_amb = load_segments(amb_flat, LABELS["ambient"], trim=False)
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(all_amb))
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        keep = idx[:cut] if split == "train" else idx[cut:]
        segs.extend([all_amb[i] for i in keep])
    else:
        print(f"  [INFO] no ambient data for {split} — skipping ambient")

    # Human speech: folded into ambient (see HUMAN_TMP_LABEL). Flat dir, split
    # 80/20 by seed like ambient, but loaded with silence-aware windowing.
    human_flat = DATA_DIR / "human"
    if human_flat.exists() and any(human_flat.rglob("*.wav")):
        all_hum = load_segments_voiced(human_flat, HUMAN_TMP_LABEL)
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(all_hum))
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        keep = idx[:cut] if split == "train" else idx[cut:]
        segs.extend([all_hum[i] for i in keep])
    else:
        print(f"  [INFO] no human data for {split} — skipping human")

    return segs


# ── Balancing ────────────────────────────────────────────────────────────────

def balance_train(train_segs: list[tuple[np.ndarray, int]],
                  rng: np.random.Generator) -> list[tuple[np.ndarray, int]]:
    """Downsample over-represented classes, augment under-represented ones."""
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for seg, lbl in train_segs:
        by_class[lbl].append(seg)

    # Downsample ambient
    amb_lbl = LABELS["ambient"]
    if amb_lbl in by_class and len(by_class[amb_lbl]) > TARGET_PER_CLASS_TRAIN:
        picks = rng.choice(len(by_class[amb_lbl]),
                           size=TARGET_PER_CLASS_TRAIN, replace=False)
        by_class[amb_lbl] = [by_class[amb_lbl][i] for i in picks]

    # Augment bark, growl, grunt & human up toward the target (all under the
    # ambient count). bark is the smallest class (~104) and was previously left
    # un-augmented, starving its decision boundary vs growl/grunt — augment it
    # too. Human is still under its sentinel label here; it gets remapped to
    # ambient after balancing, so augmenting it now keeps human voice
    # well-represented in the ambient class rather than diluted by the downsample.
    for lbl in (LABELS["bark"], LABELS["growl"], LABELS["grunt"], HUMAN_TMP_LABEL):
        if lbl not in by_class or not by_class[lbl]:
            continue
        originals = list(by_class[lbl])
        needed = TARGET_PER_CLASS_TRAIN - len(originals)
        for _ in range(max(0, needed)):
            src = originals[rng.integers(0, len(originals))]
            by_class[lbl].append(augment(src, rng))

    out: list[tuple[np.ndarray, int]] = []
    for lbl, segs in by_class.items():
        out.extend((s, lbl) for s in segs)
    rng.shuffle(out)
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def report(name: str, labels: np.ndarray) -> None:
    counts = Counter(labels.tolist())
    parts = [f"{INV_LABELS[c]}={counts.get(c, 0)}" for c in sorted(LABELS.values())]
    print(f"  {name:5s} n={len(labels):4d}  " + "  ".join(parts))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[1/4] Loading train segments…")
    train_raw = collect_split("train")
    print("[2/4] Loading test segments…")
    test_raw  = collect_split("test")

    if not train_raw or not test_raw:
        raise SystemExit("[ERR] empty train or test set — check data/ layout")

    print("[3/4] Carving val from raw train (stratified, 10%)…")
    y_full = np.array([l for _, l in train_raw])
    train_idx, val_idx = train_test_split(
        np.arange(len(train_raw)),
        test_size=VAL_FRACTION,
        stratify=y_full,
        random_state=SEED,
    )
    train_orig = [train_raw[i] for i in train_idx]
    val_segs   = [train_raw[i] for i in val_idx]

    print("[4/4] Balancing train (downsample ambient, augment growl/grunt)…")
    train_bal = balance_train(train_orig, rng)

    print("\nExtracting MFCCs…")
    X_train = np.stack([extract_features(s) for s, _ in train_bal])
    y_train = np.array([l for _, l in train_bal], dtype=np.int32)
    X_val   = np.stack([extract_features(s) for s, _ in val_segs])
    y_val   = np.array([l for _, l in val_segs], dtype=np.int32)
    X_test  = np.stack([extract_features(s) for s, _ in test_raw])
    y_test  = np.array([l for _, l in test_raw], dtype=np.int32)

    # Fold human → ambient across every split: human is not a model class, it
    # just must read as "not a dog event".
    for y in (y_train, y_val, y_test):
        y[y == HUMAN_TMP_LABEL] = LABELS["ambient"]

    np.savez_compressed(OUT_DIR / "train.npz", features=X_train, labels=y_train)
    np.savez_compressed(OUT_DIR / "val.npz",   features=X_val,   labels=y_val)
    np.savez_compressed(OUT_DIR / "test.npz",  features=X_test,  labels=y_test)

    print("\nClass balance per split:")
    report("train", y_train)
    report("val",   y_val)
    report("test",  y_test)

    print(f"\nSaved → {OUT_DIR}/{{train,val,test}}.npz")
    print(f"Per-clip shape: ({FEAT_DIM} feats × {X_train.shape[2]} frames)")


if __name__ == "__main__":
    main()
