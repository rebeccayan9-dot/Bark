"""validate_tflite.py — sanity-check a .tflite model on the test set.

Confirms that the on-disk .tflite (the one being shipped to firmware)
produces the same metrics as the sweep report. Useful to catch:
  - wrong file copied to the firmware path
  - quantization param drift
  - feature normalization mismatches
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "model" / "ds_cnn_a0.25_int8.tflite"
CLASSES = ["bark", "growl", "grunt", "ambient"]

# Expected results from sweep (α=0.25 INT8)
EXPECTED = {
    "macro_f1": 0.627,
    "recall":   [0.44, 0.65, 0.56, 1.00],
}
TOLERANCE = 0.01  # ±1pp slack


def load_test():
    d_tr = np.load(ROOT / "model" / "train.npz")
    X_tr = d_tr["features"][..., np.newaxis].astype(np.float32)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8

    d_te = np.load(ROOT / "model" / "test.npz")
    X_te = (d_te["features"][..., np.newaxis].astype(np.float32) - mean) / std
    return X_te, d_te["labels"].astype(np.int32)


def run(tflite_path):
    X, y = load_test()
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zero = inp.get("quantization", (0.0, 0))

    print(f"Model:  {tflite_path}  ({tflite_path.stat().st_size} bytes)")
    print(f"Input:  dtype={np.dtype(inp['dtype']).name} shape={list(inp['shape'])} "
          f"q=(scale={in_scale:.5g}, zero={in_zero})")
    print(f"Output: dtype={np.dtype(out['dtype']).name} shape={list(out['shape'])}")
    print(f"Running on {len(X)} test samples…\n")

    preds = []
    for x in X:
        if inp["dtype"] == np.int8:
            qx = (x / in_scale + in_zero).round().clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], qx[np.newaxis])
        else:
            interp.set_tensor(inp["index"], x[np.newaxis].astype(np.float32))
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out["index"]))))
    preds = np.array(preds)

    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    cm       = confusion_matrix(y, preds, labels=list(range(4)))
    recall   = [(cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0) for i in range(4)]

    print(classification_report(y, preds, target_names=CLASSES,
                                labels=[0, 1, 2, 3], zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"           {'  '.join(f'{c:>7s}' for c in CLASSES)}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:8s} {'  '.join(f'{cm[i,j]:>7d}' for j in range(4))}")

    print(f"\nMacro F1:        {macro_f1:.4f}   (expected ≈ {EXPECTED['macro_f1']:.3f})")
    print(f"Per-class recall: {[round(r, 3) for r in recall]}")
    print(f"          expected: {EXPECTED['recall']}")

    # Diff check
    f1_diff = abs(macro_f1 - EXPECTED["macro_f1"])
    rec_diff = max(abs(a - b) for a, b in zip(recall, EXPECTED["recall"]))
    if f1_diff <= TOLERANCE and rec_diff <= TOLERANCE:
        print(f"\n✓ Within tolerance (Δ F1={f1_diff:.4f}, max Δ recall={rec_diff:.4f})")
        return 0
    print(f"\n✗ Drift! Δ F1={f1_diff:.4f}, max Δ recall={rec_diff:.4f}  "
          f"(tolerance {TOLERANCE})")
    return 1


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        raise SystemExit(f"[ERR] {path} not found")
    sys.exit(run(path))
