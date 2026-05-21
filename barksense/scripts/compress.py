"""
compress.py — Sweep DS-CNN width multipliers × quantization flavours.

Trains DS-CNN at α ∈ {1.0, 0.75, 0.5}, converts each to float32 and INT8
TFLite, evaluates on the test set, prints a comparison table, and saves
model/tradeoff.png.

Run preprocess.py first to generate model/features.npz.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
FEAT_PATH = MODEL_DIR / "features.npz"

CLASSES     = ["bark", "human", "ambient"]
MULTIPLIERS = [1.0, 0.75, 0.5]
EPOCHS      = 30
BATCH       = 32

# ── DS-CNN (mirrors train.ipynb) ──────────────────────────────────────────────

def make_ds_cnn(input_shape, n_classes=3, alpha=1.0):
    def ch(n): return max(1, int(n * alpha))

    inp = tf.keras.Input(shape=input_shape)
    x   = tf.keras.layers.Conv2D(ch(64), (3, 3), padding="same", use_bias=False)(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Activation("relu")(x)
    x   = tf.keras.layers.Dropout(0.2)(x)

    for filters in [ch(64), ch(64), ch(128), ch(128)]:
        x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, (1, 1), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name=f"ds_cnn_a{alpha}")


# ── TFLite conversion ─────────────────────────────────────────────────────────

def to_float32(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def to_int8(model, rep_data_fn):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_fn
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluate_tflite(tflite_bytes, X_test, y_test):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    preds = []
    for x in X_test:
        xin = x[np.newaxis].astype(np.float32)
        if inp_d["dtype"] == np.int8:
            sc, zp = inp_d["quantization"]
            xin = np.clip(np.round(xin / sc) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp_d["index"], xin)
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])
        if out_d["dtype"] == np.int8:
            sc, zp = out_d["quantization"]
            raw = (raw.astype(np.float32) - zp) * sc
        preds.append(int(np.argmax(raw)))

    preds = np.array(preds)
    acc   = float(np.mean(preds == y_test))
    recalls = []
    for c in range(len(CLASSES)):
        mask = y_test == c
        recalls.append(float(np.mean(preds[mask] == c)) if mask.any() else float("nan"))
    return acc, recalls


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not FEAT_PATH.exists():
        print(f"[ERR] {FEAT_PATH} not found — run preprocess.py first")
        sys.exit(1)

    d       = np.load(str(FEAT_PATH), allow_pickle=True)
    X_train = d["X_train"][..., np.newaxis].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_val   = d["X_val"][...,   np.newaxis].astype(np.float32)
    y_val   = d["y_val"].astype(np.int32)
    X_test  = d["X_test"][...,  np.newaxis].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    input_shape = X_train.shape[1:]
    print(f"[INFO] input_shape={input_shape}  train={len(X_train)}  test={len(X_test)}")

    results = []

    for alpha in MULTIPLIERS:
        print(f"\n{'─'*60}\n[α={alpha}] training …")
        model = make_ds_cnn(input_shape, len(CLASSES), alpha)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH, verbose=1)

        # float32
        f32 = to_float32(model)
        (MODEL_DIR / f"ds_cnn_a{alpha}_f32.tflite").write_bytes(f32)
        f32_acc, f32_rec = evaluate_tflite(f32, X_test, y_test)

        # INT8
        def rep():
            for i in range(min(200, len(X_train))):
                yield [X_train[i:i+1]]

        i8 = to_int8(model, rep)
        (MODEL_DIR / f"ds_cnn_a{alpha}_int8.tflite").write_bytes(i8)
        i8_acc, i8_rec = evaluate_tflite(i8, X_test, y_test)

        for quant, acc, rec, raw in [
            ("float32", f32_acc, f32_rec, f32),
            ("int8",    i8_acc,  i8_rec,  i8),
        ]:
            size_kb = len(raw) / 1024
            ram_kb  = size_kb * 2   # rough estimate: activations ≈ model size
            row = dict(alpha=alpha, quant=quant,
                       acc=round(acc, 4),
                       recall_bark=round(rec[0], 4),
                       recall_human=round(rec[1], 4),
                       recall_ambient=round(rec[2], 4),
                       size_kb=round(size_kb, 1),
                       ram_kb_est=round(ram_kb, 1))
            results.append(row)
            print(f"  [{quant:7s}] acc={acc:.3f}  "
                  f"bark_R={rec[0]:.3f}  "
                  f"size={size_kb:.1f}KB  RAM~{ram_kb:.1f}KB")

    # Summary table
    print(f"\n{'─'*72}")
    print(f"{'α':>6}  {'quant':>7}  {'acc':>6}  {'bark_R':>6}  "
          f"{'size_KB':>8}  {'RAM_KB':>8}")
    for r in results:
        print(f"{r['alpha']:>6}  {r['quant']:>7}  {r['acc']:>6.3f}  "
              f"{r['recall_bark']:>6.3f}  {r['size_kb']:>8.1f}  {r['ram_kb_est']:>8.1f}")

    (MODEL_DIR / "sweep_results.json").write_text(json.dumps(results, indent=2))

    # Trade-off plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for quant, marker, color in [("float32", "o", "steelblue"), ("int8", "s", "tomato")]:
        rows = [r for r in results if r["quant"] == quant]
        axes[0].plot([r["size_kb"]    for r in rows],
                     [r["acc"]         for r in rows],
                     marker=marker, label=quant, color=color)
        axes[1].plot([r["ram_kb_est"] for r in rows],
                     [r["acc"]         for r in rows],
                     marker=marker, label=quant, color=color)
        for r in rows:
            axes[0].annotate(f"α={r['alpha']}", (r["size_kb"], r["acc"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=8)
            axes[1].annotate(f"α={r['alpha']}", (r["ram_kb_est"], r["acc"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=8)

    for ax, xlabel, title in [
        (axes[0], "Model size (KB)", "Accuracy vs Model Size"),
        (axes[1], "Estimated RAM (KB)", "Accuracy vs RAM"),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel("Test accuracy")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = MODEL_DIR / "tradeoff.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\n[INFO] Plot → {plot_path}")
    print(f"[INFO] Results → {MODEL_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
