"""Full compression sweep: α ∈ {1.0, 0.5, 0.25} × {float32, INT8} = 6 configs.

For each α: train DS-CNN, save Keras model, convert to TFLite (f32 + INT8),
evaluate on test set, record metrics. Plot size-vs-F1 trade-off.
"""
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
ALPHAS    = [1.0, 0.5, 0.25]
EPOCHS    = 50
BATCH     = 32
LR        = 1e-3
SEED      = 42
CLASSES   = ["bark", "growl", "grunt", "ambient"]
N_CLASSES = 4


# ── Data ─────────────────────────────────────────────────────────────────────

def load_split(name):
    d = np.load(MODEL_DIR / f"{name}.npz")
    return d["features"][..., np.newaxis].astype(np.float32), d["labels"].astype(np.int32)

X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")

mean = X_train.mean(axis=(0, 2), keepdims=True)
std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std
np.savez(MODEL_DIR / "norm_stats.npz", mean=mean, std=std)

INPUT_SHAPE = X_train.shape[1:]
print(f"Input shape: {INPUT_SHAPE}  |  train n={len(y_train)}  val n={len(y_val)}  test n={len(y_test)}")


# ── Model ────────────────────────────────────────────────────────────────────

def make_ds_cnn(input_shape, n_classes, alpha):
    def ch(n): return max(1, int(n * alpha))
    inp = tf.keras.Input(shape=input_shape, name="feat")
    x = tf.keras.layers.Conv2D(ch(64), (3, 3), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
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


def train(alpha):
    tf.random.set_seed(SEED); np.random.seed(SEED)
    model = make_ds_cnn(INPUT_SHAPE, N_CLASSES, alpha)
    cls = np.unique(y_train)
    w   = compute_class_weight("balanced", classes=cls, y=y_train)
    cw  = {int(c): float(v) for c, v in zip(cls, w)}
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, class_weight=cw,
              callbacks=cbs, verbose=2)
    return model


# ── TFLite conversion (TF 2.16 + Keras 3 needs SavedModel detour) ────────────

def to_tflite(model, int8=False):
    # TF 2.16 + Py 3.12 break model.export(); go via concrete function instead.
    spec = tf.TensorSpec(shape=(None,) + INPUT_SHAPE, dtype=tf.float32)

    @tf.function(input_signature=[spec])
    def serving_fn(x):
        return model(x, training=False)

    concrete = serving_fn.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        def rep_gen():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1].astype(np.float32)]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def maybe_load_or_train(alpha):
    cached = MODEL_DIR / f"sweep_a{alpha}.keras"
    if cached.exists():
        print(f"  [cache] loading {cached.name}")
        return tf.keras.models.load_model(str(cached), compile=False)
    return train(alpha)


def eval_tflite(tflite_bytes, X, y):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zero = inp_d.get("quantization", (0.0, 0))
    preds = []
    for x in X:
        if inp_d["dtype"] == np.int8:
            qx = (x / in_scale + in_zero).round().clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp_d["index"], qx[np.newaxis])
        else:
            interp.set_tensor(inp_d["index"], x[np.newaxis].astype(np.float32))
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out_d["index"]))))
    return np.array(preds)


def arena_kb(tflite_bytes):
    """Rough RAM estimate: sum of two largest tensor allocations."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    sizes = []
    for d in interp.get_tensor_details():
        sh = d["shape"]
        if len(sh) == 0 or 0 in sh:
            continue
        sizes.append(int(np.prod(sh) * np.dtype(d["dtype"]).itemsize))
    sizes.sort(reverse=True)
    return sum(sizes[:2]) / 1024


# ── Sweep ────────────────────────────────────────────────────────────────────

results = []
for alpha in ALPHAS:
    print("\n" + "=" * 70)
    print(f"=== α={alpha} ===")
    print("=" * 70)
    model = maybe_load_or_train(alpha)
    cached = MODEL_DIR / f"sweep_a{alpha}.keras"
    if not cached.exists():
        model.save(str(cached))

    for quant in ["float32", "int8"]:
        print(f"\n  Converting α={alpha} → {quant}…")
        tfl = to_tflite(model, int8=(quant == "int8"))
        (MODEL_DIR / f"sweep_a{alpha}_{quant}.tflite").write_bytes(tfl)

        y_pred  = eval_tflite(tfl, X_test, y_test)
        cm      = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))
        recall  = [(cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0) for i in range(N_CLASSES)]
        macroF1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        size_kb = len(tfl) / 1024
        arena   = arena_kb(tfl)
        results.append(dict(
            alpha=alpha, quant=quant, params=int(model.count_params()),
            size_kb=size_kb, arena_kb=arena,
            macro_f1=float(macroF1), per_class_recall=recall,
            cm=cm.tolist(),
        ))
        print(f"    size={size_kb:5.1f}KB arena≈{arena:5.1f}KB  "
              f"F1={macroF1:.3f}  recall={['%.2f' % r for r in recall]}")

with open(MODEL_DIR / "sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)


# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
for r in results:
    color  = "#1f77b4" if r["quant"] == "float32" else "#ff7f0e"
    marker = "o" if r["quant"] == "float32" else "s"
    ax.scatter(r["size_kb"], r["macro_f1"], s=140, c=color, marker=marker, edgecolor="black", zorder=3)
    ax.annotate(f"α={r['alpha']}", (r["size_kb"], r["macro_f1"]),
                xytext=(7, 7), textcoords="offset points", fontsize=10)

ax.set_xlabel("Model size (KB)")
ax.set_ylabel("Macro F1 (test)")
ax.set_title("BarkSense size ↔ accuracy trade-off  (6 configs)")
ax.set_xscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", mec="black", mfc="#1f77b4", markersize=11, label="float32"),
    Line2D([0],[0], marker="s", color="w", mec="black", mfc="#ff7f0e", markersize=11, label="INT8"),
], loc="best")
plt.tight_layout()
plt.savefig(MODEL_DIR / "sweep_tradeoff.png", dpi=120)


# ── Summary table ────────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print(f"{'α':>5s} {'quant':>8s} {'params':>8s} {'size(KB)':>9s} {'arena(KB)':>10s} "
      f"{'macroF1':>8s}  {'recall  bark/growl/grunt/ambient':<35s}")
print("=" * 90)
for r in results:
    rec = "/".join(f"{x:.2f}" for x in r["per_class_recall"])
    print(f"{r['alpha']:>5.2f} {r['quant']:>8s} {r['params']:>8d} {r['size_kb']:>9.1f} {r['arena_kb']:>10.1f} "
          f"{r['macro_f1']:>8.3f}  {rec}")
print("\nResults JSON → model/sweep_results.json")
print("Plot         → model/sweep_tradeoff.png")
