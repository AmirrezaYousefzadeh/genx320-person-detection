"""Counting from the detector's output grid (count_features.py), instead of counting its thresholded boxes:
  soft count   sum over the cells of the most confident anchor's objectness probability, linearly calibrated on
               the training samples (no new network)
  head         a tiny network on the grid (1x1 and 3x3 convolutions to a 12x12 density map whose sum is the
               count), trained with count supervision; a few thousand parameters, negligible next to the detector
for the single pass and the mirrored pass. Scores as count_train.py (per true count 0..5+, exact, MAE, 2->1).
Writes OUT_DIR/head_<pass>.keras and head_<pass>.json (weights as plain lists for a C port).
Usage: count_head.py FEATURES.npz OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from count_train import scores  # noqa: E402


def sig(x):
    return 1 / (1 + np.exp(-x))


def main(feat, out_dir):
    import tensorflow as tf
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(feat)
    count, val = d["count"].astype(np.float32), d["val"]
    for name in ("single", "fused"):
        g = d[name].astype(np.float32)  # N, 12, 12, 5, 6
        soft = sig(g[..., 4]).max(-1).sum((1, 2))
        a, b = np.polyfit(soft[~val], count[~val], 1)
        s = scores(a * soft[val] + b, count[val].astype(int))
        print(f"{name:6s} soft count (x{a:.2f} {b:+.2f}): exact {s['exact']:.3f} MAE {s['mae']:.3f} "
              f"per class {s['per_class']} 2->1 {s['two_as_one']}", flush=True)
        x = g.reshape(len(g), 12, 12, 30)
        mu, sd = x[~val].mean((0, 1, 2)), x[~val].std((0, 1, 2)) + 1e-6  # objectness logits reach -10: standardise
        x = (x - mu) / sd
        L = tf.keras.layers
        inp = L.Input((12, 12, 30))
        h = L.Conv2D(16, 1, activation="relu")(inp)
        h = L.Conv2D(16, 3, padding="same", activation="relu")(h)
        dens = L.Conv2D(1, 1, activation="softplus", bias_initializer=tf.keras.initializers.Constant(-4.0))(h)
        model = tf.keras.Model(inp, L.Lambda(lambda t: tf.reduce_sum(t, [1, 2, 3]), output_shape=())(dens))
        model.compile(tf.keras.optimizers.Adam(3e-3), loss="mae")
        rng = np.random.default_rng(0)
        xt, ct = x[~val], count[~val]
        flip = rng.random(len(xt)) < 0.5  # mirror: columns reversed, x offset negated
        xf = xt.copy().reshape(-1, 12, 12, 5, 6)
        xf[flip] = xf[flip, :, ::-1]
        xf[flip, ..., 0] = -xf[flip, ..., 0] + 2 * mu.reshape(5, 6)[:, 0] / sd.reshape(5, 6)[:, 0]  # negate the raw x offset
        model.fit(xf.reshape(-1, 12, 12, 30), ct, epochs=60, batch_size=128, verbose=0,
                  validation_data=(x[val], count[val]),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        s = scores(model.predict(x[val], verbose=0), count[val].astype(int))
        print(f"{name:6s} head ({model.count_params()} parameters): exact {s['exact']:.3f} MAE {s['mae']:.3f} "
              f"per class {s['per_class']} 2->1 {s['two_as_one']}", flush=True)
        model.save(out / f"head_{name}.keras")
        (out / f"head_{name}.json").write_text(json.dumps(
            {"scores": s, "mean": mu.tolist(), "std": sd.tolist(), "weights": [w.tolist() for w in model.get_weights()]}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


def head_numpy(meta, grids):
    """the head from its JSON (weights, mean, std) on grids (N, 12, 12, 30): the people count per grid"""
    k1, b1, k2, b2, k3, b3 = (np.array(w, np.float32) for w in meta["weights"])
    x = (grids - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    h1 = np.maximum(x @ k1[0, 0] + b1, 0)
    n, g = h1.shape[0], h1.shape[1]
    p = np.pad(h1, ((0, 0), (1, 1), (1, 1), (0, 0)))
    h2 = np.full((n, g, g, k2.shape[3]), b2, np.float32)
    for ky in range(3):
        for kx in range(3):
            h2 += p[:, ky:ky + g, kx:kx + g] @ k2[ky, kx]
    d = np.maximum(h2, 0) @ k3[0, 0, :, 0] + b3[0]
    return np.where(d > 20, d, np.log1p(np.exp(np.minimum(d, 20)))).sum((1, 2))
