"""Step 2 — Inspect data, train, and save weights.

Run:    python 02_train.py --data data_v1.npz --tag v1

Improved version:
  - Completed my_backward().
  - Uses a safer time-block validation split by default so validation is
    less likely to be fooled by adjacent frames from the same drive.
  - Adds --split random if your instructor specifically wants the old split.
  - Saves nav_<tag>.npz and diagnostic figures.
"""
from __future__ import annotations
import argparse
import numpy as np

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# Backpropagation for: 12 -> H1 -> H2 -> 2, ReLU/ReLU/tanh, MSE loss
# =========================================================================
def my_backward(x, y_target, w, cache):
    """Return gradients dW1, db1, ..., dW3, db3 for one mini-batch.

    x:        (N, 12) normalized input features
    y_target: (N, 2) target actions: throttle, steering
    w:        dict of model weights
    cache:    output from nn_mod.forward_all(x, w)
    """
    n = x.shape[0]
    y = cache["y"]

    # Loss: mean((y - y_target)^2)
    dy = 2.0 * (y - y_target) / (n * y.shape[1])

    # Output layer: y = tanh(z3)
    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)

    # Hidden layer 2: a2 = ReLU(z2)
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    # Hidden layer 1: a1 = ReLU(z1)
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def numerical_gradient64(x: np.ndarray, y_target: np.ndarray, w: dict,
                         key: str, idx: tuple, h: float = 1e-5) -> float:
    """Accurate finite-difference gradient for the gradient check.

    The starter nn_mod.numerical_gradient mutates float32 weights. With small
    h values, float32 roundoff can make a correct backprop look wrong. This
    helper copies the tiny check problem to float64, so the check tests the
    math rather than float32 precision.
    """
    w64 = {k: v.astype(np.float64).copy() for k, v in w.items()}
    x64 = x.astype(np.float64)
    y64 = y_target.astype(np.float64)

    w64[key][idx] += h
    loss_p = nn_mod.mse_loss(nn_mod.forward(x64, w64), y64)
    w64[key][idx] -= 2 * h
    loss_m = nn_mod.mse_loss(nn_mod.forward(x64, w64), y64)
    return (loss_p - loss_m) / (2 * h)


def gradient_check():
    rng = np.random.default_rng(0)
    w = nn_mod.init_weights(seed=0)
    # Use 9 features to match nn.py input
    x = rng.normal(size=(8, 9)).astype(np.float32)
    y = rng.uniform(-1, 1, size=(8, 1)).astype(np.float32)  # 1 output (steering)
    cache = nn_mod.forward_all(x, w)
    grads = my_backward(x, y, w, cache)

    print("\ngradient check (max relative error per parameter):")
    for key in w:
        max_err = 0.0
        flat = w[key].size
        for _ in range(5):
            idx = np.unravel_index(rng.integers(0, flat), w[key].shape)
            num = numerical_gradient64(x, y, w, key, idx)
            ana = grads[key][idx]
            denom = max(1e-12, abs(num) + abs(ana))
            max_err = max(max_err, abs(num - ana) / denom)
        flag = "OK" if max_err < 1e-4 else "BUG"
        print(f"  {key}: {max_err:.2e}   {flag}")
        assert max_err < 1e-4, (
            f"backward() gradient for {key} disagrees with numerical_gradient. "
            f"Fix it before training."
        )


def inspect_dataset(states_raw, actions, tag: str):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")
    viz.plot_action_histograms(actions, out=f"fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{tag}.png")


def make_split(X, Y, val_frac: float, seed: int, split: str):
    """Create train/validation split.

    block split is recommended for driving data because neighboring frames are
    very similar. random split is kept for comparison with the starter code.
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    n_val = max(1, int(N * val_frac))

    if split == "random":
        perm = rng.permutation(N)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        return X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    if split == "block":
        return X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]

    raise ValueError(f"Unknown split: {split}")


def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1,
          seed=0, split="block"):
    rng = np.random.default_rng(seed)
    Xtr, Ytr, Xva, Yva = make_split(X, Y, val_frac=val_frac, seed=seed, split=split)
    print(f"\nsplit     : {split}")
    print(f"train size: {len(Xtr)}")
    print(f"val size  : {len(Xva)}")

    w = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)
    train_losses, val_losses = [], []
    best_val = float("inf")
    best = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, n_b = 0.0, 0

        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i + batch_size], Ys[i:i + batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb)
            n_b += 1
            grads = my_backward(xb, yb, w, cache)
            nn_mod.adam_step(w, grads, state, lr=lr)

        val_loss = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        train_losses.append(ep_loss / max(1, n_b))
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best = {k: w[k].copy() for k in w}

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  "
                  f"val={val_loss:.4f}  best={best_val:.4f}")

    return best, train_losses, val_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1.npz",
                    help="Dataset file from 01_collect.py")
    ap.add_argument("--tag", default="v1",
                    help="Output suffix (nav_<tag>.npz, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--split", choices=["block", "random"], default="block",
                    help="Validation split. block is recommended for sequential driving data.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Training RNG seed")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    inspect_dataset(states_raw, actions, tag=args.tag)

    X = normalize_states(states_raw)
    # Select only the 9 features that match nn.py input
    # Keep columns: speed, heading_error, checkpoint_distance,
    # ray_0_front, ray_1_+45, ray_2_+90, ray_6_-90, ray_7_-45, ground_friction
    keep_cols = [0, 1, 2, 3, 4, 5, 9, 10, 11]
    X = X[:, keep_cols]
    print(f"Selected {X.shape[1]} features for network input")
    
    Y = actions.astype(np.float32)
    # Use only steering (column 1), throttle will be hardcoded
    Y = Y[:, 1:2]
    print(f"\nX range : [{X.min():+.2f}, {X.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")
    print(f"Training on steering only, shape: {Y.shape}")

    gradient_check()

    weights, tr_losses, va_losses = train(
        X, Y,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
        split=args.split,
    )

    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")
    nn_mod.save(weights, f"nav_{args.tag}.npz")
    print(f"Saved nav_{args.tag}.npz")


if __name__ == "__main__":
    main()