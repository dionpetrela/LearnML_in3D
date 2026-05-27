"""MLP forward pass, backprop, and Adam — all in NumPy."""
from __future__ import annotations
import numpy as np

H1, H2 = 128, 64
N_IN, N_OUT = 9, 1


def forward(x: np.ndarray, w: dict) -> np.ndarray:
    single = x.ndim == 1
    if single:
        x = x[None, :]
    z1 = x @ w["W1"] + w["b1"]
    a1 = np.maximum(0, z1)
    z2 = a1 @ w["W2"] + w["b2"]
    a2 = np.maximum(0, z2)
    z3 = a2 @ w["W3"] + w["b3"]
    y = np.tanh(z3)
    return y[0] if single else y


def forward_all(x: np.ndarray, w: dict) -> dict:
    z1 = x @ w["W1"] + w["b1"];   a1 = np.maximum(0, z1)
    z2 = a1 @ w["W2"] + w["b2"];  a2 = np.maximum(0, z2)
    z3 = a2 @ w["W3"] + w["b3"];  y = np.tanh(z3)
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "y": y}


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())


def backward(x: np.ndarray, y_target: np.ndarray, w: dict, cache: dict) -> dict:
    n = x.shape[0]
    y = cache["y"]
    dy = 2.0 * (y - y_target) / (n * y.shape[1])
    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def init_adam(w: dict) -> dict:
    return {
        "m": {k: np.zeros_like(v) for k, v in w.items()},
        "v": {k: np.zeros_like(v) for k, v in w.items()},
        "t": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    }


def adam_step(w: dict, grads: dict, state: dict, lr: float = 1e-3) -> None:
    state["t"] += 1
    t = state["t"]
    b1, b2, eps = state["beta1"], state["beta2"], state["eps"]
    for k in w:
        g = grads[k]
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * (g * g)
        m_hat = state["m"][k] / (1 - b1 ** t)
        v_hat = state["v"][k] / (1 - b2 ** t)
        w[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def init_weights(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / N_IN), (N_IN, H1)).astype(np.float32),
        "b1": np.zeros(H1, dtype=np.float32),
        "W2": rng.normal(0, np.sqrt(2 / H1), (H1, H2)).astype(np.float32),
        "b2": np.zeros(H2, dtype=np.float32),
        "W3": rng.normal(0, np.sqrt(1 / H2), (H2, N_OUT)).astype(np.float32),
        "b3": np.zeros(N_OUT, dtype=np.float32),
    }


def save(weights: dict, path: str) -> None:
    np.savez(path, **weights)


def load(path: str) -> dict:
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


def numerical_gradient(x: np.ndarray, y_target: np.ndarray, w: dict,
                       key: str, idx: tuple, h: float = 1e-4) -> float:
    w[key][idx] += h
    loss_p = mse_loss(forward(x, w), y_target)
    w[key][idx] -= 2 * h
    loss_m = mse_loss(forward(x, w), y_target)
    w[key][idx] += h
    return (loss_p - loss_m) / (2 * h)


def check_gradients(x: np.ndarray, y: np.ndarray, w: dict, n_samples: int = 5) -> dict:
    cache = forward_all(x, w)
    grads = backward(x, y, w, cache)
    rng = np.random.default_rng(0)
    report = {}
    for key in w:
        max_err = 0.0
        flat_size = w[key].size
        for _ in range(n_samples):
            flat_idx = rng.integers(0, flat_size)
            idx = np.unravel_index(flat_idx, w[key].shape)
            num = numerical_gradient(x, y, w, key, idx)
            ana = grads[key][idx]
            denom = max(1e-12, abs(num) + abs(ana))
            err = abs(num - ana) / denom
            max_err = max(max_err, err)
        report[key] = max_err
    return report