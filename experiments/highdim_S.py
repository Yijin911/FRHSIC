"""
T21: High-dimensional sensitive attribute experiment.

Demonstrates that FREM's per-epoch training time grows with dim(S) (curse of
dimensionality from the kernel-smoothed conditional weighting), while FRHSIC's
HSIC-based loss is essentially insensitive to dim(S).

Self-contained: stripped-down local FREM and FRHSIC training loops that handle
multi-d S directly, so we can sweep dim(S) without monkey-patching real_data.py.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from real_data import load_adult
from utils import DEVICE, Z_DIM, KERNEL_SIGMA, Encoder, Predictor, gaussian_kernel

SEED = 42
N_SUB = 2000
BATCH = 256
EPOCHS = 30
LAM = 10.0
GAMMA_S = 0.5     # FREM kernel-smoothing bandwidth on S (fixed)
SIGMA_S = 1.0     # FRHSIC HSIC bandwidth on S (fixed, no median heuristic)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hsic_md(Z, S, sigma_z=KERNEL_SIGMA, sigma_s=SIGMA_S):
    """Biased HSIC for multi-d S (S is already n x d)."""
    n = Z.shape[0]
    K = gaussian_kernel(Z, Z, sigma_z)
    L = gaussian_kernel(S, S, sigma_s)
    H = torch.eye(n, device=Z.device) - 1.0 / n
    return (H @ K @ H * (H @ L @ H)).sum() / (n ** 2)


def train_frhsic_md(X, S, Y, epochs=EPOCHS):
    """FRHSIC with multi-d S, fixed sigma_s."""
    ds = TensorDataset(X, S, Y)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    enc = Encoder(X.shape[1], z_dim=Z_DIM).to(DEVICE)
    pred = Predictor(z_dim=Z_DIM).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    last_loss = 0.0
    for _ in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = enc(X_b)
            loss = bce(pred(Z_b).squeeze(), Y_b) + LAM * hsic_md(Z_b, S_b)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = float(loss.item())
    return last_loss


def train_frem_md(X, S, Y, epochs=EPOCHS):
    """FREM with multi-d S: kernel weights on S use d-dimensional Gaussian."""
    ds = TensorDataset(X, S, Y)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    enc = Encoder(X.shape[1], z_dim=Z_DIM).to(DEVICE)
    pred = Predictor(z_dim=Z_DIM).to(DEVICE)
    opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    last_loss = 0.0
    for _ in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = enc(X_b)
            loss_pred = bce(pred(Z_b).squeeze(), Y_b)
            n = Z_b.shape[0]
            K = gaussian_kernel(Z_b, Z_b, KERNEL_SIGMA)
            W = gaussian_kernel(S_b, S_b, GAMMA_S)
            W_loo = W.clone(); W_loo.fill_diagonal_(0)
            W_loo = W_loo / (W_loo.sum(dim=1, keepdim=True) + 1e-10)
            uniform = torch.ones(n, device=DEVICE) / n
            oneK1 = uniform @ K @ uniform
            eipm = torch.tensor(0.0, device=DEVICE)
            n_anchors = min(n, 32)
            anchor_idx = torch.randperm(n, device=DEVICE)[:n_anchors]
            for i in anchor_idx:
                w_i = W_loo[i]
                mmd2 = w_i @ K @ w_i - 2 * (w_i @ K @ uniform) + oneK1
                eipm = eipm + torch.sqrt(torch.clamp(mmd2, min=1e-10))
            eipm = eipm / n_anchors
            loss = loss_pred + LAM * eipm
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss = float(loss.item())
    return last_loss


def build_S(df_X_full, d, rng):
    """Construct d-dim S from continuous Adult cols. Min-max-scaled to [0,1]."""
    # Adult numeric cols: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
    # We pick: age, capital-gain, capital-loss, hours-per-week, plus synthetic 5th feature.
    cols = [0, 3, 4, 5]  # indices into the numeric block (age, cap-gain, cap-loss, hrs)
    if d <= 4:
        S_raw = df_X_full[:, cols[:d]]
    else:
        # 5th component: synthetic Gaussian noise to extend dim without re-introducing
        # a strongly-correlated continuous Adult column.
        synth = rng.normal(size=(df_X_full.shape[0], d - 4)).astype(np.float32)
        S_raw = np.hstack([df_X_full[:, cols], synth])
    S_scaled = MinMaxScaler().fit_transform(S_raw).astype(np.float32)
    return S_scaled


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    X_np, _, Y_np, _, _ = load_adult()  # X already includes age + cap-gain etc as numeric
    # Subsample
    idx = rng.choice(len(X_np), N_SUB, replace=False)
    X_np, Y_np = X_np[idx], Y_np[idx]
    # Numeric block (first columns) holds age, fnlwgt, education-num, capital-gain,
    # capital-loss, hours-per-week. We build S from this block.
    X_full_for_S = X_np[:, :6].copy()
    # Standardize features for training (X passed to encoder)
    X_scaled = MinMaxScaler().fit_transform(X_np).astype(np.float32)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.float32)

    rows = []
    for d in [1, 2, 3, 4, 5]:
        S_np = build_S(X_full_for_S, d, rng)
        S_t = torch.tensor(S_np, dtype=torch.float32)
        for method, fn in [("FRHSIC", train_frhsic_md), ("FREM", train_frem_md)]:
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            final_loss = fn(X_t, S_t, Y_t)
            t = time.perf_counter() - t0
            rows.append({"d": d, "method": method, "time_s": t,
                         "time_per_epoch": t / EPOCHS, "final_loss": final_loss})
            print(f"d={d} {method:8s} total={t:7.2f}s "
                  f"per-epoch={t/EPOCHS:6.3f}s final_loss={final_loss:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color, marker in [("FRHSIC", "tab:red", "o"), ("FREM", "tab:green", "D")]:
        xs = [r["d"] for r in rows if r["method"] == method]
        ys = [r["time_per_epoch"] for r in rows if r["method"] == method]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=8, label=method)
    ax.set_xlabel(r"$\dim(S)$", fontsize=12)
    ax.set_ylabel("Per-epoch training time (s)", fontsize=12)
    ax.set_title(r"Adult ($n = 2000$, $\lambda = 10$, fixed bandwidths)", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_pdf = os.path.join(RESULTS_DIR, "highdim_S.pdf")
    out_png = os.path.join(RESULTS_DIR, "highdim_S.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_pdf}")

    # Summary
    print("\n=== Summary (per-epoch time, seconds) ===")
    print(f"{'d':>3} | {'FRHSIC':>10} | {'FREM':>10} | {'FREM/FRHSIC':>12}")
    for d in [1, 2, 3, 4, 5]:
        h = next(r for r in rows if r["d"] == d and r["method"] == "FRHSIC")["time_per_epoch"]
        f = next(r for r in rows if r["d"] == d and r["method"] == "FREM")["time_per_epoch"]
        print(f"{d:>3} | {h:>10.3f} | {f:>10.3f} | {f/h:>12.2f}x")


if __name__ == "__main__":
    main()
