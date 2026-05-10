"""
Ablation studies for FRHSIC.
- Kernel choice for k_Z and k_S
- Representation dimensionality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from utils import (
    DEVICE, Encoder, Predictor,
    hsic_biased, estimate_gdp, median_heuristic,
)

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_data(n=5000):
    rng = np.random.RandomState(SEED)
    S = rng.uniform(0, 1, n).astype(np.float32)
    X = np.column_stack([S + rng.normal(0, 0.3, n), rng.normal(0, 1, n)]).astype(np.float32)
    Y = rng.binomial(1, 1.0 / (1.0 + np.exp(-X[:, 0]))).astype(np.float32)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(S, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )


def train_and_eval(X, S, Y, lam=5.0, z_dim=8, kernel_z="gaussian", kernel_s="gaussian",
                   epochs=80, batch_size=256, lr=1e-3):
    n_train = int(0.8 * len(X))
    X_tr, X_te = X[:n_train], X[n_train:]
    S_tr, S_te = S[:n_train], S[n_train:]
    Y_tr, Y_te = Y[:n_train], Y[n_train:]

    dataset = TensorDataset(X_tr, S_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    sigma_z = median_heuristic(encoder(X_tr[:500].to(DEVICE)))
    sigma_s = median_heuristic(S_tr[:500].unsqueeze(1).to(DEVICE))

    for _ in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = encoder(X_b)
            loss = bce(predictor(Z_b).squeeze(), Y_b) + lam * hsic_biased(
                Z_b, S_b, sigma_z, sigma_s, kernel_z=kernel_z, kernel_s=kernel_s
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    with torch.no_grad():
        X_d, S_d, Y_d = X_te.to(DEVICE), S_te.to(DEVICE), Y_te.to(DEVICE)
        Z = encoder(X_d)
        preds = (torch.sigmoid(predictor(Z).squeeze()) > 0.5).float()
        acc = (preds == Y_d).float().mean().item()
        gdp = estimate_gdp(Z, S_d, predictor)

    return acc, gdp


def run_kernel_ablation():
    print("=" * 60)
    print("Ablation: Kernel Choice")
    print("=" * 60)

    X, S, Y = generate_data()
    kernels_z = ["gaussian", "laplacian", "imq"]
    kernels_s = ["gaussian", "laplacian", "imq"]

    print(f"\n{'k_Z':>12} {'k_S':>12} {'Accuracy':>10} {'GDP':>10}")
    print("-" * 50)

    for kz in kernels_z:
        for ks in kernels_s:
            acc, gdp = train_and_eval(X, S, Y, kernel_z=kz, kernel_s=ks)
            print(f"{kz:>12} {ks:>12} {acc:>10.4f} {gdp:>10.4f}")


def run_dim_ablation():
    print("\n" + "=" * 60)
    print("Ablation: Representation Dimensionality")
    print("=" * 60)

    X, S, Y = generate_data()
    dims = [2, 8, 32, 64]

    print(f"\n{'z_dim':>8} {'Accuracy':>10} {'GDP':>10}")
    print("-" * 35)

    for z_dim in dims:
        acc, gdp = train_and_eval(X, S, Y, z_dim=z_dim)
        print(f"{z_dim:>8} {acc:>10.4f} {gdp:>10.4f}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_kernel_ablation()
    run_dim_ablation()
