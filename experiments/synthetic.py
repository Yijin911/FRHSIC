"""
Synthetic experiment for FRHSIC paper.
Plots GDP vs. HSIC consistent with the upper bound in Theorem 3.4 of the main paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────
def generate_synthetic_data(n=5000, seed=SEED):
    """
    X = [S + eps1, eps2], Y = Bernoulli(sigmoid(X_1)), S ~ Uniform(0,1).
    """
    rng = np.random.RandomState(seed)
    S = rng.uniform(0, 1, size=n)
    eps1 = rng.normal(0, 0.3, size=n)
    eps2 = rng.normal(0, 1.0, size=n)
    X = np.column_stack([S + eps1, eps2])
    prob = 1.0 / (1.0 + np.exp(-X[:, 0]))
    Y = rng.binomial(1, prob)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(S, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )


# ──────────────────────────────────────────────
# HSIC estimator
# ──────────────────────────────────────────────
def gaussian_kernel(X, Y, sigma):
    """Gaussian (RBF) kernel matrix."""
    dist = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-dist / (2 * sigma ** 2))


def hsic_biased(Z, S, sigma_z=1.0, sigma_s=1.0):
    """Biased HSIC estimator: (1/n^2) tr(KHLH)."""
    n = Z.shape[0]
    K = gaussian_kernel(Z, Z, sigma_z)
    L = gaussian_kernel(S.unsqueeze(1), S.unsqueeze(1), sigma_s)
    H = torch.eye(n, device=Z.device) - 1.0 / n
    HKH = H @ K @ H
    return (HKH * (H @ L @ H)).sum() / (n ** 2)


# ──────────────────────────────────────────────
# GDP estimator (kernel-smoothed)
# ──────────────────────────────────────────────
def estimate_gdp(Z, S, f_pred, bandwidth=0.1, n_grid=50):
    """
    Estimate GDP = E_S | E[f(Z)|S] - E[f(Z)] |
    via kernel smoothing on a grid over S.
    """
    with torch.no_grad():
        fZ = f_pred(Z).squeeze()
        global_mean = fZ.mean()
        s_grid = torch.linspace(S.min(), S.max(), n_grid, device=Z.device)
        diffs = []
        for s in s_grid:
            weights = torch.exp(-((S - s) ** 2) / (2 * bandwidth ** 2))
            weights = weights / (weights.sum() + 1e-10)
            conditional_mean = (weights * fZ).sum()
            diffs.append(torch.abs(conditional_mean - global_mean))
        return torch.stack(diffs).mean().item()


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, z_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, z_dim=8):
        super().__init__()
        self.net = nn.Linear(z_dim, 1)

    def forward(self, z):
        return self.net(z)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(X, S, Y, lam, z_dim=8, epochs=100, batch_size=256, lr=1e-3):
    dataset = TensorDataset(X, S, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(input_dim=X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )
    bce = nn.BCEWithLogitsLoss()

    # Median heuristic for kernel bandwidths
    with torch.no_grad():
        Z_init = encoder(X[:1000].to(DEVICE))
        sigma_z = torch.median(torch.cdist(Z_init, Z_init)).item() + 1e-5
        S_sample = S[:1000].unsqueeze(1).to(DEVICE)
        sigma_s = torch.median(torch.cdist(S_sample, S_sample)).item() + 1e-5

    for epoch in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = encoder(X_b)
            logits = predictor(Z_b).squeeze()

            loss_pred = bce(logits, Y_b)
            loss_hsic = hsic_biased(Z_b, S_b, sigma_z, sigma_s)
            loss = loss_pred + lam * loss_hsic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return encoder, predictor


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(encoder, predictor, X, S, Y):
    with torch.no_grad():
        X_d, S_d, Y_d = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)
        Z = encoder(X_d)
        logits = predictor(Z).squeeze()
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == Y_d).float().mean().item()
        hsic_val = hsic_biased(Z, S_d).item()
        gdp_val = estimate_gdp(Z, S_d, predictor)
    return acc, hsic_val, gdp_val


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────
def run_experiment():
    print("Generating synthetic data...")
    X, S, Y = generate_synthetic_data(n=5000)

    # Train/test split
    n_train = 4000
    X_train, X_test = X[:n_train], X[n_train:]
    S_train, S_test = S[:n_train], S[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    lambdas = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    results = []

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}...")
        encoder, predictor = train_model(X_train, S_train, Y_train, lam=lam)
        acc, hsic_val, gdp_val = evaluate(encoder, predictor, X_test, S_test, Y_test)
        results.append({"lambda": lam, "accuracy": acc, "hsic": hsic_val, "gdp": gdp_val})
        print(f"  Accuracy: {acc:.4f}, HSIC: {hsic_val:.6f}, GDP: {gdp_val:.6f}")

    # ── Plotting ──
    lambdas_arr = [r["lambda"] for r in results]
    accs = [r["accuracy"] for r in results]
    hsics = [r["hsic"] for r in results]
    gdps = [r["gdp"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) GDP vs lambda
    axes[0].plot(lambdas_arr, gdps, "o-", color="tab:red", linewidth=2)
    axes[0].set_xlabel(r"$\lambda$", fontsize=13)
    axes[0].set_ylabel("GDP", fontsize=13)
    axes[0].set_title("(a) GDP vs. Regularization Strength", fontsize=12)
    axes[0].set_xscale("symlog", linthresh=0.05)
    axes[0].grid(True, alpha=0.3)

    # (b) Accuracy vs lambda
    axes[1].plot(lambdas_arr, accs, "s-", color="tab:blue", linewidth=2)
    axes[1].set_xlabel(r"$\lambda$", fontsize=13)
    axes[1].set_ylabel("Accuracy", fontsize=13)
    axes[1].set_title("(b) Accuracy vs. Regularization Strength", fontsize=12)
    axes[1].set_xscale("symlog", linthresh=0.05)
    axes[1].grid(True, alpha=0.3)

    # (c) GDP vs HSIC scatter, consistent with Theorem 3.4 of the main paper
    axes[2].scatter(hsics, gdps, s=80, c="tab:green", edgecolors="black", zorder=5)
    for i, lam in enumerate(lambdas_arr):
        axes[2].annotate(
            f"$\\lambda$={lam}",
            (hsics[i], gdps[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    axes[2].set_xlabel("HSIC", fontsize=13)
    axes[2].set_ylabel("GDP", fontsize=13)
    axes[2].set_title("(c) GDP vs. HSIC (Theorem 3.4)", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "synthetic_results.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "synthetic_results.png"), dpi=150, bbox_inches="tight")
    print(f"\nFigures saved to {RESULTS_DIR}/synthetic_results.{{pdf,png}}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'lambda':>8} {'Accuracy':>10} {'HSIC':>12} {'GDP':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['lambda']:>8.1f} {r['accuracy']:>10.4f} {r['hsic']:>12.6f} {r['gdp']:>10.6f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_experiment()
