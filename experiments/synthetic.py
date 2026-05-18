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
import matplotlib.ticker as mticker
import os
import json
import argparse

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Persisted raw numeric results so the figure can be restyled later
# (python synthetic.py --replot) without re-running the experiment.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "synthetic_data.json")


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
def compute_results():
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

    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw results to {CACHE}")
    return results


def run_experiment(replot=False):
    if replot:
        if not os.path.exists(CACHE):
            raise FileNotFoundError(
                f"{CACHE} not found; run without --replot once to generate it."
            )
        with open(CACHE) as f:
            results = json.load(f)
        print(f"Loaded cached results from {CACHE} (no re-run).")
    else:
        results = compute_results()

    # ── Plotting ──
    lambdas_arr = [r["lambda"] for r in results]
    accs = [r["accuracy"] for r in results]
    hsics = [r["hsic"] for r in results]
    gdps = [r["gdp"] for r in results]

    # figsize width = 1.0 * W_text, where W_text = 370.38374pt / 72.27pt/in = 5.124"
    W_text = 5.124
    fig_w = W_text          # 5.124"
    # Use taller height so titles don't overlap each other; ~1.9" per panel at this width
    fig_h = 1.9
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    # (a) GDP vs lambda
    axes[0].plot(lambdas_arr, gdps, "o-", color="tab:red",
                 linewidth=1.0, markersize=3)
    axes[0].set_xlabel(r"$\lambda$", fontsize=8)
    axes[0].set_ylabel("GDP", fontsize=8)
    axes[0].set_title("(a) GDP vs. $\\lambda$", fontsize=8)
    axes[0].set_xscale("symlog", linthresh=0.05)
    axes[0].tick_params(axis="both", labelsize=7)
    # Thin out the symlog x-ticks so the decade labels do not overlap
    axes[0].xaxis.set_major_locator(
        mticker.SymmetricalLogLocator(base=10.0, linthresh=0.05, subs=(1.0,))
    )
    axes[0].grid(True, alpha=0.3)

    # (b) Accuracy vs lambda
    axes[1].plot(lambdas_arr, accs, "s-", color="tab:blue",
                 linewidth=1.0, markersize=3)
    axes[1].set_xlabel(r"$\lambda$", fontsize=8)
    axes[1].set_ylabel("Accuracy", fontsize=8)
    axes[1].set_title("(b) Accuracy vs. $\\lambda$", fontsize=8)
    axes[1].set_xscale("symlog", linthresh=0.05)
    axes[1].tick_params(axis="both", labelsize=7)
    axes[1].xaxis.set_major_locator(
        mticker.SymmetricalLogLocator(base=10.0, linthresh=0.05, subs=(1.0,))
    )
    axes[1].grid(True, alpha=0.3)

    # (c) GDP vs HSIC scatter, consistent with Theorem 3.4 of the main
    # paper. Several lambda values map to nearly coincident (HSIC, GDP)
    # points, so per-point text labels overlap badly; instead colour the
    # markers by lambda and attach a slim colourbar.
    order_c = np.argsort(lambdas_arr)
    hsics_o = np.array(hsics)[order_c]
    gdps_o = np.array(gdps)[order_c]
    color_idx = np.arange(len(lambdas_arr))  # rank in lambda (0 -> max)
    sc = axes[2].scatter(
        hsics_o, gdps_o, s=18, c=color_idx, cmap="viridis",
        edgecolors="black", linewidths=0.5, zorder=5,
    )
    axes[2].set_xlabel("HSIC", fontsize=8)
    axes[2].set_ylabel("GDP", fontsize=8)
    axes[2].set_title("(c) GDP vs. HSIC (Thm. 3.4)", fontsize=8)
    axes[2].tick_params(axis="both", labelsize=7)
    axes[2].grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    lam_sorted = np.array(lambdas_arr)[order_c]
    cbar.set_ticks(color_idx)
    cbar.set_ticklabels([f"{v:g}" for v in lam_sorted])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r"$\lambda$", fontsize=8)

    fig.tight_layout()
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot", action="store_true",
        help="Re-draw the figure from cached results without re-running.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_experiment(replot=args.replot)
