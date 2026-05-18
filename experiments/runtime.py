"""
Computational efficiency comparison: FRHSIC vs FREM vs Reg-GDP.
Measures per-epoch training time using mini-batches (realistic setting).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import json
import argparse
import matplotlib.pyplot as plt

from utils import (
    DEVICE, Encoder, Predictor,
    hsic_biased, reg_gdp_loss, gaussian_kernel, median_heuristic,
)

SEED = 42
BATCH_SIZE = 256
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Persisted raw wall-clock measurements so the figure can be restyled
# later (python runtime.py --replot) without re-timing the experiment.
# Wall-clock numbers are NOT reproducible across machines, so the cache
# is the single source of truth once measured.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "runtime_data.json")


def generate_data(n):
    rng = np.random.RandomState(SEED)
    S = rng.uniform(0, 1, n).astype(np.float32)
    X = np.column_stack([S + rng.normal(0, 0.3, n), rng.normal(0, 1, n)]).astype(np.float32)
    Y = rng.binomial(1, 1.0 / (1.0 + np.exp(-X[:, 0]))).astype(np.float32)
    return (
        torch.tensor(X),
        torch.tensor(S),
        torch.tensor(Y),
    )


def train_one_epoch_hsic(loader, encoder, predictor, opt, bce, sigma_z, sigma_s, lam=1.0):
    for X_b, S_b, Y_b in loader:
        X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
        Z_b = encoder(X_b)
        loss = bce(predictor(Z_b).squeeze(), Y_b) + lam * hsic_biased(Z_b, S_b, sigma_z, sigma_s)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_one_epoch_frem(loader, encoder, predictor, opt, bce, sigma_z, sigma_s, lam=1.0):
    for X_b, S_b, Y_b in loader:
        X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
        Z_b = encoder(X_b)
        loss_pred = bce(predictor(Z_b).squeeze(), Y_b)

        n = Z_b.shape[0]
        K = gaussian_kernel(Z_b, Z_b, sigma_z)
        W = gaussian_kernel(S_b.unsqueeze(1), S_b.unsqueeze(1), sigma_s)
        W_loo = W.clone()
        W_loo.fill_diagonal_(0)
        W_loo = W_loo / (W_loo.sum(dim=1, keepdim=True) + 1e-10)

        uniform = torch.ones(n, device=DEVICE) / n
        eipm = torch.tensor(0.0, device=DEVICE)
        for i in range(n):  # Full EIPM: loop over all samples in batch
            w_i = W_loo[i]
            mmd2 = w_i @ K @ w_i - 2 * (w_i @ K @ uniform) + uniform @ K @ uniform
            eipm = eipm + torch.sqrt(torch.clamp(mmd2, min=1e-10))
        eipm = eipm / n

        loss = loss_pred + lam * eipm
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_one_epoch_reg_gdp(loader, encoder, predictor, opt, bce, lam=1.0):
    for X_b, S_b, Y_b in loader:
        X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
        Z_b = encoder(X_b)
        loss = bce(predictor(Z_b).squeeze(), Y_b) + lam * reg_gdp_loss(Z_b, S_b, predictor)
        opt.zero_grad()
        loss.backward()
        opt.step()


def compute_results():
    sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
    n_repeats = 3
    z_dim = 8

    results = {method: {n: [] for n in sample_sizes} for method in ["FRHSIC", "FREM", "Reg-GDP"]}

    for n in sample_sizes:
        print(f"\nn = {n} (batch_size = {BATCH_SIZE})")
        X, S, Y = generate_data(n)
        dataset = TensorDataset(X, S, Y)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for rep in range(n_repeats):
            # FRHSIC
            enc = Encoder(2, z_dim=z_dim).to(DEVICE)
            pred = Predictor(z_dim=z_dim).to(DEVICE)
            opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
            bce = nn.BCEWithLogitsLoss()
            with torch.no_grad():
                sigma_z = median_heuristic(enc(X[:min(500, n)].to(DEVICE)))
                sigma_s = median_heuristic(S[:min(500, n)].unsqueeze(1).to(DEVICE))

            t0 = time.perf_counter()
            train_one_epoch_hsic(loader, enc, pred, opt, bce, sigma_z, sigma_s)
            results["FRHSIC"][n].append(time.perf_counter() - t0)

            # FREM
            enc = Encoder(2, z_dim=z_dim).to(DEVICE)
            pred = Predictor(z_dim=z_dim).to(DEVICE)
            opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
            bce = nn.BCEWithLogitsLoss()

            t0 = time.perf_counter()
            train_one_epoch_frem(loader, enc, pred, opt, bce, sigma_z, sigma_s)
            results["FREM"][n].append(time.perf_counter() - t0)

            # Reg-GDP
            enc = Encoder(2, z_dim=z_dim).to(DEVICE)
            pred = Predictor(z_dim=z_dim).to(DEVICE)
            opt = optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
            bce = nn.BCEWithLogitsLoss()

            t0 = time.perf_counter()
            train_one_epoch_reg_gdp(loader, enc, pred, opt, bce)
            results["Reg-GDP"][n].append(time.perf_counter() - t0)

        for method in results:
            times = results[method][n]
            print(f"  {method:10s}: {np.mean(times):.4f} ± {np.std(times):.4f} sec/epoch")

    payload = {
        "sample_sizes": sample_sizes,
        "results": {
            m: {str(n): results[m][n] for n in sample_sizes} for m in results
        },
    }
    with open(CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved raw measurements to {CACHE}")
    return payload


def run_runtime_experiment(replot=False):
    if replot:
        if not os.path.exists(CACHE):
            raise FileNotFoundError(
                f"{CACHE} not found; run without --replot once to generate it."
            )
        with open(CACHE) as f:
            payload = json.load(f)
        print(f"Loaded cached measurements from {CACHE} (no re-timing).")
    else:
        payload = compute_results()

    sample_sizes = payload["sample_sizes"]
    results = {
        m: {int(n): v for n, v in payload["results"][m].items()}
        for m in payload["results"]
    }

    # Plot
    # figsize width = 0.6 * W_text, where W_text = 370.38374pt / 72.27pt/in = 5.124"
    W_text = 5.124
    fig_w = 0.6 * W_text   # 3.074"
    fig_h = fig_w * 0.82   # 2.521"
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    colors = {"FRHSIC": "tab:red", "FREM": "tab:green", "Reg-GDP": "tab:blue"}
    markers = {"FRHSIC": "o", "FREM": "s", "Reg-GDP": "^"}

    for method in results:
        means = [np.mean(results[method][n]) for n in sample_sizes]
        stds = [np.std(results[method][n]) for n in sample_sizes]
        ax.errorbar(
            sample_sizes, means, yerr=stds,
            marker=markers[method], color=colors[method],
            linewidth=1.0, markersize=3.5, capsize=2, label=method,
        )

    ax.set_xlabel("Sample size $n$", fontsize=10)
    ax.set_ylabel("Time per epoch (seconds)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Legend below the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 5), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    fig.savefig(os.path.join(RESULTS_DIR, "runtime_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "runtime_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"\nRuntime figure saved to {RESULTS_DIR}/runtime_comparison.{{pdf,png}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot", action="store_true",
        help="Re-draw the figure from cached measurements without re-timing.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_runtime_experiment(replot=args.replot)
