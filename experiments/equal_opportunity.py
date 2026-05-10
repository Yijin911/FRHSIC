"""
Equal Opportunity experiment for FRHSIC.
Tests the EO extension described in Section 3.4 of the paper:
compute HSIC only on the Y=1 subset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

from utils import (
    DEVICE, HIDDEN_DIM, Z_DIM, EPOCHS, KERNEL_SIGMA,
    Encoder, Predictor, hsic_biased, estimate_gdp,
)
from real_data import load_adult, load_compas, BATCH_SIZE, SEED

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_frhsic_eo(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS,
                     batch_size=BATCH_SIZE, lr=1e-3):
    """FRHSIC with Equal Opportunity: HSIC computed on Y=1 subset only."""
    dataset = TensorDataset(X, S, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = encoder(X_b)
            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)

            # HSIC on Y=1 subset only
            mask = Y_b == 1
            if mask.sum() > 10:
                Z_pos = Z_b[mask]
                S_pos = S_b[mask]
                loss_fair = hsic_biased(Z_pos, S_pos, KERNEL_SIGMA, KERNEL_SIGMA)
            else:
                loss_fair = torch.tensor(0.0, device=DEVICE)

            loss = loss_pred + lam * loss_fair
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def estimate_eo_gdp(Z, S, Y, predictor, bandwidth=0.1, n_grid=50):
    """GDP restricted to Y=1 subset (Equal Opportunity GDP)."""
    with torch.no_grad():
        mask = Y == 1
        if mask.sum() < 10:
            return 0.0
        Z_pos = Z[mask]
        S_pos = S[mask]
        return estimate_gdp(Z_pos, S_pos, predictor, bandwidth, n_grid)


def run_eo_experiment():
    from real_data import train_frhsic

    datasets = {"adult": load_adult, "compas": load_compas}
    lambdas = [0.1, 1.0, 5.0, 10.0, 50.0]

    all_results = {}

    for dname, load_fn in datasets.items():
        result = load_fn()
        if result is None:
            continue
        X_np, S_np, Y_np, display_name, task = result
        if task != "classification":
            continue

        S_np = (S_np - S_np.min()) / (S_np.max() - S_np.min() + 1e-8)
        scaler = MinMaxScaler()
        X_np = scaler.fit_transform(X_np)

        X = torch.tensor(X_np, dtype=torch.float32)
        S = torch.tensor(S_np, dtype=torch.float32)
        Y = torch.tensor(Y_np, dtype=torch.float32)

        rng = np.random.RandomState(SEED)
        n = len(X)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        X_tr, X_te = X[train_idx], X[test_idx]
        S_tr, S_te = S[train_idx], S[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        print(f"\n{'='*60}")
        print(f"Dataset: {display_name}")
        print(f"{'='*60}")

        dp_results = {"FRHSIC-DP": [], "FRHSIC-EO": []}
        eo_results = {"FRHSIC-DP": [], "FRHSIC-EO": []}
        accs = {"FRHSIC-DP": [], "FRHSIC-EO": []}

        for lam in lambdas:
            # DP version
            enc_dp, pred_dp = train_frhsic(X_tr, S_tr, Y_tr, lam, task)
            with torch.no_grad():
                Z_te_dp = enc_dp(X_te.to(DEVICE))
                S_te_d = S_te.to(DEVICE)
                Y_te_d = Y_te.to(DEVICE)
                logits = pred_dp(Z_te_dp).squeeze()
                acc_dp = ((torch.sigmoid(logits) > 0.5).float() == Y_te_d).float().mean().item()
                gdp_dp = estimate_gdp(Z_te_dp, S_te_d, pred_dp)
                eo_gdp_dp = estimate_eo_gdp(Z_te_dp, S_te_d, Y_te_d, pred_dp)

            dp_results["FRHSIC-DP"].append(gdp_dp)
            eo_results["FRHSIC-DP"].append(eo_gdp_dp)
            accs["FRHSIC-DP"].append(acc_dp)

            # EO version
            enc_eo, pred_eo = train_frhsic_eo(X_tr, S_tr, Y_tr, lam, task)
            with torch.no_grad():
                Z_te_eo = enc_eo(X_te.to(DEVICE))
                logits = pred_eo(Z_te_eo).squeeze()
                acc_eo = ((torch.sigmoid(logits) > 0.5).float() == Y_te_d).float().mean().item()
                gdp_eo = estimate_gdp(Z_te_eo, S_te_d, pred_eo)
                eo_gdp_eo = estimate_eo_gdp(Z_te_eo, S_te_d, Y_te_d, pred_eo)

            dp_results["FRHSIC-EO"].append(gdp_eo)
            eo_results["FRHSIC-EO"].append(eo_gdp_eo)
            accs["FRHSIC-EO"].append(acc_eo)

            print(f"  lam={lam:5.1f} | DP: Acc={acc_dp:.3f} GDP={gdp_dp:.4f} EO-GDP={eo_gdp_dp:.4f}"
                  f" | EO: Acc={acc_eo:.3f} GDP={gdp_eo:.4f} EO-GDP={eo_gdp_eo:.4f}")

        all_results[dname] = {
            "lambdas": lambdas, "dp_results": dp_results, "eo_results": eo_results,
            "accs": accs, "display_name": display_name,
        }

    # Plot
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, (dname, res) in zip(axes, all_results.items()):
        ax.plot(res["accs"]["FRHSIC-DP"], res["eo_results"]["FRHSIC-DP"],
                "o-", color="tab:red", label="FRHSIC-DP", linewidth=2, markersize=8)
        ax.plot(res["accs"]["FRHSIC-EO"], res["eo_results"]["FRHSIC-EO"],
                "s-", color="tab:blue", label="FRHSIC-EO", linewidth=2, markersize=8)
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_ylabel("EO-GDP (lower = fairer)", fontsize=12)
        ax.set_title(res["display_name"], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "equal_opportunity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "equal_opportunity.png"), dpi=150, bbox_inches="tight")
    print(f"\nEO figure saved to {RESULTS_DIR}/equal_opportunity.{{pdf,png}}")

    return all_results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_eo_experiment()
