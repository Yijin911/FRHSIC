"""
Multiple continuous sensitive attributes experiment for FRHSIC.
Demonstrates HSIC naturally handles multiple S via sum of HSIC terms
or HSIC with vector-valued S.
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
    Encoder, Predictor, hsic_biased, estimate_gdp, gaussian_kernel, median_heuristic,
)
from real_data import BATCH_SIZE, SEED

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hsic_biased_multi(Z, S_multi, sigma_z=1.0, sigma_s=1.0):
    """HSIC between Z and multi-dimensional S using product kernel on S."""
    n = Z.shape[0]
    K = gaussian_kernel(Z, Z, sigma_z)
    L = gaussian_kernel(S_multi, S_multi, sigma_s)
    H = torch.eye(n, device=Z.device) - 1.0 / n
    return (H @ K @ H * (H @ L @ H)).sum() / (n ** 2)


def load_adult_multi_s():
    """Adult dataset with age AND hours-per-week as continuous sensitive attributes."""
    import pandas as pd
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(DATA_DIR, "adult.csv")
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
        ]
        df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    S1 = df["age"].values.astype(np.float32)
    S2 = df["hours-per-week"].values.astype(np.float32)
    Y = (df["income"].str.strip() == ">50K").astype(np.float32).values

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["age", "hours-per-week"]]
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    df_cat = pd.get_dummies(df[cat_cols], columns=cat_cols)
    X = np.hstack([df[numeric_cols].values, df_cat.values]).astype(np.float32)

    S_multi = np.column_stack([S1, S2])
    return X, S_multi, Y


def train_frhsic_multi(X, S_multi, Y, lam, z_dim=Z_DIM, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, lr=1e-3):
    """FRHSIC with multi-dimensional S using product kernel + adaptive bandwidth."""
    dataset = TensorDataset(X, S_multi, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Pre-compute sigma_s from data (fixed throughout training)
    with torch.no_grad():
        S_sample = S_multi[:min(1000, len(S_multi))].to(DEVICE)
        sigma_s = median_heuristic(S_sample)

    sigma_z = KERNEL_SIGMA  # will be updated adaptively

    for epoch in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = encoder(X_b)

            if epoch % 20 == 0:
                with torch.no_grad():
                    sigma_z = median_heuristic(Z_b[:min(256, len(Z_b))])

            loss = loss_fn(predictor(Z_b).squeeze(), Y_b) + lam * hsic_biased_multi(
                Z_b, S_b, sigma_z, sigma_s)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def train_frhsic_sum(X, S_multi, Y, lam, z_dim=Z_DIM, epochs=EPOCHS,
                      batch_size=BATCH_SIZE, lr=1e-3):
    """FRHSIC with sum of per-attribute HSIC terms + adaptive bandwidth."""
    dataset = TensorDataset(X, S_multi, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Pre-compute sigma_s per attribute
    with torch.no_grad():
        sigma_s_list = []
        for d in range(S_multi.shape[1]):
            S_d = S_multi[:min(1000, len(S_multi)), d:d+1].to(DEVICE)
            sigma_s_list.append(median_heuristic(S_d))

    sigma_z = KERNEL_SIGMA

    for epoch in range(epochs):
        for X_b, S_b, Y_b in loader:
            X_b, S_b, Y_b = X_b.to(DEVICE), S_b.to(DEVICE), Y_b.to(DEVICE)
            Z_b = encoder(X_b)

            if epoch % 20 == 0:
                with torch.no_grad():
                    sigma_z = median_heuristic(Z_b[:min(256, len(Z_b))])

            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)

            loss_fair = torch.tensor(0.0, device=DEVICE)
            for d in range(S_b.shape[1]):
                loss_fair = loss_fair + hsic_biased(Z_b, S_b[:, d], sigma_z, sigma_s_list[d])

            loss = loss_pred + lam * loss_fair
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def run_multi_s_experiment():
    X_np, S_multi_np, Y_np = load_adult_multi_s()

    scaler = MinMaxScaler()
    X_np = scaler.fit_transform(X_np)
    S_multi_np = (S_multi_np - S_multi_np.min(axis=0)) / (S_multi_np.max(axis=0) - S_multi_np.min(axis=0) + 1e-8)

    X = torch.tensor(X_np, dtype=torch.float32)
    S_multi = torch.tensor(S_multi_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)

    rng = np.random.RandomState(SEED)
    n = len(X)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_tr, X_te = X[train_idx], X[test_idx]
    S_tr, S_te = S_multi[train_idx], S_multi[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    lambdas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    s_names = ["Age", "Hours/Week"]
    methods = {
        "Unfair": None,
        "FRHSIC-Joint": train_frhsic_multi,
        "FRHSIC-Sum": train_frhsic_sum,
    }

    print(f"{'='*80}")
    print("Multiple Sensitive Attributes: Adult (Age + Hours-per-Week)")
    print(f"{'='*80}")

    results = {}

    for method_name, train_fn in methods.items():
        lam_list = [0.0] if method_name == "Unfair" else lambdas
        method_results = []

        for lam in lam_list:
            if method_name == "Unfair":
                enc, pred = train_frhsic_multi(X_tr, S_tr, Y_tr, 0.0)
            else:
                enc, pred = train_fn(X_tr, S_tr, Y_tr, lam)

            enc.eval()
            pred.eval()

            with torch.no_grad():
                Z_te = enc(X_te.to(DEVICE))
                logits = pred(Z_te).squeeze()
                acc = ((torch.sigmoid(logits) > 0.5).float() == Y_te.to(DEVICE)).float().mean().item()

                gdps = []
                for d in range(S_te.shape[1]):
                    gdp_d = estimate_gdp(Z_te, S_te[:, d].to(DEVICE), pred)
                    gdps.append(gdp_d)

            method_results.append({"lam": lam, "acc": acc, "gdps": gdps})
            gdp_str = " | ".join([f"GDP({s_names[d]})={gdps[d]:.4f}" for d in range(len(gdps))])
            print(f"  {method_name} lam={lam:5.1f} | Acc={acc:.3f} | {gdp_str}")

        results[method_name] = method_results

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for d, (ax, sname) in enumerate(zip(axes, s_names)):
        for method_name, method_results in results.items():
            accs = [r["acc"] for r in method_results]
            gdps = [r["gdps"][d] for r in method_results]
            color = {"Unfair": "black", "FRHSIC-Joint": "tab:red", "FRHSIC-Sum": "tab:blue"}[method_name]
            marker = {"Unfair": "*", "FRHSIC-Joint": "o", "FRHSIC-Sum": "s"}[method_name]
            if method_name == "Unfair":
                ax.scatter(gdps, accs, marker=marker, s=200, c=color, zorder=10, label=method_name)
            else:
                order = np.argsort(gdps)
                ax.plot([gdps[i] for i in order], [accs[i] for i in order],
                        marker=marker, color=color, linewidth=2, markersize=8, label=method_name)
        ax.set_xlabel(f"GDP({sname}) (lower = fairer)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Fairness w.r.t. {sname}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multi_sensitive.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "multi_sensitive.png"), dpi=150, bbox_inches="tight")
    print(f"\nMulti-S figure saved to {RESULTS_DIR}/multi_sensitive.{{pdf,png}}")

    return results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_multi_s_experiment()
