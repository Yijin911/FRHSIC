"""
Transfer experiment for FRHSIC paper.
Tests the key claim: HSIC gives transferable fairness for any downstream head.

Protocol:
1. Train representation once with each method (freeze encoder)
2. Fit multiple different downstream heads on the frozen representation
3. Measure GDP for each head — FRHSIC should remain fair across all heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import os
import json
import argparse

from utils import (
    DEVICE, HIDDEN_DIM, Z_DIM, EPOCHS, KERNEL_SIGMA, N_REPEATS,
    Encoder, Predictor,
    hsic_biased, estimate_gdp, gaussian_kernel, reg_gdp_loss,
)
from real_data import (
    load_adult, load_communities, load_compas,
    train_frhsic, train_reg_gdp, train_frem, train_adv,
    BATCH_SIZE, SEED,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Persisted raw numeric results so the figure can be restyled later
# (python transfer.py --replot) without re-running the experiment.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "transfer_data.json")


class MLPHead(nn.Module):
    """2-layer MLP prediction head."""
    def __init__(self, z_dim=Z_DIM, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_new_head_torch(Z_train, Y_train, head, task, epochs=100, lr=1e-3):
    """Train a new PyTorch prediction head on frozen representations."""
    dataset = TensorDataset(Z_train, Y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    opt = optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    for _ in range(epochs):
        for Z_b, Y_b in loader:
            Z_b, Y_b = Z_b.to(DEVICE), Y_b.to(DEVICE)
            loss = loss_fn(head(Z_b).squeeze(), Y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def estimate_gdp_sklearn(Z_np, S_np, model, bandwidth=0.1, n_grid=50):
    """GDP estimator for sklearn models."""
    preds = model.predict(Z_np).astype(np.float64)
    global_mean = preds.mean()
    s_grid = np.linspace(S_np.min(), S_np.max(), n_grid)
    diffs = []
    for s in s_grid:
        weights = np.exp(-((S_np - s) ** 2) / (2 * bandwidth ** 2))
        weights = weights / (weights.sum() + 1e-10)
        conditional_mean = (weights * preds).sum()
        diffs.append(abs(conditional_mean - global_mean))
    return np.mean(diffs)


def compute_results():
    datasets = {
        "adult": load_adult,
        "communities": load_communities,
        "compas": load_compas,
    }

    train_methods = {
        "FRHSIC (Ours)": train_frhsic,
        "Reg-GDP": train_reg_gdp,
        "FREM": train_frem,
        "ADV": train_adv,
        "Unfair": lambda X, S, Y, lam, task, **kw: train_frhsic(X, S, Y, 0.0, task, **kw),
    }

    downstream_heads = ["Linear", "MLP", "Random Forest", "SVM"]

    lam = 10.0
    all_results = {}

    for dname, load_fn in datasets.items():
        result = load_fn()
        if result is None:
            continue
        X_np, S_np, Y_np, display_name, task = result

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

        print(f"\n{'='*70}")
        print(f"Dataset: {display_name} (n={len(X)}, task={task})")
        print(f"{'='*70}")

        dataset_results = {}

        for method_name, train_fn in train_methods.items():
            print(f"\n  Training representation: {method_name}")
            encoder, original_predictor = train_fn(X_tr, S_tr, Y_tr, lam, task)
            encoder.eval()

            # Get frozen representations
            with torch.no_grad():
                Z_tr = encoder(X_tr.to(DEVICE)).cpu()
                Z_te = encoder(X_te.to(DEVICE)).cpu()

            method_results = {}

            for head_name in downstream_heads:
                if head_name == "Linear":
                    head = Predictor(z_dim=Z_DIM).to(DEVICE)
                    head = train_new_head_torch(Z_tr, Y_tr, head, task)
                    gdp = estimate_gdp(Z_te.to(DEVICE), S_te.to(DEVICE), head)
                elif head_name == "MLP":
                    head = MLPHead(z_dim=Z_DIM).to(DEVICE)
                    head = train_new_head_torch(Z_tr, Y_tr, head, task)
                    gdp = estimate_gdp(Z_te.to(DEVICE), S_te.to(DEVICE), head)
                elif head_name == "Random Forest":
                    Z_tr_np, Z_te_np = Z_tr.numpy(), Z_te.numpy()
                    S_te_np = S_te.numpy()
                    Y_tr_np = Y_tr.numpy()
                    if task == "classification":
                        rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
                        rf.fit(Z_tr_np, Y_tr_np.astype(int))
                    else:
                        rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
                        rf.fit(Z_tr_np, Y_tr_np)
                    gdp = estimate_gdp_sklearn(Z_te_np, S_te_np, rf)
                elif head_name == "SVM":
                    Z_tr_np, Z_te_np = Z_tr.numpy(), Z_te.numpy()
                    S_te_np = S_te.numpy()
                    Y_tr_np = Y_tr.numpy()
                    if task == "classification":
                        svm = SVC(kernel="rbf", random_state=SEED)
                        n_sub = min(3000, len(Z_tr_np))
                        svm.fit(Z_tr_np[:n_sub], Y_tr_np[:n_sub].astype(int))
                    else:
                        svm = SVR(kernel="rbf")
                        n_sub = min(3000, len(Z_tr_np))
                        svm.fit(Z_tr_np[:n_sub], Y_tr_np[:n_sub])
                    gdp = estimate_gdp_sklearn(Z_te_np, S_te_np, svm)

                method_results[head_name] = float(gdp)
                print(f"    {head_name:20s} GDP: {gdp:.4f}")

            dataset_results[method_name] = method_results
        all_results[dname] = dataset_results

    payload = {
        "downstream_heads": downstream_heads,
        "all_results": all_results,
    }
    with open(CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved raw results to {CACHE}")
    return payload


def run_transfer_experiment(replot=False):
    if replot:
        if not os.path.exists(CACHE):
            raise FileNotFoundError(
                f"{CACHE} not found; run without --replot once to generate it."
            )
        with open(CACHE) as f:
            payload = json.load(f)
        print(f"Loaded cached results from {CACHE} (no re-run).")
    else:
        payload = compute_results()

    downstream_heads = payload["downstream_heads"]
    all_results = payload["all_results"]

    # Plot: grouped bar chart per dataset
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    method_colors = {
        "FRHSIC (Ours)": "tab:red", "Reg-GDP": "tab:blue", "FREM": "tab:green",
        "ADV": "tab:purple", "Unfair": "black",
    }

    for ax, (dname, dataset_results) in zip(axes, all_results.items()):
        methods = list(dataset_results.keys())
        heads = downstream_heads
        x = np.arange(len(heads))
        width = 0.15
        for i, method in enumerate(methods):
            gdps = [dataset_results[method].get(h, 0) for h in heads]
            ax.bar(x + i * width, gdps, width, label=method,
                   color=method_colors.get(method, "gray"), alpha=0.85)
        ax.set_xlabel("Downstream Head", fontsize=12)
        ax.set_ylabel(r"$\Delta_{\mathrm{GDP}}$ (lower = fairer)", fontsize=12)
        ax.set_title(dname.replace("_", " ").upper(), fontsize=13)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(heads, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "transfer_experiment.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "transfer_experiment.png"), dpi=150, bbox_inches="tight")
    print(f"\nTransfer experiment saved to {RESULTS_DIR}/transfer_experiment.{{pdf,png}}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot", action="store_true",
        help="Re-draw the figure from cached results without re-running.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_transfer_experiment(replot=args.replot)
