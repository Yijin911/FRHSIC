"""
Lambda selection via HSIC independence test (Section 3.6).
Demonstrates the principled regularization strength selection algorithm:
train with increasing lambda, stop when HSIC test fails to reject independence.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import json
import argparse

from utils import (
    DEVICE, HIDDEN_DIM, Z_DIM, EPOCHS,
    Encoder, Predictor,
    hsic_biased, hsic_test_threshold, estimate_gdp, median_heuristic,
)
from real_data import load_adult, load_communities, load_compas, train_frhsic, BATCH_SIZE, SEED

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Persisted raw numeric results so the figure can be restyled later
# (python lambda_selection.py --replot) without re-running the experiment.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "lambda_selection_data.json")


def compute_results():
    """
    For each dataset:
    1. Train FRHSIC at increasing lambda values
    2. Evaluate HSIC on held-out validation set
    3. Run HSIC independence test at alpha = 0.05
    4. Report the smallest lambda where the test fails to reject
    """
    datasets = {
        "adult": (load_adult, "classification"),
        "crime": (load_communities, "regression"),
        "compas": (load_compas, "classification"),
    }

    lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    alpha_test = 0.05
    n_repeats = 3

    all_results = {}

    for dname, (load_fn, task) in datasets.items():
        result = load_fn()
        if result is None:
            print(f"Skipping {dname}: data not available")
            continue

        X_np, S_np, Y_np, display_name, task = result
        S_np = (S_np - S_np.min()) / (S_np.max() - S_np.min() + 1e-8)
        scaler = MinMaxScaler()
        X_np = scaler.fit_transform(X_np)

        print(f"\n{'='*70}")
        print(f"Dataset: {display_name} (n={len(X_np)})")
        print(f"{'='*70}")

        dataset_results = []

        for repeat in range(n_repeats):
            rng = np.random.RandomState(SEED + repeat)
            n = len(X_np)
            idx = rng.permutation(n)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]

            X = torch.tensor(X_np, dtype=torch.float32)
            S = torch.tensor(S_np, dtype=torch.float32)
            Y = torch.tensor(Y_np, dtype=torch.float32)

            X_tr, X_val, X_te = X[train_idx], X[val_idx], X[test_idx]
            S_tr, S_val, S_te = S[train_idx], S[val_idx], S[test_idx]
            Y_tr, Y_val, Y_te = Y[train_idx], Y[val_idx], Y[test_idx]

            selected_lambda = None
            repeat_results = []

            print(f"\n  Repeat {repeat+1}/{n_repeats}")
            print(f"  {'lambda':>8s}  {'HSIC_val':>10s}  {'threshold':>10s}  {'reject':>7s}  {'GDP':>8s}  {'Perf':>8s}")
            print(f"  {'-'*60}")

            for lam in lambdas:
                enc, pred = train_frhsic(X_tr, S_tr, Y_tr, lam, task)
                enc.eval()
                pred.eval()

                with torch.no_grad():
                    Z_val = enc(X_val.to(DEVICE))
                    S_val_d = S_val.to(DEVICE)

                    # Subsample for HSIC test (n x n kernel matrix must fit in memory)
                    n_test = min(1000, len(Z_val))
                    test_perm = torch.randperm(len(Z_val))[:n_test]
                    Z_test_sub = Z_val[test_perm]
                    S_test_sub = S_val_d[test_perm]

                    # Compute sigma for test
                    sigma_z = median_heuristic(Z_test_sub[:min(500, n_test)])
                    sigma_s = median_heuristic(S_test_sub[:min(500, n_test)].unsqueeze(1))

                    # HSIC independence test on validation subsample
                    hsic_val, threshold, reject = hsic_test_threshold(
                        Z_test_sub, S_test_sub, sigma_z, sigma_s, alpha=alpha_test
                    )

                    # Evaluate on test set
                    Z_te = enc(X_te.to(DEVICE))
                    S_te_d = S_te.to(DEVICE)
                    Y_te_d = Y_te.to(DEVICE)
                    gdp = estimate_gdp(Z_te, S_te_d, pred)

                    if task == "classification":
                        logits = pred(Z_te).squeeze()
                        perf = ((torch.sigmoid(logits) > 0.5).float() == Y_te_d).float().mean().item()
                        perf_name = "Acc"
                    else:
                        preds = pred(Z_te).squeeze()
                        perf = ((preds - Y_te_d) ** 2).mean().item() * 100
                        perf_name = "MSE×100"

                status = "REJECT" if reject else "accept"
                print(f"  {lam:8.2f}  {hsic_val:10.6f}  {threshold:10.6f}  {status:>7s}  {gdp:8.4f}  {perf:8.3f}")

                repeat_results.append({
                    "lambda": float(lam), "hsic_val": float(hsic_val),
                    "threshold": float(threshold), "reject": bool(reject),
                    "gdp": float(gdp), "perf": float(perf), "perf_name": perf_name,
                })

                if selected_lambda is None and not reject:
                    selected_lambda = lam

            if selected_lambda is not None:
                print(f"  >>> Selected lambda = {selected_lambda} (first non-rejection at alpha={alpha_test})")
            else:
                print(f"  >>> No lambda achieved non-rejection at alpha={alpha_test}")

            dataset_results.append({
                "repeat": repeat,
                "selected_lambda": (
                    float(selected_lambda) if selected_lambda is not None else None
                ),
                "results": repeat_results,
            })

        all_results[dname] = {
            "display_name": display_name,
            "task": task,
            "dataset_results": dataset_results,
        }

    payload = {
        "alpha_test": alpha_test,
        "all_results": all_results,
    }
    with open(CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved raw results to {CACHE}")
    return payload


def run_lambda_selection(replot=False):
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

    alpha_test = payload["alpha_test"]
    all_results = payload["all_results"]

    # ── Plot ──
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4.5))
    if n_datasets == 1:
        axes = [axes]

    for ax, (dname, dres) in zip(axes, all_results.items()):
        # Use first repeat for the plot
        res = dres["dataset_results"][0]["results"]
        lams = [r["lambda"] for r in res]
        hsic_vals = [r["hsic_val"] for r in res]
        thresholds = [r["threshold"] for r in res]
        rejects = [r["reject"] for r in res]
        selected = dres["dataset_results"][0]["selected_lambda"]

        ax.plot(lams, hsic_vals, "o-", color="tab:red", linewidth=2, markersize=7, label="HSIC (validation)")
        ax.plot(lams, thresholds, "s--", color="tab:blue", linewidth=1.5, markersize=6, label=f"Threshold ($\\alpha={alpha_test}$)")

        # Shade reject/accept regions
        for i, (lam, rej) in enumerate(zip(lams, rejects)):
            color = "lightcoral" if rej else "lightgreen"
            ax.axvspan(
                lam * 0.7 if i == 0 else np.sqrt(lams[i-1] * lam),
                lam * 1.4 if i == len(lams) - 1 else np.sqrt(lam * lams[i+1]),
                alpha=0.15, color=color
            )

        if selected is not None:
            ax.axvline(selected, color="green", linestyle=":", linewidth=2,
                       label=f"Selected $\\lambda={selected}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\lambda$", fontsize=12)
        ax.set_ylabel("HSIC value", fontsize=12)
        ax.set_title(dres["display_name"], fontsize=13)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "lambda_selection.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "lambda_selection.png"), dpi=150, bbox_inches="tight")
    print(f"\nLambda selection figure saved to {RESULTS_DIR}/lambda_selection.{{pdf,png}}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("Summary: Selected lambda across repeats")
    print(f"{'='*70}")
    for dname, dres in all_results.items():
        selected_lambdas = [dr["selected_lambda"] for dr in dres["dataset_results"]]
        sel_str = ", ".join([str(s) if s is not None else "None" for s in selected_lambdas])
        print(f"  {dres['display_name']:20s}: {sel_str}")

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
    run_lambda_selection(replot=args.replot)
