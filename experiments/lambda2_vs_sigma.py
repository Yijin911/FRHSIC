"""
Empirical check of Remark rem:lambda2_scaling: hat_lambda_S vs sigma scaling.

Sweeps Gaussian-kernel bandwidth sigma and records the smallest positive
eigenvalue of the normalized centered Gram matrix (1/n) tilde{L} = (1/n) H L H
on S, for the Adult and Crime datasets at n = 1500. On log-log axes the curves
should be approximately linear with slope ~ -d_s (= -1 in our setup),
supporting the claim that hat_lambda_S scales polynomially with sigma.

Convention (matches Theorem 3.3): hat_lambda_S = lambda_min^+((1/n) tilde{L}).
"""
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import DEVICE, gaussian_kernel
from real_data import load_adult, load_communities, SEED
from bound_tightness import empirical_lambda2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Persisted raw numeric results so the figure can be restyled later
# (python lambda2_vs_sigma.py --replot) without re-running the experiment.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "lambda2_vs_sigma_data.json")


def compute_results():
    datasets = {
        "adult": load_adult,
        "communities": load_communities,
    }
    sigmas = [0.1, 0.3, 1.0, 3.0, 10.0]
    N_MAX = 1500

    all_results = {}

    for dname, load_fn in datasets.items():
        result = load_fn()
        if result is None:
            continue
        X_np, S_np, Y_np, display_name, _task = result
        S_np = (S_np - S_np.min()) / (S_np.max() - S_np.min() + 1e-8)
        _ = MinMaxScaler().fit_transform(X_np)  # match bound_tightness preprocessing

        S = torch.tensor(S_np, dtype=torch.float32)

        rng = np.random.RandomState(SEED)
        n_full = len(S)
        if n_full > N_MAX:
            sub = rng.choice(n_full, N_MAX, replace=False)
            S = S[sub]
        n = len(S)

        values = []
        print(f"\n{display_name} (n={n}):")
        for sigma in sigmas:
            lam_S = empirical_lambda2(S, sigma_s=sigma)
            values.append(lam_S)
            print(f"  sigma={sigma:5.2f}  hat_lambda_S={lam_S:.6e}")

        # Linear fit in log-log space: log(lam2) = slope * log(sigma) + intercept
        log_sigma = np.log(sigmas)
        log_lam = np.log(values)
        slope, intercept = np.polyfit(log_sigma, log_lam, 1)
        print(f"  Empirical log-log slope = {slope:.3f}  (expected ~ -d_s = -1)")

        all_results[dname] = {
            "sigmas": [float(s) for s in sigmas],
            "values": [float(v) for v in values],
            "slope": float(slope),
            "intercept": float(intercept),
            "display_name": display_name,
        }

    payload = {"all_results": all_results}
    with open(CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved raw results to {CACHE}")
    return payload


def run_lambda2_vs_sigma(replot=False):
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

    all_results = payload["all_results"]

    # Plot both datasets on a single log-log axis
    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {"adult": ("tab:blue", "o"), "communities": ("tab:orange", "s")}
    for dname, res in all_results.items():
        color, marker = styles.get(dname, (None, "o"))
        label = rf"{res['display_name']} (slope $\approx$ {res['slope']:.2f})"
        ax.loglog(res["sigmas"], res["values"], marker=marker, linestyle="-",
                  color=color, linewidth=2, markersize=8, label=label)
    ax.set_xlabel(r"Bandwidth $\sigma$", fontsize=13)
    ax.set_ylabel(r"$\hat\lambda_S$", fontsize=13)
    ax.set_title(r"Empirical $\hat\lambda_S$ vs $\sigma$ (log-log)", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "lambda2_vs_sigma.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "lambda2_vs_sigma.png"),
                dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {RESULTS_DIR}/lambda2_vs_sigma.{{pdf,png}}")
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
    run_lambda2_vs_sigma(replot=args.replot)
