"""
Convergence rate experiment for FRHSIC paper.
Empirically validates Corollary that HSIC estimator achieves O(n^{-1/2})
while EIPM (Kong et al. 2025) achieves O(n^{-2/5}).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from utils import gaussian_kernel, hsic_biased, eipm_weighted, DEVICE

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Synthetic data with known dependence
# ──────────────────────────────────────────────
def generate_data(n, rho=0.5, seed=0):
    """
    Generate (Z, S) jointly Gaussian with correlation rho.
    Z ~ N(0,1), S = rho*Z + sqrt(1-rho^2)*epsilon.
    """
    rng = np.random.RandomState(seed)
    Z = rng.normal(0, 1, n).astype(np.float32)
    eps = rng.normal(0, 1, n).astype(np.float32)
    S = rho * Z + np.sqrt(1 - rho ** 2) * eps
    return (
        torch.tensor(Z.reshape(-1, 1)),
        torch.tensor(S.astype(np.float32)),
    )


# ──────────────────────────────────────────────
# Compute "ground truth" via large sample size
# ──────────────────────────────────────────────
def ground_truth(rho=0.5, n_large=20000, sigma=1.0):
    Z, S = generate_data(n_large, rho=rho, seed=999)
    Z, S = Z.to(DEVICE), S.to(DEVICE)
    hsic_true = hsic_biased(Z, S, sigma, sigma).item()
    eipm_true = eipm_weighted(Z, S, sigma, sigma)
    return hsic_true, eipm_true


# ──────────────────────────────────────────────
# Convergence experiment
# ──────────────────────────────────────────────
def run_convergence_experiment():
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    n_repeats = 20
    rho = 0.5
    sigma = 1.0

    print("Computing ground truth (n=20000)...")
    hsic_true, eipm_true = ground_truth(rho=rho, n_large=20000, sigma=sigma)
    print(f"  HSIC_true ≈ {hsic_true:.6f}")
    print(f"  EIPM_true ≈ {eipm_true:.6f}")

    hsic_errors = {n: [] for n in sample_sizes}
    eipm_errors = {n: [] for n in sample_sizes}

    for n in sample_sizes:
        print(f"\nn = {n}")
        for rep in range(n_repeats):
            Z, S = generate_data(n, rho=rho, seed=rep)
            Z, S = Z.to(DEVICE), S.to(DEVICE)

            hsic_hat = hsic_biased(Z, S, sigma, sigma).item()
            hsic_errors[n].append(abs(hsic_hat - hsic_true))

            eipm_hat = eipm_weighted(Z, S, sigma, sigma)
            eipm_errors[n].append(abs(eipm_hat - eipm_true))

        print(f"  HSIC error: {np.mean(hsic_errors[n]):.6f} ± {np.std(hsic_errors[n]):.6f}")
        print(f"  EIPM error: {np.mean(eipm_errors[n]):.6f} ± {np.std(eipm_errors[n]):.6f}")

    # Fit log-log slopes
    log_n = np.log(sample_sizes)
    log_hsic = np.log([np.mean(hsic_errors[n]) for n in sample_sizes])
    log_eipm = np.log([np.mean(eipm_errors[n]) for n in sample_sizes])

    slope_hsic, intercept_hsic = np.polyfit(log_n, log_hsic, 1)
    slope_eipm, intercept_eipm = np.polyfit(log_n, log_eipm, 1)

    print(f"\nFitted convergence rates:")
    print(f"  HSIC: |error| ≈ n^{slope_hsic:.3f}  (theoretical: -1/2 = -0.500)")
    print(f"  EIPM: |error| ≈ n^{slope_eipm:.3f}  (theoretical: -2/5 = -0.400)")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    hsic_means = [np.mean(hsic_errors[n]) for n in sample_sizes]
    hsic_stds = [np.std(hsic_errors[n]) for n in sample_sizes]
    eipm_means = [np.mean(eipm_errors[n]) for n in sample_sizes]
    eipm_stds = [np.std(eipm_errors[n]) for n in sample_sizes]

    ax.errorbar(
        sample_sizes, hsic_means, yerr=hsic_stds,
        marker="o", color="tab:red", linewidth=2, markersize=8, capsize=4,
        label=f"HSIC (slope = {slope_hsic:.2f})",
    )
    ax.errorbar(
        sample_sizes, eipm_means, yerr=eipm_stds,
        marker="s", color="tab:green", linewidth=2, markersize=8, capsize=4,
        label=f"EIPM (slope = {slope_eipm:.2f})",
    )

    # Reference lines
    n_ref = np.array(sample_sizes, dtype=float)
    c_hsic = hsic_means[0] * sample_sizes[0] ** 0.5
    ax.plot(n_ref, c_hsic / np.sqrt(n_ref), "--", color="tab:red", alpha=0.5, label=r"$n^{-1/2}$ (theory)")
    c_eipm = eipm_means[0] * sample_sizes[0] ** 0.4
    ax.plot(n_ref, c_eipm / n_ref ** 0.4, "--", color="tab:green", alpha=0.5, label=r"$n^{-2/5}$ (theory)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size $n$", fontsize=13)
    ax.set_ylabel("Estimation error $|\\widehat{T} - T|$", fontsize=13)
    ax.set_title("Convergence Rate: HSIC vs.\\ EIPM Estimators", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence_rate.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "convergence_rate.png"), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {RESULTS_DIR}/convergence_rate.{{pdf,png}}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_convergence_experiment()
