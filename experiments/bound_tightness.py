"""
Empirical verification of Theorem 3.3 (empirical conditional-gap control) tightness.

The theorem is the finite-sample bound:

    (1/n) sum_i hat_delta_{f,i}^2  <=  ||f||^2 / hat_lambda_S  *  widehat_HSIC(Z, S)

where
    - hat_delta_{f,i} := f(Z_i) - (1/n) sum_j f(Z_j) for f in F_Z (the RKHS),
    - hat_lambda_S is the smallest positive eigenvalue of (1/n) tilde{L},
      where tilde{L} = H L H is the centered Gram matrix on S,
    - widehat_HSIC is the V-statistic / biased estimator (1/n^2) tr(K H L H).

This script computes both sides on the test split for each (dataset, lambda)
pair and produces figures/bound_tightness.pdf.

We use f(.) = k_Z(z_0, .) as a canonical RKHS function (so ||f||^2 = k_Z(z_0, z_0) = 1
for the Gaussian kernel). We average the LHS over a small set of anchor points z_0
drawn from the test set; the bound holds for every fixed f, hence in particular for
the worst-case anchor — we plot the maximum LHS over anchors.

Scale choices (T4 demonstration figure, not a comprehensive sweep):
    - Two datasets: Adult and Crime (Communities). Sufficient to show the
      qualitative shape (validity + informativeness) of the bound.
    - Subsample each dataset to N_MAX = 1500 points before the 80/20 split,
      so test-side eigendecomposition is on a 300-point matrix.
    - 5 lambda values spanning four orders of magnitude.
    - train_frhsic uses its module-default 200 epochs but on the smaller subset.
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import (
    DEVICE, KERNEL_SIGMA,
    hsic_biased, gaussian_kernel, median_heuristic,
)
from real_data import (
    load_adult, load_communities,
    train_frhsic, SEED,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def empirical_lambda2(S_values, sigma_s, rel_tol=1e-6):
    """Smallest numerically meaningful positive eigenvalue of (1/n) tilde{L}.

    L_ij = k_S(S_i, S_j) and tilde{L} = H L H is its centered version. The
    spectral constant in Theorem 3.3 is hat_lambda_S = lambda_min^+((1/n) tilde{L}).
    eigvalsh() returns near-zero eigenvalues at the level of floating-point noise,
    which would make the bound vacuous purely for numerical reasons. We therefore
    filter out eigenvalues smaller than `rel_tol * lambda_max`, mirroring standard
    practice for kernel methods (cf. truncated SVD / regularized inverse).

    Convention (matches Theorem 3.3 in the paper): hat_lambda_S is the smallest
    positive eigenvalue of (1/n) tilde{L}, equivalently (1/n) times the smallest
    positive eigenvalue of tilde{L}.

    S_values : 1-D torch tensor of length n.
    sigma_s  : Gaussian-kernel bandwidth on S.
    rel_tol  : eigenvalues below rel_tol * lambda_max are treated as zero.
    Returns float (hat_lambda_S).
    """
    with torch.no_grad():
        S_col = S_values.unsqueeze(1).to(DEVICE)
        L = gaussian_kernel(S_col, S_col, sigma_s)              # n x n
        n = L.shape[0]
        H = torch.eye(n, device=DEVICE) - 1.0 / n
        L_tilde_over_n = (H @ L @ H) / n                        # (1/n) H L H
        eigvals = torch.linalg.eigvalsh(L_tilde_over_n)
        lambda_max = float(eigvals.max().item())
        thresh = max(rel_tol * lambda_max, 1e-12)
        pos = eigvals[eigvals > thresh]
        if len(pos) == 0:
            return float("nan")
        return float(pos.min().item())


def empirical_lhs_rkhs(Z, S_values, sigma_z, n_anchors=32, rng=None):
    """Empirical LHS \widehat{E}_S[\hat{D}_f^2(S)] for f(.) = k_Z(z_0, .) in F_Z.

    For this choice f, the RKHS norm is ||f||^2 = k_Z(z_0, z_0) = 1
    (Gaussian kernel). The empirical conditional difference simplifies to
        \hat{D}_f(S_i) = f(Z_i) - (1/n) sum_j f(Z_j) = k_Z(z_0, Z_i) - mean_j k_Z(z_0, Z_j),
    and the LHS is the variance of the row k_Z(z_0, .) over the sample.

    We average the LHS over n_anchors random z_0 drawn from the test Z and also
    return the maximum (worst-case) anchor LHS so we can verify that the bound
    holds for every individual f (the theorem is pointwise in f).
    """
    with torch.no_grad():
        n = Z.shape[0]
        if rng is None:
            rng = np.random.RandomState(0)
        idx = rng.choice(n, size=min(n_anchors, n), replace=False)
        z0 = Z[idx]                                              # (a, d)
        K = gaussian_kernel(z0, Z, sigma_z)                     # (a, n) rows = f_a(Z_i)
        # ||f_a||^2 = k_Z(z0_a, z0_a) = 1 for Gaussian kernel.
        f_norm_sq = 1.0
        # \hat{D}_{f_a}(S_i) = K[a,i] - mean_i K[a,i]
        Dhat = K - K.mean(dim=1, keepdim=True)                   # (a, n)
        per_anchor = Dhat.pow(2).mean(dim=1)                     # (a,) — \widehat{E}_S[\hat{D}_{f_a}^2]
        return per_anchor.mean().item(), per_anchor.max().item(), f_norm_sq


def run_bound_tightness():
    datasets = {
        "adult": load_adult,
        "communities": load_communities,
    }

    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    N_MAX = 1500   # subsample cap before 80/20 split, for tractable eigendecomp

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
        n_full = len(X)
        if n_full > N_MAX:
            sub = rng.choice(n_full, N_MAX, replace=False)
            X, S, Y = X[sub], S[sub], Y[sub]
        n = len(X)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        X_tr, X_te = X[train_idx], X[test_idx]
        S_tr, S_te = S[train_idx], S[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Empirical hat_lambda_S from the test split (with median-heuristic bandwidth on S)
        sigma_s = median_heuristic(S_te.unsqueeze(1).to(DEVICE))
        hat_lambda_S = empirical_lambda2(S_te, sigma_s)
        print(f"\n{display_name}:  sigma_s (median) = {sigma_s:.4f},  "
              f"hat_lambda_S = {hat_lambda_S:.6e}")

        bounds_mean, lhs_mean, lhs_max, hsics = [], [], [], []

        for lam in lambdas:
            encoder, predictor = train_frhsic(X_tr, S_tr, Y_tr, lam, task)
            encoder.eval()

            with torch.no_grad():
                Z_te = encoder(X_te.to(DEVICE))
                S_te_d = S_te.to(DEVICE)

                # \widehat{HSIC}(Z, S) on the test split, matching kernel choices.
                hsic_val = hsic_biased(Z_te, S_te_d, KERNEL_SIGMA, sigma_s).item()

                # Empirical LHS for f in F_Z; ||f||^2 = 1 for the canonical anchor.
                lhs_avg, lhs_worst, f_norm_sq = empirical_lhs_rkhs(
                    Z_te, S_te_d, KERNEL_SIGMA, n_anchors=64,
                    rng=np.random.RandomState(SEED + 1),
                )

                # RHS = ||f||^2 / hat_lambda_S * widehat_HSIC.
                bound_val = (f_norm_sq / hat_lambda_S) * hsic_val

            bounds_mean.append(bound_val)
            lhs_mean.append(lhs_avg)
            lhs_max.append(lhs_worst)
            hsics.append(hsic_val)
            ratio = bound_val / (lhs_worst + 1e-30)
            print(f"  lambda={lam:6.1f} | HSIC={hsic_val:.4e} | "
                  f"LHS_mean={lhs_avg:.4e} | LHS_max={lhs_worst:.4e} | "
                  f"Bound={bound_val:.4e} | Ratio (bound / LHS_max)={ratio:.2f}x")

        all_results[dname] = {
            "lambdas": lambdas,
            "bounds": bounds_mean,
            "lhs_mean": lhs_mean,
            "lhs_max": lhs_max,
            "hsics": hsics,
            "hat_lambda_S": hat_lambda_S,
            "sigma_s": sigma_s,
            "display_name": display_name,
        }

    # Plot
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, (dname, res) in zip(axes, all_results.items()):
        ax.loglog(res["lambdas"], res["bounds"], "s--", color="tab:red",
                  label=r"Bound: $\|f\|^2/\hat\lambda_S \cdot \widehat{\mathrm{HSIC}}$",
                  linewidth=2)
        ax.loglog(res["lambdas"], res["lhs_max"], "o-", color="tab:blue",
                  label=r"LHS: $\frac{1}{n}\sum_i \hat\delta_{f,i}^2$ (worst $f$)",
                  linewidth=2)
        ax.loglog(res["lambdas"], res["lhs_mean"], "^:", color="tab:green",
                  label=r"LHS: $\frac{1}{n}\sum_i \hat\delta_{f,i}^2$ (mean $f$)",
                  linewidth=1.5, alpha=0.8)
        ax.set_xlabel(r"$\lambda$", fontsize=13)
        ax.set_ylabel("Value (log scale)", fontsize=13)
        ax.set_title(f"{res['display_name']}\n"
                     fr"$\hat\lambda_S = {res['hat_lambda_S']:.3e}$",
                     fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "bound_tightness.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "bound_tightness.png"), dpi=150, bbox_inches="tight")
    print(f"\nBound tightness figure saved to {RESULTS_DIR}/bound_tightness.{{pdf,png}}")

    return all_results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_bound_tightness()
