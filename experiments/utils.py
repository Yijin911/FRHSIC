"""
Shared utilities for FRHSIC experiments.
Architecture and settings match Kong et al. (2025) FREM for fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Matching FREM settings
# ──────────────────────────────────────────────
HIDDEN_DIM = 50       # FREM uses m=50
Z_DIM = 50            # FREM uses 50-dim representation
EPOCHS = 200          # FREM trains for 200 epochs
KERNEL_SIGMA = 1.0    # FREM uses fixed sigma=1.0
N_REPEATS = 5         # FREM repeats 5 times with 80/20 split


# ──────────────────────────────────────────────
# Kernels
# ──────────────────────────────────────────────
def gaussian_kernel(X, Y, sigma):
    dist = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-dist / (2 * sigma ** 2))


def laplacian_kernel(X, Y, sigma):
    dist = torch.cdist(X, Y, p=1)
    return torch.exp(-dist / sigma)


def imq_kernel(X, Y, sigma):
    """Inverse multiquadric kernel."""
    dist = torch.cdist(X, Y, p=2).pow(2)
    return 1.0 / torch.sqrt(1.0 + dist / (sigma ** 2))


KERNELS = {
    "gaussian": gaussian_kernel,
    "laplacian": laplacian_kernel,
    "imq": imq_kernel,
}


def median_heuristic(X):
    """Compute median heuristic bandwidth for kernel."""
    with torch.no_grad():
        dists = torch.cdist(X, X, p=2)
        mask = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
        return torch.median(dists[mask]).item() + 1e-5


# ──────────────────────────────────────────────
# HSIC
# ──────────────────────────────────────────────
def hsic_biased(Z, S, sigma_z=1.0, sigma_s=1.0, kernel_z="gaussian", kernel_s="gaussian"):
    """Biased HSIC estimator: (1/n^2) tr(KHLH)."""
    n = Z.shape[0]
    kfn_z = KERNELS[kernel_z]
    kfn_s = KERNELS[kernel_s]
    K = kfn_z(Z, Z, sigma_z)
    L = kfn_s(S.unsqueeze(1), S.unsqueeze(1), sigma_s)
    H = torch.eye(n, device=Z.device) - 1.0 / n
    return (H @ K @ H * (H @ L @ H)).sum() / (n ** 2)


def hsic_test_threshold(Z, S, sigma_z=1.0, sigma_s=1.0, alpha=0.05):
    """
    HSIC independence test using gamma approximation (Gretton et al. 2005).
    Returns (hsic_value, threshold, reject): reject=True means Z and S are dependent.
    Uses the test statistic n*HSIC_biased and the gamma approximation to its
    null distribution.
    """
    from scipy.stats import gamma as gamma_dist
    n = Z.shape[0]
    with torch.no_grad():
        K = gaussian_kernel(Z, Z, sigma_z)
        L = gaussian_kernel(S.unsqueeze(1), S.unsqueeze(1), sigma_s)
        H = torch.eye(n, device=Z.device) - 1.0 / n
        HKH = H @ K @ H
        HLH = H @ L @ H

        # Biased HSIC
        hsic_val = (HKH * HLH).sum() / (n ** 2)

        # Test statistic is (1/n) * tr(KHLH) = n * hsic_biased
        test_stat = n * hsic_val

        # Mean of test statistic under H0 (Gretton et al. 2005, Eq. 4)
        # E[n*HSIC] under H0 = (1/n) * (1/(n-1)) * tr(K_tilde) * tr(L_tilde)
        # where K_tilde = K with zero diagonal, etc.
        K_tilde = K.clone(); K_tilde.fill_diagonal_(0)
        L_tilde = L.clone(); L_tilde.fill_diagonal_(0)
        nf = float(n)
        mean_H0 = K_tilde.sum() * L_tilde.sum() / (nf * (nf - 1) * (nf - 2) * (nf - 3))

        # Variance under H0 (simplified)
        var_H0 = 2.0 * K_tilde.pow(2).sum() * L_tilde.pow(2).sum()
        var_H0 = var_H0 / (nf * (nf - 1) * (nf - 2) * (nf - 3) * (nf - 4) * (nf - 5))

        mean_H0 = max(mean_H0.item(), 1e-20)
        var_H0 = max(var_H0.item(), 1e-30)

        # Gamma parameters: match mean and variance
        k_param = mean_H0 ** 2 / var_H0
        theta_param = var_H0 / mean_H0

        # Threshold
        threshold = gamma_dist.ppf(1 - alpha, k_param, scale=theta_param)

        return hsic_val.item(), threshold, test_stat.item() > threshold


# ──────────────────────────────────────────────
# GDP estimator
# ──────────────────────────────────────────────
def estimate_gdp(Z, S, predictor, bandwidth=0.1, n_grid=50):
    """
    Estimate GDP = E_S |E[f(Z)|S] - E[f(Z)]| via kernel smoothing.
    """
    with torch.no_grad():
        fZ = predictor(Z).squeeze()
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
# Mutual information estimator (KSG k-NN)
# ──────────────────────────────────────────────
def estimate_mi(Z, S, k=5, max_dim=5):
    """
    Estimate mutual information I(Z; S) using the KSG estimator
    (Kraskov, Stogbauer, Grassberger 2004).
    If Z has more than max_dim dimensions, PCA is used to reduce
    dimensionality while preserving 95% of variance. This balances
    the old 1D projection (too lossy) with full-dim (curse of dimensionality).
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    with torch.no_grad():
        Z_np = Z.cpu().numpy()
        S_np = S.cpu().numpy().reshape(-1, 1)

        if Z_np.ndim == 1:
            Z_np = Z_np.reshape(-1, 1)

        # Reduce dimensionality if needed for reliable KSG estimation
        if Z_np.shape[1] > max_dim:
            Z_centered = Z_np - Z_np.mean(0)
            _, singular_vals, Vt = np.linalg.svd(Z_centered, full_matrices=False)
            # Keep components explaining 95% of variance, up to max_dim
            var_explained = np.cumsum(singular_vals ** 2) / np.sum(singular_vals ** 2)
            n_components = min(max_dim, np.searchsorted(var_explained, 0.95) + 1)
            n_components = max(n_components, 2)  # at least 2 dims
            Z_np = Z_centered @ Vt[:n_components].T

        n = Z_np.shape[0]
        joint = np.hstack([Z_np, S_np])

        # Build trees
        tree_joint = cKDTree(joint)
        tree_z = cKDTree(Z_np)
        tree_s = cKDTree(S_np)

        # For each point, find distance to k-th neighbor in joint space
        # Use Chebyshev (max-norm) distance as in KSG Algorithm 1
        dists, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
        eps = dists[:, -1]  # distance to k-th neighbor (index k since self is index 0)

        # Count neighbors within eps in marginal spaces (vectorized)
        n_z = np.array(tree_z.query_ball_point(Z_np, r=eps + 1e-15, p=np.inf, return_length=True)) - 1
        n_s = np.array(tree_s.query_ball_point(S_np, r=eps + 1e-15, p=np.inf, return_length=True)) - 1

        mi = digamma(k) - np.mean(digamma(n_z + 1) + digamma(n_s + 1)) + digamma(n)
        return max(mi, 0.0)


def estimate_mi_sklearn(Z, S):
    """
    Second MI estimator using sklearn's mutual_info_regression.
    Estimates sum of MI between each Z dimension and S individually
    (lower bound on true MI(Z;S) by data processing inequality).
    Used to corroborate KSG estimates.
    """
    from sklearn.feature_selection import mutual_info_regression
    with torch.no_grad():
        Z_np = Z.cpu().numpy()
        S_np = S.cpu().numpy()
        if Z_np.ndim == 1:
            Z_np = Z_np.reshape(-1, 1)
        # MI between each Z dimension and S, then take max
        # (sum would overcount; max gives strongest single-dim dependence)
        mi_per_dim = mutual_info_regression(Z_np, S_np, random_state=42, n_neighbors=5)
        return float(mi_per_dim.max())


# ──────────────────────────────────────────────
# EIPM estimator (Kong et al. 2025 baseline)
# ──────────────────────────────────────────────
def eipm_weighted(Z, S, sigma_z=1.0, sigma_s=0.5):
    """
    Weighted EIPM estimator from Kong et al. (2025).
    EIPM = (1/n) sum_i MMD(P^hat_{Z|S=S_i}, P^hat_Z)
    with kernel-smoothed weights.
    """
    n = Z.shape[0]
    K = gaussian_kernel(Z, Z, sigma_z)

    S_col = S.unsqueeze(1)
    W = gaussian_kernel(S_col, S_col, sigma_s)

    eipm_val = 0.0
    for i in range(n):
        w = W[i].clone()
        w[i] = 0.0
        w = w / (w.sum() + 1e-10)

        wKw = w @ K @ w
        wK1 = (w @ K).sum() / n
        oneK1 = K.sum() / (n * n)
        mmd2 = wKw - 2 * wK1 + oneK1
        eipm_val += torch.sqrt(torch.clamp(mmd2, min=0.0))

    return (eipm_val / n).item()


# ──────────────────────────────────────────────
# Reg-GDP baseline (Jiang et al. 2022)
# ──────────────────────────────────────────────
def reg_gdp_loss(Z, S, predictor, bandwidth=0.2, n_grid=30):
    """GDP regularization loss (differentiable approximation)."""
    fZ = predictor(Z).squeeze()
    global_mean = fZ.mean()
    s_grid = torch.linspace(S.min().item(), S.max().item(), n_grid, device=Z.device)
    gdp = torch.tensor(0.0, device=Z.device)
    for s in s_grid:
        weights = torch.exp(-((S - s) ** 2) / (2 * bandwidth ** 2))
        weights = weights / (weights.sum() + 1e-10)
        conditional_mean = (weights * fZ).sum()
        gdp = gdp + torch.abs(conditional_mean - global_mean)
    return gdp / n_grid


# ──────────────────────────────────────────────
# Distance Correlation (Szekely et al. 2007)
# ──────────────────────────────────────────────
def _centered_distance_matrix(X):
    """Doubly centered distance matrix for dCor."""
    D = torch.cdist(X, X, p=2)
    row_mean = D.mean(dim=1, keepdim=True)
    col_mean = D.mean(dim=0, keepdim=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def dcov_squared(Z, S):
    """Squared distance covariance (for use as loss — no normalization)."""
    if S.dim() == 1:
        S = S.unsqueeze(1)
    if Z.dim() == 1:
        Z = Z.unsqueeze(1)
    A = _centered_distance_matrix(Z)
    B = _centered_distance_matrix(S)
    n = Z.shape[0]
    return (A * B).sum() / (n * n)


# ──────────────────────────────────────────────
# Models (matching FREM architecture)
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    """2-layer encoder with SELU activation, matching FREM."""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Linear prediction head, matching FREM."""
    def __init__(self, z_dim=Z_DIM, output_dim=1):
        super().__init__()
        self.net = nn.Linear(z_dim, output_dim)

    def forward(self, z):
        return self.net(z)


class Adversary(nn.Module):
    """Adversary for ADV / LAFTR baselines."""
    def __init__(self, z_dim=Z_DIM, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, HIDDEN_DIM),
            nn.SELU(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(self, z):
        return self.net(z)
