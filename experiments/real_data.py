"""
Real data experiments for FRHSIC paper.
Experimental settings match Kong et al. (2025) FREM for fair comparison.
Datasets: Adult (classification), Communities & Crime (regression).
Baselines: Unfair, Reg-GDP, FREM, ADV, LAFTR (binned), MMD (binned).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import os
import warnings

from utils import (
    DEVICE, HIDDEN_DIM, Z_DIM, EPOCHS, KERNEL_SIGMA, N_REPEATS,
    Encoder, Predictor, Adversary,
    hsic_biased, hsic_test_threshold, estimate_gdp, estimate_mi, estimate_mi_sklearn,
    median_heuristic, reg_gdp_loss, gaussian_kernel, dcov_squared,
)

warnings.filterwarnings("ignore")

SEED = 42
BATCH_SIZE = 256
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset loading (matching FREM preprocessing)
# ──────────────────────────────────────────────
def load_adult():
    """Adult Income. Target: income >50K. Sensitive: age (continuous)."""
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

    S = df["age"].values.astype(np.float32)
    Y = (df["income"].str.strip() == ">50K").astype(np.float32).values

    # Features: numeric + one-hot encoded categoricals (excluding age and income)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "age"]
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    df_cat = pd.get_dummies(df[cat_cols], columns=cat_cols)
    X = np.hstack([df[numeric_cols].values, df_cat.values]).astype(np.float32)

    return X, S, Y, "Adult", "classification"


def load_communities():
    """Communities & Crime. Target: ViolentCrimesPerPop. Sensitive: racePctBlack."""
    path = os.path.join(DATA_DIR, "communities.csv")
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
        df = pd.read_csv(url, header=None, na_values="?")
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.shape[1] < 128:
        print(f"Warning: Communities dataset has {df.shape[1]} columns, expected 128.")
        return None

    S = df.iloc[:, 6].values.astype(np.float64)
    Y_cont = df.iloc[:, 127].values.astype(np.float64)

    # Feature columns: exclude 0-4 (non-predictive), 6 (sensitive), 127 (target)
    exclude = set(range(5)) | {6, 127}
    feat_indices = [i for i in range(df.shape[1]) if i not in exclude]
    feat_df = df.iloc[:, feat_indices].copy()

    # Drop columns with >20% missing
    thresh = int(0.8 * len(feat_df))
    feat_df = feat_df.dropna(axis=1, thresh=thresh)
    feat_df = feat_df.fillna(feat_df.median())

    # Drop rows where S or Y are NaN
    valid_mask = ~(np.isnan(S) | np.isnan(Y_cont))
    S = S[valid_mask].astype(np.float32)
    Y_cont = Y_cont[valid_mask].astype(np.float32)
    feat_df = feat_df.loc[valid_mask]

    X = feat_df.values.astype(np.float32)

    if X.shape[1] < 5:
        print("Warning: Communities dataset has too few features after cleaning.")
        return None

    return X, S, Y_cont, "Crime", "regression"


def load_law_school():
    """Law School dataset. Target: bar passage. Sensitive: LSAT (continuous)."""
    # Semi-synthetic law school data for reproducibility
    rng = np.random.RandomState(SEED)
    n = 3000
    LSAT = rng.normal(35, 8, n).astype(np.float32)
    GPA = rng.normal(3.0, 0.5, n).astype(np.float32)
    family_income = rng.lognormal(10, 1, n).astype(np.float32)
    study_hours = rng.normal(40, 10, n).astype(np.float32)

    logit = 0.05 * LSAT + 0.8 * GPA + 0.3 * np.log(family_income) / 10 + 0.02 * study_hours - 5
    prob = 1.0 / (1.0 + np.exp(-logit))
    Y = rng.binomial(1, np.clip(prob, 0.01, 0.99)).astype(np.float32)

    S = LSAT
    X = np.column_stack([GPA, family_income / 1e5, study_hours / 100])

    return X, S, Y, "Law School", "classification"


def load_acs_income():
    """ACS Income (folktables). Target: income >50K. Sensitive: age (continuous)."""
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        print("Install folktables: pip install folktables")
        return None

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                root_dir=os.path.join(DATA_DIR, "acs"))
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, labels, group = ACSIncome.df_to_numpy(acs_data)

    # group is race; we use AGEP (age) as continuous sensitive attribute
    # AGEP is the first feature in ACSIncome
    S = features[:, 0].astype(np.float32)  # AGEP
    Y = labels.astype(np.float32)
    # Remove age from features
    X = np.delete(features, 0, axis=1).astype(np.float32)

    # Handle any NaNs
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(S) | np.isnan(Y))
    X, S, Y = X[valid], S[valid], Y[valid]

    # Subsample if too large (for tractable experiments)
    if len(X) > 20000:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), 20000, replace=False)
        X, S, Y = X[idx], S[idx], Y[idx]

    return X, S, Y, "ACS Income", "classification"


def load_german_credit():
    """German Credit. Target: credit risk (good/bad). Sensitive: age (continuous)."""
    path = os.path.join(DATA_DIR, "german.csv")
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        columns = [f"feat_{i}" for i in range(20)] + ["target"]
        df = pd.read_csv(url, header=None, names=columns, sep=r"\s+")
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # Column 12 (0-indexed) is age; target is last column (1=good, 2=bad)
    S = df.iloc[:, 12].values.astype(np.float32)
    Y = (df["target"].values == 1).astype(np.float32)  # 1=good credit

    # Features: all except age (col 12) and target
    feat_cols = [i for i in range(20) if i != 12]
    X_raw = df.iloc[:, feat_cols].copy()

    # Encode categorical columns
    # German Credit: cols 0,2,3,5,6,8,9,11,13,14,16,18,19 are categorical
    cat_cols_idx = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    feat_cols_set = [i for i in range(20) if i != 12]
    cat_in_feat = [feat_cols_set.index(c) for c in cat_cols_idx if c in feat_cols_set]
    num_in_feat = [i for i in range(len(feat_cols_set)) if i not in cat_in_feat]

    X_num = X_raw.iloc[:, num_in_feat].values.astype(np.float32)
    X_cat = pd.get_dummies(X_raw.iloc[:, cat_in_feat].astype(str)).values.astype(np.float32)
    X = np.hstack([X_num, X_cat])

    return X, S, Y, "German Credit", "classification"


def load_compas():
    """COMPAS recidivism. Target: two-year recidivism. Sensitive: age (continuous)."""
    path = os.path.join(DATA_DIR, "compas.csv")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    # Filter as in ProPublica analysis
    df = df[(df["days_b_screening_arrest"] >= -30) & (df["days_b_screening_arrest"] <= 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]

    S = df["age"].values.astype(np.float32)
    Y = df["two_year_recid"].values.astype(np.float32)

    feature_cols = ["juv_fel_count", "juv_misd_count", "juv_other_count",
                    "priors_count", "days_b_screening_arrest",
                    "c_days_from_compas", "c_charge_degree", "sex", "race"]

    df_feat = df[feature_cols].copy()
    cat_cols = ["c_charge_degree", "sex", "race"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    X_num = df_feat[num_cols].values.astype(np.float32)
    X_cat = pd.get_dummies(df_feat[cat_cols]).values.astype(np.float32)
    X = np.hstack([X_num, X_cat])

    # Drop any NaN rows
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(S) | np.isnan(Y))
    X, S, Y = X[valid], S[valid], Y[valid]

    return X, S, Y, "COMPAS", "classification"


def load_meps():
    """MEPS Panel 19 (2015). Target: utilization >= 10 visits. Sensitive: age (continuous).
    Following AIF360 preprocessing of MEPS HC-181."""
    path = os.path.join(DATA_DIR, "meps.csv")
    if not os.path.exists(path):
        import zipfile, io, urllib.request
        # Download MEPS HC-181 (2015 Full Year Consolidated) as SAS transport
        url = "https://meps.ahrq.gov/mepsweb/data_files/pufs/h181ssp.zip"
        print("Downloading MEPS HC-181...")
        resp = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
            sas_name = [f for f in z.namelist() if f.lower().endswith('.ssp')][0]
            with z.open(sas_name) as sas_file:
                df = pd.read_sas(io.BytesIO(sas_file.read()), format='xport')
        # Standardize column names to uppercase
        df.columns = [c.upper() for c in df.columns]
        df.to_csv(path, index=False)
        print(f"MEPS saved: {len(df)} rows, {df.shape[1]} cols")
    else:
        df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.upper() for c in df.columns]

    # Compute utilization: sum of office-based, outpatient, ER, inpatient, home health visits
    util_cols = ['OBTOTV15', 'OPTOTV15', 'ERTOT15', 'IPNGTD15', 'HHTOTD15']
    available = [c for c in util_cols if c in df.columns]
    if not available:
        # Try without year suffix
        util_cols_alt = ['OBTOTV', 'OPTOTV', 'ERTOT', 'IPNGTD', 'HHTOTD']
        available = [c for c in util_cols_alt if c in df.columns]
    if not available:
        print("Warning: could not find utilization columns in MEPS data.")
        return None

    df['UTILIZATION'] = df[available].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    Y = (df['UTILIZATION'] >= 10).astype(np.float32).values

    # Sensitive: age
    age_col = 'AGE15X' if 'AGE15X' in df.columns else 'AGELAST' if 'AGELAST' in df.columns else 'AGE'
    if age_col not in df.columns:
        # Try any column starting with AGE
        age_candidates = [c for c in df.columns if c.startswith('AGE')]
        age_col = age_candidates[0] if age_candidates else None
    if age_col is None:
        print("Warning: could not find age column in MEPS data.")
        return None
    S = pd.to_numeric(df[age_col], errors='coerce').values.astype(np.float32)

    # Features: demographics + health conditions + insurance
    feature_cols = []
    # Demographics
    for prefix in ['SEX', 'RACE', 'MARR', 'EDUC', 'REGION', 'POVCAT', 'INSCOV']:
        candidates = [c for c in df.columns if c.startswith(prefix)]
        if candidates:
            feature_cols.append(candidates[0])
    # Health conditions (priority diagnoses)
    for prefix in ['DIABDX', 'ARTHDX', 'ASTHDX', 'CANCERDX', 'MIDX', 'STRKDX',
                    'HIBPDX', 'CHDDX', 'ANGIDX', 'OHRTDX']:
        candidates = [c for c in df.columns if c.startswith(prefix)]
        if candidates:
            feature_cols.append(candidates[0])
    # Income
    for prefix in ['TTLP', 'FAMINC']:
        candidates = [c for c in df.columns if c.startswith(prefix)]
        if candidates:
            feature_cols.append(candidates[0])

    if len(feature_cols) < 5:
        print(f"Warning: only found {len(feature_cols)} feature columns in MEPS.")
        return None

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

    # Remove rows with NaN or negative codes (MEPS uses -1, -7, -8, -9 for missing)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(S) | np.isnan(Y) | (S < 0))
    for j in range(X.shape[1]):
        valid &= (X[:, j] >= -0.5)  # exclude negative sentinel values
    X, S, Y = X[valid], S[valid], Y[valid]

    # Subsample if too large
    if len(X) > 20000:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), 20000, replace=False)
        X, S, Y = X[idx], S[idx], Y[idx]

    return X, S, Y, "MEPS", "classification"


DATASETS = {
    "adult": load_adult,
    "communities": load_communities,
    "law_school": load_law_school,
    "acs_income": load_acs_income,
    "german_credit": load_german_credit,
    "compas": load_compas,
    "meps": load_meps,
}


# ──────────────────────────────────────────────
# Training methods
# ──────────────────────────────────────────────

def _gpu_batch_iter(X_g, S_g, Y_g, batch_size, shuffle=True):
    """Iterate over (X, S, Y) in mini-batches with all data resident on GPU.

    Replaces DataLoader+TensorDataset where the dataset fits in GPU memory.
    Avoids per-batch host->device transfers (the main overhead at n~30k).
    """
    n = X_g.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=X_g.device)
    else:
        perm = torch.arange(n, device=X_g.device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield X_g[idx], S_g[idx], Y_g[idx]


def train_frhsic(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3,
                  adaptive_bandwidth=True):
    """Our method: FRHSIC. Uses median heuristic bandwidth by default."""
    X_g, S_g, Y_g = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    # Pre-compute sigma_s from data (fixed throughout training)
    if adaptive_bandwidth:
        with torch.no_grad():
            sigma_s = median_heuristic(S_g[:min(1000, len(S_g))].unsqueeze(1))
    else:
        sigma_s = KERNEL_SIGMA

    for epoch in range(epochs):
        for X_b, S_b, Y_b in _gpu_batch_iter(X_g, S_g, Y_g, batch_size):
            Z_b = encoder(X_b)

            # Adaptive sigma_z: recompute periodically from current representations
            if adaptive_bandwidth and (epoch % 20 == 0):
                with torch.no_grad():
                    sigma_z = median_heuristic(Z_b[:min(256, len(Z_b))])
            elif not adaptive_bandwidth:
                sigma_z = KERNEL_SIGMA

            loss = loss_fn(predictor(Z_b).squeeze(), Y_b) + lam * hsic_biased(Z_b, S_b, sigma_z, sigma_s)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def train_reg_gdp(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: Reg-GDP (Jiang et al. 2022)."""
    X_g, S_g, Y_g = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    for _ in range(epochs):
        for X_b, S_b, Y_b in _gpu_batch_iter(X_g, S_g, Y_g, batch_size):
            Z_b = encoder(X_b)
            loss = loss_fn(predictor(Z_b).squeeze(), Y_b) + lam * reg_gdp_loss(Z_b, S_b, predictor)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def train_frem(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: FREM (Kong et al. 2025) — weighted EIPM with MMD.

    The anchor-loop is vectorized: instead of iterating over anchors and
    computing scalar w_i @ K @ w_i, we form a (n_anchors, n) weight matrix
    and compute all per-anchor MMD^2 values in a single batched matmul.
    """
    X_g, S_g, Y_g = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    # FREM uses fixed bandwidth gamma for kernel smoothing on S
    # and fixed sigma=1.0 for the MMD kernel on Z
    gamma = 0.5

    for _ in range(epochs):
        for X_b, S_b, Y_b in _gpu_batch_iter(X_g, S_g, Y_g, batch_size):
            Z_b = encoder(X_b)
            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)

            n = Z_b.shape[0]
            K = gaussian_kernel(Z_b, Z_b, KERNEL_SIGMA)            # (n, n)
            W = gaussian_kernel(S_b.unsqueeze(1), S_b.unsqueeze(1), gamma)  # (n, n)
            W_loo = W - torch.diag_embed(torch.diagonal(W))
            W_loo = W_loo / (W_loo.sum(dim=1, keepdim=True) + 1e-10)

            uniform = torch.full((n,), 1.0 / n, device=DEVICE)
            oneK1 = uniform @ K @ uniform                          # scalar
            uniform_K = uniform @ K                                # (n,)

            n_anchors = min(n, 32)
            anchor_idx = torch.randperm(n, device=DEVICE)[:n_anchors]
            W_anchors = W_loo[anchor_idx]                          # (n_anchors, n)
            # term1[i] = w_i @ K @ w_i, computed as diag(W_anchors @ K @ W_anchors.T)
            WK = W_anchors @ K                                     # (n_anchors, n)
            term1 = (WK * W_anchors).sum(dim=1)                    # (n_anchors,)
            term2 = (W_anchors @ uniform_K)                        # (n_anchors,)
            mmd2 = term1 - 2.0 * term2 + oneK1
            eipm = torch.sqrt(torch.clamp(mmd2, min=1e-10)).mean()

            loss = loss_pred + lam * eipm
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def train_adv(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: ADV — adversarial approach for continuous S (Zhang et al. 2018)."""
    X_g, S_g, Y_g = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    adversary = Adversary(z_dim=z_dim, output_dim=1).to(DEVICE)

    opt_enc = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    opt_adv = optim.Adam(adversary.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    mse = nn.MSELoss()

    for _ in range(epochs):
        for X_b, S_b, Y_b in _gpu_batch_iter(X_g, S_g, Y_g, batch_size):
            Z_b = encoder(X_b)

            # Adversary step: predict S from Z
            s_pred = adversary(Z_b.detach()).squeeze()
            loss_adv = mse(s_pred, S_b)
            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()

            # Encoder + predictor step: predict Y, fool adversary
            Z_b = encoder(X_b)
            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)
            s_pred = adversary(Z_b).squeeze()
            loss_fair = -mse(s_pred, S_b)
            loss = loss_pred + lam * loss_fair
            opt_enc.zero_grad()
            loss.backward()
            opt_enc.step()

    return encoder, predictor


def train_mmd_binned(X, S, Y, lam, task, n_bins=10, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: MMD with binned S (Deka & Sutherland 2023)."""
    S_np = S.numpy()
    bins = np.quantile(S_np, np.linspace(0, 1, n_bins + 1))
    S_binned = torch.tensor(np.digitize(S_np, bins[1:-1]), dtype=torch.long)

    X_g, Sb_g, Y_g = X.to(DEVICE), S_binned.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    for _ in range(epochs):
        for X_b, Sb_b, Y_b in _gpu_batch_iter(X_g, Sb_g, Y_g, batch_size):
            Z_b = encoder(X_b)
            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)

            mmd_total = torch.tensor(0.0, device=DEVICE)
            mu_all = Z_b.mean(0)
            for b in range(n_bins):
                mask = Sb_b == b
                if mask.sum() > 1:
                    mu_b = Z_b[mask].mean(0)
                    mmd_total = mmd_total + (mu_b - mu_all).pow(2).sum()

            loss = loss_pred + lam * mmd_total
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def train_laftr(X, S, Y, lam, task, n_bins=4, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: LAFTR with binned S (Madras et al. 2018)."""
    S_np = S.numpy()
    bins = np.quantile(S_np, np.linspace(0, 1, n_bins + 1))
    S_binned = torch.tensor(np.digitize(S_np, bins[1:-1]), dtype=torch.long)

    X_g, Sb_g, Y_g = X.to(DEVICE), S_binned.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    adversary_mc = nn.Sequential(
        nn.Linear(z_dim, HIDDEN_DIM), nn.SELU(), nn.Linear(HIDDEN_DIM, n_bins)
    ).to(DEVICE)

    opt_enc = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    opt_adv = optim.Adam(adversary_mc.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_b, Sb_b, Y_b in _gpu_batch_iter(X_g, Sb_g, Y_g, batch_size):
            Z_b = encoder(X_b)

            adv_logits = adversary_mc(Z_b.detach())
            loss_adv = ce(adv_logits, Sb_b)
            opt_adv.zero_grad()
            loss_adv.backward()
            opt_adv.step()

            Z_b = encoder(X_b)
            loss_pred = loss_fn(predictor(Z_b).squeeze(), Y_b)
            adv_logits = adversary_mc(Z_b)
            loss_fair = -ce(adv_logits, Sb_b)
            loss = loss_pred + lam * loss_fair
            opt_enc.zero_grad()
            loss.backward()
            opt_enc.step()

    return encoder, predictor


def train_dcor(X, S, Y, lam, task, z_dim=Z_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3):
    """Baseline: Distance Covariance regularizer (Szekely et al. 2007)."""
    X_g, S_g, Y_g = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)

    encoder = Encoder(X.shape[1], z_dim=z_dim).to(DEVICE)
    predictor = Predictor(z_dim=z_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    for _ in range(epochs):
        for X_b, S_b, Y_b in _gpu_batch_iter(X_g, S_g, Y_g, batch_size):
            Z_b = encoder(X_b)
            loss = loss_fn(predictor(Z_b).squeeze(), Y_b) + lam * dcov_squared(Z_b, S_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return encoder, predictor


def select_lambda_adaptive(X_train, S_train, Y_train, X_val, S_val, task,
                           lambda_candidates=None, alpha_test=0.05):
    """
    Select lambda via HSIC independence test.
    Returns smallest lambda for which HSIC test fails to reject H0: Z perp S.
    """
    if lambda_candidates is None:
        lambda_candidates = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    for lam in lambda_candidates:
        encoder, predictor = train_frhsic(X_train, S_train, Y_train, lam, task)
        encoder.eval()
        with torch.no_grad():
            Z_val = encoder(X_val.to(DEVICE))
            sigma_z = median_heuristic(Z_val[:min(500, len(Z_val))])
            sigma_s = median_heuristic(S_val[:min(500, len(S_val))].unsqueeze(1).to(DEVICE))
            hsic_val, threshold, reject = hsic_test_threshold(
                Z_val, S_val.to(DEVICE), sigma_z, sigma_s, alpha=alpha_test)
        print(f"  lambda={lam:6.1f} | HSIC={hsic_val:.6f} | threshold={threshold:.6f} | "
              f"reject={reject}")
        if not reject:
            print(f"  -> Selected lambda={lam} (independence not rejected at alpha={alpha_test})")
            return lam, encoder, predictor
    print(f"  -> No lambda achieved independence; using largest: {lambda_candidates[-1]}")
    return lambda_candidates[-1], encoder, predictor


METHODS = {
    "Unfair": lambda X, S, Y, lam, task, **kw: train_frhsic(X, S, Y, 0.0, task, **kw),
    "FRHSIC (Ours)": train_frhsic,
    "Reg-GDP": train_reg_gdp,
    "FREM": train_frem,
    "ADV": train_adv,
    "MMD (binned)": train_mmd_binned,
    "LAFTR (binned)": train_laftr,
    "dCor": train_dcor,
}


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_model(encoder, predictor, X, S, Y, task):
    with torch.no_grad():
        X_d, S_d, Y_d = X.to(DEVICE), S.to(DEVICE), Y.to(DEVICE)
        Z = encoder(X_d)
        logits = predictor(Z).squeeze()

        if task == "classification":
            preds = (torch.sigmoid(logits) > 0.5).float()
            perf = (preds == Y_d).float().mean().item()
            perf_name = "Acc"
        else:
            perf = nn.MSELoss()(logits, Y_d).item()
            perf_name = "MSE"

        gdp = estimate_gdp(Z, S_d, predictor)
        mi = estimate_mi(Z, S_d)
        mi_sk = estimate_mi_sklearn(Z, S_d)
    return {"perf": perf, "perf_name": perf_name, "gdp": gdp, "mi": mi, "mi_sk": mi_sk}


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────
def run_single_dataset(dataset_name, lambdas=None):
    if lambdas is None:
        lambdas = [0.1, 1.0, 10.0, 50.0, 100.0]

    load_fn = DATASETS[dataset_name]
    result = load_fn()
    if result is None:
        print(f"Skipping {dataset_name}: data loading failed.")
        return None
    X_np, S_np, Y_np, display_name, task = result

    # Preprocessing: the min-max scalers for X and S are fit on the training
    # split only and then applied to both train and test (see the per-split
    # scaling below), so no test-split statistics enter training.
    Y_np = np.asarray(Y_np)

    print(f"\n{'='*70}")
    print(f"Dataset: {display_name} (n={len(X_np)}, d={X_np.shape[1]}, task={task})")
    print(f"{'='*70}")

    all_results = {}

    for method_name, train_fn in METHODS.items():
        lam_to_use = lambdas if method_name != "Unfair" else [0.0]

        for lam in lam_to_use:
            key = f"{method_name} (lam={lam})" if method_name != "Unfair" else "Unfair"
            repeat_results = []

            for rep in range(N_REPEATS):  # 5 repeats
                # 80/20 split (matching FREM)
                rng = np.random.RandomState(SEED + rep)
                n = len(X_np)
                idx = rng.permutation(n)
                n_train = int(0.8 * n)
                train_idx, test_idx = idx[:n_train], idx[n_train:]

                X_tr_raw, X_te_raw = X_np[train_idx], X_np[test_idx]
                S_tr_raw, S_te_raw = S_np[train_idx], S_np[test_idx]
                Y_tr_np, Y_te_np = Y_np[train_idx], Y_np[test_idx]

                # Fit scalers on TRAIN split only, then apply to train and test.
                x_scaler = MinMaxScaler()
                X_tr_np = x_scaler.fit_transform(X_tr_raw)
                X_te_np = x_scaler.transform(X_te_raw)

                s_min = S_tr_raw.min()
                s_max = S_tr_raw.max()
                s_den = (s_max - s_min) + 1e-8
                S_tr_np = (S_tr_raw - s_min) / s_den
                S_te_np = (S_te_raw - s_min) / s_den

                X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
                X_te = torch.tensor(X_te_np, dtype=torch.float32)
                S_tr = torch.tensor(S_tr_np, dtype=torch.float32)
                S_te = torch.tensor(S_te_np, dtype=torch.float32)
                Y_tr = torch.tensor(Y_tr_np, dtype=torch.float32)
                Y_te = torch.tensor(Y_te_np, dtype=torch.float32)

                encoder, predictor = train_fn(X_tr, S_tr, Y_tr, lam, task)
                metrics = evaluate_model(encoder, predictor, X_te, S_te, Y_te, task)
                repeat_results.append(metrics)

            avg = {k: np.mean([r[k] for r in repeat_results]) for k in repeat_results[0] if k != "perf_name"}
            std = {k: np.std([r[k] for r in repeat_results]) for k in repeat_results[0] if k != "perf_name"}
            perf_name = repeat_results[0]["perf_name"]
            all_results[key] = {"mean": avg, "std": std, "perf_name": perf_name}
            print(f"  {key:35s} | {perf_name}: {avg['perf']:.4f}+/-{std['perf']:.4f} | "
                  f"GDP: {avg['gdp']:.4f}+/-{std['gdp']:.4f} | MI: {avg['mi']:.4f} | MI_sk: {avg['mi_sk']:.4f}")

    return all_results


def run_all():
    import matplotlib.pyplot as plt

    all_dataset_results = {}
    for name in ["adult", "communities", "acs_income", "meps", "compas"]:
        try:
            res = run_single_dataset(name, lambdas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
            if res is not None:
                all_dataset_results[name] = res
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Plot Pareto frontiers (GDP vs performance)
    fig, axes = plt.subplots(1, len(all_dataset_results), figsize=(7 * len(all_dataset_results), 5))
    if len(all_dataset_results) == 1:
        axes = [axes]

    colors = {
        "FRHSIC (Ours)": "tab:red", "Reg-GDP": "tab:blue", "FREM": "tab:green",
        "ADV": "tab:purple", "MMD (binned)": "tab:orange", "LAFTR (binned)": "tab:brown",
        "dCor": "tab:cyan",
    }
    markers = {
        "FRHSIC (Ours)": "o", "Reg-GDP": "s", "FREM": "D",
        "ADV": "^", "MMD (binned)": "v", "LAFTR (binned)": "<",
        "dCor": ">",
    }

    for ax, (dname, results) in zip(axes, all_dataset_results.items()):
        perf_name = None
        for method in colors:
            perfs, gdps = [], []
            for key, val in results.items():
                if key.startswith(method):
                    perfs.append(val["mean"]["perf"])
                    gdps.append(val["mean"]["gdp"])
                    perf_name = val["perf_name"]
            if perfs:
                order = np.argsort(gdps)
                ax.plot(
                    [gdps[i] for i in order], [perfs[i] for i in order],
                    marker=markers.get(method, "o"), color=colors.get(method, "gray"),
                    linewidth=2, markersize=7, label=method,
                )
        # Add unfair baseline
        if "Unfair" in results:
            ax.scatter(
                results["Unfair"]["mean"]["gdp"], results["Unfair"]["mean"]["perf"],
                marker="*", s=200, c="black", zorder=10, label="Unfair",
            )
        ax.set_xlabel(r"$\Delta_{\mathrm{GDP}}$ (lower = fairer)", fontsize=12)
        y_label = perf_name if perf_name else "Performance"
        if y_label == "MSE":
            ax.set_ylabel(f"{y_label} (lower = better)", fontsize=12)
        else:
            ax.set_ylabel(f"{y_label} (higher = better)", fontsize=12)
        ax.set_title(dname.replace("_", " ").upper(), fontsize=13)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pareto_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "pareto_curves.png"), dpi=150, bbox_inches="tight")
    print(f"\nPareto curves saved to {RESULTS_DIR}/pareto_curves.{{pdf,png}}")

    # Also plot MI(Z,S) vs performance (matching FREM Fig. 5)
    fig2, axes2 = plt.subplots(1, len(all_dataset_results), figsize=(7 * len(all_dataset_results), 5))
    if len(all_dataset_results) == 1:
        axes2 = [axes2]

    for ax, (dname, results) in zip(axes2, all_dataset_results.items()):
        perf_name = None
        for method in colors:
            perfs, mis = [], []
            for key, val in results.items():
                if key.startswith(method):
                    perfs.append(val["mean"]["perf"])
                    mis.append(val["mean"]["mi"])
                    perf_name = val["perf_name"]
            if perfs:
                order = np.argsort(mis)
                ax.plot(
                    [mis[i] for i in order], [perfs[i] for i in order],
                    marker=markers.get(method, "o"), color=colors.get(method, "gray"),
                    linewidth=2, markersize=7, label=method,
                )
        if "Unfair" in results:
            ax.scatter(
                results["Unfair"]["mean"]["mi"], results["Unfair"]["mean"]["perf"],
                marker="*", s=200, c="black", zorder=10, label="Unfair",
            )
        ax.set_xlabel(r"$\mathrm{MI}(Z, S)$ (lower = fairer)", fontsize=12)
        y_label = perf_name if perf_name else "Performance"
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(dname.replace("_", " ").upper(), fontsize=13)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "mi_curves.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(RESULTS_DIR, "mi_curves.png"), dpi=150, bbox_inches="tight")
    print(f"MI curves saved to {RESULTS_DIR}/mi_curves.{{pdf,png}}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_all()
