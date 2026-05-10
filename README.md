# FRHSIC: A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes

This repository contains the code and data for the paper **"A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes"** (Ni & Huo, submitted to *SIAM Journal on Mathematics of Data Science*).

## Summary

Fair representation learning with a continuous sensitive attribute is often operationalized through conditional-distribution discrepancies $\mathbb{E}_S[d(P_{Z|S}, P_Z)]$, as in Generalized Demographic Parity (GDP) and the Expectation of Integral Probability Metrics (EIPM). This conditional route is conceptually natural but requires smoothing over $S$, bandwidth choices, leave-one-out corrections, and repeated conditional-distribution estimates.

This paper proposes a **joint-distribution route**. Rather than estimating the conditional family $\{P_{Z|S=s}\}_s$, FRHSIC penalizes the discrepancy between the joint $P_{Z, S}$ and the product $P_Z \otimes P_S$ using the Hilbert–Schmidt Independence Criterion (HSIC). Under characteristic kernels, the joint criterion is zero-equivalent to EIPM and targets the same population independence condition; beyond zero-level equivalence, empirical HSIC controls a projected empirical conditional-gap surrogate through a spectral inequality.

## Repository layout

```
.
├── main.pdf                # Main paper (26 pages)
├── supplement.pdf          # Supplementary material (19 pages: proofs, extra experiments)
├── experiments/            # Training and analysis scripts
│   ├── real_data.py        # Core training loops for all methods + dataset loaders
│   ├── run_all_figures.py  # Driver: runs all real datasets and writes results to txt
│   ├── synthetic.py        # Synthetic-data theory-validation experiment (Figure 2)
│   ├── runtime.py          # Computational efficiency comparison (Figure 5)
│   ├── bound_tightness.py  # Empirical tightness of Theorem 3.4 (supplement)
│   ├── lambda2_vs_sigma.py # Spectral-gap sensitivity to kernel bandwidth (supplement)
│   ├── highdim_S.py        # High-dimensional sensitive attribute experiment (supplement)
│   ├── lambda_selection.py # Validation-based lambda selection (supplement)
│   ├── equal_opportunity.py# Equal Opportunity extension (supplement)
│   ├── multi_sensitive.py  # Multiple sensitive attributes (supplement)
│   ├── ablation.py         # Kernel-choice and architecture ablations
│   └── convergence.py      # Estimator convergence-rate verification
├── data/                   # Datasets (Adult, Communities/Crime, COMPAS)
│   ├── adult.csv
│   ├── communities.csv
│   └── compas.csv
└── figures/                # Generated figures (PDF, referenced by main + supplement)
```

## Methods compared

| Method                | Type                       | Reference               |
|-----------------------|----------------------------|-------------------------|
| **FRHSIC (Ours)**     | Joint-distribution route   | This paper              |
| Unfair                | No fairness penalty        | —                       |
| FREM                  | EIPM (conditional route)   | Kong, Kim, Kim (2025)   |
| Reg-GDP               | GDP (conditional route)    | Jiang et al. (2022)     |
| ADV                   | Adversarial / kHGR proxy   | Grari et al. (2022)     |
| MMD (binned)          | Two-sample MMD on bins     | Standard                |
| LAFTR (binned)        | Adversarial latent fair    | Madras et al. (2018)    |
| dCor                  | Distance covariance        | Székely et al. (2007)   |

## Datasets

- **Adult** (UCI): $n \approx 30{,}000$, classification, sensitive attribute = age.
- **Communities and Crime** (UCI): $n \approx 2{,}000$, regression, sensitive attribute = % African-American population.
- **COMPAS** (ProPublica): $n \approx 6{,}000$, classification, sensitive attribute = age (ordinal integer values).

Continuous sensitive attributes are kept in raw scale and min–max scaled fit on the training split only (no test-set leakage).

## Reproducing the main results

### Requirements

- Python 3.10 or 3.11
- PyTorch 2.0 or later (GPU recommended; A100 or similar tested)
- NumPy, SciPy, scikit-learn, pandas, matplotlib

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib
```

### Driver

To reproduce the main Table 2 and the Pareto-frontier figures across all real datasets:

```bash
cd experiments
python run_all_figures.py
```

This iterates over Adult, ACS Income, MEPS, Communities/Crime, and COMPAS, sweeping $\lambda \in \{0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500\}$ across 5 random 80/20 train–test splits per dataset and writing results to `../results_all_figures.txt`. Total runtime: roughly 12–24 hours on a single A100.

To reproduce a single dataset more quickly:

```python
from experiments.real_data import run_single_dataset
run_single_dataset("adult", lambdas=[0.01, 1.0, 10.0, 100.0, 500.0])
```

### Synthetic validation (Figure 2)

```bash
cd experiments
python synthetic.py
```

Output is saved to `../figures/synthetic_results.pdf`. Runs in ~1 minute on CPU.

### Runtime comparison (Figure 5)

```bash
cd experiments
python runtime.py
```

### Supplementary experiments

Each script in `experiments/` is self-contained. See the per-script docstrings.

## Reproducibility settings

- Random seed: `SEED = 42` (see `experiments/real_data.py` and `experiments/utils.py`).
- Train/test split: 80/20 stratified (for classification) / random (for regression), 5 random restarts.
- Optimizer: Adam with $\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, weight decay 0.
- Encoder architecture: 2-layer MLP, hidden dim 50, SELU activation, output dim 50.
- Predictor architecture: linear $50 \to 1$.
- Batch size: 256. Epochs: 200.
- Kernel bandwidths:
  - **FRHSIC**: median heuristic on the centered features.
  - **FREM**: $\sigma_Z = 1.0$, smoothing bandwidth $\gamma = 0.5$ on $S$.
  - **Reg-GDP**: $\sigma_Z = 1.0$, Nadaraya–Watson smoothing bandwidth $0.2$ on $S$.
  - **dCor**: no bandwidth parameter.
- FREM EIPM uses 32-anchor subsampling per batch for computational tractability.

Full reproducibility details are in **Appendix SM5** of the supplement.

## Citation

A BibTeX entry will be added upon acceptance. For now, please cite as:

> Yijin Ni and Xiaoming Huo. A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes. Manuscript submitted to SIAM Journal on Mathematics of Data Science, 2026.

## License

MIT License (see `LICENSE`).

## Contact

- Yijin Ni — yni64@gatech.edu
- Xiaoming Huo — huo@gatech.edu

H. Milton Stewart School of Industrial and Systems Engineering, Georgia Institute of Technology, Atlanta, GA, USA.
