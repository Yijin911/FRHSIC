# FRHSIC: A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes

This repository contains the code, data, and pre-computed results for the paper **"A Joint-Distribution Route to Fair Representations with Continuous Sensitive Attributes"** (Ni & Huo, submitted to *SIAM Journal on Mathematics of Data Science*).

## Summary

Fair representation learning with a continuous sensitive attribute is often operationalized through conditional-distribution discrepancies $\mathbb{E}_S[d(P_{Z|S}, P_Z)]$, as in Generalized Demographic Parity (GDP) and the Expectation of Integral Probability Metrics (EIPM). This conditional route is conceptually natural but requires smoothing over $S$, bandwidth choices, leave-one-out corrections, and repeated conditional-distribution estimates.

This paper proposes a **joint-distribution route**. Rather than estimating the conditional family $\{P_{Z|S=s}\}_s$, FRHSIC penalizes the discrepancy between the joint $P_{Z, S}$ and the product $P_Z \otimes P_S$ using the Hilbert–Schmidt Independence Criterion (HSIC). Under characteristic kernels, the joint criterion is zero-equivalent to EIPM and targets the same population independence condition; beyond zero-level equivalence, empirical HSIC controls a projected empirical conditional-gap surrogate through a spectral inequality.

## Repository layout

```
.
├── main.pdf                       # Main paper (26 pages)
├── supplement.pdf                 # Supplementary material (19 pages)
├── requirements.txt
├── experiments/                   # Training and analysis scripts
│   ├── real_data.py               # Training loops, dataset loaders, evaluation
│   ├── utils.py                   # HSIC / EIPM / dCor / MMD kernels and metrics
│   ├── run_single_method.py       # Canonical per-(dataset, method) sweep launcher
│   ├── run_all_figures.py         # Convenience driver: all datasets in sequence
│   ├── regen_pareto_mi.py         # Rebuild Pareto/MI figures from result txts
│   ├── compute_pareto_summary.py  # Pareto-summary table (lowest in-band GDP)
│   ├── synthetic.py               # Synthetic validation (Figure 2)
│   ├── runtime.py                 # Per-epoch runtime comparison (Figure 5)
│   ├── convergence.py             # Estimator convergence-rate experiment
│   ├── bound_tightness.py         # Spectral-bound tightness (supplement)
│   ├── lambda2_vs_sigma.py        # Spectral-gap vs bandwidth (supplement)
│   ├── highdim_S.py               # High-dimensional sensitive attribute (supplement)
│   ├── lambda_selection.py        # Validation-based lambda selection (supplement)
│   ├── equal_opportunity.py       # Equal Opportunity extension (supplement)
│   ├── multi_sensitive.py         # Multiple sensitive attributes (supplement)
│   ├── transfer.py                # Transfer to held-out heads (supplement)
│   └── ablation.py                # Kernel/architecture ablations
├── data/                          # Local datasets
│   ├── adult.csv
│   ├── communities.csv
│   └── compas.csv                 # (ACS Income via folktables; MEPS auto-downloaded)
├── figures/                       # Generated figures (PDF + PNG previews)
└── results/                       # Pre-computed sweep results (40 files)
    └── results_split_<dataset>_<method>.txt
```

## Methods compared

| Method            | Type                       | Reference               |
|-------------------|----------------------------|-------------------------|
| **FRHSIC (Ours)** | Joint-distribution route   | This paper              |
| Unfair            | No fairness penalty        | —                       |
| FREM              | EIPM (conditional route)   | Kong, Kim, Kim (2025)   |
| Reg-GDP           | GDP (conditional route)    | Jiang et al. (2022)     |
| ADV               | Adversarial / kHGR proxy   | Grari et al. (2022)     |
| MMD (binned)      | Two-sample MMD on bins     | Standard                |
| LAFTR (binned)    | Adversarial latent fair    | Madras et al. (2018)    |
| dCor              | Distance covariance        | Székely et al. (2007)   |

## Datasets

- **Adult** (UCI): $n \approx 30{,}000$, classification, sensitive attribute = age. Local CSV.
- **Communities & Crime** (UCI): $n \approx 2{,}000$, regression, sensitive attribute = % African-American population. Local CSV.
- **COMPAS** (ProPublica): $n \approx 6{,}000$, classification, sensitive attribute = age (ordinal integers). Local CSV.
- **ACS Income** (Folktables, 2018 CA): $n \approx 200{,}000$, classification, sensitive attribute = age. Downloaded via `folktables` on first run.
- **MEPS HC-181** (AHRQ, 2015): $n \approx 30{,}000$, classification (utilization $\geq 10$), sensitive attribute = age. Auto-downloaded on first run.

Continuous sensitive attributes are kept in raw scale and min–max scaled with the scaler fit on the training split only (no test-set leakage). See Appendix SM5 of the supplement.

## Reproducing the results

### Requirements

- Python 3.10 or 3.11
- PyTorch 2.0+ (GPU recommended; A100 tested)
- NumPy, SciPy, scikit-learn, pandas, matplotlib, folktables

```bash
pip install -r requirements.txt
```

### Pre-computed results

The `results/` directory holds the 40 result files (5 datasets × 8 methods) used to generate Table 1 and the Pareto/MI figures in the paper. Each file contains 9 lines (one per $\lambda$) with `Acc` (or `MSE`) ± std, GDP ± std, and MI averaged over 5 random 80/20 splits.

To regenerate the Pareto and MI figures from these files without re-running training:

```bash
python experiments/regen_pareto_mi.py
python experiments/compute_pareto_summary.py
```

### Per-(dataset, method) sweep

To reproduce a single column of Table 1:

```bash
python experiments/run_single_method.py <dataset> <method> [<gpu_id>]
# dataset: adult | communities | acs_income | meps | compas
# method:  FRHSIC_Ours | FREM | Reg-GDP | ADV | MMD_binned | LAFTR_binned | dCor | Unfair
```

Writes `results_split_<dataset>_<method>.txt` to the working directory. Roughly 20–90 minutes per call on a single A100, depending on dataset size.

### All datasets at once

```bash
python experiments/run_all_figures.py
```

Iterates over all five datasets in sequence and writes both per-method results and the Pareto/MI figures. Total runtime: roughly 12–24 hours on a single A100.

### Synthetic validation (Figure 2)

```bash
python experiments/synthetic.py
```

Output saved to `figures/synthetic_results.pdf`. Runs in ~1 minute on CPU.

### Runtime comparison (Figure 5)

```bash
python experiments/runtime.py
```

### Supplementary experiments

Each script in `experiments/` is self-contained. See per-script docstrings for outputs.

## Reproducibility settings

- Random seed: `SEED = 42` (see `experiments/real_data.py` and `experiments/utils.py`); 5 random restarts use seeds 42, 43, …, 46.
- Train/test split: 80/20, 5 random restarts.
- Scalers: `MinMaxScaler` fit on training split only.
- Optimizer: Adam, $\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, weight decay 0.
- Encoder: 2-layer MLP, hidden dim 50, SELU activation, output dim 50.
- Predictor: linear $50 \to 1$.
- Batch size: 256. Epochs: 200.
- Kernel bandwidths:
  - **FRHSIC**: median heuristic on the centered features.
  - **FREM**: $\sigma_Z = 1.0$, smoothing bandwidth $\gamma = 0.5$ on $S$.
  - **Reg-GDP**: $\sigma_Z = 1.0$, Nadaraya–Watson smoothing bandwidth $0.2$ on $S$.
  - **dCor**: no bandwidth parameter.
- FREM EIPM uses 32-anchor subsampling per batch for tractability.

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
