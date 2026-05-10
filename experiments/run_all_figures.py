"""
Run all 5 datasets with key lambdas and generate Pareto + MI figures.
Uses 5 lambdas instead of 9 to save time.
"""
import numpy as np
import torch
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from real_data import run_single_dataset, SEED, RESULTS_DIR

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lambdas = [0.1, 1.0, 10.0, 100.0, 500.0]
    datasets = ["adult", "acs_income", "meps", "communities", "compas"]

    all_dataset_results = {}
    for name in datasets:
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"Starting {name}...")
        print(f"{'='*70}", flush=True)
        try:
            res = run_single_dataset(name, lambdas=lambdas)
            if res is not None:
                all_dataset_results[name] = res
            elapsed = time.time() - t0
            print(f">>> {name} done in {elapsed/60:.1f} min", flush=True)
        except Exception as e:
            print(f"Error on {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ── Generate Pareto curves ──
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

    n_ds = len(all_dataset_results)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5))
    if n_ds == 1:
        axes = [axes]

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
        if "Unfair" in results:
            ax.scatter(
                results["Unfair"]["mean"]["gdp"], results["Unfair"]["mean"]["perf"],
                marker="*", s=200, c="black", zorder=10, label="Unfair",
            )
        ax.set_xlabel(r"$\Delta_{\mathrm{GDP}}$ (lower = fairer)", fontsize=11)
        y_label = perf_name if perf_name else "Performance"
        if y_label == "MSE":
            ax.set_ylabel(f"{y_label} (lower = better)", fontsize=11)
        else:
            ax.set_ylabel(f"{y_label} (higher = better)", fontsize=11)
        display = {"adult": "Adult", "acs_income": "ACS Income", "meps": "MEPS",
                    "communities": "Crime", "compas": "COMPAS"}.get(dname, dname)
        ax.set_title(display, fontsize=12)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pareto_curves.pdf"), bbox_inches="tight")
    print(f"\nPareto curves saved.", flush=True)

    # ── Generate MI curves ──
    fig2, axes2 = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5))
    if n_ds == 1:
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
        ax.set_xlabel(r"$\mathrm{MI}(Z, S)$ (lower = fairer)", fontsize=11)
        y_label = perf_name if perf_name else "Performance"
        ax.set_ylabel(y_label, fontsize=11)
        display = {"adult": "Adult", "acs_income": "ACS Income", "meps": "MEPS",
                    "communities": "Crime", "compas": "COMPAS"}.get(dname, dname)
        ax.set_title(display, fontsize=12)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "mi_curves.pdf"), bbox_inches="tight")
    print(f"MI curves saved.", flush=True)
    print("\nALL DONE.", flush=True)
