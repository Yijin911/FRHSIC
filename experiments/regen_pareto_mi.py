"""Regenerate pareto_curves.{pdf,png} and mi_curves.{pdf,png} from the
results_split_*.txt files produced by run_single_method.py.

This bypasses re-running the full training sweep when only the figures
need to be refreshed (e.g., after a leakage-free re-run of the splits).
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

DATASETS = [
    ("adult", "Adult"),
    ("communities", "Communities"),
    ("acs_income", "ACS Income"),
    ("meps", "MEPS"),
    ("compas", "COMPAS"),
]

METHOD_ORDER = [
    "FRHSIC (Ours)", "FREM", "Reg-GDP", "ADV",
    "MMD (binned)", "LAFTR (binned)", "dCor",
]

COLORS = {
    "FRHSIC (Ours)": "tab:red", "Reg-GDP": "tab:blue", "FREM": "tab:green",
    "ADV": "tab:purple", "MMD (binned)": "tab:orange",
    "LAFTR (binned)": "tab:brown", "dCor": "tab:cyan",
}
MARKERS = {
    "FRHSIC (Ours)": "o", "Reg-GDP": "s", "FREM": "D",
    "ADV": "^", "MMD (binned)": "v", "LAFTR (binned)": "<",
    "dCor": ">",
}

METHOD_FILES = {
    "FRHSIC (Ours)": "FRHSIC_Ours",
    "FREM": "FREM",
    "Reg-GDP": "Reg-GDP",
    "ADV": "ADV",
    "MMD (binned)": "MMD_binned",
    "LAFTR (binned)": "LAFTR_binned",
    "dCor": "dCor",
    "Unfair": "Unfair",
}

ROW_RE = re.compile(
    r"\(lam=([0-9.eE+-]+)\)\s*\|\s*([A-Za-z]+):\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)\s*"
    r"\|\s*GDP:\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)\s*\|\s*MI:\s*([0-9.eE+-]+)"
)


def parse_file(path):
    """Return list of dicts with keys: lam, perf_name, perf, perf_std, gdp, gdp_std, mi."""
    rows = []
    perf_name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = ROW_RE.search(line)
            if not m:
                continue
            lam, perf_name, perf, perf_std, gdp, gdp_std, mi = m.groups()
            rows.append({
                "lam": float(lam),
                "perf_name": perf_name,
                "perf": float(perf),
                "perf_std": float(perf_std),
                "gdp": float(gdp),
                "gdp_std": float(gdp_std),
                "mi": float(mi),
            })
    return rows


def collect_dataset(dname):
    """Return dict: method -> list of row dicts (sorted by lam)."""
    out = {}
    for method, suffix in METHOD_FILES.items():
        path = RESULTS_DIR / f"results_split_{dname}_{suffix}.txt"
        if not path.exists():
            continue
        rows = parse_file(path)
        rows.sort(key=lambda r: r["lam"])
        out[method] = rows
    return out


def plot_pareto(all_data):
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, DATASETS):
        if key not in all_data:
            continue
        results = all_data[key]
        perf_name = None
        for method in METHOD_ORDER:
            rows = results.get(method)
            if not rows:
                continue
            gdps = np.array([r["gdp"] for r in rows])
            perfs = np.array([r["perf"] for r in rows])
            perf_name = rows[0]["perf_name"]
            order = np.argsort(gdps)
            ax.plot(
                gdps[order], perfs[order],
                marker=MARKERS[method], color=COLORS[method],
                linewidth=2, markersize=6, label=method,
            )
        unfair = results.get("Unfair")
        if unfair:
            ax.scatter(
                unfair[0]["gdp"], unfair[0]["perf"],
                marker="*", s=200, c="black", zorder=10, label="Unfair",
            )
        ax.set_xlabel(r"$\Delta_{\mathrm{GDP}}$ (lower = fairer)", fontsize=11)
        y_lab = perf_name if perf_name else "Performance"
        if y_lab == "MSE":
            ax.set_ylabel(f"{y_lab} (lower = better)", fontsize=11)
        else:
            ax.set_ylabel(f"{y_lab} (higher = better)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(FIGS / "pareto_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "pareto_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'pareto_curves.pdf'} and .png")


def plot_mi(all_data):
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, DATASETS):
        if key not in all_data:
            continue
        results = all_data[key]
        perf_name = None
        for method in METHOD_ORDER:
            rows = results.get(method)
            if not rows:
                continue
            mis = np.array([r["mi"] for r in rows])
            perfs = np.array([r["perf"] for r in rows])
            perf_name = rows[0]["perf_name"]
            order = np.argsort(mis)
            ax.plot(
                mis[order], perfs[order],
                marker=MARKERS[method], color=COLORS[method],
                linewidth=2, markersize=6, label=method,
            )
        unfair = results.get("Unfair")
        if unfair:
            ax.scatter(
                unfair[0]["mi"], unfair[0]["perf"],
                marker="*", s=200, c="black", zorder=10, label="Unfair",
            )
        ax.set_xlabel(r"$\mathrm{MI}(Z, S)$ (lower = fairer)", fontsize=11)
        y_lab = perf_name if perf_name else "Performance"
        if y_lab == "MSE":
            ax.set_ylabel(f"{y_lab} (lower = better)", fontsize=11)
        else:
            ax.set_ylabel(f"{y_lab} (higher = better)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig.savefig(FIGS / "mi_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "mi_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'mi_curves.pdf'} and .png")


def main():
    all_data = {}
    for key, _ in DATASETS:
        data = collect_dataset(key)
        if data:
            all_data[key] = data
            n_methods = sum(1 for m in METHOD_FILES if m in data)
            print(f"{key}: parsed {n_methods}/{len(METHOD_FILES)} method files")
    plot_pareto(all_data)
    plot_mi(all_data)


if __name__ == "__main__":
    main()
