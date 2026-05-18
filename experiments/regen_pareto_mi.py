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
    ("communities", "Crime"),
    ("acs_income", "ACS Income"),
    ("meps", "MEPS"),
    ("compas", "COMPAS"),
]

METHOD_ORDER = [
    "FRHSIC (Ours)", "FREM", "Reg-GDP", "ADV",
    "MMD (binned)", "LAFTR (binned)",
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
    # figsize width = 1.0 * W_text (LaTeX inclusion width changed to \textwidth)
    # W_text = 370.38374pt / 72.27pt/in = 5.124"
    # Layout: 3 columns x 2 rows (last cell empty for 5 datasets)
    W_text = 5.124
    fig_w = W_text          # 5.124"
    # Each panel aspect ~1:1 at ~1.7" wide; 2 rows with shared legend below
    fig_h = fig_w * 0.72    # 3.689"
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten()
    # Hide the unused last cell (5 datasets in 6 slots)
    if len(DATASETS) < ncols * nrows:
        for extra in axes_flat[len(DATASETS):]:
            extra.set_visible(False)

    all_handles, all_labels = {}, {}
    for ax, (key, label) in zip(axes_flat, DATASETS):
        if key not in all_data:
            ax.set_visible(False)
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
            h, = ax.plot(
                gdps[order], perfs[order],
                marker=MARKERS[method], color=COLORS[method],
                linewidth=0.9, markersize=2.5, label=method,
            )
            if method not in all_labels:
                all_handles[method] = h
                all_labels[method] = method
        unfair = results.get("Unfair")
        if unfair:
            h = ax.scatter(
                unfair[0]["gdp"], unfair[0]["perf"],
                marker="*", s=45, c="black", zorder=10, label="Unfair",
            )
            if "Unfair" not in all_labels:
                all_handles["Unfair"] = h
                all_labels["Unfair"] = "Unfair"
        ax.set_xlabel(r"$\Delta_{\mathrm{GDP}}$", fontsize=9)
        y_lab = perf_name if perf_name else "Perf."
        ax.set_ylabel(y_lab, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)

    handles_list = list(all_handles.values())
    labels_list = list(all_labels.values())
    fig.legend(handles_list, labels_list, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=len(labels_list),
               fontsize=6.5, frameon=False, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(FIGS / "pareto_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "pareto_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'pareto_curves.pdf'} and .png")


def plot_mi(all_data):
    # COMPAS is excluded: the KSG MI estimator collapses to ~0 on its
    # low-dimensional, discrete-valued feature space (the same reason
    # Table 1 marks COMPAS MI as "---"), so the COMPAS MI panel is a
    # degenerate vertical line and is not shown.
    mi_datasets = [(k, lbl) for (k, lbl) in DATASETS if k != "compas"]
    # figsize width = 1.0 * W_text (LaTeX inclusion width = \textwidth)
    # W_text = 370.38374pt / 72.27pt/in = 5.124"
    # Layout: 2 columns x 2 rows for the 4 retained datasets.
    W_text = 5.124
    fig_w = W_text          # 5.124"
    fig_h = fig_w * 0.78
    ncols, nrows = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten()
    if len(mi_datasets) < ncols * nrows:
        for extra in axes_flat[len(mi_datasets):]:
            extra.set_visible(False)

    all_handles, all_labels = {}, {}
    for ax, (key, label) in zip(axes_flat, mi_datasets):
        if key not in all_data:
            ax.set_visible(False)
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
            h, = ax.plot(
                mis[order], perfs[order],
                marker=MARKERS[method], color=COLORS[method],
                linewidth=0.9, markersize=2.5, label=method,
            )
            if method not in all_labels:
                all_handles[method] = h
                all_labels[method] = method
        unfair = results.get("Unfair")
        if unfair:
            h = ax.scatter(
                unfair[0]["mi"], unfair[0]["perf"],
                marker="*", s=45, c="black", zorder=10, label="Unfair",
            )
            if "Unfair" not in all_labels:
                all_handles["Unfair"] = h
                all_labels["Unfair"] = "Unfair"
        ax.set_xlabel(r"$\mathrm{MI}(Z, S)$", fontsize=9)
        y_lab = perf_name if perf_name else "Perf."
        ax.set_ylabel(y_lab, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)

    handles_list = list(all_handles.values())
    labels_list = list(all_labels.values())
    fig.legend(handles_list, labels_list, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=len(labels_list),
               fontsize=6.5, frameon=False, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
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
