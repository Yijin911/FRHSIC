"""Regenerate the fairness-accuracy frontier figures and the MI
diagnostic from the cached results_split_*.txt sweeps (no retraining).

Outputs (vector PDF, publication style shared via plot_style.py):
  figures/pareto_classification.pdf  Adult, ACS Income, MEPS, COMPAS (Accuracy)
  figures/pareto_crime.pdf           Crime (MSE)
  figures/mi_curves.pdf              MI diagnostic (supplement)
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_style import (
    set_pub_style, METHOD_ORDER, PRIMARY, SCATTER, style, scatter_kw,
    pareto_idx, ordered_legend, despine, LBL_GDP, LBL_ACC, LBL_MSE,
)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
W_TEXT = 5.124  # \textwidth in inches (370.38374pt / 72.27)

DATASET_NAME = {
    "adult": "Adult", "communities": "Crime", "acs_income": "ACS Income",
    "meps": "MEPS", "compas": "COMPAS",
}
CLASSIFICATION = ["adult", "acs_income", "meps", "compas"]
REGRESSION = ["communities"]
ALL_DATASETS = CLASSIFICATION + REGRESSION

METHOD_FILES = {
    "FRHSIC (Ours)": "FRHSIC_Ours",
    "FREM": "FREM",
    "Reg-GDP": "Reg-GDP",
    "ADV": "ADV",
    "MMD (binned)": "MMD_binned",
    "LAFTR (binned)": "LAFTR_binned",
    "Unfair": "Unfair",
}

ROW_RE = re.compile(
    r"\(lam=([0-9.eE+-]+)\)\s*\|\s*([A-Za-z]+):\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)\s*"
    r"\|\s*GDP:\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)\s*\|\s*MI:\s*([0-9.eE+-]+)"
)


def parse_file(path):
    rows = []
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
                "lam": float(lam), "perf_name": perf_name,
                "perf": float(perf), "perf_std": float(perf_std),
                "gdp": float(gdp), "gdp_std": float(gdp_std),
                "mi": float(mi),
            })
    return rows


def collect_dataset(dname):
    out = {}
    for method, suffix in METHOD_FILES.items():
        path = RESULTS_DIR / f"results_split_{dname}_{suffix}.txt"
        if not path.exists():
            continue
        rows = parse_file(path)
        rows.sort(key=lambda r: r["lam"])
        out[method] = rows
    return out


def _legend_proxy(method):
    """A Line2D the legend can show (line + marker), or a lone star
    for the Unfair reference (no connecting line)."""
    if method == "Unfair":
        return Line2D([], [], color="black", marker="*", linestyle="None",
                      markersize=12, markeredgecolor="white",
                      markeredgewidth=0.8)
    st = style(method)
    return Line2D([], [], color=st["color"], marker=st["marker"],
                  linestyle=st["linestyle"], linewidth=st["linewidth"],
                  markersize=st["markersize"])


def _draw_panel(ax, results, xkey, maximize_y, ylabel=None):
    """All lambda values are drawn as points; a line connects only the
    Pareto-efficient points (lower x, and higher or lower y depending
    on the metric). Returns {method: legend-proxy handle}."""
    handles = {}
    for method in METHOD_ORDER:
        if method == "Unfair":
            continue
        rows = results.get(method)
        if not rows:
            continue
        x = np.array([r[xkey] for r in rows])
        y = np.array([r["perf"] for r in rows])
        st = style(method)
        primary = method == PRIMARY
        # all-lambda cloud (points only, no connecting line)
        ax.scatter(x, y, marker=st["marker"], facecolor=st["color"],
                   edgecolor="white", linewidths=0.3,
                   s=SCATTER["primary"] if primary else SCATTER["secondary"],
                   alpha=0.85 if primary else 0.6,
                   zorder=st["zorder"])
        # connect only the Pareto-efficient subset
        idx = pareto_idx(x, y, minimize_x=True, maximize_y=maximize_y)
        if len(idx) >= 2:
            ax.plot(x[idx], y[idx], color=st["color"],
                    linestyle=st["linestyle"], linewidth=st["linewidth"],
                    alpha=st["alpha"], zorder=st["zorder"] + 0.5,
                    solid_capstyle="round")
        handles[method] = _legend_proxy(method)
    unfair = results.get("Unfair")
    if unfair:
        ax.scatter([unfair[0][xkey]], [unfair[0]["perf"]],
                   **scatter_kw("Unfair"))   # star only, no line
        handles["Unfair"] = _legend_proxy("Unfair")
    ax.grid(True, alpha=0.3)
    despine(ax)
    if ylabel:
        ax.set_ylabel(ylabel)
    return handles


def _grid_figure(keys, all_data, xkey, maxy_fn, x_label,
                 y_label=None, per_panel_ylabel_fn=None):
    """2x2 grid with shared axis labels (set once at the figure level)
    and a single ordered legend below."""
    fig_w = W_TEXT
    fig_h = fig_w * 0.82
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    axes = axes.flatten()
    handles_by_method = {}
    for ax, key in zip(axes, keys):
        if key not in all_data:
            ax.set_visible(False)
            continue
        yl = per_panel_ylabel_fn(key) if per_panel_ylabel_fn else None
        hs = _draw_panel(ax, all_data[key], xkey, maxy_fn(key), ylabel=yl)
        ax.set_title(DATASET_NAME[key])
        for m, h in hs.items():
            handles_by_method.setdefault(m, h)
    # constrained_layout positions the shared super-labels correctly;
    # the legend is placed below the figure and the saved canvas is
    # expanded to include it via bbox_inches="tight".
    fig.supxlabel(x_label, fontsize=10)
    if y_label:
        fig.supylabel(y_label, fontsize=10)
    leg = ordered_legend(fig, handles_by_method, ncol=4,
                         loc="upper center", bbox_to_anchor=(0.5, -0.02))
    return fig, leg


def plot_pareto_classification(all_data):
    set_pub_style()
    fig, leg = _grid_figure(
        CLASSIFICATION, all_data, "gdp",
        maxy_fn=lambda k: True, x_label=LBL_GDP, y_label=LBL_ACC)
    fig.savefig(FIGS / "pareto_classification.pdf",
                bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(FIGS / "pareto_classification.png", dpi=300,
                bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)
    print(f"Saved {FIGS/'pareto_classification.pdf'}")


def plot_pareto_crime(all_data):
    set_pub_style()
    if "communities" not in all_data:
        print("Crime data missing; skipped")
        return
    fig_w = 0.55 * W_TEXT
    fig_h = fig_w * 1.05
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Crime is regression: lower MSE is better, so maximize_y=False.
    hs = _draw_panel(ax, all_data["communities"], "gdp", maximize_y=False)
    ax.set_xlabel(LBL_GDP)
    ax.set_ylabel(LBL_MSE)
    ax.set_title(DATASET_NAME["communities"])
    # Legend outside, below the panel, so it never covers the curves.
    fig.subplots_adjust(left=0.17, right=0.97, top=0.92, bottom=0.32)
    leg = ordered_legend(fig, hs, ncol=4, loc="lower center",
                         bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.savefig(FIGS / "pareto_crime.pdf",
                bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(FIGS / "pareto_crime.png", dpi=300,
                bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)
    print(f"Saved {FIGS/'pareto_crime.pdf'}")


def plot_mi(all_data):
    # COMPAS excluded: the KSG MI estimator collapses to ~0 on its
    # low-dimensional discrete feature space (Table 1 marks it "---").
    set_pub_style()
    mi_keys = [k for k in ALL_DATASETS if k != "compas"]
    fig, leg = _grid_figure(
        mi_keys, all_data, "mi",
        maxy_fn=lambda k: k != "communities",
        x_label=r"$\mathrm{MI}(Z, S)\,\downarrow$",
        per_panel_ylabel_fn=lambda k: LBL_MSE if k == "communities" else LBL_ACC)
    fig.savefig(FIGS / "mi_curves.pdf",
                bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(FIGS / "mi_curves.png", dpi=300,
                bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)
    print(f"Saved {FIGS/'mi_curves.pdf'}")


def main():
    all_data = {}
    for key in ALL_DATASETS:
        data = collect_dataset(key)
        if data:
            all_data[key] = data
            n = sum(1 for m in METHOD_FILES if m in data)
            print(f"{key}: parsed {n}/{len(METHOD_FILES)} method files")
    plot_pareto_classification(all_data)
    plot_pareto_crime(all_data)
    plot_mi(all_data)


if __name__ == "__main__":
    main()
