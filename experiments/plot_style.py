"""Shared publication style for the FRHSIC main-paper figures.

One source of truth for the color-blind-safe palette (Okabe--Ito),
marker shapes, method ordering, FRHSIC emphasis, and font sizes, so
that every figure in the paper is visually consistent. All figures are
exported as vector PDF.
"""
import matplotlib

# Method ordering used in every legend.
METHOD_ORDER = [
    "FRHSIC (Ours)", "FREM", "Reg-GDP", "ADV",
    "MMD (binned)", "LAFTR (binned)", "Unfair",
]

# Short legend label (the results files use "FRHSIC (Ours)").
LEGEND_LABEL = {
    "FRHSIC (Ours)": "FRHSIC",
    "FREM": "FREM",
    "Reg-GDP": "Reg-GDP",
    "ADV": "ADV",
    "MMD (binned)": "Binned MMD",
    "LAFTR (binned)": "Binned LAFTR",
    "Unfair": "Unfair",
}

# Okabe--Ito color-blind-safe palette. Distinct hues AND distinct
# markers, so the encoding never relies on color alone.
COLORS = {
    "FRHSIC (Ours)": "#D55E00",   # vermillion (primary)
    "FREM":          "#0072B2",   # blue
    "Reg-GDP":       "#009E73",   # bluish green
    "ADV":           "#CC79A7",   # reddish purple
    "MMD (binned)":  "#E69F00",   # orange
    "LAFTR (binned)":"#56B4E9",   # sky blue
    "Unfair":        "#000000",   # black
}
MARKERS = {
    "FRHSIC (Ours)": "o",
    "FREM":          "s",
    "Reg-GDP":       "D",
    "ADV":           "^",
    "MMD (binned)":  "v",
    "LAFTR (binned)":"<",
    "Unfair":        "*",
}
# Only FRHSIC is solid; every baseline uses a distinct, pronounced
# dash pattern so methods are easy to tell apart (also in grayscale
# and where short frontier segments overlap).
LINESTYLE = {
    "FRHSIC (Ours)": "-",
    "FREM":          (0, (7, 2)),                 # long dash
    "Reg-GDP":       (0, (6, 2, 1.5, 2)),         # dash-dot
    "ADV":           (0, (2, 2)),                 # dotted
    "MMD (binned)":  (0, (7, 2, 1.5, 2, 1.5, 2)), # dash-dot-dot
    "LAFTR (binned)":(0, (4, 2.5)),               # short dash
    "Unfair":        "None",
}

# FRHSIC is visually primary; baselines are secondary (thinner,
# smaller, slightly transparent).
PRIMARY = "FRHSIC (Ours)"
LINEWIDTH = {"primary": 2.4, "secondary": 1.8}
MARKERSIZE = {"primary": 5.5, "secondary": 3.5}      # line plots
SCATTER = {"primary": 42, "secondary": 26}           # all-lambda cloud
# Baselines are semi-transparent so overlapping frontiers remain
# distinguishable; FRHSIC stays opaque and on top but is no longer
# so large that it hides nearby methods.
ALPHA = {"primary": 1.0, "secondary": 0.80}
ZORDER = {"primary": 7, "secondary": 3}


def pareto_idx(x, y, minimize_x=True, maximize_y=True):
    """Indices of the Pareto-efficient points, ordered along the
    frontier (ascending x). A point is efficient if no other point is
    at least as good on both axes and strictly better on one."""
    import numpy as np
    a = np.asarray(x, float) if minimize_x else -np.asarray(x, float)
    b = -np.asarray(y, float) if maximize_y else np.asarray(y, float)
    n = len(a)
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))
        if dom.any():
            eff[i] = False
    idx = np.where(eff)[0]
    return idx[np.argsort(a[idx])]


def style(method):
    """Per-method plot kwargs with FRHSIC emphasized."""
    tier = "primary" if method == PRIMARY else "secondary"
    return dict(
        color=COLORS[method],
        marker=MARKERS[method],
        linestyle=LINESTYLE[method],
        linewidth=LINEWIDTH[tier],
        markersize=MARKERSIZE[tier],
        alpha=ALPHA[tier],
        zorder=ZORDER[tier],
    )


def scatter_kw(method):
    # The Unfair reference must stay clearly visible on top of every
    # frontier: a large black star with a white outline, highest
    # z-order, fully opaque.
    if method == "Unfair":
        return dict(
            color="black", marker="*", s=170, alpha=1.0,
            zorder=30, linewidths=1.0, edgecolors="white",
        )
    tier = "primary" if method == PRIMARY else "secondary"
    return dict(
        color=COLORS[method],
        marker=MARKERS[method],
        s=SCATTER[tier],
        alpha=ALPHA[tier],
        zorder=ZORDER[tier] + 1,
        linewidths=0.6,
        edgecolors="white",
    )


def despine(ax):
    """Remove the top and right plot frame (spines)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_pub_style():
    """Font sizes legible after insertion at the paper's column width:
    ticks >= 9pt, axis labels >= 10pt, legend >= 9pt, titles >= 10pt.
    Fonts are embedded (Type-42) for a clean vector PDF."""
    matplotlib.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# Directional axis labels (mathtext arrows render in the PDF).
LBL_GDP = r"Fairness violation $\Delta_{\mathrm{GDP}}\,\downarrow$"
LBL_ACC = r"Accuracy $\uparrow$"
LBL_MSE = r"MSE $\downarrow$"
LBL_TIME = r"Time per epoch (s) $\downarrow$"


def ordered_legend(fig, handles_by_method, **kw):
    """Place a single figure-level legend below the panels, with
    methods in METHOD_ORDER and short labels."""
    handles, labels = [], []
    for m in METHOD_ORDER:
        if m in handles_by_method:
            handles.append(handles_by_method[m])
            labels.append(LEGEND_LABEL[m])
    defaults = dict(
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(labels), 7), frameon=False,
        handlelength=1.6, columnspacing=1.2, handletextpad=0.5,
    )
    defaults.update(kw)
    return fig.legend(handles, labels, **defaults)
