"""Matched-operating-point + area-under-Pareto (AUP) summary.

For each (dataset, method) we report two apples-to-apples quantities,
both computed from the per-lambda sweep files results_split_*.txt:

1. Matched operating point: the point with the LOWEST GDP subject to the
   accuracy staying within 1% of the Unfair baseline
   (classification: Acc >= 0.99 * Acc_Unfair; regression:
   MSE <= 1.01 * MSE_Unfair). Accuracy is therefore held ~constant,
   so GDP is directly comparable across methods.

2. Normalized area under the accuracy-fairness Pareto envelope (nAUP),
   in [0, 1], higher = better. Utility u is Acc (classification) or
   -MSE (regression), min-max normalized per dataset across every
   point of every method. Fairness budget is g = GDP / GDP_Unfair,
   clipped to [0, 1]. The utility envelope U(g) = max normalized
   utility among that method's points with budget <= g. nAUP is the
   trapezoidal integral of U over g in [0, 1]: the mean utility the
   method retains across the whole fairness range. It is insensitive
   to the 1% band and to where any single lambda lands.
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# (key, display, perf_name, is_mse)
DATASETS = [
    ("adult", "Adult", "Acc", False),
    ("acs_income", "ACS Income", "Acc", False),
    ("meps", "MEPS", "Acc", False),
    ("communities", "Crime", "MSE", True),
    ("compas", "COMPAS", "Acc", False),
]

# Method set matches Table 1 of the paper (no dCor).
METHOD_ORDER = [
    "FRHSIC (Ours)", "FREM", "Reg-GDP", "ADV",
    "MMD (binned)", "LAFTR (binned)",
]
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
    r"\|\s*GDP:\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)"
)


def parse_file(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = ROW_RE.search(line)
            if not m:
                continue
            lam, _n, perf, _ps, gdp, _gs = m.groups()
            rows.append((float(lam), float(perf), float(gdp)))
    return rows


def matched_point(rows, unfair_perf, is_mse):
    if not rows:
        return None
    if is_mse:
        feas = [r for r in rows if r[1] <= 1.01 * unfair_perf]
    else:
        feas = [r for r in rows if r[1] >= 0.99 * unfair_perf]
    if not feas:
        return None
    feas.sort(key=lambda r: r[2])  # lowest GDP
    return feas[0]  # (lam, perf, gdp)


def naup(rows, gdp_unfair, u_min, u_max, is_mse):
    """Trapezoidal integral of the utility envelope over g in [0,1]."""
    if not rows or gdp_unfair <= 0 or u_max <= u_min:
        return None
    pts = []
    for _lam, perf, gdp in rows:
        u = (-perf if is_mse else perf)
        u_norm = (u - u_min) / (u_max - u_min)
        g = min(max(gdp / gdp_unfair, 0.0), 1.0)
        pts.append((g, u_norm))
    pts.sort()
    grid = np.linspace(0.0, 1.0, 201)
    env = np.full_like(grid, -np.inf)
    for g, u in pts:
        env[grid >= g] = np.maximum(env[grid >= g], u)
    if not np.all(np.isfinite(env)):
        # no point with budget <= some g: backfill with running max from right
        best = -np.inf
        for i in range(len(grid)):
            if np.isfinite(env[i]):
                best = max(best, env[i])
            env[i] = best if best > -np.inf else 0.0
    return float(np.trapz(env, grid))


def main():
    print("\nMatched operating point (Perf, GDP) at <=1% acc band  +  nAUP\n")
    for dkey, dname, pname, is_mse in DATASETS:
        unfair_rows = parse_file(RESULTS_DIR / f"results_split_{dkey}_Unfair.txt")
        if not unfair_rows:
            print(f"{dname}: no Unfair reference, skipped")
            continue
        unfair_perf, gdp_unfair = unfair_rows[0][1], unfair_rows[0][2]
        # utility normalization range across all methods+unfair on this dataset
        all_u = []
        per_method_rows = {}
        for m in METHOD_ORDER:
            r = parse_file(RESULTS_DIR / f"results_split_{dkey}_{METHOD_FILES[m]}.txt")
            per_method_rows[m] = r
            for _l, perf, _g in r:
                all_u.append(-perf if is_mse else perf)
        all_u.append(-unfair_perf if is_mse else unfair_perf)
        u_min, u_max = min(all_u), max(all_u)

        print(f"== {dname} ({pname}{'↓' if is_mse else '↑'}) | "
              f"Unfair=({unfair_perf:.3f}, GDP {gdp_unfair:.3f}) ==")
        for m in METHOD_ORDER:
            rows = per_method_rows[m]
            mp = matched_point(rows, unfair_perf, is_mse)
            a = naup(rows, gdp_unfair, u_min, u_max, is_mse)
            if mp is None:
                cell = "matched=---"
            else:
                _l, perf, gdp = mp
                pv = f"{perf*100:.2f}" if is_mse else f"{perf:.3f}"
                cell = f"matched=({pv}, {gdp:.3f})"
            astr = f"{a:.3f}" if a is not None else "---"
            print(f"  {m:<16} {cell:<24} nAUP={astr}")
        print()


if __name__ == "__main__":
    main()
