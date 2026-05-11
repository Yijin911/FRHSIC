"""Compute the Pareto-summary table from the results_split_*.txt files.

For each (method, dataset) pair, find the operating point with the LOWEST GDP
subject to:
  classification: Acc >= 0.99 * Acc_Unfair
  regression:     MSE <= 1.01 * MSE_Unfair
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

DATASETS = [
    ("adult", "Adult", "Acc", False),
    ("acs_income", "ACS Income", "Acc", False),
    ("meps", "MEPS", "Acc", False),
    ("communities", "Crime", "MSE", True),
    ("compas", "COMPAS", "Acc", False),
]

METHOD_ORDER = [
    "Unfair", "FRHSIC (Ours)", "FREM", "Reg-GDP", "ADV",
    "MMD (binned)", "LAFTR (binned)", "dCor",
]

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
    r"\|\s*GDP:\s*([0-9.eE+-]+)\+/-([0-9.eE+-]+)"
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
            lam, _name, perf, _ps, gdp, _gs = m.groups()
            rows.append((float(lam), float(perf), float(gdp)))
    return rows


def best_op(dname, mname, unfair_perf, is_mse):
    path = RESULTS_DIR / f"results_split_{dname}_{METHOD_FILES[mname]}.txt"
    if not path.exists():
        return None
    rows = parse_file(path)
    if not rows:
        return None
    if is_mse:
        bound = 1.01 * unfair_perf
        feasible = [r for r in rows if r[1] <= bound]
    else:
        bound = 0.99 * unfair_perf
        feasible = [r for r in rows if r[1] >= bound]
    if not feasible:
        return None
    # lowest GDP among feasible
    feasible.sort(key=lambda r: r[2])
    return feasible[0]  # (lam, perf, gdp)


def main():
    print("\nPareto summary (lowest GDP within 1% accuracy band):\n")
    header = f"{'Method':<18}" + "".join(f"{d[1]:<22}" for d in DATASETS)
    print(header)
    print("-" * len(header))
    # Get unfair refs
    unfair = {}
    for dkey, _dn, _pn, is_mse in DATASETS:
        rows = parse_file(RESULTS_DIR / f"results_split_{dkey}_Unfair.txt")
        if rows:
            unfair[dkey] = rows[0]
    # Print Unfair row
    line = f"{'Unfair (ref.)':<18}"
    for dkey, _dn, _pn, is_mse in DATASETS:
        if dkey in unfair:
            _, perf, gdp = unfair[dkey]
            if is_mse:
                cell = f"({perf*100:.2f}, {gdp:.3f})"
            else:
                cell = f"({perf:.3f}, {gdp:.3f})"
            line += f"{cell:<22}"
        else:
            line += f"{'---':<22}"
    print(line)
    # Other methods
    for mname in METHOD_ORDER[1:]:
        line = f"{mname:<18}"
        for dkey, _dn, _pn, is_mse in DATASETS:
            if dkey not in unfair:
                line += f"{'---':<22}"
                continue
            unfair_perf = unfair[dkey][1]
            op = best_op(dkey, mname, unfair_perf, is_mse)
            if op is None:
                line += f"{'---':<22}"
            else:
                _, perf, gdp = op
                if is_mse:
                    cell = f"({perf*100:.2f}, {gdp:.3f})"
                else:
                    cell = f"({perf:.3f}, {gdp:.3f})"
                line += f"{cell:<22}"
        print(line)


if __name__ == "__main__":
    main()
