"""Re-run FRHSIC only, with a denser lambda grid that resolves the
fairness-accuracy frontier (the original grid jumped 50 -> 100 -> 500,
under-sampling exactly the region that determines the matched operating
point and the area under the Pareto envelope).

Split / scaling / seeding / bandwidth protocol are byte-identical to
run_single_method.py (median-heuristic bandwidth, the cited Gretton
2012 default). Only the lambda grid is denser. Baselines are unchanged;
the Pareto/AUP comparison is envelope-based, so a denser FRHSIC grid
only resolves FRHSIC's own true frontier, it does not alter the others.

Writes the standard results_split_<ds>_FRHSIC_Ours.txt files (the
originals were backed up to results/_backup_frhsic_orig/).
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data import SEED, N_REPEATS, METHODS, DATASETS, evaluate_model
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(SEED)
np.random.seed(SEED)

PAPER_DATASETS = ["adult", "communities", "acs_income", "meps", "compas"]
# Seeding/splits are byte-identical to the original sweep, so the 9
# original lambda points reproduce exactly. We therefore compute ONLY
# the new infill points here (halves compute, and keeps the
# previously-reported points byte-identical from the backup), then
# merge_dense() interleaves them with the backed-up original rows.
ORIG_LAMBDAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
LAMBDAS = [2.0, 20.0, 30.0, 70.0, 150.0, 200.0, 300.0, 750.0, 1000.0]
BACKUP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "_backup_frhsic_orig",
)

METHOD = "FRHSIC (Ours)"
train_fn = METHODS[METHOD]
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


import re

_LAM_RE = re.compile(r"\(lam=([0-9.eE+-]+)\)")


def merge_dense(dataset):
    """Interleave backed-up original rows with the freshly computed
    infill rows, sorted by lambda, into the final results_split file."""
    orig = os.path.join(BACKUP_DIR, f"results_split_{dataset}_FRHSIC_Ours.txt")
    infill = os.path.join(RESULTS_DIR, f"results_split_{dataset}_FRHSIC_Ours_infill.txt")
    final = os.path.join(RESULTS_DIR, f"results_split_{dataset}_FRHSIC_Ours.txt")
    header = None
    rows = []
    for path in (orig, infill):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    if header is None:
                        header = line
                    continue
                m = _LAM_RE.search(line)
                if m:
                    rows.append((float(m.group(1)), line))
    rows.sort(key=lambda r: r[0])
    with open(final, "w") as f:
        f.write(header or f"# dataset={dataset} method={METHOD} grid=dense\n")
        for _lam, line in rows:
            f.write(line)
    print(f"MERGED -> {final} ({len(rows)} lambda points)", flush=True)


def run_dataset(dataset):
    X_np, S_np, Y_np, display_name, task = DATASETS[dataset]()
    n = X_np.shape[0]
    outfile = os.path.join(RESULTS_DIR, f"results_split_{dataset}_FRHSIC_Ours_infill.txt")
    print(f"\n=== {dataset} ({display_name}, task={task}, n={n}) -> {outfile}", flush=True)
    with open(outfile, "w") as f:
        f.write(f"# dataset={dataset} method={METHOD} gpu=1 task={task} "
                f"n={n} d={X_np.shape[1]} grid=infill\n")
    for lam in LAMBDAS:
        perfs, gdps, mis, mi_sks = [], [], [], []
        for rep in range(N_REPEATS):
            torch.manual_seed(SEED + rep)
            np.random.seed(SEED + rep)
            idx = np.random.permutation(n)
            n_train = int(0.8 * n)
            ti, te = idx[:n_train], idx[n_train:]
            xs = MinMaxScaler()
            Xtr = torch.from_numpy(xs.fit_transform(X_np[ti])).float()
            Xte = torch.from_numpy(xs.transform(X_np[te])).float()
            smin = S_np[ti].min()
            denom = max(S_np[ti].max() - smin, 1e-12)
            Str = torch.from_numpy((S_np[ti] - smin) / denom).float()
            Ste = torch.from_numpy((S_np[te] - smin) / denom).float()
            Ytr = torch.from_numpy(Y_np[ti]).float()
            Yte = torch.from_numpy(Y_np[te]).float()
            t0 = time.time()
            enc, pred = train_fn(Xtr, Str, Ytr, lam, task)
            elapsed = time.time() - t0
            m = evaluate_model(enc, pred, Xte, Ste, Yte, task)
            perfs.append(m["perf"]); gdps.append(m["gdp"])
            mis.append(m["mi"]); mi_sks.append(m["mi_sk"])
            print(f"  [lam={lam} rep={rep}] {elapsed:.1f}s "
                  f"perf={m['perf']:.4f} gdp={m['gdp']:.4f}", flush=True)
        line = (f"  {METHOD:34s} (lam={lam}) | "
                f"{m['perf_name']}: {np.mean(perfs):.4f}+/-{np.std(perfs):.4f} | "
                f"GDP: {np.mean(gdps):.4f}+/-{np.std(gdps):.4f} | "
                f"MI: {np.mean(mis):.4f} | MI_sk: {np.mean(mi_sks):.4f}\n")
        print(line.strip(), flush=True)
        with open(outfile, "a") as f:
            f.write(line)
    print(f"DONE: {dataset} / {METHOD}", flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else PAPER_DATASETS
    for ds in targets:
        run_dataset(ds)
        merge_dense(ds)
    print("ALL_FRHSIC_DENSE_DONE", flush=True)
