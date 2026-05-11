"""Run one (dataset, method) pair across the lambda grid + 5 splits."""
import sys, os, time
import numpy as np
import torch

if len(sys.argv) < 3:
    print("Usage: python run_single_method.py <dataset> <method> [<gpu_id>]")
    sys.exit(1)
dataset = sys.argv[1]
method = sys.argv[2]
gpu = sys.argv[3] if len(sys.argv) > 3 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data import (
    SEED, N_REPEATS, METHODS, DATASETS, evaluate_model,
)
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(SEED)
np.random.seed(SEED)

if method not in METHODS:
    print(f"Unknown method: {method}; available: {list(METHODS.keys())}")
    sys.exit(1)
if dataset not in DATASETS:
    print(f"Unknown dataset: {dataset}; available: {list(DATASETS.keys())}")
    sys.exit(1)

result = DATASETS[dataset]()
X_np, S_np, Y_np, display_name, task = result
n = X_np.shape[0]
lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
train_fn = METHODS[method]

safe = method.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
outfile = os.path.join(RESULTS_DIR, f"results_split_{dataset}_{safe}.txt")
print(f"Writing to {outfile}", flush=True)
with open(outfile, "w") as f:
    f.write(f"# dataset={dataset} method={method} gpu={gpu} task={task} n={n} d={X_np.shape[1]}\n")

for lam in ([0.0] if method == "Unfair" else lambdas):
    perfs, gdps, mis, mi_sks = [], [], [], []
    for rep in range(N_REPEATS):
        torch.manual_seed(SEED + rep)
        np.random.seed(SEED + rep)
        idx = np.random.permutation(n)
        n_train = int(0.8 * n)
        ti, te = idx[:n_train], idx[n_train:]
        Xtr_raw, Xte_raw = X_np[ti], X_np[te]
        Str_raw, Ste_raw = S_np[ti], S_np[te]
        Ytr_np, Yte_np = Y_np[ti], Y_np[te]
        xs = MinMaxScaler()
        Xtr = torch.from_numpy(xs.fit_transform(Xtr_raw)).float()
        Xte = torch.from_numpy(xs.transform(Xte_raw)).float()
        smin, smax = Str_raw.min(), Str_raw.max()
        denom = max(smax - smin, 1e-12)
        Str = torch.from_numpy((Str_raw - smin) / denom).float()
        Ste = torch.from_numpy((Ste_raw - smin) / denom).float()
        Ytr = torch.from_numpy(Ytr_np).float()
        Yte = torch.from_numpy(Yte_np).float()
        t0 = time.time()
        if method == "Unfair":
            enc, pred = train_fn(Xtr, Str, Ytr, 0.0, task)
        else:
            enc, pred = train_fn(Xtr, Str, Ytr, lam, task)
        elapsed = time.time() - t0
        m = evaluate_model(enc, pred, Xte, Ste, Yte, task)
        perfs.append(m["perf"]); gdps.append(m["gdp"]); mis.append(m["mi"]); mi_sks.append(m["mi_sk"])
        print(f"  [{method} lam={lam} rep={rep}] {elapsed:.1f}s perf={m['perf']:.4f} gdp={m['gdp']:.4f} mi={m['mi']:.4f}", flush=True)
    line = f"  {method:34s} (lam={lam}) | {m['perf_name']}: {np.mean(perfs):.4f}+/-{np.std(perfs):.4f} | GDP: {np.mean(gdps):.4f}+/-{np.std(gdps):.4f} | MI: {np.mean(mis):.4f} | MI_sk: {np.mean(mi_sks):.4f}\n"
    print(line.strip(), flush=True)
    with open(outfile, "a") as f:
        f.write(line)
print(f"DONE: {dataset} / {method}", flush=True)
