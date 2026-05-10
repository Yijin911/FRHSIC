"""
Run experiments on the two datasets whose raw logs are missing from the repo:
Crime and COMPAS. Uses the same 9-lambda protocol as the other datasets.
Results saved to ../results_missing_datasets.txt
"""
import numpy as np
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from real_data import run_single_dataset, SEED

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    for name in ["communities", "compas"]:
        t0 = time.time()
        print(f"\n{'#'*70}", flush=True)
        print(f"# Starting {name}", flush=True)
        print(f"{'#'*70}", flush=True)
        try:
            res = run_single_dataset(name, lambdas=lambdas)
            elapsed = time.time() - t0
            print(f"\n>>> {name} completed in {elapsed/3600:.2f} hours ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f"Error on {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nALL DONE", flush=True)
