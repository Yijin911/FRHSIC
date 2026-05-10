"""
Run experiments on the two new datasets: ACS Income and MEPS.
Results are printed to stdout and figures are saved to ../figures/.
"""
import numpy as np
import torch
import sys
import os
import time

# Add experiments dir to path
sys.path.insert(0, os.path.dirname(__file__))

from real_data import run_single_dataset, SEED

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    for name in ["acs_income", "meps"]:
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"# Starting {name}")
        print(f"{'#'*70}")
        try:
            res = run_single_dataset(name, lambdas=lambdas)
            elapsed = time.time() - t0
            print(f"\n>>> {name} completed in {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()
