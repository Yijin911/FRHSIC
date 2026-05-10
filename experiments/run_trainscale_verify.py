"""Re-run Adult and Crime with train-split-only min-max scaling."""
import numpy as np
import torch
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from real_data import run_single_dataset, SEED

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    for name in ["adult", "communities"]:
        t0 = time.time()
        sep = "#" * 70
        print("\n" + sep, flush=True)
        print("# " + name, flush=True)
        print(sep, flush=True)
        try:
            run_single_dataset(name, lambdas=lambdas)
            elapsed = (time.time() - t0) / 60.0
            print(">>> %s done in %.1f min" % (name, elapsed), flush=True)
        except Exception as e:
            print("ERROR on %s: %s" % (name, e), flush=True)
            import traceback
            traceback.print_exc()
    print("\nALL DONE", flush=True)
