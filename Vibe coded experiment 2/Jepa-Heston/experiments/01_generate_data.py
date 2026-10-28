"""
experiments/01_generate_data.py

Run this first — generates all path data and saves to data/*.npz

    python experiments/01_generate_data.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heston_sim.generate import generate_split, load_split
import numpy as np

if __name__ == "__main__":
    print("=" * 55)
    print("Step 1 — Generating Heston market paths")
    print("=" * 55)

    generate_split(
        n_train     = 100_000,
        n_val       = 20_000,
        n_test      = 50_000,
        seed_train  = 15,
        seed_val    = 25,
        seed_test   = 35,
        save_dir    = "data",
    )

    print("\nSanity check on training data:")
    train = load_split("train", "data")
    N, T = train["log_returns"].shape
    print(f"  Paths x Timesteps  : {N:,} x {T}")
    print(f"  Variance  mean/std : {train['variance'].mean():.4f} / {train['variance'].std():.4f}")
    print(f"  log_return mean    : {train['log_returns'].mean():.6f}  (should be ~0)")
    print(f"  log_return std     : {train['log_returns'].std():.4f}")
    print(f"  Payoff > 0         : {(train['payoff'] > 0).mean()*100:.1f}% of paths")