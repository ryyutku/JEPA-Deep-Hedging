"""
heston_sim/generate.py

Thin wrapper around the repo's simulate_market().
Generates train / val / test path arrays and saves them as .npz files.

Nothing in this file touches PyTorch or JEPA — it is purely a data
generation utility.  All downstream code loads from the saved .npz files,
so you only need to run this once (or when you want a different Heston regime).

Usage (from project root):
    python -m heston_sim.generate
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Make the cloned repo importable regardless of how the project is installed.
# Adjust this path if your folder layout is different.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "deep-hedging-heston")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deep_hedging_heston.simulate import simulate_market          # noqa: E402
from deep_hedging_heston.parameters import HestonParameters       # noqa: E402


# ---------------------------------------------------------------------------
# Default Heston parameters  (matches the paper's reference experiment)
# ---------------------------------------------------------------------------
DEFAULT_HESTON = HestonParameters()   # v0=0.04, S0=100, K=100, T=30/365, etc.


def generate_split(
    heston_params=None,
    n_train: int = 100_000,
    n_val:   int = 20_000,
    n_test:  int = 50_000,
    seed_train: int = 15,
    seed_val:   int = 25,
    seed_test:  int = 35,
    save_dir: str = "data",
) -> dict:
    """
    Generate train / val / test Heston path splits and save each as a .npz.

    Each .npz contains:
        stocks        (N, T+1)   raw stock prices S_t
        log_stocks    (N, T+1)   log prices  ln S_t
        log_returns   (N, T)     log returns ln(S_{t+1}/S_t)   ← JEPA input
        variance      (N, T+1)   true variance V_t              ← probe target
        variance_swap (N, T+1)   variance swap S^2_t
        payoff        (N,)       max(S_T - K, 0)

    Returns a dict with keys 'train', 'val', 'test', each holding the arrays.
    """
    if heston_params is None:
        heston_params = DEFAULT_HESTON

    os.makedirs(save_dir, exist_ok=True)

    splits = {
        "train": (n_train, seed_train),
        "val":   (n_val,   seed_val),
        "test":  (n_test,  seed_test),
    }

    results = {}
    for name, (n_paths, seed) in splits.items():
        print(f"  Simulating {name:5s}: {n_paths:>7,} paths  (seed={seed}) ...", end=" ")

        paths = simulate_market(heston_params, seed=seed, nr_of_simulated_paths=n_paths)

        log_stocks  = np.log(paths.Stocks)                       # (N, T+1)
        log_returns = log_stocks[:, 1:] - log_stocks[:, :-1]    # (N, T)

        arrays = dict(
            stocks        = paths.Stocks.astype(np.float32),
            log_stocks    = log_stocks.astype(np.float32),
            log_returns   = log_returns.astype(np.float32),
            variance      = paths.Variance.astype(np.float32),
            variance_swap = paths.Variance_swap.astype(np.float32),
            payoff        = paths.Payoff.astype(np.float32),
        )

        out_path = os.path.join(save_dir, f"{name}.npz")
        np.savez(out_path, **arrays)
        print(f"saved → {out_path}")

        results[name] = arrays

    # Save a small metadata file so downstream code knows what was generated
    meta_path = os.path.join(save_dir, "meta.npz")
    np.savez(meta_path,
             v0        = np.float32(heston_params.v0),
             S0        = np.float32(heston_params.S0),
             K         = np.float32(heston_params.K),
             T         = np.float32(heston_params.T),
             timesteps = np.int32(heston_params.timesteps),
             kappa     = np.float32(heston_params.kappa),
             theta     = np.float32(heston_params.theta),
             xi        = np.float32(heston_params.xi),
             rho       = np.float32(heston_params.rho))
    print(f"  Metadata  → {meta_path}")

    return results


def load_split(name: str, data_dir: str = "data") -> dict:
    """
    Load a previously saved split by name ('train', 'val', or 'test').
    Returns a plain dict of numpy arrays.
    """
    path = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at '{path}'. Run generate_split() first."
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# CLI entry point:  python -m heston_sim.generate
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Heston path data …")
    print(f"  Using default parameters: {DEFAULT_HESTON}")
    generate_split()
    print("\nDone. Arrays in data/train.npz, data/val.npz, data/test.npz")
    print("\nQuick sanity check:")
    train = load_split("train")
    print(f"  log_returns shape : {train['log_returns'].shape}")   # (100000, 30)
    print(f"  variance    shape : {train['variance'].shape}")       # (100000, 31)
    print(f"  variance mean     : {train['variance'].mean():.4f}") # should be ~0.04
    print(f"  log_returns std   : {train['log_returns'].std():.4f}")