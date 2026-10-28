"""
jepa/dataset.py

Converts flat Heston path arrays into (context_window, target_window) pairs
that the JEPA model trains on.

Key design choices
------------------
* Context  : the L time-steps BEFORE position t  →  what the encoder sees
* Target   : the H time-steps STARTING at t      →  what the predictor must match
* Windows overlap across time, so one path of length T gives (T - L - H + 1)
  training samples.

Two observation modes are supported (set at construction time):

    mode='prices'   input features = [log_return_t]          shape (window, 1)
    mode='oracle'   input features = [log_return_t, V_t]     shape (window, 2)

'oracle' lets you check the upper bound of what is learnable (when the model
can see V_t directly).  'prices' is the realistic / thesis-relevant setting.

The variance at the TARGET start step (variance[:, t]) is stored separately as
`target_variance` so the probe can use it as a regression label without it
ever touching the JEPA loss.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HestonWindowDataset(Dataset):
    """
    Parameters
    ----------
    data       : dict returned by heston_sim.load_split()
    context_len: L  — how many past steps the encoder sees
    target_len : H  — how many future steps the predictor must match
    mode       : 'prices' (log returns only) or 'oracle' (log returns + V_t)
    """

    def __init__(
        self,
        data: dict,
        context_len: int = 10,
        target_len:  int = 5,
        mode:        str = "prices",   # 'prices' | 'oracle'
    ):
        assert mode in ("prices", "oracle"), f"Unknown mode: {mode!r}"

        log_returns = data["log_returns"]   # (N, T)
        variance    = data["variance"]      # (N, T+1)

        N, T = log_returns.shape

        # Build the feature matrix that will be windowed
        # ------------------------------------------------
        # For both modes we align features at the same time index as variance.
        # variance[:, t] is known AT step t (before the next return is observed).
        # log_returns[:, t] = ln(S_{t+1}/S_t) is the return OVER step t.
        #
        # We treat the feature at position t as:
        #   prices : log_returns[:, t]           shape (N, T, 1)
        #   oracle : [log_returns[:, t], V[:, t]] shape (N, T, 2)
        #
        # Note: we use V[:, t] (start of step) not V[:, t+1] (end of step)
        # to avoid any look-ahead of the next-step variance.

        if mode == "prices":
            features = log_returns[:, :, np.newaxis]           # (N, T, 1)
        else:  # oracle
            v_aligned = variance[:, :T]                        # (N, T)  drop last
            features  = np.stack([log_returns, v_aligned], axis=-1)  # (N, T, 2)

        self.features        = features.astype(np.float32)    # (N, T, C)
        self.variance        = variance.astype(np.float32)    # (N, T+1)
        self.context_len     = context_len
        self.target_len      = target_len
        self.mode            = mode
        self.n_paths         = N
        self.T               = T
        self.feature_dim     = features.shape[-1]             # 1 or 2

        # Every valid (path, start) pair is a training sample
        # start is the first index of the context window
        # so context = [start : start+L], target = [start+L : start+L+H]
        # constraint: start+L+H <= T
        self.max_start = T - context_len - target_len         # inclusive upper bound
        assert self.max_start >= 0, (
            f"Path length T={T} is too short for "
            f"context_len={context_len} + target_len={target_len}"
        )

        # Pre-compute flat index → (path_idx, start_idx) mapping
        # This allows __getitem__ to work with a single integer index.
        n_starts = self.max_start + 1
        self.n_samples = N * n_starts
        self._n_starts = n_starts

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        path_idx  = idx // self._n_starts
        start_idx = idx %  self._n_starts

        L = self.context_len
        H = self.target_len
        t = start_idx           # start of context window
        t_target = t + L        # start of target window

        context = self.features[path_idx, t : t + L]          # (L, C)
        target  = self.features[path_idx, t_target : t_target + H]  # (H, C)

        # True variance at the TARGET start step — used by the linear probe,
        # NOT fed into the JEPA loss.
        target_variance = self.variance[path_idx, t_target]   # scalar

        return {
            "context":         torch.from_numpy(context),         # (L, C)
            "target":          torch.from_numpy(target),           # (H, C)
            "target_variance": torch.tensor(target_variance),      # ()
            "path_idx":        torch.tensor(path_idx),
            "t":               torch.tensor(t),
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from heston_sim.generate import load_split

    data = load_split("train", data_dir="data")
    ds = HestonWindowDataset(data, context_len=10, target_len=5, mode="prices")

    print(f"Dataset size       : {len(ds):,}")
    sample = ds[0]
    print(f"context shape      : {sample['context'].shape}")          # (10, 1)
    print(f"target  shape      : {sample['target'].shape}")           # (5, 1)
    print(f"target_variance    : {sample['target_variance'].item():.5f}")

    ds_oracle = HestonWindowDataset(data, context_len=10, target_len=5, mode="oracle")
    sample_o  = ds_oracle[0]
    print(f"\noracle context     : {sample_o['context'].shape}")      # (10, 2)