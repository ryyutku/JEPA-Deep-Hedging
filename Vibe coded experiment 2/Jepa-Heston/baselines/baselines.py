"""
baselines/baselines.py

Three comparisons against the JEPA probe result (linear R²=0.754):

1. Realized Variance (RV)
   Rolling window estimator of variance from log returns.
   No learning at all — purely statistical.
   Formula: RV_t = (1/L) * sum(r_{t-L..t}²)
   If this already scores R²=0.75, JEPA learned nothing special.
   If it scores R²=0.3, JEPA is doing real work.

2. Supervised LSTM
   Trained WITH access to V_t as a label.
   This is the upper bound — the best a sequential model can do
   when it knows exactly what to predict.
   If JEPA (unsupervised) is close to this, that's the thesis argument.

3. PCA embedding visualisation
   Projects JEPA's z_t down to 2D and colours each point by V_t.
   If the coloring forms a smooth gradient, the embedding space is
   geometrically organised around volatility — visual proof.

Run from project root:
    python baselines/baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from heston_sim.generate  import load_split
from jepa.dataset         import HestonWindowDataset
from jepa.model           import HestonJEPA

# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/jepa_prices.pt"
DATA_DIR     = "data"
MODE         = "prices"
CONTEXT_LEN  = 10
TARGET_LEN   = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR     = "results"

# LSTM training
LSTM_LR         = 1e-3
LSTM_EPOCHS     = 20
LSTM_BATCH      = 512
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
# ──────────────────────────────────────────────────────────────────────────


# ==========================================================================
# 1. Realized Variance baseline
# ==========================================================================

def realized_variance(log_returns: np.ndarray, context_len: int) -> np.ndarray:
    """
    For each position t in [context_len, T], compute:
        RV_t = mean(r_{t-L}² , ..., r_{t-1}²)

    log_returns : (N, T)
    returns       (N, T - context_len)   one RV estimate per valid window
    """
    # Rolling mean of squared returns over a window of size context_len
    r2 = log_returns ** 2                           # (N, T)
    # Use cumsum for efficiency
    cs = np.cumsum(r2, axis=1)                      # (N, T)
    # cs[:, t] = sum of r2 up to and including index t
    # window sum from t-L to t-1 (exclusive of t):
    # = cs[:, t-1] - cs[:, t-L-1]   for t >= L
    T  = log_returns.shape[1]
    L  = context_len
    rv = np.zeros((log_returns.shape[0], T - L))
    for t in range(L, T):
        window_sum = cs[:, t-1] - (cs[:, t-L-1] if t-L-1 >= 0 else 0.0)
        rv[:, t - L] = window_sum / L
    return rv                                       # (N, T-L)


def evaluate_realized_variance(train_data, test_data, context_len, target_len):
    """
    Align RV estimates with the variance labels used by the JEPA probe.

    The probe's target_variance at sample (path, start) is:
        variance[path, start + context_len]    i.e. V at the target start step

    So the matching RV estimate is:
        RV computed over [start, start+context_len-1]  →  index start+context_len-1 in rv
    Which is rv[:, start + context_len - context_len] = rv[:, start]
    → rv[:, start]  for start in [0, T-context_len-target_len]
    """
    def _get_rv_and_labels(data):
        rv     = realized_variance(data["log_returns"], context_len)   # (N, T-L)
        var    = data["variance"]                                       # (N, T+1)
        n_paths, T_rv = rv.shape
        max_start = T_rv - target_len
        rv_flat  = []
        var_flat = []
        for start in range(max_start + 1):
            rv_flat.append(rv[:, start])
            var_flat.append(var[:, start + context_len])
        return np.concatenate(rv_flat), np.concatenate(var_flat)

    rv_train, v_train = _get_rv_and_labels(train_data)
    rv_test,  v_test  = _get_rv_and_labels(test_data)

    r2  = r2_score(v_test, rv_test)
    mae = mean_absolute_error(v_test, rv_test)
    return r2, mae, rv_test, v_test


# ==========================================================================
# 2. Supervised LSTM baseline
# ==========================================================================

class VolatilityLSTM(nn.Module):
    """
    Takes a context window of log returns and directly predicts V_t.
    Trained with full supervision on V_t — this is the upper bound.
    """
    def __init__(self, input_dim=1, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()          # variance must be positive
        )

    def forward(self, x):
        # x : (B, L, 1)
        out, (h, _) = self.lstm(x)
        last = out[:, -1, :]       # take last hidden state
        return self.head(last).squeeze(-1)   # (B,)


def build_lstm_dataset(data, context_len, target_len):
    """
    Build (context_window, V_target) pairs for LSTM training.
    Same windowing as the JEPA dataset so comparisons are fair.
    """
    log_returns = data["log_returns"]   # (N, T)
    variance    = data["variance"]      # (N, T+1)
    N, T = log_returns.shape

    X_list, Y_list = [], []
    for start in range(T - context_len - target_len + 1):
        ctx = log_returns[:, start : start + context_len]      # (N, L)
        v_t = variance[:,  start + context_len]                # (N,)
        X_list.append(ctx)
        Y_list.append(v_t)

    X = np.concatenate(X_list, axis=0)[:, :, np.newaxis]      # (M, L, 1)
    Y = np.concatenate(Y_list, axis=0)                         # (M,)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32)
    )


def train_lstm(train_data, val_data, context_len, target_len, device,
               hidden_dim=64, n_layers=2, lr=1e-3, n_epochs=20, batch_size=512):

    print("\n  Building LSTM datasets …")
    # Subsample to keep training fast
    sub = np.random.default_rng(0).choice(
        train_data["log_returns"].shape[0], size=50_000, replace=False
    )
    train_sub = {k: v[sub] for k, v in train_data.items()}

    X_tr, Y_tr = build_lstm_dataset(train_sub, context_len, target_len)
    X_va, Y_va = build_lstm_dataset(val_data,  context_len, target_len)
    print(f"  Train: {X_tr.shape}   Val: {X_va.shape}")

    tr_ds = TensorDataset(X_tr, Y_tr)
    va_ds = TensorDataset(X_va, Y_va)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VolatilityLSTM(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val, best_state = float("inf"), None
    for epoch in range(1, n_epochs + 1):
        model.train()
        for X_b, Y_b in tr_ld:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X_b), Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, Y_b in va_ld:
                val_losses.append(
                    nn.functional.mse_loss(model(X_b.to(device)), Y_b.to(device)).item()
                )
        val_loss = np.mean(val_losses)
        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  LSTM Epoch {epoch:02d}/{n_epochs}  val_mse={val_loss:.6f}")

    model.load_state_dict(best_state)
    return model


def evaluate_lstm(model, test_data, context_len, target_len, device, batch_size=1024):
    X_te, Y_te = build_lstm_dataset(test_data, context_len, target_len)
    ds = TensorDataset(X_te, Y_te)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    model.eval()
    with torch.no_grad():
        for X_b, _ in ld:
            preds.append(model(X_b.to(device)).cpu().numpy())
    Y_pred = np.concatenate(preds)
    Y_true = Y_te.numpy()
    return r2_score(Y_true, Y_pred), mean_absolute_error(Y_true, Y_pred), Y_pred, Y_true


# ==========================================================================
# 3. PCA embedding visualisation
# ==========================================================================

def extract_jepa_embeddings(model, test_data, context_len, target_len, mode, device):
    ds = HestonWindowDataset(test_data, context_len, target_len, mode)
    ld = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    Z_list, V_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in ld:
            z = model.encode(batch["context"].to(device))
            Z_list.append(z.cpu().numpy())
            V_list.append(batch["target_variance"].numpy())
    return np.concatenate(Z_list), np.concatenate(V_list)


def plot_pca(Z, V_true, save_dir):
    pca  = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)

    var_explained = pca.explained_variance_ratio_

    # Subsample for plot clarity
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z_2d), size=min(8000, len(Z_2d)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: PCA coloured by V_t ──────────────────────────────────────
    sc = axes[0].scatter(
        Z_2d[idx, 0], Z_2d[idx, 1],
        c=V_true[idx], cmap="plasma",
        s=4, alpha=0.4, vmin=0, vmax=np.percentile(V_true, 95)
    )
    plt.colorbar(sc, ax=axes[0], label="True variance V_t")
    axes[0].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    axes[0].set_title(
        "JEPA embedding space (PCA)\nColoured by true variance V_t"
    )
    axes[0].grid(alpha=0.2)

    # ── Right: PC1 vs V_t scatter  ─────────────────────────────────────
    axes[1].scatter(
        Z_2d[idx, 0], V_true[idx],
        s=4, alpha=0.3, color="#3498db"
    )
    # Fit and plot a line
    m, b  = np.polyfit(Z_2d[idx, 0], V_true[idx], 1)
    x_lin = np.linspace(Z_2d[idx, 0].min(), Z_2d[idx, 0].max(), 200)
    axes[1].plot(x_lin, m * x_lin + b, "r-", lw=2,
                 label=f"Linear fit  R²={r2_score(V_true[idx], m*Z_2d[idx,0]+b):.3f}")
    r2_pc1 = r2_score(V_true[idx], m * Z_2d[idx, 0] + b)
    axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    axes[1].set_ylabel("True variance V_t")
    axes[1].set_title(
        f"PC1 vs V_t\n(R²={r2_pc1:.3f} — how much variance is in one direction)"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "PCA of JEPA Embeddings: Is V_t encoded in a single direction?",
        fontsize=13, fontweight="bold"
    )
    out = os.path.join(save_dir, "pca_embeddings.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  PCA figure → {out}")
    plt.show()
    return r2_pc1


# ==========================================================================
# 4. Final comparison plot
# ==========================================================================

def plot_comparison(results: dict, save_dir: str):
    """
    Bar chart comparing R² across all methods.
    results = {label: {"r2": float, "mae": float}}
    """
    labels = list(results.keys())
    r2s    = [results[l]["r2"]  for l in labels]
    maes   = [results[l]["mae"] for l in labels]

    colors = ["#2ecc71", "#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    # green = JEPA variants, blue = supervised, orange = stat, red = baseline
    hatches = ["", "///", "", "", ""]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(labels, r2s, color=colors[:len(labels)],
                       hatch=None, width=0.55, edgecolor="white")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R²  (higher = better)", fontsize=12)
    axes[0].set_title("Variance Explained (R²)\nz_t → V_t across methods", fontsize=12)
    for bar, val in zip(bars, r2s):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    bars2 = axes[1].bar(labels, maes, color=colors[:len(labels)],
                        width=0.55, edgecolor="white")
    axes[1].set_ylabel("MAE  (lower = better)", fontsize=12)
    axes[1].set_title("Mean Absolute Error\nV_t prediction across methods", fontsize=12)
    for bar, val in zip(bars2, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.0002,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "JEPA vs Baselines: Recovering Heston Variance V_t",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out = os.path.join(save_dir, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Comparison figure → {out}")
    plt.show()


# ==========================================================================
# Main
# ==========================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data …")
    train_data = load_split("train", DATA_DIR)
    val_data   = load_split("val",   DATA_DIR)
    test_data  = load_split("test",  DATA_DIR)

    # ── Load JEPA model ────────────────────────────────────────────────────
    print(f"\nLoading JEPA: {CHECKPOINT}")
    ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    jepa   = HestonJEPA(
        feature_dim   = config["feature_dim"],
        d_model       = config["d_model"],
        n_heads       = config["n_heads"],
        n_layers      = config["n_layers"],
        ema_momentum  = config["ema_momentum"],
        var_reg_coeff = config["var_reg_coeff"],
    ).to(DEVICE)
    jepa.load_state_dict(ckpt["model_state"])
    jepa.eval()
    print(f"  JEPA loaded (val_loss={ckpt['val_loss']:.4f})")

    # ── 1. Realized Variance ───────────────────────────────────────────────
    print("\n── 1. Realized Variance baseline ──")
    rv_r2, rv_mae, rv_pred, rv_true = evaluate_realized_variance(
        train_data, test_data, CONTEXT_LEN, TARGET_LEN
    )
    print(f"  RV   R²={rv_r2:.4f}   MAE={rv_mae:.6f}")

    # ── 2. Supervised LSTM ─────────────────────────────────────────────────
    print("\n── 2. Supervised LSTM baseline ──")
    lstm = train_lstm(
        train_data, val_data, CONTEXT_LEN, TARGET_LEN, DEVICE,
        hidden_dim=LSTM_HIDDEN, n_layers=LSTM_LAYERS,
        lr=LSTM_LR, n_epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH
    )
    lstm_r2, lstm_mae, lstm_pred, lstm_true = evaluate_lstm(
        lstm, test_data, CONTEXT_LEN, TARGET_LEN, DEVICE
    )
    print(f"  LSTM R²={lstm_r2:.4f}   MAE={lstm_mae:.6f}")

    # ── 3. PCA visualisation ───────────────────────────────────────────────
    print("\n── 3. PCA of JEPA embeddings ──")
    Z, V = extract_jepa_embeddings(
        jepa, test_data, CONTEXT_LEN, TARGET_LEN, MODE, DEVICE
    )
    r2_pc1 = plot_pca(Z, V, SAVE_DIR)
    print(f"  PC1 vs V_t  R²={r2_pc1:.4f}")

    # ── 4. Comparison summary ──────────────────────────────────────────────
    print("\n── Summary ────────────────────────────────────────────────────")
    all_results = {
        "JEPA\n(linear probe)":    {"r2": 0.754,    "mae": 0.02149},
        "JEPA\n(MLP probe)":       {"r2": 0.788,    "mae": 0.01725},
        "Supervised\nLSTM":        {"r2": lstm_r2,  "mae": lstm_mae},
        "Realized\nVariance":      {"r2": rv_r2,    "mae": rv_mae},
        "Baseline\n(mean)":        {"r2": 0.000,    "mae": 0.05515},
    }
    for name, m in all_results.items():
        print(f"  {name.replace(chr(10), ' '):30s}  R²={m['r2']:.4f}   MAE={m['mae']:.6f}")

    plot_comparison(all_results, SAVE_DIR)

    # Save all metrics
    np.savez(os.path.join(SAVE_DIR, "all_metrics.npz"),
             jepa_linear_r2 = np.float32(0.754),
             jepa_mlp_r2    = np.float32(0.788),
             lstm_r2        = np.float32(lstm_r2),
             rv_r2          = np.float32(rv_r2),
             r2_pc1         = np.float32(r2_pc1))
    print(f"\nAll metrics → {SAVE_DIR}/all_metrics.npz")


if __name__ == "__main__":
    main()