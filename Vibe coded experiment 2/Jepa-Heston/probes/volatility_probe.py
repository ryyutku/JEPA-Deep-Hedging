"""
probes/volatility_probe.py

Answers the core thesis question:
    "Does the JEPA latent vector z_t encode the true Heston variance V_t?"

Method
------
1. Load the frozen JEPA context encoder (no fine-tuning).
2. Encode every context window in the TEST set → z  shape (N_samples, d_model).
3. Fit a LINEAR regression:  z → V_t  (the true variance at the target start).
   Linear-only, because if the relationship is linear we know z explicitly
   encodes V_t as a feature, not just in some entangled nonlinear way.
4. Report R², MAE, and a scatter plot.

We also fit a NONLINEAR probe (3-layer MLP) to see how much extra information
is accessible with more capacity — the gap tells you whether V_t is encoded
linearly or only nonlinearly.

Run from project root:
    python probes/volatility_probe.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from heston_sim.generate import load_split
from jepa.dataset        import HestonWindowDataset
from jepa.model          import HestonJEPA

# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINT  = "checkpoints/jepa_prices.pt"
DATA_DIR    = "data"
MODE        = "prices"
CONTEXT_LEN = 10
TARGET_LEN  = 5
BATCH_SIZE  = 1024
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "results"
# ──────────────────────────────────────────────────────────────────────────


def extract_embeddings(model, dataset, device, batch_size=1024):
    """
    Run the frozen context encoder over the full dataset.
    Returns:
        Z      : (N, d_model)  embeddings
        V_true : (N,)          true variance at target start step
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    Z_list, V_list = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            context = batch["context"].to(device)
            z = model.encode(context)                   # (B, d_model)
            Z_list.append(z.cpu().numpy())
            V_list.append(batch["target_variance"].numpy())

    Z      = np.concatenate(Z_list, axis=0)             # (N, d_model)
    V_true = np.concatenate(V_list, axis=0)             # (N,)
    return Z, V_true


def run_probes(Z_train, V_train, Z_test, V_test):
    """Fit linear and nonlinear probes, return results dict."""
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Z_train)
    Z_te = scaler.transform(Z_test)

    results = {}

    # ── Linear probe (Ridge regression) ───────────────────────────────────
    linear = Ridge(alpha=1.0)
    linear.fit(Z_tr, V_train)
    V_pred_lin = linear.predict(Z_te)
    results["linear"] = {
        "r2":      r2_score(V_test, V_pred_lin),
        "mae":     mean_absolute_error(V_test, V_pred_lin),
        "V_pred":  V_pred_lin,
        "label":   "Linear probe (Ridge)",
    }

    # ── Nonlinear probe (MLP) ─────────────────────────────────────────────
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(Z_tr, V_train)
    V_pred_mlp = mlp.predict(Z_te)
    results["mlp"] = {
        "r2":      r2_score(V_test, V_pred_mlp),
        "mae":     mean_absolute_error(V_test, V_pred_mlp),
        "V_pred":  V_pred_mlp,
        "label":   "Nonlinear probe (MLP)",
    }

    # ── Naive baseline: predict the mean variance ──────────────────────────
    V_pred_mean = np.full_like(V_test, V_train.mean())
    results["baseline"] = {
        "r2":      r2_score(V_test, V_pred_mean),
        "mae":     mean_absolute_error(V_test, V_pred_mean),
        "V_pred":  V_pred_mean,
        "label":   "Baseline (predict mean)",
    }

    return results


def plot_results(V_test, results, training_log, save_dir):
    """Four-panel figure: training curve + 3 probe scatter plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 0: Training curve ───────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    epochs = training_log["epoch"]
    ax0.plot(epochs, training_log["train_loss"], label="Train loss", lw=2)
    ax0.plot(epochs, training_log["val_loss"],   label="Val loss",   lw=2, linestyle="--")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("JEPA loss (MSE in latent space)")
    ax0.set_title("JEPA Training Curve")
    ax0.legend()
    ax0.grid(alpha=0.3)

    # ── Panel 1: R² bar chart ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    names  = [r["label"].split("(")[0].strip() for r in results.values()]
    r2vals = [r["r2"] for r in results.values()]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax1.bar(names, r2vals, color=colors, width=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R²  (higher = better)")
    ax1.set_title("Probe R²: z_t → V_t")
    for bar, val in zip(bars, r2vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panels 2-4: Scatter plots for each probe ──────────────────────────
    probe_keys = ["linear", "mlp", "baseline"]
    positions  = [gs[1, 0], gs[1, 1], gs[1, 2]]

    # subsample for scatter readability
    idx = np.random.choice(len(V_test), size=min(5000, len(V_test)), replace=False)

    for pos, key in zip(positions, probe_keys):
        ax = fig.add_subplot(pos)
        r  = results[key]
        ax.scatter(V_test[idx], r["V_pred"][idx],
                   alpha=0.15, s=4, color="#3498db" if key != "baseline" else "#e74c3c")
        vmin = V_test.min(); vmax = V_test.max()
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("True variance V_t")
        ax.set_ylabel("Predicted variance")
        ax.set_title(f"{r['label']}\nR²={r['r2']:.3f}  MAE={r['mae']:.5f}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "JEPA Volatility Probe: Does z_t encode the Heston variance V_t?",
        fontsize=14, fontweight="bold", y=1.01
    )

    out = os.path.join(save_dir, "probe_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Figure saved → {out}")
    plt.show()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"  Trained config: {config}")

    model = HestonJEPA(
        feature_dim   = config["feature_dim"],
        d_model       = config["d_model"],
        n_heads       = config["n_heads"],
        n_layers      = config["n_layers"],
        ema_momentum  = config["ema_momentum"],
        var_reg_coeff = config["var_reg_coeff"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded (val_loss={ckpt['val_loss']:.4f})")

    # ── Load training log ─────────────────────────────────────────────────
    log_path = CHECKPOINT.replace(".pt", "_log.npz")
    log      = np.load(log_path)
    training_log = {k: log[k] for k in log.files}

    # ── Build datasets ────────────────────────────────────────────────────
    # Use a subsample of train for probe fitting (full set is too slow for sklearn)
    train_data = load_split("train", DATA_DIR)
    test_data  = load_split("test",  DATA_DIR)

    # Subsample training paths for probe fitting (50k paths is plenty)
    sub_idx = np.random.default_rng(42).choice(
        train_data["log_returns"].shape[0], size=50_000, replace=False
    )
    train_sub = {k: v[sub_idx] for k, v in train_data.items()}

    train_ds = HestonWindowDataset(train_sub,  CONTEXT_LEN, TARGET_LEN, MODE)
    test_ds  = HestonWindowDataset(test_data,  CONTEXT_LEN, TARGET_LEN, MODE)

    # ── Extract embeddings ────────────────────────────────────────────────
    print("\nExtracting train embeddings …")
    Z_train, V_train = extract_embeddings(model, train_ds, DEVICE, BATCH_SIZE)
    print(f"  Z_train: {Z_train.shape}   V_train: {V_train.shape}")

    print("Extracting test  embeddings …")
    Z_test,  V_test  = extract_embeddings(model, test_ds,  DEVICE, BATCH_SIZE)
    print(f"  Z_test:  {Z_test.shape}   V_test:  {V_test.shape}")

    # ── Run probes ────────────────────────────────────────────────────────
    print("\nFitting probes …")
    results = run_probes(Z_train, V_train, Z_test, V_test)

    print("\n── Probe Results ──────────────────────────────────────────")
    for key, r in results.items():
        print(f"  {r['label']:35s}  R²={r['r2']:.4f}   MAE={r['mae']:.6f}")

    # Save numeric results
    np.savez(
        os.path.join(SAVE_DIR, "probe_metrics.npz"),
        linear_r2   = results["linear"]["r2"],
        linear_mae  = results["linear"]["mae"],
        mlp_r2      = results["mlp"]["r2"],
        mlp_mae     = results["mlp"]["mae"],
        baseline_r2 = results["baseline"]["r2"],
        baseline_mae= results["baseline"]["mae"],
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nPlotting …")
    plot_results(V_test, results, training_log, SAVE_DIR)

    print("\nDone. Results in results/")


if __name__ == "__main__":
    main()