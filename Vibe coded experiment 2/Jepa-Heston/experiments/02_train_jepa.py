"""
experiments/02_train_jepa.py

Trains the JEPA model on the simulated Heston paths.
Run from the project root:

    python experiments/02_train_jepa.py

Checkpoints are saved to  checkpoints/jepa_<mode>.pt
Training curves are saved to  checkpoints/jepa_<mode>_log.npz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from heston_sim.generate import load_split
from jepa.dataset        import HestonWindowDataset
from jepa.model          import HestonJEPA

# ---------------------------------------------------------------------------
# Config  — edit these to experiment
# ---------------------------------------------------------------------------
MODE         = "prices"   # "prices" or "oracle"
CONTEXT_LEN  = 10         # L : steps the encoder sees
TARGET_LEN   = 5          # H : steps the predictor must match
D_MODEL      = 64         # latent dimension
N_HEADS      = 4
N_LAYERS     = 2
EMA_MOMENTUM = 0.99
VAR_REG      = 0.01       # variance collapse penalty weight

LR           = 3e-4
BATCH_SIZE   = 512
N_EPOCHS     = 30
LOG_EVERY    = 100        # batches
VAL_EVERY    = 1          # epochs

SAVE_DIR     = "checkpoints"
DATA_DIR     = "data"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Mode   : {MODE}  (context={CONTEXT_LEN}, target={TARGET_LEN})")

    # ── Data ──────────────────────────────────────────────────────────────
    train_data = load_split("train", DATA_DIR)
    val_data   = load_split("val",   DATA_DIR)

    train_ds = HestonWindowDataset(train_data, CONTEXT_LEN, TARGET_LEN, MODE)
    val_ds   = HestonWindowDataset(val_data,   CONTEXT_LEN, TARGET_LEN, MODE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples : {len(train_ds):,}")
    print(f"Val   samples : {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    feature_dim = train_ds.feature_dim   # 1 for prices, 2 for oracle
    model = HestonJEPA(
        feature_dim   = feature_dim,
        d_model       = D_MODEL,
        n_heads       = N_HEADS,
        n_layers      = N_LAYERS,
        ema_momentum  = EMA_MOMENTUM,
        var_reg_coeff = VAR_REG,
    ).to(DEVICE)
    print(f"Trainable params : {model.count_parameters():,}")

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=N_EPOCHS
    )

    # ── Training loop ─────────────────────────────────────────────────────
    log = {"train_loss": [], "val_loss": [], "epoch": []}
    best_val = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for step, batch in enumerate(train_loader):
            context = batch["context"].to(DEVICE)   # (B, L, C)
            target  = batch["target"].to(DEVICE)    # (B, H, C)

            optimiser.zero_grad()
            loss, metrics, _ = model(context, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            model.ema_update()

            epoch_losses.append(metrics["total_loss"])

            if (step + 1) % LOG_EVERY == 0:
                print(f"  E{epoch:02d} step {step+1:>5d} | "
                      f"loss={metrics['total_loss']:.4f}  "
                      f"pred={metrics['pred_loss']:.4f}  "
                      f"var_reg={metrics['var_reg']:.4f}")

        train_loss = np.mean(epoch_losses)
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        if epoch % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    context = batch["context"].to(DEVICE)
                    target  = batch["target"].to(DEVICE)
                    loss, metrics, _ = model(context, target)
                    val_losses.append(metrics["total_loss"])
            val_loss = np.mean(val_losses)

            print(f"Epoch {epoch:02d}/{N_EPOCHS} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["val_loss"].append(val_loss)

            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(SAVE_DIR, f"jepa_{MODE}.pt")
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                    "config": dict(
                        feature_dim   = feature_dim,
                        d_model       = D_MODEL,
                        n_heads       = N_HEADS,
                        n_layers      = N_LAYERS,
                        ema_momentum  = EMA_MOMENTUM,
                        var_reg_coeff = VAR_REG,
                        context_len   = CONTEXT_LEN,
                        target_len    = TARGET_LEN,
                        mode          = MODE,
                    ),
                }, ckpt_path)
                print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    # Save training log
    log_path = os.path.join(SAVE_DIR, f"jepa_{MODE}_log.npz")
    np.savez(log_path, **{k: np.array(v) for k, v in log.items()})
    print(f"\nTraining log → {log_path}")
    print(f"Best val loss : {best_val:.4f}")


if __name__ == "__main__":
    main()