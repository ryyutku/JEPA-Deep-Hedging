"""
jepa/model.py

Assembles the full JEPA model:

    context_encoder  (trained by gradient descent)
    target_encoder   (EMA copy of context_encoder — never directly optimised)
    predictor        (trained by gradient descent)

Training step (called from the training script):
    1.  z_c  = context_encoder(context_window)           grad flows
    2.  ẑ_t  = predictor(z_c)                            grad flows
    3.  z_t  = target_encoder(target_window)             NO grad (stop-gradient)
    4.  loss = MSE(ẑ_t, z_t) / d_model                  normalised
    5.  loss.backward()  →  update context_encoder + predictor params
    6.  ema_update()     →  soft-update target_encoder from context_encoder

Why EMA (not shared weights, not a second optimiser)?
------------------------------------------------------
If both encoders were identical, the model could collapse — both would map
everything to the same constant vector and the loss would trivially be zero.
The EMA target encoder moves slowly, giving the context encoder a stable
prediction target that still evolves as training progresses.
Momentum τ ≈ 0.99 is standard (BYOL, I-JEPA).

Variance collapse guard
-----------------------
We add a very small variance regulariser that penalises embeddings whose
variance across the batch is near zero.  This is a lightweight alternative
to VICReg / Barlow Twins — enough for a thesis-scale experiment.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder   import JEPAEncoder
from .predictor import JEPAPredictor


class HestonJEPA(nn.Module):
    """
    Parameters
    ----------
    feature_dim   : int    1 (prices mode) or 2 (oracle mode)
    d_model       : int    latent dimension
    n_heads       : int    Transformer attention heads
    n_layers      : int    Transformer encoder layers
    ema_momentum  : float  τ in  θ_target ← τ·θ_target + (1-τ)·θ_context
    var_reg_coeff : float  weight of variance-collapse penalty (set 0 to disable)
    dropout       : float
    """

    def __init__(
        self,
        feature_dim:   int   = 1,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 2,
        ema_momentum:  float = 0.99,
        var_reg_coeff: float = 0.01,
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.d_model       = d_model
        self.ema_momentum  = ema_momentum
        self.var_reg_coeff = var_reg_coeff

        # Trainable components
        self.context_encoder = JEPAEncoder(feature_dim, d_model, n_heads, n_layers, dropout)
        self.predictor       = JEPAPredictor(d_model, dropout=dropout)

        # EMA target encoder — deep copy, never in the optimiser
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward pass — returns loss + diagnostic dict
    # ------------------------------------------------------------------
    def forward(self, context: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        context : (B, L, C)   context window
        target  : (B, H, C)   target window

        Returns
        -------
        loss     : scalar Tensor
        metrics  : dict with 'pred_loss', 'var_reg', 'total_loss'
        z_context: (B, d_model) — the useful embedding, returned for probes
        """
        # Step 1 & 2 : context → embedding → prediction
        z_c   = self.context_encoder(context)      # (B, d_model)
        z_hat = self.predictor(z_c)                # (B, d_model) — predicted

        # Step 3 : target embedding (no gradient)
        with torch.no_grad():
            z_t = self.target_encoder(target)      # (B, d_model)

        # Step 4 : prediction loss (normalised MSE)
        pred_loss = F.mse_loss(z_hat, z_t)

        # Variance regulariser — penalise collapsed dimensions
        var_reg = self._variance_reg(z_c)

        loss = pred_loss + self.var_reg_coeff * var_reg

        metrics = {
            "pred_loss":  pred_loss.item(),
            "var_reg":    var_reg.item(),
            "total_loss": loss.item(),
        }

        return loss, metrics, z_c.detach()

    # ------------------------------------------------------------------
    # EMA update  — call AFTER every optimiser step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self):
        """Soft-update target encoder weights from context encoder."""
        τ = self.ema_momentum
        for p_c, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.mul_(τ).add_((1.0 - τ) * p_c.data)

    # ------------------------------------------------------------------
    # Inference: encode a context window → z  (no target needed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, context: torch.Tensor) -> torch.Tensor:
        """
        context : (B, L, C)
        returns   (B, d_model)
        Used at eval time and by the probe.
        """
        self.context_encoder.eval()
        z = self.context_encoder(context)
        return z

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _variance_reg(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encourage each embedding dimension to have non-trivial variance
        across the batch.  Loss = mean over dims of max(0, 1 - std_d).
        Adapted from VICReg (Barron et al. 2022).
        """
        std = z.std(dim=0)                          # (d_model,)
        return F.relu(1.0 - std).mean()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = HestonJEPA(feature_dim=1, d_model=64)
    print(f"Trainable params: {model.count_parameters():,}")

    ctx = torch.randn(32, 10, 1)   # batch=32, L=10, C=1
    tgt = torch.randn(32,  5, 1)   # batch=32, H=5,  C=1

    loss, metrics, z_c = model(ctx, tgt)
    print(f"Loss    : {loss.item():.4f}")
    print(f"Metrics : {metrics}")
    print(f"z_c     : {z_c.shape}")   # (32, 64)

    model.ema_update()
    print("EMA update OK")