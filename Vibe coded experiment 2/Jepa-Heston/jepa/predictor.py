"""
jepa/predictor.py

The predictor maps the context embedding z_context → ẑ_target.

It is deliberately small — a narrow MLP — so the heavy lifting of
understanding the sequence is done by the encoder, not the predictor.
This is a key JEPA design principle: the predictor should not be able
to "cheat" by doing all the work itself.

Architecture
------------
    z_context : (B, d_model)
    → Linear → GELU → Linear → GELU → Linear
    ẑ_target  : (B, d_model)
"""

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    A 3-layer MLP mapping context embedding to predicted target embedding.

    Parameters
    ----------
    d_model    : int   must match the encoder's d_model
    hidden_dim : int   width of the hidden layers (default: d_model // 2)
    dropout    : float
    """

    def __init__(
        self,
        d_model:    int = 64,
        hidden_dim: int = None,
        dropout:    float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """
        z_context : (B, d_model)
        returns   : (B, d_model)   predicted target embedding
        """
        return self.net(z_context)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pred = JEPAPredictor(d_model=64)
    z    = torch.randn(32, 64)
    z_hat = pred(z)
    print(f"Input  : {z.shape}")       # (32, 64)
    print(f"Output : {z_hat.shape}")   # (32, 64)
    print(f"Params : {sum(p.numel() for p in pred.parameters()):,}")