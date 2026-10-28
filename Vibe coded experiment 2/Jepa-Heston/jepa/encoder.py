"""
jepa/encoder.py

The context encoder and target encoder share the same architecture.
They map a variable-length window of market features to a single latent vector z.

Architecture
------------
    input  : (B, window_len, feature_dim)
    → linear projection to d_model
    → positional encoding
    → N Transformer encoder layers
    → mean-pool across the time axis
    output : (B, d_model)

Why a Transformer?
  The Heston variance process has memory (mean-reversion).
  A Transformer can in principle learn to attend to whichever
  past steps are most informative about the current variance state,
  without baking in a fixed decay schedule.

Why mean-pool?
  We want a fixed-size z regardless of window length.
  Mean-pooling is simple and works well in practice for time series.
  A [CLS] token is an alternative if you want to experiment later.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class JEPAEncoder(nn.Module):
    """
    Maps a window of market observations → a latent vector z.

    Parameters
    ----------
    feature_dim : int   number of input features per timestep (1 or 2)
    d_model     : int   internal & output embedding dimension
    n_heads     : int   attention heads (d_model must be divisible by n_heads)
    n_layers    : int   Transformer encoder layers
    dropout     : float
    """

    def __init__(
        self,
        feature_dim: int = 1,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Project raw features up to d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = n_heads,
            dim_feedforward = d_model * 4,
            dropout    = dropout,
            batch_first = True,        # (B, T, d_model) convention
            norm_first  = True,        # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, window_len, feature_dim)
        returns z : (B, d_model)
        """
        x = self.input_proj(x)         # (B, T, d_model)
        x = self.pos_enc(x)            # add positional info
        x = self.transformer(x)        # (B, T, d_model)
        z = x.mean(dim=1)              # mean-pool → (B, d_model)
        z = self.norm(z)
        return z


# ---------------------------------------------------------------------------
# Quick shape check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enc = JEPAEncoder(feature_dim=1, d_model=64, n_heads=4, n_layers=2)
    x   = torch.randn(32, 10, 1)      # batch=32, context_len=10, features=1
    z   = enc(x)
    print(f"Input  : {x.shape}")      # (32, 10, 1)
    print(f"Output : {z.shape}")      # (32, 64)
    print(f"Params : {sum(p.numel() for p in enc.parameters()):,}")