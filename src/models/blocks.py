# src/models/blocks.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise FeedForward network:
        Linear(d_model -> d_ff)
        GELU
        Dropout
        Linear(d_ff -> d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Multi-head attention
        attn_out, attn_weights = self.mha(
            x,
            attention_mask=attention_mask,
            return_attn=return_attn,
        )

        # Residual + Norm
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FeedForward
        ffn_out = self.ffn(x)

        # Residual + Norm
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        if return_attn:
            return x, attn_weights
        return x, None


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model = 2, 6, 32
    H = 4
    d_ff = 64

    x = torch.randn(B, T, d_model)
    mask = torch.ones(B, T)

    block = TransformerBlock(d_model=d_model, num_heads=H, d_ff=d_ff)
    out, attn = block(x, attention_mask=mask, return_attn=True)

    print("block out shape:", out.shape)      # (B,T,d_model)
    print("attn shape:", attn.shape)          # (B,H,T,T)
