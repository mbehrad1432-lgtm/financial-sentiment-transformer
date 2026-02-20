# src/models/attention.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _apply_attention_mask(scores: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    scores: (B, H, T, T)
    attention_mask: (B, T) with 1 for real tokens and 0 for padding
    We mask keys (last dimension T) so that attention does not attend to padding positions.
    """
    if attention_mask is None:
        return scores

    # (B, 1, 1, T) for broadcasting over heads and query positions
    key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
    scores = scores.masked_fill(~key_mask, float("-inf"))
    return scores


class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    This module operates on *projected* Q,K,V already shaped for multi-head:
        Q,K,V: (B, H, T, d_k)
    Returns:
        context: (B, H, T, d_k)
        attn_weights: (B, H, T, T)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k,v: (B,H,T,d_k)
        d_k = q.size(-1)

        # scores: (B,H,T,T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # mask padding tokens (keys)
        scores = _apply_attention_mask(scores, attention_mask)

        # softmax over keys dimension
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: (B,H,T,d_k)
        context = torch.matmul(attn_weights, v)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention:
    - Projects x to Q,K,V
    - Splits into heads
    - Applies scaled dot-product attention
    - Concatenates heads and projects back
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = SelfAttention(dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,d_model) -> (B,H,T,d_k)
        """
        b, t, _ = x.shape
        x = x.view(b, t, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B,H,T,d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,H,T,d_k) -> (B,T,d_model)
        """
        b, h, t, d_k = x.shape
        x = x.transpose(1, 2).contiguous()  # (B,T,H,d_k)
        return x.view(b, t, h * d_k)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B,T,d_model)
        attention_mask: (B,T) with 1/0
        return_attn: if True, also return attn_weights (B,H,T,T)
        """
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        context, attn_weights = self.attn(q, k, v, attention_mask=attention_mask)
        merged = self._merge_heads(context)  # (B,T,d_model)

        out = self.out_proj(merged)
        out = self.out_dropout(out)

        if return_attn:
            return out, attn_weights
        return out, None


if __name__ == "__main__":
    # quick sanity check
    torch.manual_seed(0)
    B, T, d_model, H = 2, 5, 16, 4
    x = torch.randn(B, T, d_model)
    mask = torch.tensor([[1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1]], dtype=torch.long)

    mha = MultiHeadAttention(d_model=d_model, num_heads=H, dropout=0.0)
    y, attn = mha(x, attention_mask=mask, return_attn=True)

    print("y shape:", y.shape)          # (B,T,d_model)
    print("attn shape:", attn.shape)    # (B,H,T,T)
    # Check masked positions: keys at padding columns should be ~0 after softmax
    print("attn[0] sum over keys:", attn[0, 0, 0].sum().item())
    print("attn[0] padding key probs:", attn[0, 0, 0, 3:].detach().cpu().numpy())
