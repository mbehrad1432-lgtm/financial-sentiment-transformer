# src/models/model.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from .blocks import TransformerBlock


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding:
      PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # register_buffer => moved with device, not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,d_model)
        """
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


def masked_mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B,T,d_model)
    attention_mask: (B,T) 1 for tokens, 0 for padding
    returns: (B,d_model)
    """
    mask = attention_mask.unsqueeze(-1).to(x.dtype)  # (B,T,1)
    x = x * mask
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B,1)
    return x.sum(dim=1) / denom


@dataclass
class ModelConfig:
    vocab_size: int
    max_length: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    num_classes: int = 3


class FinancialTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, embedding_matrix: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg

        # --- Token embeddings ---
        # If we pass a pretrained matrix (GloVe-based), use it.
        # Otherwise fall back to a normal nn.Embedding.
        if embedding_matrix is None:
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self._use_pretrained = False
        else:
            # embedding_matrix should already be on the correct device
            self.token_emb = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=False
            )
            self._use_pretrained = True

        # --- Positional encoding (this is what was missing!) ---
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.max_length,
            dropout=cfg.dropout,
        )

        # --- Transformer encoder blocks ---
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.num_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # Only re-init token_emb if we're NOT using pretrained embeddings.
        if not getattr(self, "_use_pretrained", False):
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: (B,T)
        attention_mask: (B,T)
        returns:
          logits: (B,num_classes)
          last_attn: (B,H,T,T) if return_attn else None
        """
        # (B,T,d_model)
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        last_attn = None
        for i, block in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            if return_attn and is_last:
                x, last_attn = block(
                    x,
                    attention_mask=attention_mask,
                    return_attn=True,
                )
            else:
                x, _ = block(
                    x,
                    attention_mask=attention_mask,
                    return_attn=False,
                )

        pooled = masked_mean_pooling(x, attention_mask)  # (B,d_model)
        logits = self.classifier(pooled)  # (B,C)

        return logits, last_attn

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 8
    vocab_size = 30522

    cfg = ModelConfig(vocab_size=vocab_size, max_length=64, d_model=32, num_heads=4, num_layers=2, d_ff=64)
    model = FinancialTransformer(cfg)

    input_ids = torch.randint(0, vocab_size, (B, T))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    logits, attn = model(input_ids, mask, return_attn=True)
    print("logits shape:", logits.shape)  # (B,3)
    print("attn shape:", attn.shape)      # (B,H,T,T)
