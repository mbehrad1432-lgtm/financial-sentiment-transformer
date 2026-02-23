# src/attention_viz.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from models.model import ModelConfig, FinancialTransformer

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/attention_viz.py -> root


@dataclass
class AttnVizConfig:
    checkpoint_path: str = str(PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt")
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = str(PROJECT_ROOT / "outputs" / "figures" / "attn_heatmaps")
    num_examples: int = 5

    test_csv_path: Optional[str] = str(PROJECT_ROOT / "data" / "sentences.csv")
    test_text_col: str = "sentence"


def load_model(ckpt_path: str, device: str) -> FinancialTransformer:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = FinancialTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def read_test_sentences(cfg: AttnVizConfig) -> List[str]:
    """
    طبق توضیح شما: در پروژه معمولاً train در دسترس نیست و تست یک فایل جداست.
    پس اگر test_csv_path را دادید، از همان می‌خوانیم.
    """
    if cfg.test_csv_path is None:
        raise RuntimeError(
            "برای بخش ۶ باید ۵ جمله از داده TEST داشته باشیم. "
            "مسیر فایل تست را در test_csv_path تنظیم کن."
        )

    import pandas as pd
    df = pd.read_csv(cfg.test_csv_path)
    if cfg.test_text_col not in df.columns:
        raise ValueError(f"Column '{cfg.test_text_col}' not found in test CSV. Columns: {df.columns.tolist()}")

    texts = [str(x) for x in df[cfg.test_text_col].tolist()]
    return texts


def plot_heatmap(attn_2d: np.ndarray, tokens: List[str], save_path: str, title: str):
    """
    attn_2d: (T, T)
    tokens: length T (فقط توکن‌های non-pad)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_2d, aspect="auto")
    plt.title(title)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")

    # برای خوانایی: اگر توکن‌ها زیادند، تعداد برچسب‌ها را محدود کنیم
    T = len(tokens)
    step = max(1, T // 25)  # حدوداً 25 برچسب
    idx = list(range(0, T, step))

    plt.xticks(idx, [tokens[i] for i in idx], rotation=90)
    plt.yticks(idx, [tokens[i] for i in idx])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def main():
    cfg = AttnVizConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    model = load_model(cfg.checkpoint_path, cfg.device)

    test_texts = read_test_sentences(cfg)
    examples = test_texts[: cfg.num_examples]

    for i, text in enumerate(examples, start=1):
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(cfg.device)
        attention_mask = enc["attention_mask"].to(cfg.device)

        logits, last_attn = model(input_ids, attention_mask, return_attn=True)
        pred = logits.argmax(dim=-1).item()

        if last_attn is None:
            raise RuntimeError("مدل last_attn را برنگرداند. بررسی کن TransformerBlock return_attn را درست پیاده کرده باشد.")

        # last_attn: (B, H, T, T) -> (H, T, T)
        attn = last_attn[0].detach().cpu().numpy()

        # میانگین روی headها: (T, T)
        attn_mean = attn.mean(axis=0)

        # فقط توکن‌های واقعی (non-pad) را نگه داریم
        mask = attention_mask[0].detach().cpu().numpy().astype(bool)
        valid_idx = np.where(mask)[0]
        attn_mean = attn_mean[np.ix_(valid_idx, valid_idx)]

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        tokens = [tokens[j] for j in valid_idx.tolist()]

        save_path = os.path.join(cfg.save_dir, f"attn_example_{i}.png")
        title = f"Example {i} | pred_id={pred} | text: {text[:60]}..."
        plot_heatmap(attn_mean, tokens, save_path, title)

        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()