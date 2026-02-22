# src/predict_test.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from models.model import ModelConfig, FinancialTransformer
from data.dataset import ID2LABEL


@dataclass
class PredictConfig:
    # بهترین مدل ذخیره‌شده در train.py
    checkpoint_path: str = "outputs/checkpoints/best.pt"

    # مسیر فایل تست (بدون label)
    test_path: str = "src/data/sentences.csv"

    # اسم ستون متن داخل فایل تست
    text_col: str = "sentence"  # اگر فایل شما "text" دارد این را عوض کن

    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # مسیر خروجی submission
    save_path: str = "outputs/submissions/test_predictions.csv"

    # اگر بخواهی precision مثبت را بالاتر ببری
    # مثلا 0.60 بگذار — اگر نخواستی None بگذار
    positive_threshold: float | None = None


def load_model_and_tokenizer(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = FinancialTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer_name = ckpt.get("tokenizer_name", "bert-base-uncased")
    max_length = ckpt.get("max_length", model_cfg.max_length)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    return model, tokenizer, max_length


@torch.no_grad()
def batch_predict(
    model: FinancialTransformer,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
    positive_threshold: float | None,
) -> List[int]:

    preds_all: List[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="predict"):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits, _ = model(input_ids, attention_mask, return_attn=False)

        if positive_threshold is None:
            preds = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            # کلاس positive = 2 ، neutral = 1
            pos_prob = probs[:, 2]
            low_conf_pos = (preds == 2) & (pos_prob < positive_threshold)
            preds[low_conf_pos] = 1

        preds_all.extend(preds.detach().cpu().tolist())

    return preds_all


def main():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    cfg = PredictConfig()

    if not os.path.exists(cfg.test_path):
        raise FileNotFoundError(f"Test file not found: {cfg.test_path}")

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    # 1) load test file
    test_df = pd.read_csv(cfg.test_path)

    if cfg.text_col not in test_df.columns:
        raise ValueError(
            f"Column '{cfg.text_col}' not found. Available columns: {list(test_df.columns)}"
        )

    test_texts = test_df[cfg.text_col].astype(str).tolist()

    # 2) load best model + tokenizer from checkpoint
    model, tokenizer, max_length = load_model_and_tokenizer(
        cfg.checkpoint_path, cfg.device
    )

    # 3) predict
    pred_ids = batch_predict(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=cfg.device,
        max_length=max_length,
        batch_size=cfg.batch_size,
        positive_threshold=cfg.positive_threshold,
    )

    pred_labels = [ID2LABEL[i] for i in pred_ids]

    # 4) add predictions next to sentence
    out_df = test_df.copy()
    out_df["prediction"] = pred_labels
    out_df["prediction_id"] = pred_ids

    out_df.to_csv(cfg.save_path, index=False, encoding="utf-8")
    print(f"Saved predictions to: {cfg.save_path}")


if __name__ == "__main__":
    main()
