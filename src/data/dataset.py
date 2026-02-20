# src/data/dataset.py
from __future__ import annotations
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class DataConfig:
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 64
    train_frac: float = 0.8
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0


def load_phrasebank_dataframe(
    seed: int = 42,
    variant: str = "sentences_allagree",
) -> pd.DataFrame:
    """
    Loads FinancialPhraseBank from HuggingFace and returns a clean DataFrame
    with columns: text, label_str, label.
    variant examples:
      - "sentences_allagree"
      - "sentences_75agree"
      - "sentences_66agree"
      - "sentences_50agree"
    """
    ds = load_dataset("financial_phrasebank", variant, trust_remote_code=True)

    # ds["train"] contains the full set for that variant
    df = pd.DataFrame(ds["train"])

    # Expected columns: "sentence", "label"
    # label is usually int (0,1,2) in HF dataset; sentence is text
    # But we'll normalize to label_str + label_id anyway for robustness.
    if "sentence" not in df.columns:
        raise ValueError(f"Expected 'sentence' column, got: {df.columns.tolist()}")

    # HF dataset typically: label 0: negative, 1: neutral, 2: positive
    # We'll store both numeric and string forms.
    df = df.rename(columns={"sentence": "text"})
    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column, got: {df.columns.tolist()}")

    # ensure int labels
    df["label"] = df["label"].astype(int)

    # map to strings (assumes HF order: 0 neg, 1 neu, 2 pos)
    hf_id2label = {0: "negative", 1: "neutral", 2: "positive"}
    df["label_str"] = df["label"].map(hf_id2label)

    # reorder columns
    df = df[["text", "label_str", "label"]].copy()

    # shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# def train_val_split(
#     df: pd.DataFrame,
#     train_frac: float = 0.8,
#     seed: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Deterministic split (shuffle already done in load, but we keep it robust).
#     """
#     if not (0.0 < train_frac < 1.0):
#         raise ValueError("train_frac must be in (0, 1)")

#     df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
#     n_train = int(len(df) * train_frac)
#     train_df = df.iloc[:n_train].reset_index(drop=True)
#     val_df = df.iloc[n_train:].reset_index(drop=True)
#     return train_df, val_df
def train_val_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split to keep class distribution similar in train/val.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    train_df, val_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=df["label"],   # <-- این مهم‌ترین خطه
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)




class PhraseBankDataset(Dataset):
    """
    Returns dict with:
      - input_ids: LongTensor [max_length]
      - attention_mask: LongTensor [max_length]
      - labels: LongTensor []
      - text (optional, for debugging/visualization)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 64,
        return_text: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_text = return_text

        # basic checks
        for col in ["text", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in df.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0).long(),
            "attention_mask": enc["attention_mask"].squeeze(0).long(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        if self.return_text:
            # keep as python string (not tensor)
            item["text"] = text
        return item


def build_dataloaders(
    cfg: DataConfig,
    variant: str = "sentences_allagree",
    return_text_in_val: bool = True,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer, Dict[int, str]]:
    """
    Creates train/val loaders + tokenizer.
    """
    df = load_phrasebank_dataframe(seed=cfg.seed, variant=variant)
    train_df, val_df = train_val_split(df, train_frac=cfg.train_frac, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

    train_ds = PhraseBankDataset(
        train_df, tokenizer=tokenizer, max_length=cfg.max_length, return_text=False
    )
    val_ds = PhraseBankDataset(
        val_df, tokenizer=tokenizer, max_length=cfg.max_length, return_text=return_text_in_val
    )
   #################################################################################

        # ---- WeightedRandomSampler (balance classes in train batches) ----
    labels_tensor = torch.tensor(train_df["label"].values, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=3).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels_tensor]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    ################################################################################## 



    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,          # <-- به جای shuffle
        shuffle=False,            # <-- باید False باشد
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, tokenizer, ID2LABEL


if __name__ == "__main__":
    cfg = DataConfig()
    train_loader, val_loader, tokenizer, id2label = build_dataloaders(cfg)

    batch = next(iter(train_loader))
    print("train batch shapes:")
    print("input_ids:", tuple(batch["input_ids"].shape))
    print("attention_mask:", tuple(batch["attention_mask"].shape))
    print("labels:", tuple(batch["labels"].shape))

    vbatch = next(iter(val_loader))
    print("\nval batch keys:", vbatch.keys())
    if "text" in vbatch:
        print("example text:", vbatch["text"][0][:80], "...")
