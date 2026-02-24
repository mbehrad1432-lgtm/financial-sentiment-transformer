# src/train.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sympy import gamma
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from data.dataset import DataConfig, build_dataloaders
from models import model
from models.model import ModelConfig, FinancialTransformer

import torch.nn.functional as F

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss
    - gamma: شدت تمرکز روی نمونه‌های سخت (معمولاً 1 تا 3؛ پیشنهاد: 2)
    - alpha: وزن کلاس‌ها (Tensor شکل [C]) مثل همون weights که ساختی (اختیاری)
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=-1)          # [B, C]
        probs = torch.exp(log_probs)                       # [B, C]

        # log(p_t) and p_t for the true class
        targets = targets.long()
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B]
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)          # [B]

        # alpha weight per sample
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=0, index=targets)  # [B]
        else:
            alpha_t = 1.0

        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt   # [B]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainConfig:
    variant: str = "sentences_allagree"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 120   #10
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    save_dir: str = str(PROJECT_ROOT / "outputs" / "checkpoints")
    best_name: str = "best.pt"
    log_csv: str = str(PROJECT_ROOT / "outputs" / "logs" / "train_log.csv")
    curve_path: str = str(PROJECT_ROOT / "outputs" / "figures" / "training_curves.png")

# @torch.no_grad()
# def evaluate(model: nn.Module, loader, device: str, criterion: nn.Module) -> Dict[str, float]:
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_count = 0

# #    criterion = nn.CrossEntropyLoss()

#     for batch in loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         logits, _ = model(input_ids, attention_mask, return_attn=False)
#         loss = criterion(logits, labels)

#         preds = logits.argmax(dim=-1)
#         total_correct += (preds == labels).sum().item()
#         total_loss += loss.item() * labels.size(0)
#         total_count += labels.size(0)

#     return {
#         "loss": total_loss / max(total_count, 1),
#         "acc": total_correct / max(total_count, 1),
#     }
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, criterion=None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds = []
    all_labels = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
       
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask, return_attn=False)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)

        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
        "macro_f1": float(macro_f1),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: str,
    grad_clip: float = 1.0,
    criterion: nn.Module = None,
) -> Dict[str, float]:
    model.train()
    if criterion is None:



        #criterion = nn.CrossEntropyLoss()
        gamma = 2.0
        criterion = FocalLoss(gamma=gamma, alpha=weights)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(input_ids, attention_mask, return_attn=False)
        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=loss.item())
    train_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
        "macro_f1": float(train_macro_f1),
    }

def main():
    # make torch import stable on some windows setups
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    train_cfg = TrainConfig()
    data_cfg = DataConfig(
        tokenizer_name="bert-base-uncased",
        max_length=128, #128
        train_frac=0.8,
        seed=train_cfg.seed,
        batch_size=32,
        num_workers=0,
    )

    set_seed(train_cfg.seed)
    device = train_cfg.device  # IMPORTANT: define early

    # --------------------
    # 1) Data
    # --------------------
    train_loader, val_loader, tokenizer, id2label = build_dataloaders(
        data_cfg, variant=train_cfg.variant, return_text_in_val=False
    )

    # --------------------
    # 2) Pretrained embeddings (GloVe)  <-- allowed part
    # --------------------
    embedding_dim = 300  # GloVe 100d
    vocab_size = tokenizer.vocab_size

    # default random init for all tokens
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.01

    # IMPORTANT: use an absolute path relative to PROJECT_ROOT
    glove_path = PROJECT_ROOT / "data" / "embeddings" / "glove.42B.300d.txt"
    print("Loading GloVe from:", glove_path)

    # Fail fast with a helpful message
    if not glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found at: {glove_path}\n"
            f"Put glove.42B.300d.txt in: {PROJECT_ROOT / 'data' / 'embeddings'}"
        )

    glove = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype="float32")
            glove[word] = vec

    # fill matrix for tokens that exist in GloVe
    for token, idx in tokenizer.get_vocab().items():
        w = token
        if w.startswith("##"):
            w = w[2:]
        w = w.lower()
        if w in glove:
            embedding_matrix[idx] = torch.from_numpy(glove[w])

    embedding_matrix = embedding_matrix.to(device)

    # --------------------
    # 3) Model
    # --------------------
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        max_length=data_cfg.max_length,
        d_model=embedding_dim,       # MUST match embedding_dim
        num_heads=6,
        num_layers=3,                # slightly smaller to reduce overfit 3
        d_ff=4 * embedding_dim , # 4*
        dropout=0.3, # 0.3
        num_classes=3,
    )

    model = FinancialTransformer(model_cfg, embedding_matrix=embedding_matrix).to(device)
    # Freeze token embeddings for first few epochs
    freeze_epochs = 5
    for param in model.token_emb.parameters():
        param.requires_grad = False

    print(f"Token embeddings frozen for first {freeze_epochs} epochs.")
    # --------------------
    # 4) Loss + Optimizer
    # --------------------
    # alpha = torch.tensor([1.7, 0.43, 0.86], device=device)
    # criterion = FocalLoss(gamma=2, alpha=alpha)
    # If you want the simpler/cleaner option (often more stable with sampler), use:
    # weights=torch.tensor([1.15, 1.0, 1.10], device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay
    )

    # Optional scheduler (you already import it)
    # scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.epochs, eta_min=1e-6)

    # --------------------
    # 5) Outputs / paths
    # --------------------
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(train_cfg.log_csv), exist_ok=True)
    os.makedirs(os.path.dirname(train_cfg.curve_path), exist_ok=True)

    best_path = os.path.join(train_cfg.save_dir, train_cfg.best_name)

    # --------------------
    # 6) Training loop (best by val macro-F1)
    # --------------------
    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        
        

    # Unfreeze embeddings after freeze_epochs
        if epoch == freeze_epochs + 1:
            for param in model.token_emb.parameters():
                param.requires_grad = True
            print("Token embeddings unfrozen.")

        
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=train_cfg.grad_clip,
            criterion=criterion,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )

        # scheduler.step()  # if you enabled scheduler

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} macro_f1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} macro_f1 {val_metrics['macro_f1']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        # save best checkpoint by macro-F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "tokenizer_name": data_cfg.tokenizer_name,
                    "max_length": data_cfg.max_length,
                    "id2label": id2label,
                    "best_val_macro_f1": best_val_macro_f1,
                    "data_cfg": data_cfg.__dict__,
                    "variant": train_cfg.variant,
                    "seed": train_cfg.seed,
                },
                best_path,
            )

            pd.DataFrame(history).to_csv(train_cfg.log_csv, index=False)
            print(f"  ✓ saved best checkpoint (macro_f1={best_val_macro_f1:.4f}) to: {best_path}")

    # --------------------
    # 7) Save logs + plots
    # --------------------
    df = pd.DataFrame(history)
    df.to_csv(train_cfg.log_csv, index=False)

    # Loss curve
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(train_cfg.curve_path.replace(".png", "_loss.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Accuracy curve
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(train_cfg.curve_path.replace(".png", "_acc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved log CSV to: {train_cfg.log_csv}")
    print(f"Saved curves to: {train_cfg.curve_path.replace('.png','_loss.png')} and _acc.png")
    print("Done.")

    
if __name__ == "__main__":
    main()

