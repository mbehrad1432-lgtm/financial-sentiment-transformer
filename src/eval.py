# src/eval.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

from data.dataset import DataConfig, build_dataloaders, ID2LABEL
from models.model import ModelConfig, FinancialTransformer


@dataclass
class EvalConfig:
    checkpoint_path: str = "outputs/checkpoints/best.pt"
    variant: str = "sentences_allagree"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "outputs/figures"


@torch.no_grad()
def predict_on_loader(model: torch.nn.Module, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask, return_attn=False)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # numbers in cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_model_from_checkpoint(ckpt_path: str, device: str) -> Tuple[FinancialTransformer, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)

    model_cfg_dict = ckpt["model_cfg"]
    model_cfg = ModelConfig(**model_cfg_dict)

    model = FinancialTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    cfg = EvalConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # load model + ckpt first
    model, ckpt = load_model_from_checkpoint(cfg.checkpoint_path, cfg.device)

    # build loaders using the EXACT data config used in training
    if "data_cfg" in ckpt:
        data_cfg = DataConfig(**ckpt["data_cfg"])
    else:
        # fallback (old checkpoints)
        data_cfg = DataConfig(
            tokenizer_name="bert-base-uncased",
            max_length=64,
            train_frac=0.8,
            seed=cfg.seed,
            batch_size=64,
            num_workers=0,
        )

    # eval batch size can be larger without changing the split
    data_cfg.batch_size = 64

    variant = ckpt.get("variant", cfg.variant)
    _, val_loader, _, _ = build_dataloaders(data_cfg, variant=variant, return_text_in_val=False)

    print("Evaluating checkpoint:", cfg.checkpoint_path)
    print("Using variant:", variant)
    print("Using data_cfg:", data_cfg)

    # load model
    model, ckpt = load_model_from_checkpoint(cfg.checkpoint_path, cfg.device)

    # predict
    y_true, y_pred = predict_on_loader(model, val_loader, cfg.device)

    # metrics (macro + weighted)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("=== Validation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro     P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted  P/R/F1: {p_weight:.4f} / {r_weight:.4f} / {f1_weight:.4f}")
    print("\nClassification report (per class):")
    class_names = [ID2LABEL[i] for i in range(3)]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_path = os.path.join(cfg.save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nSaved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
