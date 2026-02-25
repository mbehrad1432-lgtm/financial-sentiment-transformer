# Financial Sentiment Analysis (Custom Transformer)

This repository implements a **Transformer-based sentiment classifier built from scratch in PyTorch** (i.e., it does **not** use pretrained Transformer encoders like BERT for the model).  
A **BERT tokenizer** (`bert-base-uncased`) is used only for tokenization and vocabulary.

The goal is to classify financial sentences into **three** sentiment classes:

- **negative**
- **neutral**
- **positive**

Dataset: `financial_phrasebank` (Hugging Face Datasets)

---

## Model Architecture

### Transformer Sentiment Classifier (your model)

```text
Input batch of sentences
        |
        v
+-------------------------------+
| Tokenizer (BERT uncased)      |
| - WordPiece tokenize          |
| - pad/truncate to T=128       |
+-------------------------------+
        |
        |  input_ids:      (B, 128)
        |  attention_mask: (B, 128)
        v
+-------------------------------+
| Token Embedding (GloVe-init)  |
| Embedding[vocab_size, 300]    |
+-------------------------------+
        |
        |  X0 shape: (B, 128, 300)
        v
+-------------------------------+
| + Positional Encoding         |
| (sinusoidal)                  |
+-------------------------------+
        |
        v
========================================================
Stack of Transformer Encoder Blocks (num_layers = 3)
Each block keeps the SAME shape: (B, 128, 300)
========================================================

   Block (repeated 3x):
   -------------------
   +-------------------------+
   | Multi-Head Self-Attn    |   heads = 6
   | attention_mask used     |   (pads masked out)
   +-------------------------+
            |
       + Residual + LayerNorm
            |
   +-------------------------+
   | FeedForward (FFN)       |
   | 300 -> 1200 -> 300      |
   +-------------------------+
            |
       + Residual + LayerNorm
            |
   Output: (B, 128, 300)

========================================================
        |
        v
+-------------------------------+
| Masked Mean Pooling           |
| mean over tokens where mask=1 |
| (ignore [PAD])                |
+-------------------------------+
        |
        |  pooled: (B, 300)
        v
+-------------------------------+
| Classifier (MLP)              |
| Linear 300->300 + GELU        |
| Dropout                       |
| Linear 300->3                 |
+-------------------------------+
        |
        | logits: (B, 3)
        v
Prediction = argmax(logits)  -> {negative, neutral, positive}
```

### Components

- **Token Embedding**: initialized from **GloVe 42B 300d** vectors when available (trainable after initial freeze period)
- **Sinusoidal Positional Encoding**
- **Multi-Head Self-Attention**
- **Transformer Encoder Blocks** (stacked)
- **Masked Mean Pooling** (ignores padding)
- **MLP Classification Head** (3-class logits)
---

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Place GloVe embeddings (required)

Training expects the file:

```text
data/embeddings/glove.42B.300d.txt
```

If the file is missing, training will raise a `FileNotFoundError`.

---

## Training

Run:

```bash
python src/train.py
```

### Outputs

- **Best checkpoint** (selected by highest validation **Macro F1**):  
  `outputs/checkpoints/best.pt`

- **Training curves**:  
  `outputs/figures/*_loss.png`, `outputs/figures/*_acc.png`

- **Training logs**:  
  `outputs/logs/train_log.csv`

Notes:
- The model freezes the token embedding layer for the first few epochs, then unfreezes it for full fine-tuning.
- Training uses a **stratified train/val split** and **weighted random sampling** to mitigate class imbalance.

---

## Evaluation (Validation Set)

Run:

```bash
python src/eval.py
```

Metrics reported (on validation set):
- Accuracy
- Precision / Recall
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix

The validation set is used **only** for evaluation/model selection.

---

## Test Prediction (Submission Generation)

Run:

```bash
python src/predict_test.py
```

Output file:
- `outputs/submissions/test_predictions.csv`

---

## Techniques Used

- Cross Entropy Loss + **Label Smoothing**
- **WeightedRandomSampler** (class imbalance)
- **Masked Mean Pooling**
- **Xavier** initialization (linear layers)
- Embedding freeze → unfreeze schedule

---

## Model Selection Strategy

- Train only on the training split
- Select best checkpoint using validation **Macro F1**
- Test set is never used during training or tuning

---

## License

Developed for educational and academic purposes.
