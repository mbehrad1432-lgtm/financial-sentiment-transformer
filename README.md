Financial Sentiment Analysis using Transformer

This project implements a custom Transformer-based model for Financial Sentiment Analysis.
The model is built from scratch using PyTorch and does not rely on pretrained transformer architectures such as BERT.

The goal is to classify financial sentences into sentiment categories using a self-implemented attention mechanism and transformer blocks

Project Objective

Classify financial text into three sentiment classes:

Negative

Neutral

Positive

Dataset used:

financial_phrasebank


Model Architecture

The model consists of the following components:

Token Embedding Layer

Sinusoidal Positional Encoding

Multi-Head Self Attention

Transformer Blocks

Masked Mean Pooling

MLP Classification Hea

Pipeline:

Embedding → Positional Encoding → Transformer Layers → Pooling → Classifier

Project Structure:

financial-sentiment-transformer/
│
├── src/
│   ├── data/
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── attention.py
│   │   ├── blocks.py
│   │   └── model.py
│   │
│   ├── train.py
│   ├── eval.py
│   └── predict_test.py
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── submissions/
│
├── .gitignore
└── README.md




Installation:

pip install torch transformers datasets scikit-learn matplotlib pandas tqdm



Training

Run:

python src/train.py

Outputs:

Best model checkpoint:

outputs/checkpoints/best.pt

Training curves:

outputs/figures/

Training logs:

outputs/logs/train_log.csv

The best model is selected based on highest Macro F1 score on validation set

Evaluation

Run:

python src/eval.py

Metrics reported:

Accuracy

Precision

Recall

Macro F1

Weighted F1

Classification Report

Confusion Matrix

The evaluation is performed only on the validation set.
No validation data is used during training

Test Prediction

Run:

python src/predict_test.py

Output file:

outputs/submissions/test_predictions.csv

Output format:

sentence	prediction	prediction_id

The script loads the best checkpoint and generates predictions for the test set.

📈 Evaluation Metrics

The following metrics are used:

Accuracy

Precision

Recall

Macro F1 (Primary selection metric)

Weighted F1

Confusion Matrix

Macro F1 is used as the main model selection criterion to ensure balanced performance across classes.

🛠 Techniques Used

Cross Entropy Loss

Label Smoothing

Weighted Random Sampling

Learning Rate Scheduler

Masked Mean Pooling

Xavier Initialization

🧩 Model Selection Strategy

The model is trained only on the training set.

Validation set is used exclusively for model selection.

The best checkpoint is selected based on highest validation Macro F1.

The test set is never used during training or hyperparameter tuning.

🤝 Team Workflow

Development workflow:

Each feature is implemented in a separate branch.

Pull Requests are created for merging into main.

Code is reviewed before merging.

Team members synchronize changes using git pull.

📊 Best Validation Performance

Macro F1 ≈ 0.77

Accuracy ≈ 0.83

🔮 Future Improvements

Improve Positive class precision

Experiment with Focal Loss

Test alternative pooling strategies (CLS / Attention Pooling)

Increase Transformer depth

Integrate pretrained embeddings

📜 License

This project is developed for educational and academic purposes