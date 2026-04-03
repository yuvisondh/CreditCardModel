# Credit Card Fraud Detection — Deep Learning

A binary classification pipeline that detects fraudulent credit card transactions using a deep neural network trained on highly imbalanced real-world data.

---

## Problem Statement

Credit card fraud is rare (~0.17 % of transactions) but extremely costly. Standard classifiers trained naively on imbalanced data simply predict "legitimate" every time and achieve 99 %+ accuracy while catching zero fraud. This project tackles that challenge head-on.

---

## Highlights

| Technique | Purpose |
|---|---|
| **Class Weights** (balanced) | Handles 0.17% fraud imbalance without synthetic data — avoids SMOTE artifacts |
| **L2 Regularization** (1e-3) | Prevents large weights that saturate the sigmoid to exactly 0 or 1 |
| **Label Smoothing** (0.05) | Trains against soft targets (0.025/0.975), producing calibrated probabilities |
| **StandardScaler** | Normalises `Time` and `Amount` so they don't dominate gradient updates |
| **Deep Neural Network** (TensorFlow / Keras) | Learns non-linear decision boundaries across 30 PCA features |
| **Dropout (0.3)** | Regularises each hidden layer to reduce overfitting |
| **Early Stopping** | Halts training when validation loss stops improving and restores best weights |
| **Threshold tuning (0.5)** | Classification threshold for fraud vs legit |
| **Precision-Recall AUC** | Primary evaluation metric — more informative than ROC-AUC on imbalanced data |

---

## Model Architecture

```
Input (30 features)
    │
Dense(32, ReLU, L2=1e-3) → Dropout(0.3)
    │
Dense(16, ReLU, L2=1e-3) → Dropout(0.3)
    │
Dense(1, Sigmoid)  ← fraud probability
```

Compiled with **Adam** optimiser (lr=1e-3) and **binary cross-entropy** loss with **label smoothing (0.05)**.  
Tracked metrics: Accuracy, Precision, Recall, ROC-AUC, PR-AUC.

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions — 492 fraud (0.172 %)
- Features V1–V28 are PCA-transformed (anonymised); `Time` and `Amount` are raw
- Target: `Class` — 0 = Legitimate, 1 = Fraud

> The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the project root before running.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/CreditCardModel.git
cd CreditCardModel

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Download creditcard.csv from Kaggle and place it in the project root

# 5. Train the model
python main.py
```

After training, the following files are generated in the project root:

| File | Description |
|---|---|
| `fraud_model.keras` | Saved Keras model |
| `fraud_model.pkl` | Pickled model for Flask API deployment |
| `scaler.pkl` | Fitted StandardScaler for inference |
| `training_history.png` | Loss & accuracy curves |
| `confusion_matrix.png` | Confusion matrix on the test set |
| `precision_recall_curve.png` | PR curve with average precision score |

---

## Running the Tests

```bash
pytest
```

The test suite covers data loading, preprocessing correctness (shape, class weights, scaling), model architecture, and an end-to-end evaluation smoke test — all using a synthetic dataset so the large CSV is not required.

---

## Project Structure

```
CreditCardModel/
├── main.py            # Full pipeline: load → preprocess → train → evaluate → save
├── test_main.py       # Unit tests (pytest)
├── pytest.ini         # Pytest configuration
├── requirements.txt   # Pinned dependencies
└── README.md
```

---

## Results

Evaluation is performed on a held-out 20% test set (real-world distribution).  
A classification threshold of **0.5** is used with class-weighted training.

Key metrics (representative run):

| Metric | Fraud class |
|---|---|
| Precision | ~0.18 |
| Recall | ~0.89 |
| F1-score | ~0.29 |
| ROC-AUC | ~0.989 |
| PR-AUC | ~0.74 |

> **Note:** Low precision is expected — the model is tuned for high recall to catch as many fraud cases as possible. With class weights heavily favouring the minority class (289:1), the model flags more transactions for review. In production, this is the preferred trade-off for fraud detection. The previous SMOTE-based model had artificially inflated precision but produced saturated (0% or 100%) confidence scores, making it useless for risk scoring.

---

## Tech Stack

- **Python 3.12**
- **TensorFlow 2.20 / Keras 3**
- **scikit-learn 1.8** — train/test split, StandardScaler, class weights, metrics
- **pandas 3 / NumPy 2** — data manipulation
- **Matplotlib / Seaborn** — visualisation

---

## License

This project is released under the [MIT License](LICENSE).
