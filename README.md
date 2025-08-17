# Credit Card Fraud Detection using LightGBM & BiLSTM

This repository provides **two separate approaches** for detecting fraudulent credit card transactions:

* **LightGBM (Machine Learning)**
* **BiLSTM (Deep Learning)**

Both models were trained and evaluated **independently** on the same dataset.
The goal is to provide a **clear comparison** between a fast, interpretable ML approach and a sequence-learning DL model.

---

## 📌 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Models](#models)
* [Evaluation & Comparison](#evaluation--comparison)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

---

## 🔍 Introduction

Fraud detection is a **critical challenge in financial systems** due to the highly imbalanced nature of transaction data.
This project explores two different modeling approaches:

* **LightGBM**: Gradient boosting decision tree framework, optimized for speed and imbalance handling.
* **BiLSTM**: A recurrent neural network that learns sequential dependencies across features.

👉 Both implementations are included in this repository for researchers, data scientists, and engineers to **train, test, and compare**.

---

## 📊 Dataset

* **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Records**: 284,807 transactions
* **Fraudulent cases**: 492 (0.172%)
* **Features**:

  * 28 anonymized PCA-transformed features (`V1`–`V28`)
  * `Time`, `Amount`
  * `Class` (Target: 0 = Legitimate, 1 = Fraud)

⚠️ Highly imbalanced dataset — requires special handling for fair evaluation.

---

## 📂 Project Structure

```
credit_card_fraud_detection_lightgbm_bilstm/
│── README.md
│── requirements.txt
│── train_lightgbm.py        # LightGBM training & evaluation
│── train_bilstm.py          # BiLSTM training & evaluation
│── evaluate_model.py        # Compare both models
│── predict_batch.py         # Batch predictions
│── predict_single.py        # Single transaction prediction
│── utils.py                 # Data preprocessing & helpers
│── data/                    # Dataset (creditcard.csv here)
│── models/                  # Saved models (.pkl, .h5)
```

---

## 🤖 Models

### 🔹 LightGBM (Machine Learning)

* Handles imbalanced data using **class weights**.
* Provides **feature importance** insights.
* Very **fast training & inference**.

### 🔹 BiLSTM (Deep Learning)

* Learns **temporal patterns** in transaction sequences.
* Uses dropout & early stopping to prevent overfitting.
* Achieves better fraud-class recall compared to baseline ML models.

---

## 📈 Evaluation & Comparison

Both models were trained **independently** and tested on the same dataset.

| Metric                      | LightGBM (ML) | BiLSTM (DL) |
| --------------------------- | ------------- | ----------- |
| **Precision (Fraud class)** | 0.27          | 0.21        |
| **Recall (Fraud class)**    | 0.79          | 0.84        |
| **F1-score (Fraud class)**  | 0.41          | 0.33        |
| **ROC-AUC**                 | 0.91          | 0.92        |
| **Speed**                   | Fast          | Slower      |
| **Interpretability**        | High          | Low         |

📌 **Key Insights**:

* LightGBM is **faster** and provides better **precision & F1-score**.
* BiLSTM achieves **slightly higher recall** and ROC-AUC, making it better at **catching frauds**.
* Both models highlight the trade-off between **accuracy, interpretability, and recall**.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/credit_card_fraud_detection_lightgbm_bilstm.git
cd credit_card_fraud_detection_lightgbm_bilstm
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Place `creditcard.csv` dataset inside the `data/` folder.
2. Train LightGBM model:

   ```bash
   python train_lightgbm.py
   ```
3. Train BiLSTM model:

   ```bash
   python train_bilstm.py
   ```
4. Compare both models:

   ```bash
   python evaluate_model.py
   ```
5. Run predictions:

   ```bash
   python predict_batch.py lightgbm
   python predict_batch.py bilstm
   python predict_single.py lightgbm
   python predict_single.py bilstm
   ```

---

## 🤝 Contributing

Contributions are welcome! You can extend this repo by:

* Adding new models (XGBoost, Transformers, etc.)
* Experimenting with oversampling (SMOTE, ADASYN) or cost-sensitive learning
* Improving feature engineering

---

## 📜 License

This project is licensed under the **MIT License**.

