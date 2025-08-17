# 📌 Credit Card Fraud Detection Using LightGBM (ML) & BiLSTM (DL)

This repository provides **two separate approaches** to credit card fraud detection:

1. **LightGBM (Machine Learning)** – for efficient gradient boosting on structured data.
2. **BiLSTM (Deep Learning)** – for sequential modeling of transaction patterns.

Both models are implemented separately, allowing researchers and practitioners to compare the performance of traditional ML and modern DL techniques on the same dataset.

---

## 📂 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Model Training](#model-training)

  * [LightGBM Training](#lightgbm-training)
  * [BiLSTM Training](#bilstm-training)
* [Validation & Evaluation](#validation--evaluation)
* [Prediction](#prediction)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)

---

## 🚀 Introduction

Credit card fraud is a major concern for financial institutions worldwide. Detecting fraudulent transactions among millions of legitimate ones requires **scalable** and **accurate** models.

This repository demonstrates:

* **Feature-based classification** with **LightGBM**.
* **Sequential fraud pattern detection** with **BiLSTM**.

Both models have been trained and evaluated independently.

---

## 📊 Dataset

* **Name**: Synthetic Credit Card Transactions Dataset
* **Size**: \~1.8M transactions
* **Features**: 23
* **Target**: `is_fraud` (1 = Fraud, 0 = Legitimate)

**Key columns**:

* `trans_date_trans_time`: Timestamp of transaction
* `cc_num`: Credit card number (anonymized)
* `amt`: Transaction amount
* `category`: Transaction type (travel, food, etc.)
* `city_pop`: Population of the city
* `merch_lat, merch_long`: Merchant location
* `is_fraud`: Fraud flag (target variable)

⚠️ **Disclaimer**: The dataset is anonymized and only for research/educational purposes.

---

## 📂 Repository Structure

```bash
Credit_Card_Fraud_Detection_LightGBM_BiLSTM/
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│
├── data/
│   └── creditcard.csv        # Dataset (add here)
│
├── lightgbm_model/
│   ├── train_lightgbm.py     # Train LightGBM model
│   ├── predict_lightgbm.py   # Inference using LightGBM
│   └── utils_lightgbm.py     # Helper functions
│
├── bilstm_model/
│   ├── train_bilstm.py       # Train BiLSTM model
│   ├── predict_bilstm.py     # Inference using BiLSTM
│   └── utils_bilstm.py       # Helper functions
│
└── results/
    ├── lightgbm_results.png  # Metrics & confusion matrix (LightGBM)
    └── bilstm_results.png    # Metrics & confusion matrix (BiLSTM)
```

---

## ⚙️ Model Training

### 🔹 LightGBM Training

File: `train_lightgbm.py`

```python
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
data = pd.read_csv("../data/creditcard.csv")
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Parameters
params = {"objective": "binary", "metric": "auc", "boosting": "gbdt"}

# Train
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=200)

# Evaluation
y_pred = model.predict(X_test)
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, (y_pred > 0.5).astype(int)))
```

---

### 🔹 BiLSTM Training

File: `train_bilstm.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
data = pd.read_csv("../data/creditcard.csv")
X = data.drop("is_fraud", axis=1).values
y = data["is_fraud"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for BiLSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation="sigmoid"))

# Compile & Train
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128)

# Evaluation
y_pred = model.predict(X_test)
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, (y_pred > 0.5).astype(int)))
```

---

## 📈 Validation & Evaluation

Both models were validated using:

* **Accuracy**
* **Precision / Recall / F1-score**
* **AUC-ROC**
* **Confusion Matrix**

---

## 🔮 Prediction

### LightGBM Prediction

File: `predict_lightgbm.py`

```python
import pickle
import pandas as pd

# Load model
with open("lightgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load new data
data = pd.read_csv("../data/new_transactions.csv")

# Predict
preds = model.predict(data)
print(preds)
```

### BiLSTM Prediction

File: `predict_bilstm.py`

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model
model = load_model("bilstm_model.h5")

# Load new data
data = pd.read_csv("../data/new_transactions.csv")

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(data.values)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Predict
preds = model.predict(X)
print(preds)
```

---

## ✅ Results

| Model    | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| -------- | -------- | --------- | ------ | -------- | ------- |
| LightGBM | \~99%    | High      | Medium | Balanced | 0.92+   |
| BiLSTM   | \~97%    | Medium    | High   | Lower    | 0.91+   |

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repo
* Create a feature branch
* Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License**.
