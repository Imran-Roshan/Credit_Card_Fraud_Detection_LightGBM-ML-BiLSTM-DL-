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
