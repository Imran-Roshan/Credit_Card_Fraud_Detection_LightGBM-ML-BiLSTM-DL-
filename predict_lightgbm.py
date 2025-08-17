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
