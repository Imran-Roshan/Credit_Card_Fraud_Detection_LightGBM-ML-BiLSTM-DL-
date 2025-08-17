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
