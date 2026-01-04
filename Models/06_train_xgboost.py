import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIG
INPUT_FILE = "../Databases/DIPLOMA_FUSION_DATA_ROBERTA.csv"
RESULT_FILE = "../Results/result_xgboost_with_sentiment_roberta.txt"

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(RESULT_FILE, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

print("--- TRAINING XGBOOST ---")

# 1. Load Data
df = pd.read_csv(INPUT_FILE, index_col=0)
# === FIX INFINITY ERRORS ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop('Target', axis=1)
y = df['Target']

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# 4. Train
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Model: XGBoost")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))