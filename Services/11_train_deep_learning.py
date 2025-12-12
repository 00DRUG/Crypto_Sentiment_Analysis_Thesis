import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIG
INPUT_FILE = "../Databases/DIPLOMA_FUSION_DATA_ROBERTA.csv"
RESULT_FILE = "../Results/result_deep_learning_with_sentiment_roberta.txt"
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Logger to save results to file
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

print(f"--- TRAINING MULTI-MODAL FUSION MODEL ---")

# 1. SETUP GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processing Unit: {device}")
if device.type == 'cuda':
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# 2. LOAD DATA
try:
    df = pd.read_csv(INPUT_FILE, index_col=0)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}")
    sys.exit()

# Cleanup
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print(f"Data Loaded: {len(df)} samples")

# 3. SEPARATE MODALITIES Social and Financial
social_cols = ['Sentiment', 'Tweet_Volume']
financial_cols = [c for c in df.columns if c not in social_cols and c != 'Target']

print(f"Financial Features: {len(financial_cols)} {financial_cols}")
print(f"Social Features: {len(social_cols)} {social_cols}")

X_fin = df[financial_cols].values
X_soc = df[social_cols].values
y = df['Target'].values

# 4. SCALE DATA
scaler_fin = StandardScaler()
X_fin_scaled = scaler_fin.fit_transform(X_fin)

scaler_soc = StandardScaler()
X_soc_scaled = scaler_soc.fit_transform(X_soc)

# 5. SPLIT DATA in order
split_idx = int(len(df) * 0.8)

X_fin_train = X_fin_scaled[:split_idx]
X_fin_test = X_fin_scaled[split_idx:]

X_soc_train = X_soc_scaled[:split_idx]
X_soc_test = X_soc_scaled[split_idx:]

y_train = y[:split_idx]
y_test = y[split_idx:]

# 6. CONVERT TO PYTORCH TENSORS
t_fin_train = torch.Tensor(X_fin_train).to(device)
t_soc_train = torch.Tensor(X_soc_train).to(device)
t_y_train = torch.Tensor(y_train).long().to(device)

t_fin_test = torch.Tensor(X_fin_test).to(device)
t_soc_test = torch.Tensor(X_soc_test).to(device)
t_y_test = torch.Tensor(y_test).long().to(device)

# Dataset & Loader
train_dataset = TensorDataset(t_fin_train, t_soc_train, t_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=False)


# 7. DEFINE THE FUSION ARCHITECTURE
class FusionNet(nn.Module):
    def __init__(self, input_dim_fin, input_dim_soc):
        super(FusionNet, self).__init__()

        # --- Financial ---
        self.fin_layer1 = nn.Linear(input_dim_fin, 128)
        self.fin_relu = nn.ReLU()
        self.fin_drop = nn.Dropout(0.3)

        # --- Social ---
        self.soc_layer1 = nn.Linear(input_dim_soc, 32)
        self.soc_relu = nn.ReLU()

        # --- Fusion Layer ---
        self.fusion_layer1 = nn.Linear(160,64)
        self.fusion_relu = nn.ReLU()
        self.fusion_drop = nn.Dropout(0.2)

        self.output = nn.Linear(64, 2)  # UP or DOWN

    def forward(self, x_fin, x_soc):
        # Process Financial
        f = self.fin_layer1(x_fin)
        f = self.fin_relu(f)
        f = self.fin_drop(f)

        # Process Social
        s = self.soc_layer1(x_soc)
        s = self.soc_relu(s)

        # Fuse
        combined = torch.cat((f, s), dim=1)  # Concatenate

        x = self.fusion_layer1(combined)
        x = self.fusion_relu(x)
        x = self.fusion_drop(x)

        return self.output(x)


# Initialize Model
model = FusionNet(input_dim_fin=X_fin_train.shape[1], input_dim_soc=X_soc_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8. TRAINING LOOP
print(f"\nStarting Training ({EPOCHS} Epochs)...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_fin, batch_soc, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_fin, batch_soc)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss:.4f}")

# 9. EVALUATION
print("\nEvaluating...")
model.eval()
with torch.no_grad():
    outputs = model(t_fin_test, t_soc_test)
    _, predicted = torch.max(outputs.data, 1)

    y_true = t_y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

acc = accuracy_score(y_true, y_pred)

print("=" * 50)
print(f"FINAL RESULT - MULTI-MODAL FUSION")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print("=" * 50)
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))