import copy

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
EPOCHS = 350
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

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

# 6. CLASS WEIGHTS
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"Class Weights: {class_weights}")

# 7. CONVERT TO PYTORCH TENSORS
t_fin_train = torch.Tensor(X_fin_train).to(device)
t_soc_train = torch.Tensor(X_soc_train).to(device)
t_y_train = torch.Tensor(y_train).long().to(device)

t_fin_test = torch.Tensor(X_fin_test).to(device)
t_soc_test = torch.Tensor(X_soc_test).to(device)
t_y_test = torch.Tensor(y_test).long().to(device)

# Dataset & Loader
train_dataset = TensorDataset(t_fin_train, t_soc_train, t_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)


# 8. DEFINE THE FUSION ARCHITECTURE
class FusionNet(nn.Module):
    def __init__(self, input_dim_fin, input_dim_soc):
        super(FusionNet, self).__init__()

        # --- Financial ---
        self.fin_layer = nn.Sequential(
            nn.Linear(input_dim_fin, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4)
        )

        # --- Social ---
        self.soc_layer = nn.Sequential(
            nn.Linear(input_dim_soc, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01)
        )

        # --- Fusion Layer ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(288, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x_fin, x_soc):
        f = self.fin_layer(x_fin)
        s = self.soc_layer(x_soc)
        combined = torch.cat((f, s), dim=1)
        return self.fusion_layer(combined)


# Initialize Model
model = FusionNet(input_dim_fin=X_fin_train.shape[1], input_dim_soc=X_soc_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25)

# 9. TRAINING LOOP
print(f"\nStarting Training ({EPOCHS} Epochs)...")

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_fin, batch_soc, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_fin, batch_soc)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    # Validate
    model.eval()
    with torch.no_grad():
        test_outputs = model(t_fin_test, t_soc_test)
        _, test_preds = torch.max(test_outputs.data, 1)
        test_acc = accuracy_score(y_test, test_preds.cpu().numpy())

    # Scheduler Step
    scheduler.step(test_acc)

    # Save Best
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc * 100:.2f}% | LR: {current_lr}")

# 10. EVALUATION
print("\nEvaluating...")
model.load_state_dict(best_model_wts)
model.eval()
with torch.no_grad():
    outputs = model(t_fin_test, t_soc_test)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.cpu().numpy()

acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"FINAL RESULT - MULTI-MODAL FUSION")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print("=" * 50)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))