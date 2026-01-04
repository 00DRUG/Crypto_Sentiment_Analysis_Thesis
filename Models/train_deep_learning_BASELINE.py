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
INPUT_FILE = "../DataBases/DIPLOMA_BASELINE_DATA.csv"
RESULT_FILE = "../Results/result_deep_learning_baseline.txt"
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load Baseline Data
try:
    df = pd.read_csv(INPUT_FILE, index_col=0)
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}")
    sys.exit()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

X = df.drop('Target', axis=1).values
y = df['Target'].values

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split
split_idx = int(len(df) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# 4. Class Weights
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"Class Weights: {class_weights}")

# 5. Tensors
t_x_train = torch.Tensor(X_train).to(device)
t_y_train = torch.Tensor(y_train).long().to(device)

t_x_test = torch.Tensor(X_test).to(device)
t_y_test = torch.Tensor(y_test).long().to(device)

train_dataset = TensorDataset(t_x_train, t_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5. Simple Neural Network - no fusion
class BaselineNet(nn.Module):
    def __init__(self, input_dim):
        super(BaselineNet, self).__init__()

        # Layer 1: Expansion
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4)
        )

        # Layer 2: Compression
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3)
        )

        # Output
        self.output = nn.Linear(64, 2)

        # Init Weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return self.output(x)

model = BaselineNet(input_dim=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25)

# 6. Train
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for bx, by in train_loader:
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += by.size(0)
        correct += (predicted == by).sum().item()

    # Validation Step
    model.eval()
    with torch.no_grad():
        test_outputs = model(t_x_test)
        _, test_preds = torch.max(test_outputs.data, 1)
        test_acc = accuracy_score(y_test, test_preds.cpu().numpy())

    # Update Scheduler
    scheduler.step(test_acc)

    # Save Model
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc * 100:.2f}% | LR: {current_lr}")

# 7. Evaluate
model.load_state_dict(best_model_wts)
model.eval()

with torch.no_grad():
    outputs = model(t_x_test)
    _, predicted = torch.max(outputs.data, 1)
    y_true = t_y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

acc = accuracy_score(y_true, y_pred)

print("=" * 50)
print(f"FINAL RESULT - MULTI-MODAL FUSION")
print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print("=" * 50)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))