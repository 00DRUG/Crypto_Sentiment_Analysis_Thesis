import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# CONFIG
INPUT_FILE = "../DataBases/DIPLOMA_BASELINE_DATA.csv"
RESULT_FILE = "../Results/result_deep_learning_baseline.txt"
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Logger
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

# 4. Tensors
t_x_train = torch.Tensor(X_train).to(device)
t_y_train = torch.Tensor(y_train).long().to(device)
t_x_test = torch.Tensor(X_test).to(device)
t_y_test = torch.Tensor(y_test).long().to(device)

train_dataset = TensorDataset(t_x_train, t_y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Simple Neural Network - no fusion
class BaselineNet(nn.Module):
    def __init__(self, input_dim):
        super(BaselineNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        return self.output(x)

model = BaselineNet(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Train
model.train()
for epoch in range(EPOCHS):
    for bx, by in train_loader:
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()

# 7. Evaluate
model.eval()
with torch.no_grad():
    outputs = model(t_x_test)
    _, predicted = torch.max(outputs.data, 1)
    y_true = t_y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

acc = accuracy_score(y_true, y_pred)
print("="*50)
print(f"BASELINE DEEP LEARNING (PRICE ONLY) RESULT")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("="*50)
print(classification_report(y_true, y_pred))