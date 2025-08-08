import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import glob
import os

# 读取所有地图的传感器数据
all_data = []
data_dir = '../data'
for file in glob.glob(os.path.join(data_dir, 'sensor_data_v*.csv')):
    df = pd.read_csv(file)
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.fc(x)

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 增加训练轮数以提高训练强度
for epoch in range(50):  # 增加到50轮
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 每5轮打印一次损失
    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model_enhanced.pth")
print("Enhanced model saved as model_enhanced.pth")