import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD REAL IMD DATA
# ============================================================
print("="*50)
print("📥 Loading Real IMD Data")
print("="*50)

DATA_FOLDER = r'C:\Users\HP\OneDrive\Desktop\imd_data'

nc_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.nc')]
nc_files.sort()
print(f"Found {len(nc_files)} files: {nc_files}")

all_data = []
for file in nc_files:
    file_path = os.path.join(DATA_FOLDER, file)
    print(f"  Loading: {file}")
    ds = xr.open_dataset(file_path)
    
    # Check variable name
    if 'RAINFALL' in ds.variables:
        rain = ds['RAINFALL'].values
    elif 'rain' in ds.variables:
        rain = ds['rain'].values
    else:
        var_name = list(ds.data_vars.keys())[0]
        rain = ds[var_name].values
    
    all_data.append(rain)
    ds.close()

rain_data = np.vstack(all_data)
print(f"Total time steps: {rain_data.shape[0]}")
print(f"Grid size: {rain_data.shape[1]} x {rain_data.shape[2]}")

# Crop to region
lat_start, lon_start = 80, 80
region_size = 10
rain_cropped = rain_data[:, lat_start:lat_start+region_size, lon_start:lon_start+region_size]
print(f"Cropped shape: {rain_cropped.shape}")

# Normalize
mean_val = np.mean(rain_cropped)
std_val = np.std(rain_cropped)
rain_norm = (rain_cropped - mean_val) / (std_val + 1e-8)

print(f"Mean: {mean_val:.4f}, Std: {std_val:.4f}")

# Create sequences
seq_len = 30
X, y = [], []
for i in range(seq_len, len(rain_norm)):
    X.append(rain_norm[i-seq_len:i])
    y.append(rain_norm[i])

X = np.array(X)
y = np.array(y)
y_avg = np.mean(y, axis=(1, 2))

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
y_tensor = torch.tensor(y_avg, dtype=torch.float32).unsqueeze(1)

print(f"X_tensor shape: {X_tensor.shape}")
print(f"y_tensor shape: {y_tensor.shape}")

# Split
split = int(0.8 * len(X_tensor))
X_train, X_val = X_tensor[:split], X_tensor[split:]
y_train, y_val = y_tensor[:split], y_tensor[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")

# ============================================================
# CNN-LSTM MODEL
# ============================================================
class ClimateCNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.lstm = nn.LSTM(128*4*4, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        cnn_in = x.view(batch_size * seq_len, C, H, W)
        cnn_out = self.cnn(cnn_in)
        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)
        return self.fc(lstm_out[:, -1, :])

# ============================================================
# TRAIN
# ============================================================
print("\n" + "="*50)
print("🏋️ Training on Real IMD Data")
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ClimateCNN_LSTM(input_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1:2d}/30 | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

# ============================================================
# SAVE MODEL
# ============================================================
save_path = 'model/climate_model_real.pth'
torch.save(model.state_dict(), save_path)
print(f"\n✅ Model saved to: {save_path}")

# Also save as the main model (overwrite)
torch.save(model.state_dict(), 'model/climate_model.pth')
print(f"✅ Model saved to: model/climate_model.pth")

print("\n" + "="*50)
print("🎉 Retraining complete! You can now restart the API.")
print("="*50)