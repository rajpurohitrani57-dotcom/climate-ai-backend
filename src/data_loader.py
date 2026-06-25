import xarray as xr
import numpy as np
import torch
import os

def load_real_imd_data(data_folder):
    """Load real IMD NetCDF files."""
    print(f"\n📥 Loading real IMD data from: {data_folder}")
    
    nc_files = [f for f in os.listdir(data_folder) if f.endswith('.nc')]
    nc_files.sort()
    
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {data_folder}")
    
    print(f"Found {len(nc_files)} files: {nc_files}")
    
    all_data = []
    for file in nc_files:
        file_path = os.path.join(data_folder, file)
        ds = xr.open_dataset(file_path)
        rain = ds['rain'].values
        all_data.append(rain)
        ds.close()
    
    rain_data = np.vstack(all_data)
    print(f"Total time steps: {rain_data.shape[0]}")
    print(f"Grid size: {rain_data.shape[1]} x {rain_data.shape[2]}")
    
    return rain_data

def preprocess_imd_data(rain_data, seq_len=30, region_size=10):
    """Preprocess IMD data: crop, normalize, create sequences."""
    print("🔄 Preprocessing data...")
    
    # Crop to region (adjust these indices for your pilot area)
    lat_start = 80
    lat_end = lat_start + region_size
    lon_start = 80
    lon_end = lon_start + region_size
    
    rain_cropped = rain_data[:, lat_start:lat_end, lon_start:lon_end]
    print(f"Cropped shape: {rain_cropped.shape}")
    
    mean_val = np.mean(rain_cropped)
    std_val = np.std(rain_cropped)
    rain_norm = (rain_cropped - mean_val) / (std_val + 1e-8)
    
    X, y = [], []
    for i in range(seq_len, len(rain_norm)):
        X.append(rain_norm[i-seq_len:i])
        y.append(rain_norm[i])
    
    X = np.array(X)
    y = np.array(y)
    y_avg = np.mean(y, axis=(1, 2))
    
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y_tensor = torch.tensor(y_avg, dtype=torch.float32).unsqueeze(1)
    
    print(f"✅ X_tensor shape: {X_tensor.shape}")
    print(f"✅ y_tensor shape: {y_tensor.shape}")
    
    return X_tensor, y_tensor, mean_val, std_val