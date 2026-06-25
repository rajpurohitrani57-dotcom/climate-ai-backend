import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_architecture import ClimateCNN_LSTM

# Try to import explain module
try:
    from src.explain import model_prediction_with_explanation
    EXPLAIN_AVAILABLE = True
    print("✅ Explain module loaded!")
except ImportError:
    EXPLAIN_AVAILABLE = False
    print("⚠️ Explain module not found. Run: pip install captum")

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
USE_REAL_DATA = True
DATA_FOLDER = r'C:\Users\HP\OneDrive\Desktop\imd_data'

# ============================================================
# LOAD MODEL
# ============================================================
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'climate_model.pth')

print(f"🔍 Looking for model at: {model_path}")

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    exit(1)

model = ClimateCNN_LSTM(input_channels=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("✅ Model loaded successfully!")

# ============================================================
# LOAD REAL IMD DATA
# ============================================================
def load_real_imd_data(data_folder):
    """Load real IMD NetCDF files."""
    print(f"\n📥 Loading real IMD data from: {data_folder}")
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Folder not found: {data_folder}")
    
    nc_files = [f for f in os.listdir(data_folder) if f.endswith('.nc')]
    nc_files.sort()
    
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {data_folder}")
    
    print(f"Found {len(nc_files)} files: {nc_files}")
    
    import xarray as xr
    all_data = []
    
    for file in nc_files:
        file_path = os.path.join(data_folder, file)
        print(f"  Loading: {file}")
        ds = xr.open_dataset(file_path)
        
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
    
    return rain_data

def preprocess_imd_data(rain_data, seq_len=30, region_size=10):
    """Preprocess IMD data."""
    print("🔄 Preprocessing data...")
    
    lat_start = 80
    lat_end = lat_start + region_size
    lon_start = 80
    lon_end = lon_start + region_size
    
    rain_cropped = rain_data[:, lat_start:lat_end, lon_start:lon_end]
    rain_cropped = np.nan_to_num(rain_cropped, nan=0.0)
    print(f"Cropped shape: {rain_cropped.shape}")
    
    mean_val = np.mean(rain_cropped)
    std_val = np.std(rain_cropped)
    if std_val < 1e-8:
        std_val = 1.0
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

# ============================================================
# LOAD REAL DATA
# ============================================================
if USE_REAL_DATA:
    try:
        rain_data = load_real_imd_data(DATA_FOLDER)
        X_tensor, y_tensor, mean_val, std_val = preprocess_imd_data(rain_data)
        print(f"✅ Real data loaded: {X_tensor.shape[0]} samples")
        
        app.config['X_tensor'] = X_tensor
        app.config['y_tensor'] = y_tensor
        app.config['data_loaded'] = True
        print("✅ Real IMD data ready!")
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        print("⚠️ Falling back to synthetic data...")
        app.config['data_loaded'] = False
else:
    print("ℹ️ Using synthetic data")
    app.config['data_loaded'] = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_with_uncertainty(model, input_tensor, n_samples=10):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(input_tensor)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    if np.isnan(std) or std[0][0] == 0:
        std = np.array([[0.01]])
    
    return {
        'prediction': float(mean[0][0]),
        'uncertainty': float(std[0][0]),
        'confidence_interval': [
            float(mean[0][0] - 1.96 * std[0][0]),
            float(mean[0][0] + 1.96 * std[0][0])
        ]
    }

def climate_risk_score(temp_pred, rain_pred, temp_threshold=35, rain_threshold=100):
    temp_risk = min(10, max(0, (temp_pred - 25) / (temp_threshold - 25) * 10))
    rain_risk = min(10, max(0, (rain_threshold - rain_pred) / rain_threshold * 10))
    total_risk = 0.5 * temp_risk + 0.5 * rain_risk

    if total_risk < 3:
        severity = "LOW"
    elif total_risk < 6:
        severity = "MODERATE"
    elif total_risk < 8:
        severity = "HIGH"
    else:
        severity = "SEVERE"

    return {
        'risk_score': round(total_risk, 2),
        'severity': severity,
        'temperature_risk': round(temp_risk, 2),
        'rainfall_risk': round(rain_risk, 2)
    }

def simulate_whatif(model, input_tensor, temp_delta=0, rain_delta=0):
    modified_input = input_tensor.clone()
    if modified_input.shape[2] >= 2:
        modified_input[:, :, 0, :, :] += temp_delta
        modified_input[:, :, 1, :, :] *= (1 + rain_delta / 100)
    else:
        modified_input += temp_delta
    with torch.no_grad():
        new_pred = model(modified_input)
    return float(new_pred[0][0])

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'data_source': 'real' if app.config.get('data_loaded', False) else 'synthetic'
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working!', 'endpoints': ['/health', '/predict', '/whatif', '/explain_human']})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    if app.config.get('data_loaded', False) and 'features' not in data:
        X_data = app.config['X_tensor']
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
        temp = data.get('temperature', 30.0)
        rain = data.get('rainfall', 60.0)
        print(f"ℹ️ Using real data sample {idx}")
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
            temp = data.get('temperature', 30.0)
            rain = data.get('rainfall', 50.0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        return jsonify({
            'error': 'Missing "features" field in request body',
            'example': {
                'features': [[[[0.1]*10]*10]*30],
                'temperature': 32,
                'rainfall': 60
            }
        }), 400
    
    result = predict_with_uncertainty(model, input_tensor)
    risk = climate_risk_score(temp, rain)
    
    return jsonify({
        'prediction': result['prediction'],
        'uncertainty': result['uncertainty'],
        'confidence_interval': result['confidence_interval'],
        'risk_score': risk,
        'data_source': 'real' if app.config.get('data_loaded', False) else 'synthetic',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/whatif', methods=['POST'])
def whatif():
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    use_real_sample = data.get('use_real_sample', False)
    
    if use_real_sample and app.config.get('data_loaded', False):
        X_data = app.config['X_tensor']
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
        print(f"ℹ️ What-if using real data sample {idx}")
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        return jsonify({
            'error': 'Missing "features" field or use_real_sample',
            'example': {
                'use_real_sample': True,
                'temp_delta': 2.0,
                'rain_delta': -10.0
            }
        }), 400
    
    new_pred = simulate_whatif(
        model,
        input_tensor,
        temp_delta=data.get('temp_delta', 0.0),
        rain_delta=data.get('rain_delta', 0.0)
    )
    return jsonify({
        'whatif_prediction': new_pred,
        'temp_delta': data.get('temp_delta', 0.0),
        'rain_delta': data.get('rain_delta', 0.0),
        'data_source': 'real' if app.config.get('data_loaded', False) else 'synthetic'
    })

# ============================================================
# EXPLAINABLE AI ENDPOINT - HUMAN READABLE
# ============================================================
@app.route('/explain_human', methods=['POST'])
def explain_human():
    """Explain why the model made a prediction - Human readable"""
    if not EXPLAIN_AVAILABLE:
        return jsonify({
            'error': 'Explain module not available',
            'fix': 'Run: pip install captum'
        }), 503
    
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    use_real_sample = data.get('use_real_sample', False)
    method = data.get('method', 'integrated_gradients')
    
    if use_real_sample and app.config.get('data_loaded', False):
        X_data = app.config['X_tensor']
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
        print(f"ℹ️ Explaining real data sample {idx}")
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        return jsonify({
            'error': 'Missing "features" or "use_real_sample"',
            'example': {
                'use_real_sample': True,
                'method': 'integrated_gradients'
            }
        }), 400
    
    # Get explanation
    try:
        result = model_prediction_with_explanation(model, input_tensor, method)
    except Exception as e:
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500
    
    temp = data.get('temperature', 30.0)
    rain = data.get('rainfall', 50.0)
    risk = climate_risk_score(temp, rain)
    
    return jsonify({
        'prediction': result['prediction'],
        'prediction_meaning': result['prediction_meaning'],
        'human_explanation': result['explanation']['human_explanation'],
        'summary': result['explanation']['summary'],
        'feature_summary': result['feature_summary']['description'],
        'risk_score': risk,
        'data_source': 'real' if app.config.get('data_loaded', False) else 'synthetic',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌤️ Climate AI Module API Server")
    print("="*50)
    print(f"📊 Data source: {'REAL IMD' if app.config.get('data_loaded', False) else 'Synthetic'}")
    print(f"🧠 Explainable AI: {'✅ Available' if EXPLAIN_AVAILABLE else '❌ Not available'}")
    print("="*50)
    print("📋 Available Endpoints:")
    print("  GET  /health")
    print("  GET  /test")
    print("  POST /predict")
    print("  POST /whatif")
    print("  POST /explain_human  ← Human-readable explanation")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)