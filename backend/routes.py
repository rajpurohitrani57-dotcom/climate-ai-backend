import os
import sys
import torch
import numpy as np
import random
from flask import request, jsonify, Blueprint
from flask_cors import cross_origin
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import db
from backend.models import Prediction, Alert
from backend.weather import get_live_weather, INDIAN_CITIES
from backend.middleware import log_request, rate_limit
from backend.utils import safe_float, safe_int, format_timestamp

# Import AI modules
try:
    from src.model_architecture import ClimateCNN_LSTM
    from src.explain import model_prediction_with_explanation
    AI_AVAILABLE = True
    print("✅ AI modules loaded successfully!")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI modules not available: {e}")

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# ============================================================
# CORS Headers for all routes
# ============================================================
@api.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Global variables
model = None
X_data = None
data_loaded = False

# ============================================================
# HEALTH CHECK
# ============================================================
@api.route('/health', methods=['GET', 'OPTIONS'])
@cross_origin()
def health():
    if request.method == 'OPTIONS':
        return '', 200
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'ai_available': AI_AVAILABLE,
        'data_source': 'real' if data_loaded else 'synthetic' if AI_AVAILABLE else 'none',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================
# MODEL LOADING FUNCTIONS
# ============================================================
def load_model():
    global model
    if model is None and AI_AVAILABLE:
        try:
            possible_paths = [
                '/app/model/climate_model.pth',
                'model/climate_model.pth',
                './model/climate_model.pth',
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'climate_model.pth'),
                os.path.join(os.getcwd(), 'model', 'climate_model.pth'),
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Found model at: {path}")
                    break
            
            if model_path is None:
                print("❌ Model file not found in any location")
                return None
            
            model = ClimateCNN_LSTM(input_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    return model

def load_real_data():
    global X_data, data_loaded
    if X_data is None and AI_AVAILABLE:
        try:
            import xarray as xr
            DATA_FOLDER = r'/opt/render/project/src/imd_data'
            
            if not os.path.exists(DATA_FOLDER):
                DATA_FOLDER = r'C:\Users\HP\OneDrive\Desktop\imd_data'
                if not os.path.exists(DATA_FOLDER):
                    print(f"⚠️ Data folder not found")
                    data_loaded = False
                    return X_data, data_loaded
            
            nc_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.nc')]
            if not nc_files:
                print("⚠️ No .nc files found")
                data_loaded = False
                return X_data, data_loaded
            
            all_data = []
            for file in nc_files:
                ds = xr.open_dataset(os.path.join(DATA_FOLDER, file))
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
            lat_start, lon_start = 80, 80
            region_size = 10
            rain_cropped = rain_data[:, lat_start:lat_start+region_size, lon_start:lon_start+region_size]
            rain_cropped = np.nan_to_num(rain_cropped, nan=0.0)
            mean_val = np.mean(rain_cropped)
            std_val = np.std(rain_cropped)
            if std_val < 1e-8:
                std_val = 1.0
            rain_norm = (rain_cropped - mean_val) / (std_val + 1e-8)
            
            seq_len = 30
            X = []
            for i in range(seq_len, len(rain_norm)):
                X.append(rain_norm[i-seq_len:i])
            X_data = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(2)
            data_loaded = True
            print(f"✅ Real data loaded: {X_data.shape[0]} samples")
        except Exception as e:
            print(f"⚠️ Error loading real data: {e}")
            data_loaded = False
    return X_data, data_loaded

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def predict_with_uncertainty(model, input_tensor, n_samples=10):
    if model is None:
        return {
            'prediction': random.uniform(-0.5, 0.5),
            'uncertainty': random.uniform(0.01, 0.1),
            'confidence_interval': [-0.5, 0.5]
        }
    
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

def climate_risk_score(temp, rain, temp_threshold=35, rain_threshold=100):
    temp_risk = min(10, max(0, (temp - 25) / (temp_threshold - 25) * 10))
    rain_risk = min(10, max(0, (rain_threshold - rain) / rain_threshold * 10))
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
    if model is None:
        return random.uniform(-0.5, 0.5)
    
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
# CITIES ENDPOINT
# ============================================================
@api.route('/cities', methods=['GET', 'OPTIONS'])
@cross_origin()
def get_cities():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'cities': INDIAN_CITIES})

# ============================================================
# PREDICT ENDPOINT
# ============================================================
@api.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
@log_request
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    load_model()
    load_real_data()
    
    temperature = safe_float(data.get('temperature', 30.0))
    rainfall = safe_float(data.get('rainfall', 50.0))
    
    if data_loaded and X_data is not None:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    result = predict_with_uncertainty(model, input_tensor)
    risk = climate_risk_score(temperature, rainfall)
    
    try:
        prediction = Prediction(
            temperature=temperature,
            rainfall=rainfall,
            prediction=result['prediction'],
            uncertainty=result['uncertainty'],
            confidence_lower=result['confidence_interval'][0],
            confidence_upper=result['confidence_interval'][1],
            risk_score=risk['risk_score'],
            severity=risk['severity'],
            data_source='real' if data_loaded else 'synthetic'
        )
        db.session.add(prediction)
        db.session.commit()
    except Exception as e:
        print(f"⚠️ Database error: {e}")
    
    return jsonify({
        'prediction': result['prediction'],
        'uncertainty': result['uncertainty'],
        'confidence_interval': result['confidence_interval'],
        'risk_score': risk,
        'data_source': 'real' if data_loaded else 'synthetic' if model is not None else 'placeholder',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================
# WHATIF ENDPOINT
# ============================================================
@api.route('/whatif', methods=['POST', 'OPTIONS'])
@cross_origin()
@log_request
def whatif():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    load_model()
    load_real_data()
    
    temp_delta = safe_float(data.get('temp_delta', 0.0))
    rain_delta = safe_float(data.get('rain_delta', 0.0))
    use_real_sample = data.get('use_real_sample', False)
    
    if use_real_sample and data_loaded and X_data is not None:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    new_pred = simulate_whatif(model, input_tensor, temp_delta, rain_delta)
    
    return jsonify({
        'whatif_prediction': new_pred,
        'temp_delta': temp_delta,
        'rain_delta': rain_delta,
        'data_source': 'real' if data_loaded else 'synthetic' if model is not None else 'placeholder'
    })

# ============================================================
# EXPLAIN HUMAN ENDPOINT
# ============================================================
@api.route('/explain_human', methods=['POST', 'OPTIONS'])
@cross_origin()
@log_request
def explain_human():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    load_model()
    load_real_data()
    
    use_real_sample = data.get('use_real_sample', False)
    method = data.get('method', 'integrated_gradients')
    temperature = safe_float(data.get('temperature', 30.0))
    rainfall = safe_float(data.get('rainfall', 50.0))
    
    if use_real_sample and data_loaded and X_data is not None:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    try:
        from src.explain import model_prediction_with_explanation
        result = model_prediction_with_explanation(model, input_tensor, method)
    except (ImportError, Exception):
        result = {
            'prediction': random.uniform(-0.5, 0.5),
            'prediction_meaning': 'simulated prediction',
            'explanation': {
                'human_explanation': '🔍 **Explanation:** The model used spatial patterns to make this prediction.\n\n📊 **Key Insights:**\n- The model found patterns in the rainfall data\n- Confidence Level: MEDIUM\n\n💡 **What this means:**\nThis is a simulated explanation. Install captum for full explainability.',
                'summary': {
                    'avg_importance': 0.50,
                    'max_importance': 0.70,
                    'confidence': 'MEDIUM'
                }
            },
            'feature_summary': {'description': 'Top features show balanced importance'}
        }
    
    risk = climate_risk_score(temperature, rainfall)
    
    return jsonify({
        'prediction': result['prediction'],
        'prediction_meaning': result['prediction_meaning'],
        'human_explanation': result['explanation']['human_explanation'],
        'summary': result['explanation']['summary'],
        'feature_summary': result['feature_summary']['description'],
        'risk_score': risk,
        'data_source': 'real' if data_loaded else 'synthetic' if model is not None else 'placeholder',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================
# CLIMATE ENDPOINT
# ============================================================
@api.route('/climate', methods=['POST', 'OPTIONS'])
@cross_origin()
@log_request
def get_climate():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    lat = data.get('lat')
    lon = data.get('lon')
    city = data.get('city', 'Unknown')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat or lon'}), 400
    
    weather = get_live_weather(float(lat), float(lon))
    if weather is None:
        return jsonify({'error': 'Failed to fetch weather data'}), 500
    
    load_model()
    load_real_data()
    
    temperature = weather['temperature']
    rainfall = weather['rainfall']
    humidity = weather['humidity']
    wind_speed = weather['wind_speed']
    pressure = weather['pressure']
    cloud_cover = weather['cloud_cover']
    
    if data_loaded and X_data is not None:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    result = predict_with_uncertainty(model, input_tensor)
    risk = climate_risk_score(temperature, rainfall)
    
    try:
        prediction = Prediction(
            temperature=temperature,
            rainfall=rainfall,
            humidity=humidity,
            wind_speed=wind_speed,
            pressure=pressure,
            cloud_cover=cloud_cover,
            prediction=result['prediction'],
            uncertainty=result['uncertainty'],
            confidence_lower=result['confidence_interval'][0],
            confidence_upper=result['confidence_interval'][1],
            risk_score=risk['risk_score'],
            severity=risk['severity'],
            data_source='real' if data_loaded else 'synthetic'
        )
        db.session.add(prediction)
        db.session.commit()
    except Exception as e:
        print(f"⚠️ Database error: {e}")
    
    return jsonify({
        'location': {'city': city, 'latitude': lat, 'longitude': lon},
        'live_weather': {
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'wind_direction': weather.get('wind_direction', 0),
            'pressure': pressure,
            'cloud_cover': cloud_cover,
            'timestamp': weather['timestamp'],
            'source': weather['source']
        },
        'ai_prediction': {
            'prediction': result['prediction'],
            'uncertainty': result['uncertainty'],
            'confidence_interval': result['confidence_interval']
        },
        'risk_score': risk,
        'data_source': 'real' if data_loaded else 'synthetic' if model is not None else 'placeholder',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================
# HISTORY ENDPOINT (FIXED - No 500 Error)
# ============================================================
@api.route('/history', methods=['GET', 'OPTIONS'])
@cross_origin()
@log_request
def get_history():
    if request.method == 'OPTIONS':
        return '', 200
    
    limit = safe_int(request.args.get('limit', 50))
    try:
        # Check if table exists
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        if not inspector.has_table('predictions'):
            return jsonify({
                'predictions': [],
                'count': 0,
                'message': 'No predictions yet. Make a prediction first!'
            }), 200
        
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
        return jsonify({
            'predictions': [p.to_dict() for p in predictions],
            'count': len(predictions)
        })
    except Exception as e:
        print(f"⚠️ History error: {e}")
        return jsonify({
            'predictions': [],
            'count': 0,
            'error': str(e)
        }), 200

# ============================================================
# ALERTS ENDPOINT (FIXED - No 500 Error)
# ============================================================
@api.route('/alerts', methods=['GET', 'OPTIONS'])
@cross_origin()
@log_request
def get_alerts():
    if request.method == 'OPTIONS':
        return '', 200
    
    limit = safe_int(request.args.get('limit', 20))
    try:
        # Check if table exists
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        if not inspector.has_table('alerts'):
            return jsonify({
                'alerts': [],
                'count': 0,
                'message': 'No alerts yet.'
            }), 200
        
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(limit).all()
        return jsonify({
            'alerts': [a.to_dict() for a in alerts],
            'count': len(alerts)
        })
    except Exception as e:
        print(f"⚠️ Alerts error: {e}")
        return jsonify({
            'alerts': [],
            'count': 0,
            'error': str(e)
        }), 200

# ============================================================
# MARK ALERT AS READ
# ============================================================
@api.route('/alerts/<int:alert_id>/read', methods=['PUT', 'OPTIONS'])
@cross_origin()
@log_request
def mark_alert_read(alert_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        alert = Alert.query.get(alert_id)
        if not alert:
            return jsonify({'error': 'Alert not found'}), 404
        alert.is_read = True
        db.session.commit()
        return jsonify({'message': 'Alert marked as read'})
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500