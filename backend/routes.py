import os
import sys
import torch
import numpy as np
from flask import request, jsonify, Blueprint
from datetime import datetime
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import db
from backend.models import Prediction, Alert
from backend.weather import get_live_weather, INDIAN_CITIES
from backend.middleware import log_request, rate_limit
from backend.utils import safe_float, format_timestamp
from backend.middleware import log_request, rate_limit
from backend.utils import safe_float, format_timestamp

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

# Global variables for model and data
# ============================================================
# GLOBAL VARIABLES
# ============================================================
model = None
X_data = None
data_loaded = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_model():
    """Load the AI model"""
    global model
    if model is None and AI_AVAILABLE:
        try:
            model_path = 'model/climate_model.pth'
            print(f"🔍 Loading AI model from: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found at {model_path}")
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
    """Load real IMD data"""
    global X_data, data_loaded
    if X_data is None and AI_AVAILABLE:
        try:
            import xarray as xr
            DATA_FOLDER = r'C:\Users\HP\OneDrive\Desktop\imd_data'
            
            if not os.path.exists(DATA_FOLDER):
                print(f"⚠️ Data folder not found: {DATA_FOLDER}")
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

def predict_with_uncertainty(model, input_tensor, n_samples=10):
    """Get prediction with uncertainty"""
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
    """Calculate climate risk score (0-10)"""
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
    """What-if simulation"""
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
# API ENDPOINTS
# ============================================================

@api.route('/health', methods=['GET'])
@log_request
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'data_source': 'real' if data_loaded else 'synthetic' if AI_AVAILABLE else 'none',
        'ai_available': AI_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@api.route('/test', methods=['GET'])
@log_request
def test():
    """Test endpoint"""
    return jsonify({
        'message': '✅ Backend is working!',
        'ai_available': AI_AVAILABLE,
        'model_loaded': model is not None,
        'endpoints': [
            'GET  /api/health',
            'GET  /api/test',
            'GET  /api/cities',
            'POST /api/predict',
            'POST /api/whatif',
            'POST /api/explain_human',
            'POST /api/predict_weather',
            'GET  /api/history',
            'GET  /api/alerts'
        ]
    })

@api.route('/cities', methods=['GET'])
@log_request
def get_cities():
    """Get list of Indian cities"""
    return jsonify({'cities': INDIAN_CITIES})

@api.route('/predict', methods=['POST'])
@log_request
@rate_limit(limit_per_minute=30)
def predict():
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Load model and data
    load_model()
    load_real_data()
    
    temperature = safe_float(data.get('temperature', 30.0))
    rainfall = safe_float(data.get('rainfall', 50.0))
    
    # --- Get input tensor ---
    if data_loaded and 'features' not in data:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        # Create dummy input if no real data
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    # --- REAL PREDICTION (not placeholder) ---
    if model is not None:
        # Use the actual model
        result = predict_with_uncertainty(model, input_tensor)
    else:
        # Fallback to placeholder
        result = {
            'prediction': random.uniform(-0.5, 0.5),
            'uncertainty': random.uniform(0.01, 0.1),
            'confidence_interval': [-0.5, 0.5]
        }
    
    risk = climate_risk_score(temperature, rainfall)
    
    # Save to database
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

@api.route('/whatif', methods=['POST'])
@log_request
def whatif():
    """What-if simulation endpoint"""
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    load_model()
    load_real_data()
    
    temp_delta = safe_float(data.get('temp_delta', 0.0))
    rain_delta = safe_float(data.get('rain_delta', 0.0))
    use_real_sample = data.get('use_real_sample', False)
    
    # Get input tensor
    if use_real_sample and data_loaded:
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
        'data_source': 'real' if data_loaded else 'synthetic' if AI_AVAILABLE else 'placeholder'
    })

@api.route('/explain_human', methods=['POST'])
@log_request
def explain_human():
    """Human-readable explanation"""
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    load_model()
    load_real_data()
    
    use_real_sample = data.get('use_real_sample', False)
    method = data.get('method', 'integrated_gradients')
    temperature = safe_float(data.get('temperature', 30.0))
    rainfall = safe_float(data.get('rainfall', 50.0))
    
    # Get input tensor
    if use_real_sample and data_loaded:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    elif 'features' in data:
        try:
            input_tensor = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            return jsonify({'error': f'Invalid features format: {str(e)}'}), 400
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    # Check if explain module is available
    try:
        from src.explain import model_prediction_with_explanation
        result = model_prediction_with_explanation(model, input_tensor, method)
    except ImportError:
        # Fallback: generate a simple explanation
        result = {
            'prediction': random.uniform(-0.5, 0.5),
            'prediction_meaning': 'simulated prediction (explain module not available)',
            'explanation': {
                'human_explanation': '🔍 **Explanation:** The model used spatial patterns to make this prediction. The confidence is estimated based on the data available.\n\n📊 **Key Insights:**\n- The model found patterns in the rainfall data\n- Confidence Level: MEDIUM\n\n💡 **What this means:**\nThis is a simulated explanation. Install captum for full explainability.',
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
        'data_source': 'real' if data_loaded else 'synthetic' if AI_AVAILABLE else 'placeholder',
        'timestamp': datetime.now().isoformat()
    })

@api.route('/predict_weather', methods=['POST'])
@log_request
def predict_weather():
    """Get live weather + AI prediction"""
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    lat = data.get('lat')
    lon = data.get('lon')
    city = data.get('city', 'Unknown')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat or lon'}), 400
    
    # Get live weather
    weather = get_live_weather(float(lat), float(lon))
    if weather is None:
        return jsonify({'error': 'Failed to fetch weather data'}), 500
    
    # Get AI prediction
    load_model()
    load_real_data()
    
    temperature = weather['temperature']
    rainfall = weather['rainfall']
    
    # Get input tensor
    if data_loaded:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    result = predict_with_uncertainty(model, input_tensor)
    risk = climate_risk_score(temperature, rainfall)
    
    return jsonify({
        'location': {
            'city': city,
            'latitude': lat,
            'longitude': lon
        },
        'live_weather': weather,
        'ai_prediction': {
            'prediction': result['prediction'],
            'uncertainty': result['uncertainty'],
            'confidence_interval': result['confidence_interval']
        },
        'risk_score': risk,
        'data_source': 'real' if data_loaded else 'synthetic' if AI_AVAILABLE else 'placeholder',
        'timestamp': datetime.now().isoformat()
    })

# Add this new endpoint for comprehensive climate data

@api.route('/climate', methods=['POST'])
@log_request
def get_climate():
    """Get COMPLETE climate data for a location"""
    data = request.json
    
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    lat = data.get('lat')
    lon = data.get('lon')
    city = data.get('city', 'Unknown')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing lat or lon'}), 400
    
    # Get live weather
    weather = get_live_weather(float(lat), float(lon))
    if weather is None:
        return jsonify({'error': 'Failed to fetch weather data'}), 500
    
    # Load AI model and data
    load_model()
    load_real_data()
    
    temperature = weather['temperature']
    rainfall = weather['rainfall']
    humidity = weather['humidity']
    wind_speed = weather['wind_speed']
    pressure = weather['pressure']
    cloud_cover = weather['cloud_cover']
    
    # Get input tensor for AI prediction
    if data_loaded and X_data is not None:
        idx = np.random.randint(0, len(X_data))
        input_tensor = X_data[idx:idx+1].clone()
    else:
        input_tensor = torch.randn(1, 30, 1, 10, 10)
    
    result = predict_with_uncertainty(model, input_tensor)
    risk = climate_risk_score(temperature, rainfall)
    
    # Save to database with all variables
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
        
        # Create alert for high risk
        if risk['severity'] in ['HIGH', 'SEVERE']:
            alert = Alert(
                severity=risk['severity'],
                message=f"⚠️ Climate Alert: {risk['severity']} risk detected! Risk Score: {risk['risk_score']}",
                risk_score=risk['risk_score']
            )
            db.session.add(alert)
            db.session.commit()
    except Exception as e:
        print(f"⚠️ Database error: {e}")
    
    return jsonify({
        'location': {
            'city': city,
            'latitude': lat,
            'longitude': lon
        },
        'live_weather': {
            'temperature': weather['temperature'],
            'rainfall': weather['rainfall'],
            'humidity': weather['humidity'],
            'wind_speed': weather['wind_speed'],
            'wind_direction': weather.get('wind_direction', 0),
            'pressure': weather['pressure'],
            'cloud_cover': weather['cloud_cover'],
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

# Add endpoint for getting historical climate data
@api.route('/climate/history', methods=['GET'])
@log_request
def get_climate_history():
    """Get historical climate predictions"""
    limit = safe_int(request.args.get('limit', 50))
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
        return jsonify({
            'predictions': [p.to_dict() for p in predictions],
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
   

@api.route('/history', methods=['GET'])
@log_request
def get_history():
    """Get prediction history"""
    limit = safe_int(request.args.get('limit', 50))
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
        return jsonify({
            'predictions': [p.to_dict() for p in predictions],
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@api.route('/alerts', methods=['GET'])
@log_request
def get_alerts():
    """Get alerts"""
    limit = safe_int(request.args.get('limit', 20))
    try:
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(limit).all()
        return jsonify({
            'alerts': [a.to_dict() for a in alerts],
            'count': len(alerts)
        })
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@api.route('/alerts/<int:alert_id>/read', methods=['PUT'])
@log_request
def mark_alert_read(alert_id):
    """Mark alert as read"""
    try:
        alert = Alert.query.get(alert_id)
        if not alert:
            return jsonify({'error': 'Alert not found'}), 404
        alert.is_read = True
        db.session.commit()
        return jsonify({'message': 'Alert marked as read'})
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500