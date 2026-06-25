import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import Config
from backend.database import init_db
from backend.routes import api
from backend.middleware import logger

# Create Flask app with explicit instance path
app = Flask(__name__, instance_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'instance'))
app.config.from_object(Config)

# --- FIX: Set database URI using absolute path ---
db_path = os.path.join(app.instance_path, 'climate.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

# Ensure instance folder exists
os.makedirs(app.instance_path, exist_ok=True)
print(f"📁 Instance folder: {app.instance_path}")

# Initialize database
init_db(app)

# Enable CORS
CORS(app, origins=app.config['CORS_ORIGINS'])

# Register routes
app.register_blueprint(api)

# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.route('/')
def home():
    return jsonify({
        'name': '🌤️ Climate AI Backend',
        'version': '1.0',
        'status': 'running',
        'database': 'SQLite',
        'endpoints': {
            'health': '/api/health',
            'test': '/api/test',
            'cities': '/api/cities',
            'predict': '/api/predict',
            'whatif': '/api/whatif',
            'explain_human': '/api/explain_human',
            'predict_weather': '/api/predict_weather',
            'history': '/api/history',
            'alerts': '/api/alerts'
        }
    })

@app.route('/list_routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({'routes': routes})
# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌤️ CLIMATE AI BACKEND SERVER")
    print("="*60)
    print(f"📊 Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("📋 Available Endpoints:")
    print("  GET    /")
    print("  GET    /api/health")
    print("  GET    /api/test")
    print("  GET    /api/cities")
    print("  POST   /api/predict")
    print("  POST   /api/whatif")
    print("  POST   /api/explain_human")
    print("  POST   /api/predict_weather")
    print("  GET    /api/history")
    print("  GET    /api/alerts")
    print("  PUT    /api/alerts/<id>/read")
    print("="*60)
    app.run(host='0.0.0.0', port=5001, debug=True)