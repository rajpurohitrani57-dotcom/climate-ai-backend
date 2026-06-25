import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-2024')
    
    # We will set SQLALCHEMY_DATABASE_URI in app.py
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    MODEL_PATH = os.getenv('MODEL_PATH', 'model/climate_model.pth')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))