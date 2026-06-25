from backend.database import db
from datetime import datetime

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Input variables
    temperature = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    humidity = db.Column(db.Float)          # ✅ NEW
    wind_speed = db.Column(db.Float)        # ✅ NEW
    pressure = db.Column(db.Float)          # ✅ NEW
    cloud_cover = db.Column(db.Float)       # ✅ NEW
    
    # AI Output
    prediction = db.Column(db.Float)
    uncertainty = db.Column(db.Float)
    confidence_lower = db.Column(db.Float)
    confidence_upper = db.Column(db.Float)
    
    # Risk Assessment
    risk_score = db.Column(db.Float)
    severity = db.Column(db.String(20))
    data_source = db.Column(db.String(20))
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'temperature': self.temperature,
            'rainfall': self.rainfall,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'pressure': self.pressure,
            'cloud_cover': self.cloud_cover,
            'prediction': self.prediction,
            'uncertainty': self.uncertainty,
            'confidence_interval': [self.confidence_lower, self.confidence_upper],
            'risk_score': self.risk_score,
            'severity': self.severity,
            'data_source': self.data_source,
            'timestamp': self.timestamp.isoformat()
        }

class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    severity = db.Column(db.String(20))
    message = db.Column(db.String(500))
    risk_score = db.Column(db.Float)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'severity': self.severity,
            'message': self.message,
            'risk_score': self.risk_score,
            'is_read': self.is_read,
            'timestamp': self.timestamp.isoformat()
        }