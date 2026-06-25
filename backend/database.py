import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    # Ensure instance folder exists (just in case)
    os.makedirs(app.instance_path, exist_ok=True)
    
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")