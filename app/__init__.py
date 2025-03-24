"""
حزمة تطبيق مطابقة البصمات
"""
from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Register blueprints
    from app.routes.fingerprint_routes import fingerprint_bp
    app.register_blueprint(fingerprint_bp)
    
    return app

app = create_app() 