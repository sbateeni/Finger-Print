from flask import Flask
import os
import logging
from app.routes.fingerprint_routes import fingerprint_bp
from app.config.config import (
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    RESULTS_FOLDER,
    OUTPUT_FOLDER,
    MAX_FILE_SIZE
)

# إعداد التسجيل
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # إعداد التكوين
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    # إنشاء المجلدات المطلوبة
    for directory in [UPLOAD_FOLDER, PROCESSED_FOLDER, RESULTS_FOLDER, OUTPUT_FOLDER, 'logs']:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    
    # تسجيل Blueprint
    app.register_blueprint(fingerprint_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 