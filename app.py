from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import traceback
import logging
from preprocessing.image_processing import preprocess_image
from features.minutiae_extraction import extract_minutiae
from matching.matcher import match_fingerprints

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f"File saved successfully: {filepath}")
            return jsonify({'filename': filename, 'filepath': filepath})
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_fingerprints():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
            
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not data or 'original' not in data or 'partial' not in data:
            return jsonify({'error': 'Missing original or partial filename'}), 400
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], data['original'])
        partial_path = os.path.join(app.config['UPLOAD_FOLDER'], data['partial'])
        
        logger.debug(f"Original path: {original_path}")
        logger.debug(f"Partial path: {partial_path}")
        
        if not os.path.exists(original_path) or not os.path.exists(partial_path):
            return jsonify({'error': 'One or both files not found'}), 404
        
        # Read and preprocess images
        original_img = cv2.imread(original_path)
        partial_img = cv2.imread(partial_path)
        
        if original_img is None or partial_img is None:
            return jsonify({'error': 'Failed to read image files'}), 400
        
        logger.debug(f"Original image shape: {original_img.shape}")
        logger.debug(f"Partial image shape: {partial_img.shape}")
        
        # Preprocess images
        original_processed = preprocess_image(original_img)
        partial_processed = preprocess_image(partial_img)
        
        logger.debug(f"Preprocessing completed")
        
        # Extract minutiae
        original_minutiae = extract_minutiae(original_processed)
        partial_minutiae = extract_minutiae(partial_processed)
        
        logger.debug(f"Original minutiae count: {len(original_minutiae)}")
        logger.debug(f"Partial minutiae count: {len(partial_minutiae)}")
        
        # Match fingerprints
        match_result = match_fingerprints(original_minutiae, partial_minutiae)
        
        logger.debug(f"Match result: {match_result}")
        return jsonify(match_result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 