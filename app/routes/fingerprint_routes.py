from flask import Blueprint, request, jsonify, render_template, url_for, send_file
import os
from datetime import datetime
import cv2
import json
import logging
from werkzeug.utils import secure_filename

from ..utils.image_processing import preprocess_image, enhance_image, remove_noise, normalize_ridges
from ..utils.minutiae_extraction import extract_minutiae, visualize_minutiae
from ..utils.matcher import match_fingerprints, visualize_matches
from ..utils.feature_extraction import extract_features
from ..config.config import *

# إنشاء Blueprint
fingerprint_bp = Blueprint('fingerprint', __name__)

# إعداد التسجيل
logger = logging.getLogger(__name__)

@fingerprint_bp.route('/')
def index():
    return render_template('index.html')

@fingerprint_bp.route('/normal_compare')
def normal_compare():
    return render_template('normal_compare.html')

@fingerprint_bp.route('/partial_compare')
def partial_compare():
    return render_template('partial_compare.html')

@fingerprint_bp.route('/advanced_compare')
def advanced_compare():
    return render_template('advanced_compare.html')

@fingerprint_bp.route('/grid_cutter')
def grid_cutter():
    return render_template('grid_cutter.html')

@fingerprint_bp.route('/grid_compare')
def grid_compare():
    return render_template('grid_compare.html')

@fingerprint_bp.route('/reports')
def reports():
    return render_template('reports.html')

@fingerprint_bp.route('/settings')
def settings():
    return render_template('settings.html')

@fingerprint_bp.route('/upload', methods=['POST'])
def upload_fingerprint():
    try:
        logger.info("Starting fingerprint upload and processing...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'Both fingerprint images are required'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # الحصول على المعلمات
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        is_partial_mode = request.form.get('matchingMode') == 'true'
        
        logger.info(f"Parameters: minutiae_count={minutiae_count}, is_partial_mode={is_partial_mode}")
        
        # التحقق من الملفات
        if fingerprint1.filename == '' or fingerprint2.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(fingerprint1.filename) and allowed_file(fingerprint2.filename)):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # حفظ الملفات
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename1 = f"{timestamp}_1_{secure_filename(fingerprint1.filename)}"
        filename2 = f"{timestamp}_2_{secure_filename(fingerprint2.filename)}"
        
        filepath1 = os.path.join(UPLOAD_FOLDER, filename1)
        filepath2 = os.path.join(UPLOAD_FOLDER, filename2)
        
        fingerprint1.save(filepath1)
        fingerprint2.save(filepath2)
        
        logger.info("Files saved successfully")
        
        # معالجة الصور
        processed1 = preprocess_image(filepath1)
        processed2 = preprocess_image(filepath2)
        
        if processed1 is None or processed2 is None:
            return jsonify({'error': 'Error processing images'}), 400
        
        # استخراج نقاط التفاصيل
        minutiae1 = extract_minutiae(processed1, minutiae_count)
        minutiae2 = extract_minutiae(processed2, minutiae_count)
        
        # استخراج الخصائص
        features1 = extract_features(processed1)
        features2 = extract_features(processed2)
        
        # مطابقة البصمات
        match_result = match_fingerprints(minutiae1, minutiae2, features1, features2)
        
        # إنشاء الصور التوضيحية
        min1_img = visualize_minutiae(processed1, minutiae1)
        min2_img = visualize_minutiae(processed2, minutiae2)
        match_img = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
        
        # حفظ الصور
        proc1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_processed.png')
        proc2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_processed.png')
        min1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_minutiae.png')
        min2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_minutiae.png')
        match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
        
        cv2.imwrite(proc1_path, processed1)
        cv2.imwrite(proc2_path, processed2)
        cv2.imwrite(min1_path, min1_img)
        cv2.imwrite(min2_path, min2_img)
        cv2.imwrite(match_path, match_img)
        
        logger.info("Images processed and saved successfully")
        
        # تحضير البيانات للرد
        response_data = {
            'processed_images': {
                'img1': url_for('static', filename=f'images/processed/{timestamp}_1_processed.png'),
                'img2': url_for('static', filename=f'images/processed/{timestamp}_2_processed.png')
            },
            'minutiae_images': {
                'img1': url_for('static', filename=f'images/processed/{timestamp}_1_minutiae.png'),
                'img2': url_for('static', filename=f'images/processed/{timestamp}_2_minutiae.png')
            },
            'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
            'score': match_result['score'] * 100,
            'quality_score': match_result['quality_score'] * 100,
            'is_match': match_result['score'] >= MATCHING_THRESHOLD / 100,
            'minutiae_count': {
                'img1': len(minutiae1),
                'img2': len(minutiae2)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 