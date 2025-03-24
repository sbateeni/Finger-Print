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

@fingerprint_bp.route('/save_grid_cells', methods=['POST'])
def save_grid_cells():
    try:
        data = request.get_json()
        cells = data.get('cells', [])
        
        if not cells:
            return jsonify({'error': 'No cells selected'}), 400
        
        # حفظ الخلايا في ملف
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'grid_cells_{timestamp}.json'
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'cells': cells,
                'total_cells': len(cells)
            }, f, ensure_ascii=False, indent=4)
        
        return jsonify({
            'message': 'Grid cells saved successfully',
            'filename': filename,
            'total_cells': len(cells)
        })
        
    except Exception as e:
        logger.error(f"Error in save_grid_cells: {str(e)}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@fingerprint_bp.route('/upload_normal', methods=['POST'])
def upload_normal_fingerprint():
    try:
        logger.info("Starting normal fingerprint upload and processing...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'Both fingerprint images are required'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # الحصول على المعلمات
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        
        logger.info(f"Parameters: minutiae_count={minutiae_count}")
        
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
        logger.error(f"Error in upload_normal_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@fingerprint_bp.route('/upload_partial', methods=['POST'])
def upload_partial_fingerprint():
    try:
        logger.info("Starting partial fingerprint upload and processing...")
        
        # التحقق من وجود الملفات
        if 'fullFingerprint' not in request.files or 'partialFingerprint' not in request.files:
            return jsonify({'error': 'Both full and partial fingerprint images are required'}), 400
        
        full_fingerprint = request.files['fullFingerprint']
        partial_fingerprint = request.files['partialFingerprint']
        
        # الحصول على المعلمات
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        matching_threshold = float(request.form.get('matchingThreshold', 80)) / 100
        
        logger.info(f"Parameters: minutiae_count={minutiae_count}, matching_threshold={matching_threshold}")
        
        # التحقق من الملفات
        if full_fingerprint.filename == '' or partial_fingerprint.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(full_fingerprint.filename) and allowed_file(partial_fingerprint.filename)):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # حفظ الملفات
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"{timestamp}_full_{secure_filename(full_fingerprint.filename)}"
        partial_filename = f"{timestamp}_partial_{secure_filename(partial_fingerprint.filename)}"
        
        full_filepath = os.path.join(UPLOAD_FOLDER, full_filename)
        partial_filepath = os.path.join(UPLOAD_FOLDER, partial_filename)
        
        full_fingerprint.save(full_filepath)
        partial_fingerprint.save(partial_filepath)
        
        logger.info("Files saved successfully")
        
        # معالجة الصور
        processed_full = preprocess_image(full_filepath)
        processed_partial = preprocess_image(partial_filepath)
        
        if processed_full is None or processed_partial is None:
            return jsonify({'error': 'Error processing images'}), 400
        
        # استخراج نقاط التفاصيل
        minutiae_full = extract_minutiae(processed_full, minutiae_count)
        minutiae_partial = extract_minutiae(processed_partial, minutiae_count)
        
        # استخراج الخصائص
        features_full = extract_features(processed_full)
        features_partial = extract_features(processed_partial)
        
        # مطابقة البصمات
        match_result = match_fingerprints(minutiae_full, minutiae_partial, features_full, features_partial)
        
        # إنشاء الصور التوضيحية
        min_full_img = visualize_minutiae(processed_full, minutiae_full)
        min_partial_img = visualize_minutiae(processed_partial, minutiae_partial)
        match_img = visualize_matches(processed_full, processed_partial, match_result['matched_minutiae'])
        
        # حفظ الصور
        proc_full_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_full_processed.png')
        proc_partial_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_partial_processed.png')
        min_full_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_full_minutiae.png')
        min_partial_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_partial_minutiae.png')
        match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
        
        cv2.imwrite(proc_full_path, processed_full)
        cv2.imwrite(proc_partial_path, processed_partial)
        cv2.imwrite(min_full_path, min_full_img)
        cv2.imwrite(min_partial_path, min_partial_img)
        cv2.imwrite(match_path, match_img)
        
        logger.info("Images processed and saved successfully")
        
        # تحضير البيانات للرد
        response_data = {
            'processed_images': {
                'full': url_for('static', filename=f'images/processed/{timestamp}_full_processed.png'),
                'partial': url_for('static', filename=f'images/processed/{timestamp}_partial_processed.png')
            },
            'minutiae_images': {
                'full': url_for('static', filename=f'images/processed/{timestamp}_full_minutiae.png'),
                'partial': url_for('static', filename=f'images/processed/{timestamp}_partial_minutiae.png')
            },
            'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
            'score': match_result['score'] * 100,
            'quality_score': match_result['quality_score'] * 100,
            'is_match': match_result['score'] >= matching_threshold,
            'minutiae_count': {
                'full': len(minutiae_full),
                'partial': len(minutiae_partial)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_partial_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@fingerprint_bp.route('/upload_advanced', methods=['POST'])
def upload_advanced_fingerprint():
    try:
        logger.info("Starting advanced fingerprint upload and processing...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'Both fingerprint images are required'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # الحصول على المعلمات المتقدمة
        image_quality = float(request.form.get('imageQuality', 80)) / 100
        contrast = float(request.form.get('contrast', 100)) / 100
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        minutiae_quality = float(request.form.get('minutiaeQuality', 80)) / 100
        matching_threshold = float(request.form.get('matchingThreshold', 80)) / 100
        rotation_tolerance = float(request.form.get('rotationTolerance', 30))
        algorithm = request.form.get('algorithm', 'minutiae')
        is_partial_mode = request.form.get('matchingMode') == 'true'
        
        logger.info(f"Advanced parameters: image_quality={image_quality}, contrast={contrast}, "
                   f"minutiae_count={minutiae_count}, minutiae_quality={minutiae_quality}, "
                   f"matching_threshold={matching_threshold}, rotation_tolerance={rotation_tolerance}, "
                   f"algorithm={algorithm}, is_partial_mode={is_partial_mode}")
        
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
        
        # معالجة الصور مع المعلمات المتقدمة
        processed1 = preprocess_image(filepath1, image_quality, contrast)
        processed2 = preprocess_image(filepath2, image_quality, contrast)
        
        if processed1 is None or processed2 is None:
            return jsonify({'error': 'Error processing images'}), 400
        
        # استخراج نقاط التفاصيل
        minutiae1 = extract_minutiae(processed1, minutiae_count, minutiae_quality)
        minutiae2 = extract_minutiae(processed2, minutiae_count, minutiae_quality)
        
        # استخراج الخصائص
        features1 = extract_features(processed1)
        features2 = extract_features(processed2)
        
        # مطابقة البصمات
        match_result = match_fingerprints(
            minutiae1, minutiae2, features1, features2,
            threshold=matching_threshold,
            rotation_tolerance=rotation_tolerance,
            algorithm=algorithm,
            is_partial=is_partial_mode
        )
        
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
            'is_match': match_result['score'] >= matching_threshold / 100,
            'minutiae_count': {
                'img1': len(minutiae1),
                'img2': len(minutiae2)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_advanced_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@fingerprint_bp.route('/upload_grid', methods=['POST'])
def upload_grid_fingerprint():
    try:
        logger.info("Starting grid fingerprint upload and processing...")
        
        # التحقق من وجود الملف
        if 'fingerprint' not in request.files:
            return jsonify({'error': 'Fingerprint image is required'}), 400
        
        fingerprint = request.files['fingerprint']
        
        # الحصول على معلمات الشبكة
        grid_rows = int(request.form.get('gridRows', 3))
        grid_cols = int(request.form.get('gridCols', 3))
        overlap = float(request.form.get('overlap', 20)) / 100
        cell_size = int(request.form.get('cellSize', 200))
        
        logger.info(f"Parameters: grid_rows={grid_rows}, grid_cols={grid_cols}, overlap={overlap}, cell_size={cell_size}")
        
        # التحقق من الملف
        if fingerprint.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(fingerprint.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # حفظ الملف
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(fingerprint.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        fingerprint.save(filepath)
        
        logger.info("File saved successfully")
        
        # معالجة الصورة
        processed = preprocess_image(filepath)
        
        if processed is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # إنشاء الشبكة
        height, width = processed.shape[:2]
        cell_width = int(width / grid_cols)
        cell_height = int(height / grid_rows)
        
        # إنشاء صورة الشبكة
        grid_img = processed.copy()
        
        # رسم خطوط الشبكة
        for i in range(grid_rows + 1):
            y = i * cell_height
            cv2.line(grid_img, (0, y), (width, y), (0, 255, 0), 1)
        
        for j in range(grid_cols + 1):
            x = j * cell_width
            cv2.line(grid_img, (x, 0), (x, height), (0, 255, 0), 1)
        
        # حفظ صورة الشبكة
        grid_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_grid.png')
        cv2.imwrite(grid_path, grid_img)
        
        logger.info("Grid image created and saved successfully")
        
        # تحضير البيانات للرد
        response_data = {
            'grid_image': url_for('static', filename=f'images/processed/{timestamp}_grid.png'),
            'grid_info': {
                'rows': grid_rows,
                'cols': grid_cols,
                'total_cells': grid_rows * grid_cols,
                'cell_width': cell_width,
                'cell_height': cell_height,
                'overlap': overlap
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_grid_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@fingerprint_bp.route('/upload_grid_compare', methods=['POST'])
def upload_grid_compare_fingerprint():
    try:
        logger.info("Starting grid comparison fingerprint upload and processing...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'Both fingerprint images are required'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # الحصول على معلمات الشبكة
        grid_rows = int(request.form.get('gridRows', 3))
        grid_cols = int(request.form.get('gridCols', 3))
        overlap = float(request.form.get('overlap', 20)) / 100
        matching_threshold = float(request.form.get('matchingThreshold', 80)) / 100
        
        logger.info(f"Parameters: grid_rows={grid_rows}, grid_cols={grid_cols}, overlap={overlap}, matching_threshold={matching_threshold}")
        
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
        
        # إنشاء الشبكة
        height1, width1 = processed1.shape[:2]
        height2, width2 = processed2.shape[:2]
        
        cell_width1 = int(width1 / grid_cols)
        cell_height1 = int(height1 / grid_rows)
        cell_width2 = int(width2 / grid_cols)
        cell_height2 = int(height2 / grid_rows)
        
        # إنشاء صور الشبكة
        grid_img1 = processed1.copy()
        grid_img2 = processed2.copy()
        
        # رسم خطوط الشبكة
        for i in range(grid_rows + 1):
            y1 = i * cell_height1
            y2 = i * cell_height2
            cv2.line(grid_img1, (0, y1), (width1, y1), (0, 255, 0), 1)
            cv2.line(grid_img2, (0, y2), (width2, y2), (0, 255, 0), 1)
        
        for j in range(grid_cols + 1):
            x1 = j * cell_width1
            x2 = j * cell_width2
            cv2.line(grid_img1, (x1, 0), (x1, height1), (0, 255, 0), 1)
            cv2.line(grid_img2, (x2, 0), (x2, height2), (0, 255, 0), 1)
        
        # حفظ صور الشبكة
        grid_path1 = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_grid.png')
        grid_path2 = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_grid.png')
        cv2.imwrite(grid_path1, grid_img1)
        cv2.imwrite(grid_path2, grid_img2)
        
        logger.info("Grid images created and saved successfully")
        
        # تحضير البيانات للرد
        response_data = {
            'grid_images': {
                'img1': url_for('static', filename=f'images/processed/{timestamp}_1_grid.png'),
                'img2': url_for('static', filename=f'images/processed/{timestamp}_2_grid.png')
            },
            'grid_info': {
                'rows': grid_rows,
                'cols': grid_cols,
                'total_cells': grid_rows * grid_cols,
                'cell_width1': cell_width1,
                'cell_height1': cell_height1,
                'cell_width2': cell_width2,
                'cell_height2': cell_height2,
                'overlap': overlap,
                'matching_threshold': matching_threshold
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in upload_grid_compare_fingerprint: {str(e)}")
        return jsonify({'error': str(e)}), 500 