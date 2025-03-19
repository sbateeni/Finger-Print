from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from preprocessing.image_processing import preprocess_image, detect_ridges, analyze_ridge_patterns
from features.minutiae_extraction import extract_minutiae, analyze_ridge_characteristics
from matching.matcher import match_fingerprints
import logging
import traceback

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# التأكد من وجود مجلد التحميل
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(image):
    if image is None:
        return False
    if image.size == 0:
        return False
    if len(image.shape) != 2:
        return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'يرجى تحميل ملفين'}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400
        
        # حفظ الملفات
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        # قراءة الصور
        img1 = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
        
        if not validate_image(img1) or not validate_image(img2):
            return jsonify({'error': 'فشل في قراءة الصور أو الصور غير صالحة'}), 400
        
        # معالجة الصور
        try:
            processed_img1, direction1 = preprocess_image(img1)
            processed_img2, direction2 = preprocess_image(img2)
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return jsonify({'error': 'فشل في معالجة الصور'}), 500
        
        # استخراج الميزات
        try:
            minutiae1 = extract_minutiae(processed_img1)
            minutiae2 = extract_minutiae(processed_img2)
        except Exception as e:
            logger.error(f"Error in minutiae extraction: {str(e)}")
            return jsonify({'error': 'فشل في استخراج الميزات'}), 500
        
        # تحليل أنماط الخطوط
        try:
            ridge_patterns1 = analyze_ridge_patterns(processed_img1, direction1)
            ridge_patterns2 = analyze_ridge_patterns(processed_img2, direction2)
        except Exception as e:
            logger.error(f"Error in ridge pattern analysis: {str(e)}")
            return jsonify({'error': 'فشل في تحليل أنماط الخطوط'}), 500
        
        # مطابقة البصمات
        try:
            match_result = match_fingerprints(minutiae1, minutiae2)
        except Exception as e:
            logger.error(f"Error in fingerprint matching: {str(e)}")
            return jsonify({'error': 'فشل في مطابقة البصمات'}), 500
        
        # تحليل خصائص الخطوط بين النقاط المتطابقة
        ridge_analysis = []
        if match_result.get('matches'):
            try:
                ridge_analysis = analyze_ridge_characteristics(
                    processed_img1, 
                    processed_img2, 
                    match_result['matches']
                )
            except Exception as e:
                logger.error(f"Error in ridge analysis: {str(e)}")
        
        # تنظيف الملفات المؤقتة
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except Exception as e:
            logger.warning(f"Error removing temporary files: {str(e)}")
        
        return jsonify({
            'match_score': match_result.get('match_score', 0),
            'num_matches': len(match_result.get('matches', [])),
            'ridge_patterns1': ridge_patterns1,
            'ridge_patterns2': ridge_patterns2,
            'ridge_analysis': ridge_analysis,
            'details': match_result.get('details', {})
        })
        
    except Exception as e:
        logger.error(f"Error processing fingerprints: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'حدث خطأ أثناء معالجة البصمات'}), 500

if __name__ == '__main__':
    app.run(debug=True) 