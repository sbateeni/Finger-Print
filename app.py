from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from preprocessing.image_processing import preprocess_image, detect_ridges, analyze_ridge_patterns
from features.minutiae_extraction import extract_minutiae, analyze_ridge_characteristics
from matching.matcher import match_fingerprints
import logging

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
        
        # معالجة الصور
        img1 = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return jsonify({'error': 'فشل في قراءة الصور'}), 400
        
        # معالجة الصور
        processed_img1 = preprocess_image(img1)
        processed_img2 = preprocess_image(img2)
        
        # استخراج الميزات
        minutiae1 = extract_minutiae(processed_img1)
        minutiae2 = extract_minutiae(processed_img2)
        
        # تحليل أنماط الخطوط
        ridge_patterns1 = analyze_ridge_patterns(processed_img1)
        ridge_patterns2 = analyze_ridge_patterns(processed_img2)
        
        # مطابقة البصمات
        match_result = match_fingerprints(minutiae1, minutiae2)
        
        # تحليل خصائص الخطوط بين النقاط المتطابقة
        ridge_analysis = []
        if match_result['matches']:
            ridge_analysis = analyze_ridge_characteristics(
                processed_img1, 
                processed_img2, 
                match_result['matches']
            )
        
        # تنظيف الملفات المؤقتة
        os.remove(filepath1)
        os.remove(filepath2)
        
        return jsonify({
            'match_score': match_result['match_score'],
            'num_matches': len(match_result['matches']),
            'ridge_patterns1': ridge_patterns1,
            'ridge_patterns2': ridge_patterns2,
            'ridge_analysis': ridge_analysis,
            'details': match_result['details']
        })
        
    except Exception as e:
        logger.error(f"Error processing fingerprints: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 