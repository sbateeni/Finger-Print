import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from utils.fingerprint_processor import FingerprintProcessor
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import shutil

print("Starting application...")

print("Creating required directories...")
# إنشاء المجلدات المطلوبة إذا لم تكن موجودة
os.makedirs('uploads', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# إعداد التسجيل
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("Initializing Flask app...")
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # حد أقصى 16 ميجابايت
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

print("Creating FingerprintProcessor instance...")
fingerprint_processor = FingerprintProcessor()

# Create fingerprint database directory if it doesn't exist
FINGERPRINT_DB_DIR = 'fingerprint_database'
if not os.path.exists(FINGERPRINT_DB_DIR):
    os.makedirs(FINGERPRINT_DB_DIR)

def allowed_file(filename):
    """التحقق من امتداد الملف"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """عرض الصفحة الرئيسية"""
    print("Rendering index page...")
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_fingerprints():
    """مقارنة البصمتين المرفوعتين"""
    try:
        print("Processing fingerprint comparison request...")
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'يرجى تحميل صورتي البصمات'}), 400

        file1 = request.files['fingerprint1']
        file2 = request.files['fingerprint2']

        # التحقق من اختيار الملفات
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'لم يتم اختيار الملفات'}), 400

        # التحقق من صحة امتدادات الملفات
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

        # إنشاء مجلد التحميلات إذا لم يكن موجودًا
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)

        try:
            # حفظ الملفات
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            
            filepath1 = os.path.join(upload_dir, filename1)
            filepath2 = os.path.join(upload_dir, filename2)
            
            file1.save(filepath1)
            file2.save(filepath2)

            if not (os.path.exists(filepath1) and os.path.exists(filepath2)):
                raise FileNotFoundError("فشل في حفظ الملفات المحملة")

            print("Processing images...")
            # معالجة الصور
            img1 = fingerprint_processor.preprocess_image(filepath1)
            img2 = fingerprint_processor.preprocess_image(filepath2)

            print("Comparing fingerprints...")
            # مقارنة البصمات
            match_score, details = fingerprint_processor.compare_fingerprints(img1, img2)

            # حذف الملفات المؤقتة
            try:
                os.remove(filepath1)
                os.remove(filepath2)
            except OSError as e:
                print(f"Warning: Could not remove temporary files: {e}")

            return jsonify({
                'match_score': float(match_score),
                'details': {
                    'ssim_score': float(details['ssim_score']),
                    'feature_score': float(details['feature_score']),
                    'minutiae_score': float(details['minutiae_score']),
                    'orientation_score': float(details['orientation_score']),
                    'core_score': float(details['core_score']),
                    'frequency_score': float(details['frequency_score']),
                    'quality_score1': float(details['quality_score1']),
                    'quality_score2': float(details['quality_score2'])
                }
            })

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # حذف الملفات في حالة حدوث خطأ
            for filepath in [filepath1, filepath2]:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    pass
            raise e

    except Exception as e:
        logger.error(f"خطأ في مقارنة البصمات: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/compare_multiple', methods=['POST'])
def compare_multiple_fingerprints():
    """تحليل ومقارنة البصمات المتعددة في صورة واحدة"""
    try:
        print("Processing multiple fingerprints request...")
        # التحقق من وجود الملف
        if 'image' not in request.files:
            return jsonify({'error': 'يرجى تحميل صورة'}), 400

        file = request.files['image']

        # التحقق من اختيار الملف
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار الملف'}), 400

        # التحقق من صحة امتداد الملف
        if not allowed_file(file.filename):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

        # إنشاء مجلد التحميلات إذا لم يكن موجودًا
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)

        try:
            # حفظ الملف
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            if not os.path.exists(filepath):
                raise FileNotFoundError("فشل في حفظ الملف المحمل")

            print("Processing image for multiple fingerprints...")
            # معالجة الصورة واكتشاف البصمات المتعددة
            fingerprints = fingerprint_processor.detect_multiple_fingerprints(filepath)

            if len(fingerprints) < 2:
                raise ValueError("لم يتم العثور على بصمات كافية في الصورة. يجب أن تحتوي الصورة على بصمتين على الأقل.")

            print(f"Found {len(fingerprints)} fingerprints")
            
            # مقارنة كل البصمات مع بعضها
            comparisons = []
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    print(f"Comparing fingerprint {i+1} with fingerprint {j+1}")
                    match_score, details = fingerprint_processor.compare_fingerprints(
                        fingerprints[i]['image'],
                        fingerprints[j]['image']
                    )
                    comparisons.append({
                        'fingerprint1_id': i + 1,
                        'fingerprint2_id': j + 1,
                        'match_score': float(match_score),
                        'quality_score1': float(details['quality_score1']),
                        'quality_score2': float(details['quality_score2'])
                    })

            # حذف الملف المؤقت
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Warning: Could not remove temporary file: {e}")

            return jsonify({
                'comparisons': comparisons
            })

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # حذف الملف في حالة حدوث خطأ
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass
            raise e

    except Exception as e:
        logger.error(f"خطأ في تحليل البصمات المتعددة: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cross_compare', methods=['POST'])
def cross_compare_fingerprints():
    """مقارنة البصمات المتعددة بين صورتين"""
    try:
        print("Processing cross comparison request...")
        # التحقق من وجود الملفات
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'يرجى تحميل الصورتين'}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        # التحقق من اختيار الملفات
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'لم يتم اختيار الملفات'}), 400

        # التحقق من صحة امتدادات الملفات
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

        # إنشاء مجلد التحميلات إذا لم يكن موجودًا
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)

        try:
            # حفظ الملفات
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            
            filepath1 = os.path.join(upload_dir, filename1)
            filepath2 = os.path.join(upload_dir, filename2)
            
            file1.save(filepath1)
            file2.save(filepath2)

            if not (os.path.exists(filepath1) and os.path.exists(filepath2)):
                raise FileNotFoundError("فشل في حفظ الملفات المحملة")

            print("Processing images for multiple fingerprints...")
            # اكتشاف البصمات في كلا الصورتين
            fingerprints1 = fingerprint_processor.detect_multiple_fingerprints(filepath1)
            fingerprints2 = fingerprint_processor.detect_multiple_fingerprints(filepath2)

            if len(fingerprints1) == 0 or len(fingerprints2) == 0:
                raise ValueError("لم يتم العثور على بصمات في إحدى الصور أو كلتيهما")

            print(f"Found {len(fingerprints1)} fingerprints in image 1 and {len(fingerprints2)} fingerprints in image 2")
            
            # مقارنة كل البصمات من الصورة الأولى مع كل البصمات من الصورة الثانية
            comparisons = []
            for i, fp1 in enumerate(fingerprints1):
                for j, fp2 in enumerate(fingerprints2):
                    print(f"Comparing fingerprint {i+1} from image 1 with fingerprint {j+1} from image 2")
                    match_score, details = fingerprint_processor.compare_fingerprints(
                        fp1['image'],
                        fp2['image']
                    )
                    comparisons.append({
                        'fingerprint1_id': i + 1,
                        'fingerprint2_id': j + 1,
                        'match_score': float(match_score),
                        'quality_score1': float(details['quality_score1']),
                        'quality_score2': float(details['quality_score2'])
                    })

            # حذف الملفات المؤقتة
            try:
                os.remove(filepath1)
                os.remove(filepath2)
            except OSError as e:
                print(f"Warning: Could not remove temporary files: {e}")

            return jsonify({
                'comparisons': comparisons
            })

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # حذف الملفات في حالة حدوث خطأ
            for filepath in [filepath1, filepath2]:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    pass
            raise e

    except Exception as e:
        logger.error(f"خطأ في مقارنة البصمات المتعددة: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_to_database', methods=['POST'])
def upload_to_database():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fingerprint_{timestamp}_{file.filename}'
        filepath = os.path.join(FINGERPRINT_DB_DIR, filename)
        
        # Save the file
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })

@app.route('/search_fingerprint', methods=['POST'])
def search_fingerprint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file temporarily
    temp_path = 'temp_search_fingerprint.jpg'
    file.save(temp_path)
    
    # Load the search fingerprint
    search_img = cv2.imread(temp_path)
    search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    
    matches = []
    # Search through all files in the database
    for filename in os.listdir(FINGERPRINT_DB_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            db_img_path = os.path.join(FINGERPRINT_DB_DIR, filename)
            db_img = cv2.imread(db_img_path)
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(search_img, db_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # If match is found (threshold can be adjusted)
            if max_val > 0.8:
                matches.append({
                    'filename': filename,
                    'confidence': float(max_val),
                    'path': db_img_path
                })
    
    # Clean up temporary file
    os.remove(temp_path)
    
    # Sort matches by confidence
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        'matches': matches,
        'total_matches': len(matches)
    })

if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(debug=True) 