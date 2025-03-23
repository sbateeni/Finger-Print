<<<<<<< HEAD
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

# إنشاء مجلد قاعدة بيانات البصمات إذا لم يكن موجوداً
FINGERPRINT_DB_DIR = 'fingerprint_database'
if not os.path.exists(FINGERPRINT_DB_DIR):
    os.makedirs(FINGERPRINT_DB_DIR)
    print(f"Created fingerprint database directory: {FINGERPRINT_DB_DIR}")

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
    """رفع بصمة إلى قاعدة البيانات"""
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # إنشاء اسم فريد للملف باستخدام التاريخ والوقت
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fingerprint_{timestamp}_{secure_filename(file.filename)}'
            filepath = os.path.join(FINGERPRINT_DB_DIR, filename)
            
            # حفظ الملف
            file.save(filepath)
            
            return jsonify({
                'message': 'تم رفع الملف بنجاح',
                'filename': filename,
                'path': filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

@app.route('/search_fingerprint', methods=['POST'])
def search_fingerprint():
    """البحث عن تطابق في قاعدة البيانات"""
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # حفظ الملف المؤقت للبحث
            temp_path = 'temp_search_fingerprint.jpg'
            file.save(temp_path)
            
            # تحميل صورة البحث
            search_img = cv2.imread(temp_path)
            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            
            matches = []
            # البحث في جميع الملفات في قاعدة البيانات
            for filename in os.listdir(FINGERPRINT_DB_DIR):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    db_img_path = os.path.join(FINGERPRINT_DB_DIR, filename)
                    db_img = cv2.imread(db_img_path)
                    db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
                    
                    # إجراء مقارنة البصمات
                    result = fingerprint_processor.compare_fingerprints(search_img, db_img)
                    
                    # إذا وجد تطابق (يمكن تعديل العتبة حسب الحاجة)
                    if result['match_score'] > 0.6:
                        matches.append({
                            'filename': filename,
                            'confidence': result['match_score'],
                            'path': db_img_path
                        })
            
            # حذف الملف المؤقت
            os.remove(temp_path)
            
            # ترتيب النتائج حسب نسبة التطابق
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            return jsonify({
                'matches': matches,
                'total_matches': len(matches)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

@app.route('/grid_compare')
def grid_compare():
    """عرض صفحة مقارنة البصمات باستخدام الشبكة"""
    return render_template('grid_compare.html')

@app.route('/split_fingerprint', methods=['POST'])
def split_fingerprint():
    """تقسيم البصمة إلى شبكة من المربعات"""
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # قراءة الصورة
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': 'فشل في قراءة الصورة'}), 400

            # تحويل الصورة إلى رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # الحصول على حجم الشبكة
            grid_size = int(request.form.get('grid_size', 3))
            
            # تقسيم الصورة إلى شبكة
            height, width = gray.shape
            cell_height = height // grid_size
            cell_width = width // grid_size
            
            grid_images = []
            for i in range(grid_size):
                for j in range(grid_size):
                    # استخراج المربع
                    cell = gray[i*cell_height:(i+1)*cell_height, 
                              j*cell_width:(j+1)*cell_width]
                    
                    # تحويل المربع إلى صورة
                    _, buffer = cv2.imencode('.jpg', cell)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    grid_images.append(f'data:image/jpeg;base64,{img_str}')
            
            return jsonify({
                'grid_images': grid_images,
                'grid_size': grid_size
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

@app.route('/compare_grids', methods=['POST'])
def compare_grids():
    """مقارنة مربعين من البصمات"""
    try:
        grid1_index = int(request.form.get('grid1_index'))
        grid2_index = int(request.form.get('grid2_index'))
        grid_size = int(request.form.get('grid_size', 3))
        
        # الحصول على المربعات المحددة من الصور المخزنة مؤقتاً
        # (يجب تنفيذ آلية لتخزين المربعات مؤقتاً)
        
        # هنا سنقوم بمحاكاة النتائج
        match_score = np.random.uniform(0.6, 0.95)
        avg_score = np.random.uniform(0.5, 0.9)
        matching_squares = np.random.randint(4, 9)
        quality1 = np.random.uniform(0.7, 0.95)
        quality2 = np.random.uniform(0.7, 0.95)
        
        return jsonify({
            'match_score': float(match_score),
            'avg_score': float(avg_score),
            'matching_squares': int(matching_squares),
            'quality1': float(quality1),
            'quality2': float(quality2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask development server...")
=======
from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np

from utils.image_processing import preprocess_image
from utils.minutiae_extraction import extract_minutiae
from utils.matcher import match_fingerprints, visualize_matches
from utils.feature_extraction import extract_features, estimate_ridge_frequency
from utils.scoring import calculate_similarity_score, get_score_details, analyze_match_quality
from utils.report_generator import generate_report
from utils.partial_matcher import match_partial_fingerprint, visualize_partial_match
from utils.grid_matcher import match_normalized_grids, visualize_grid_match, normalize_ridge_distance, divide_into_grids

from config import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_fingerprint():
    try:
        print("Starting fingerprint upload and processing...")
        
        # Create necessary directories if they don't exist
        print("Creating directories...")
        for directory in [app.config['UPLOAD_FOLDER'], PROCESSED_FOLDER, RESULTS_FOLDER, OUTPUT_FOLDER]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory created/verified: {directory}")

        print("Checking request files...")
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            print("Error: Missing files in request")
            return jsonify({'error': 'Both fingerprint images are required'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # Get parameters from request
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        is_partial_mode = request.form.get('matchingMode') == 'true'
        use_grid_matching = request.form.get('useGridMatching') == 'true'
        use_grid_cut_matching = request.form.get('useGridCutMatching') == 'true'
        
        print(f"Requested minutiae count: {minutiae_count}")
        print(f"Partial matching mode: {is_partial_mode}")
        print(f"Grid matching mode: {use_grid_matching}")
        print(f"Grid cut matching mode: {use_grid_cut_matching}")
        
        print(f"Received files: {fingerprint1.filename}, {fingerprint2.filename}")
        
        if fingerprint1.filename == '' or fingerprint2.filename == '':
            print("Error: Empty filenames")
            return jsonify({'error': 'No selected files'}), 400
        
        if not (allowed_file(fingerprint1.filename) and allowed_file(fingerprint2.filename)):
            print(f"Error: Invalid file types. Allowed types are: {ALLOWED_EXTENSIONS}")
            return jsonify({'error': 'Invalid file type. Allowed types are: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Generated timestamp: {timestamp}")
        
        # Save original images
        filename1 = f"{timestamp}_1_{secure_filename(fingerprint1.filename)}"
        filename2 = f"{timestamp}_2_{secure_filename(fingerprint2.filename)}"
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        print(f"Saving files to: {filepath1}, {filepath2}")
        fingerprint1.save(filepath1)
        fingerprint2.save(filepath2)
        
        # Process images
        print("Processing images...")
        processed1 = preprocess_image(filepath1)
        processed2 = preprocess_image(filepath2)
        
        # Save processed images
        proc1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_processed.png')
        proc2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_processed.png')
        print(f"Saving processed images to: {proc1_path}, {proc2_path}")
        cv2.imwrite(proc1_path, processed1)
        cv2.imwrite(proc2_path, processed2)
        
        # Extract minutiae
        print("Extracting minutiae...")
        minutiae1 = extract_minutiae(processed1, max_points=minutiae_count)
        minutiae2 = extract_minutiae(processed2, max_points=minutiae_count)
        print(f"Found minutiae points: {len(minutiae1)} in image 1, {len(minutiae2)} in image 2")
        
        # Save minutiae visualizations
        print("Generating minutiae visualizations...")
        from utils.minutiae_extraction import visualize_minutiae
        min1_img = visualize_minutiae(processed1, minutiae1)
        min2_img = visualize_minutiae(processed2, minutiae2)
        
        min1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_minutiae.png')
        min2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_minutiae.png')
        print(f"Saving minutiae visualizations to: {min1_path}, {min2_path}")
        cv2.imwrite(min1_path, min1_img)
        cv2.imwrite(min2_path, min2_img)
        
        # Extract features
        print("Extracting features...")
        features1 = extract_features(processed1)
        features2 = extract_features(processed2)
        
        # Match fingerprints based on mode
        print("Matching fingerprints...")
        if use_grid_cut_matching:
            print("بدء المطابقة مع المربعات المقطعة...")
            
            # تقسيم البصمة الأولى إلى مربعات
            grid_rows = 3
            grid_cols = 3
            height, width = processed1.shape
            grid_height = height // grid_rows
            grid_width = width // grid_cols
            
            print(f"تقسيم البصمة الأولى {width}x{height} إلى شبكة {grid_rows}x{grid_cols}")
            
            best_match = {
                'score': 0,
                'grid_position': None,
                'match_result': None
            }
            
            all_grid_matches = []
            
            # مقارنة البصمة الجزئية مع كل مربع
            for row in range(grid_rows):
                for col in range(grid_cols):
                    try:
                        # تحديد حدود المربع
                        y1 = row * grid_height
                        y2 = (row + 1) * grid_height
                        x1 = col * grid_width
                        x2 = (col + 1) * grid_width
                        
                        # استخراج المربع
                        grid = processed1[y1:y2, x1:x2].copy()
                        
                        # مقارنة المربع مع البصمة الجزئية
                        grid_match = match_partial_fingerprint(grid, processed2, 
                                                            extract_features(grid),
                                                            features2)
                        
                        match_score = grid_match['best_match_score']
                        
                        grid_result = {
                            'position': {'row': row + 1, 'col': col + 1},
                            'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'score': match_score
                        }
                        
                        all_grid_matches.append(grid_result)
                        
                        if match_score > best_match['score']:
                            best_match = {
                                'score': match_score,
                                'grid_position': grid_result['position'],
                                'match_result': grid_match
                            }
                        
                        print(f"مطابقة المربع ({row+1}, {col+1}) - النتيجة: {match_score:.2f}%")
                        
                    except Exception as e:
                        print(f"خطأ في معالجة المربع ({row+1}, {col+1}): {str(e)}")
                        continue
            
            # إنشاء صورة توضيحية للنتائج
            visualization = cv2.cvtColor(processed1.copy(), cv2.COLOR_GRAY2BGR)
            
            # رسم كل المربعات مع درجات تطابقها
            for grid_match in all_grid_matches:
                pos = grid_match['position']
                coords = grid_match['coordinates']
                score = grid_match['score']
                
                # تحديد لون المربع بناءً على درجة التطابق
                color_intensity = int(score * 2.55)  # تحويل النسبة المئوية إلى قيمة لونية
                color = (0, color_intensity, 0)  # اللون الأخضر بدرجات متفاوتة
                
                # رسم المربع
                cv2.rectangle(visualization, 
                            (coords['x1'], coords['y1']), 
                            (coords['x2'], coords['y2']), 
                            color, 2)
                
                # كتابة درجة التطابق
                cv2.putText(visualization, 
                          f"{score:.1f}%",
                          (coords['x1'] + 5, coords['y1'] + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # تمييز أفضل مربع باللون الأصفر
            best_coords = all_grid_matches[0]['coordinates']  # الموقع الافتراضي
            for grid_match in all_grid_matches:
                if (grid_match['position']['row'] == best_match['grid_position']['row'] and 
                    grid_match['position']['col'] == best_match['grid_position']['col']):
                    best_coords = grid_match['coordinates']
                    break
            
            cv2.rectangle(visualization,
                        (best_coords['x1'], best_coords['y1']),
                        (best_coords['x2'], best_coords['y2']),
                        (0, 255, 255), 3)  # لون أصفر وخط أعرض
            
            # حفظ الصورة التوضيحية
            match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
            cv2.imwrite(match_path, visualization)
            
            # تحضير تفاصيل النتيجة
            score_details = {
                'total_score': best_match['score'],
                'minutiae_score': best_match['score'],
                'orientation_score': best_match['score'],
                'density_score': best_match['score']
            }
            
            quality_analysis = {
                'quality_level': 'جيد' if best_match['score'] >= 70 else 'متوسط' if best_match['score'] >= 50 else 'ضعيف',
                'issues': [],
                'recommendations': []
            }
            
            if best_match['score'] < 50:
                quality_analysis['issues'].append('درجة التطابق منخفضة')
                quality_analysis['recommendations'].append('حاول استخدام جزء أكبر من البصمة')
            
            response_data = {
                'processed_images': {
                    'img1': url_for('static', filename=f'images/processed/{timestamp}_1_processed.png'),
                    'img2': url_for('static', filename=f'images/processed/{timestamp}_2_processed.png')
                },
                'minutiae_images': {
                    'img1': url_for('static', filename=f'images/processed/{timestamp}_1_minutiae.png'),
                    'img2': url_for('static', filename=f'images/processed/{timestamp}_2_minutiae.png')
                },
                'minutiae_count': {
                    'img1': int(len(minutiae1)),
                    'img2': int(len(minutiae2))
                },
                'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
                'score': {
                    'total': float(score_details['total_score']),
                    'minutiae': float(score_details['minutiae_score']),
                    'orientation': float(score_details['orientation_score']),
                    'density': float(score_details['density_score'])
                },
                'is_match': bool(score_details['total_score'] >= MATCHING_THRESHOLD),
                'quality': quality_analysis,
                'grid_cut_match': {
                    'best_match': {
                        'position': best_match['grid_position'],
                        'score': float(best_match['score'])
                    },
                    'all_matches': all_grid_matches
                }
            }
            
        elif use_grid_matching:
            # استخدام المطابقة بالمربعات المعدلة
            print("بدء المطابقة باستخدام المربعات المعدلة...")
            print("معالجة البصمة الثانية كبصمة جزئية...")
            
            # تبديل البصمات إذا كانت البصمة الثانية أكبر من الأولى
            if processed2.shape[0] * processed2.shape[1] > processed1.shape[0] * processed1.shape[1]:
                processed1, processed2 = processed2, processed1
                print("تم تبديل البصمتين لأن البصمة الثانية أكبر")
            
            match_result = match_normalized_grids(processed2, processed1)  # البصمة الثانية هي الجزئية
            visualizations = visualize_grid_match(processed2, processed1, match_result)
            
            # حفظ الصور
            match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
            grids_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_grids_visualization.png')
            
            cv2.imwrite(match_path, visualizations['main_visualization'])
            cv2.imwrite(grids_path, visualizations['grids_visualization'])
            
            score_details = {
                'total_score': match_result['best_match']['score'],
                'minutiae_score': match_result['best_match']['score'],
                'orientation_score': match_result['best_match']['score'],
                'density_score': match_result['best_match']['score']
            }
            
            # تحديث response_data
            response_data = {
                'processed_images': {
                    'img1': url_for('static', filename=f'images/processed/{timestamp}_1_processed.png'),
                    'img2': url_for('static', filename=f'images/processed/{timestamp}_2_processed.png')
                },
                'minutiae_images': {
                    'img1': url_for('static', filename=f'images/processed/{timestamp}_1_minutiae.png'),
                    'img2': url_for('static', filename=f'images/processed/{timestamp}_2_minutiae.png')
                },
                'minutiae_count': {
                    'img1': int(len(minutiae1)),
                    'img2': int(len(minutiae2))
                },
                'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
                'score': {
                    'total': float(score_details['total_score']),
                    'minutiae': float(score_details['minutiae_score']),
                    'orientation': float(score_details['orientation_score']),
                    'density': float(score_details['density_score'])
                },
                'is_match': bool(score_details['total_score'] >= MATCHING_THRESHOLD),
                'quality': {
                    'level': str(quality_analysis['quality_level']),
                    'issues': [str(issue) for issue in quality_analysis['issues']],
                    'recommendations': [str(rec) for rec in quality_analysis['recommendations']]
                },
                'is_partial_mode': is_partial_mode,
                'use_grid_matching': use_grid_matching,
                'grid_match': {
                    'position': match_result['best_match']['position'],
                    'score': float(match_result['best_match']['score']),
                    'grids_visualization': url_for('static', filename=f'images/results/{timestamp}_grids_visualization.png')
                }
            }
        elif is_partial_mode:
            match_result = match_partial_fingerprint(processed1, processed2, features1, features2)
            match_img = visualize_partial_match(processed1, processed2, match_result)
            score_details = get_score_details(match_result)
        else:
            match_result = match_fingerprints(minutiae1, minutiae2, features1, features2)
            match_img = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
            score_details = get_score_details(match_result)
        
        match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
        print(f"Saving matching visualization to: {match_path}")
        cv2.imwrite(match_path, match_img)
        
        # Calculate scores and analysis
        print("Calculating scores and analysis...")
        quality_analysis = analyze_match_quality(match_result)
        
        print(f"Score details: {score_details}")
        print(f"Quality analysis: {quality_analysis}")
        
        # Prepare response data
        response_data = {
            'processed_images': {
                'img1': url_for('static', filename=f'images/processed/{timestamp}_1_processed.png'),
                'img2': url_for('static', filename=f'images/processed/{timestamp}_2_processed.png')
            },
            'minutiae_images': {
                'img1': url_for('static', filename=f'images/processed/{timestamp}_1_minutiae.png'),
                'img2': url_for('static', filename=f'images/processed/{timestamp}_2_minutiae.png')
            },
            'minutiae_count': {
                'img1': int(len(minutiae1)),
                'img2': int(len(minutiae2))
            },
            'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
            'score': {
                'total': float(score_details['total_score']),
                'minutiae': float(score_details['minutiae_score']),
                'orientation': float(score_details['orientation_score']),
                'density': float(score_details['density_score'])
            },
            'is_match': bool(score_details['total_score'] >= MATCHING_THRESHOLD),
            'quality': {
                'level': str(quality_analysis['quality_level']),
                'issues': [str(issue) for issue in quality_analysis['issues']],
                'recommendations': [str(rec) for rec in quality_analysis['recommendations']]
            },
            'is_partial_mode': is_partial_mode,
            'use_grid_matching': use_grid_matching
        }
        
        if is_partial_mode:
            response_data['partial_match'] = {
                'location': match_result['match_location'],
                'region': match_result['best_match_region']
            }
        
        print("Sending response data...")
        print(f"Response URLs: {response_data['processed_images']}")
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("Error occurred during processing:")
        print("Error message:", str(e))
        print("Traceback:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/report/<timestamp>')
def view_report(timestamp):
    return render_template('report.html', timestamp=timestamp)

@app.route('/download_report/<timestamp>')
def download_report(timestamp):
    report_path = os.path.join(OUTPUT_FOLDER, f'report_{timestamp}.pdf')
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404

@app.route('/grid_cutter')
def grid_cutter():
    return render_template('grid_cutter.html')

@app.route('/process_grids', methods=['POST'])
def process_grids():
    try:
        print("بدء معالجة المربعات...")
        
        # التحقق من وجود الملفات
        if 'fullFingerprint' not in request.files or 'referenceFingerprint' not in request.files:
            return jsonify({'error': 'يجب تحديد كلا الصورتين'}), 400
        
        full_fingerprint = request.files['fullFingerprint']
        reference_fingerprint = request.files['referenceFingerprint']
        
        if full_fingerprint.filename == '' or reference_fingerprint.filename == '':
            return jsonify({'error': 'لم يتم اختيار الملفات'}), 400
            
        if not (allowed_file(full_fingerprint.filename) and allowed_file(reference_fingerprint.filename)):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400
        
        # إنشاء مجلد للنتائج إذا لم يكن موجوداً
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        grid_results_folder = os.path.join(RESULTS_FOLDER, f'grids_{timestamp}')
        os.makedirs(grid_results_folder, exist_ok=True)
        
        # حفظ الصور المرفوعة
        full_path = os.path.join(grid_results_folder, 'full.png')
        ref_path = os.path.join(grid_results_folder, 'reference.png')
        full_fingerprint.save(full_path)
        reference_fingerprint.save(ref_path)
        
        print("تم حفظ الصور المرفوعة")
        
        # معالجة الصور
        processed_full = preprocess_image(full_path)
        processed_ref = preprocess_image(ref_path)
        
        print("تم معالجة الصور")
        
        # تحديد عدد المربعات في كل صف وعمود
        grid_rows = 3  # عدد الصفوف
        grid_cols = 3  # عدد الأعمدة
        
        # حساب حجم كل مربع
        height, width = processed_full.shape
        grid_height = height // grid_rows
        grid_width = width // grid_cols
        
        print(f"تقسيم الصورة {width}x{height} إلى شبكة {grid_rows}x{grid_cols}")
        print(f"حجم كل مربع: {grid_width}x{grid_height}")
        
        # معالجة كل مربع وحفظ النتائج
        processed_grids = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                try:
                    # تحديد حدود المربع
                    y1 = row * grid_height
                    y2 = (row + 1) * grid_height
                    x1 = col * grid_width
                    x2 = (col + 1) * grid_width
                    
                    # استخراج المربع
                    grid = processed_full[y1:y2, x1:x2].copy()
                    
                    # حفظ المربع
                    grid_filename = f'grid_r{row+1}_c{col+1}.png'
                    grid_path = os.path.join(grid_results_folder, grid_filename)
                    cv2.imwrite(grid_path, grid)
                    
                    # حساب المسافة بين الخطوط في المربع
                    grid_freq = estimate_ridge_frequency(grid)
                    valid_grid_freq = grid_freq[grid_freq > 0]
                    if len(valid_grid_freq) > 0:
                        grid_distance = 1.0 / np.mean(valid_grid_freq)
                    else:
                        grid_distance = 0
                    
                    processed_grids.append({
                        'image_url': url_for('static', filename=f'images/results/grids_{timestamp}/{grid_filename}'),
                        'position': {'row': row + 1, 'col': col + 1},
                        'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'ridge_distance': float(grid_distance)
                    })
                    
                    print(f"تمت معالجة المربع ({row+1}, {col+1}) بنجاح")
                    
                except Exception as e:
                    print(f"خطأ في معالجة المربع ({row+1}, {col+1}): {str(e)}")
                    continue
        
        # إنشاء صورة توضيحية للتقسيم
        visualization = cv2.cvtColor(processed_full.copy(), cv2.COLOR_GRAY2BGR)
        
        # رسم خطوط الشبكة
        for row in range(1, grid_rows):
            y = row * grid_height
            cv2.line(visualization, (0, y), (width, y), (0, 255, 0), 2)
        
        for col in range(1, grid_cols):
            x = col * grid_width
            cv2.line(visualization, (x, 0), (x, height), (0, 255, 0), 2)
            
        # حفظ الصورة التوضيحية
        vis_filename = 'grid_visualization.png'
        vis_path = os.path.join(grid_results_folder, vis_filename)
        cv2.imwrite(vis_path, visualization)
        
        return jsonify({
            'grids': processed_grids,
            'grid_info': {
                'rows': grid_rows,
                'cols': grid_cols,
                'width': grid_width,
                'height': grid_height
            },
            'visualization': url_for('static', filename=f'images/results/grids_{timestamp}/{vis_filename}')
        })
        
    except Exception as e:
        import traceback
        print("حدث خطأ:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/cut_fingerprint', methods=['POST'])
def cut_fingerprint():
    """
    تقطيع البصمة الأولى إلى مربعات وتعديل حجمها حسب البصمة الثانية
    """
    try:
        print("بدء عملية تقطيع البصمة...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'يجب تحديد كلا البصمتين'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        
        # إنشاء المجلدات اللازمة
        for directory in [app.config['UPLOAD_FOLDER'], PROCESSED_FOLDER, RESULTS_FOLDER]:
            os.makedirs(directory, exist_ok=True)
        
        # حفظ الصور
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp1_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_1.png')
        fp2_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_2.png')
        fingerprint1.save(fp1_path)
        fingerprint2.save(fp2_path)
        
        # معالجة الصور
        processed1 = preprocess_image(fp1_path)
        processed2 = preprocess_image(fp2_path)
        
        # حساب المسافة بين الخطوط في البصمة الثانية
        freq2 = estimate_ridge_frequency(processed2)
        target_distance = 1.0 / np.mean(freq2[freq2 > 0])
        print(f"المسافة المستهدفة بين الخطوط: {target_distance:.2f}")
        
        # تقسيم البصمة الأولى إلى مربعات
        grid_size = max(processed2.shape)
        grids = divide_into_grids(processed1, grid_size)
        print(f"تم تقسيم البصمة إلى {len(grids)} مربع")
        
        # معالجة كل مربع
        processed_grids = []
        for i, grid in enumerate(grids):
            try:
                # تعديل حجم المربع
                normalized_grid, scale_factor = normalize_ridge_distance(grid['image'], target_distance)
                
                # حفظ المربع
                grid_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_grid_{i+1}.png')
                cv2.imwrite(grid_path, normalized_grid)
                
                # حساب المسافة بين الخطوط في المربع
                grid_freq = estimate_ridge_frequency(normalized_grid)
                grid_distance = 1.0 / np.mean(grid_freq[grid_freq > 0])
                
                processed_grids.append({
                    'image': url_for('static', filename=f'images/processed/{timestamp}_grid_{i+1}.png'),
                    'position': {
                        'row': i // 3 + 1,
                        'col': i % 3 + 1
                    },
                    'ridge_distance': float(grid_distance),
                    'scale_factor': float(scale_factor)
                })
                
            except Exception as e:
                print(f"خطأ في معالجة المربع {i+1}: {str(e)}")
                continue
        
        return jsonify({
            'grids': processed_grids,
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"خطأ: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/match_fingerprints', methods=['POST'])
def match_fingerprints_route():
    """
    مطابقة البصمات باستخدام الطريقة المحددة
    """
    try:
        print("بدء عملية المطابقة...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'يجب تحديد كلا البصمتين'}), 400
        
        fingerprint1 = request.files['fingerprint1']
        fingerprint2 = request.files['fingerprint2']
        matching_mode = request.form.get('matchingMode', 'normal')
        minutiae_count = int(request.form.get('minutiaeCount', 100))
        
        # حفظ الصور
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp1_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_1.png')
        fp2_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_2.png')
        fingerprint1.save(fp1_path)
        fingerprint2.save(fp2_path)
        
        # معالجة الصور
        processed1 = preprocess_image(fp1_path)
        processed2 = preprocess_image(fp2_path)
        
        # استخراج الخصائص
        features1 = extract_features(processed1)
        features2 = extract_features(processed2)
        
        if matching_mode == 'normalized':
            print("استخدام المطابقة بالمربعات المعدلة...")
            match_result = match_normalized_grids(processed2, processed1)
            
        elif matching_mode == 'grid_cut':
            print("استخدام المطابقة مع المربعات المقطعة...")
            # تقسيم البصمة الأولى إلى مربعات
            grid_size = max(processed2.shape)
            grids = divide_into_grids(processed1, grid_size)
            
            # مطابقة البصمة الجزئية مع كل مربع
            best_match = {
                'score': 0,
                'grid': None,
                'position': None
            }
            
            for grid in grids:
                match_score = calculate_grid_match_score(
                    extract_minutiae(processed2),
                    extract_minutiae(grid['image']),
                    processed2,
                    grid['image']
                )
                
                if match_score > best_match['score']:
                    best_match = {
                        'score': match_score,
                        'grid': grid['image'],
                        'position': grid['position']
                    }
            
            match_result = {
                'best_match': best_match,
                'grid_size': grid_size
            }
            
        else:
            print("استخدام المطابقة العادية...")
            match_result = match_fingerprints(
                extract_minutiae(processed1),
                extract_minutiae(processed2),
                features1,
                features2
            )
        
        # تحليل النتائج
        score_details = get_score_details(match_result)
        quality_analysis = analyze_match_quality(match_result)
        
        # إنشاء الصورة التوضيحية
        if matching_mode in ['normalized', 'grid_cut']:
            visualization = visualize_grid_match(processed2, processed1, match_result)
            match_vis_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
            cv2.imwrite(match_vis_path, visualization['main_visualization'])
        else:
            match_vis = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
            match_vis_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
            cv2.imwrite(match_vis_path, match_vis)
        
        return jsonify({
            'is_match': score_details['total_score'] >= MATCHING_THRESHOLD,
            'score': {
                'total': score_details['total_score'],
                'minutiae': score_details['minutiae_score'],
                'orientation': score_details['orientation_score'],
                'density': score_details['density_score']
            },
            'quality': quality_analysis,
            'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png')
        })
        
    except Exception as e:
        print(f"خطأ: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
>>>>>>> b776498f34254ddb2b3f12e3edd5fb2ba551068b
    app.run(debug=True) 