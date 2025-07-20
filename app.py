from flask import Flask, request, render_template, jsonify, send_file, url_for, redirect
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
import zipfile

from utils.image_processing import preprocess_image
from utils.minutiae_extraction import extract_minutiae
from utils.matcher import match_fingerprints, visualize_matches
from utils.feature_extraction import extract_features, estimate_ridge_frequency
from utils.scoring import calculate_similarity_score, get_score_details, analyze_match_quality
from utils.report_generator import generate_report
from utils.partial_matcher import match_partial_fingerprint, visualize_partial_match
from utils.grid_matcher import match_normalized_grids, visualize_grid_match, normalize_ridge_distance, divide_into_grids, calculate_grid_match_score

from config import *

app = Flask(__name__)

# تكوين المسارات
UPLOAD_FOLDER = os.path.join('static', 'images', 'uploaded')
PROCESSED_FOLDER = os.path.join('static', 'images', 'processed')
RESULTS_FOLDER = os.path.join('static', 'images', 'results')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# التأكد من وجود المجلدات
for directory in [UPLOAD_FOLDER, PROCESSED_FOLDER, RESULTS_FOLDER]:
    os.makedirs(directory, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('grid_cutter.html')

@app.route('/match')
def match():
    return render_template('match.html')

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
                'density_score': match_result['best_match']['score'],
                'matched_count': len(minutiae1),
                'confidence': 'High' if match_result['best_match']['score'] >= 80 else 'Medium' if match_result['best_match']['score'] >= 60 else 'Low'
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

@app.route('/grid_result/<timestamp>')
def grid_result(timestamp):
    try:
        # استرجاع معلومات النتائج من الملفات المحفوظة
        original_image = url_for('static', filename=f'images/uploaded/{timestamp}_original.png')
        visualization = url_for('static', filename=f'images/results/{timestamp}_visualization.png')
        
        # قراءة معلومات المربعات
        grids = []
        for row in range(3):  # 3x3 grid
            for col in range(3):
                grid_url = url_for('static', filename=f'images/results/{timestamp}_grid_{row}_{col}.png')
                grids.append({
                    'image_url': grid_url,
                    'position': {'row': row, 'col': col}
                })
        
        # إنشاء رابط تحميل الملف المضغوط
        download_url = url_for('static', filename=f'images/results/{timestamp}_grids.zip')
        
        return render_template('grid_result.html',
                             original_image=original_image,
                             visualization=visualization,
                             grids=grids,
                             grid_info={'rows': 3, 'cols': 3},
                             download_url=download_url)
                             
    except Exception as e:
        return redirect(url_for('grid_cutter'))

@app.route('/process_grids', methods=['POST'])
def process_grids():
    try:
        if 'fullFingerprint' not in request.files:
            return jsonify({'error': 'لم يتم تحديد ملف'}), 400
            
        file = request.files['fullFingerprint']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

        # إنشاء طابع زمني فريد
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # حفظ الصورة الأصلية
        original_filename = f'{timestamp}_original.png'
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(original_path)
        
        # معالجة الصورة
        img = cv2.imread(original_path)
        height, width = img.shape[:2]
        
        # حساب حجم المربع
        grid_size = min(height, width) // 3
        
        # إنشاء صورة توضيحية
        visualization = img.copy()
        
        # تقطيع الصورة إلى مربعات وحفظها
        grids = []
        for row in range(3):
            for col in range(3):
                # حساب إحداثيات المربع
                y = row * grid_size
                x = col * grid_size
                
                # قص المربع
                grid = img[y:y+grid_size, x:x+grid_size]
                
                # حفظ المربع
                grid_filename = f'{timestamp}_grid_{row}_{col}.png'
                grid_path = os.path.join(RESULTS_FOLDER, grid_filename)
                cv2.imwrite(grid_path, grid)
                
                # رسم مستطيل حول المربع في الصورة التوضيحية
                cv2.rectangle(visualization, (x, y), (x+grid_size, y+grid_size), (0, 255, 0), 2)
                # إضافة رقم المربع
                cv2.putText(visualization, f'{row},{col}', (x+10, y+30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # حفظ الصورة التوضيحية
        vis_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_visualization.png')
        cv2.imwrite(vis_path, visualization)

        # إنشاء ملف مضغوط يحتوي على جميع المربعات
        zip_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_grids.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for row in range(3):
                for col in range(3):
                    grid_filename = f'{timestamp}_grid_{row}_{col}.png'
                    grid_path = os.path.join(RESULTS_FOLDER, grid_filename)
                    zipf.write(grid_path, grid_filename)

        # إعداد قائمة المربعات
        grids = [
            {
                'image_url': url_for('static', filename=f'images/results/{timestamp}_grid_{row}_{col}.png'),
                'position': {'row': row, 'col': col}
            }
            for row in range(3) for col in range(3)
        ]

        # إرجاع جميع البيانات المطلوبة
        return jsonify({
            'grids': grids,
            'grid_info': {'rows': 3, 'cols': 3},
            'visualization': url_for('static', filename=f'images/results/{timestamp}_visualization.png'),
            'original': url_for('static', filename=f'images/uploaded/{timestamp}_original.png'),
            'result_url': url_for('grid_result', timestamp=timestamp)
        })
        
    except Exception as e:
        print(f"خطأ: {str(e)}")
        import traceback
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
            grid_match_result = match_normalized_grids(processed2, processed1)
            
            # تحويل النتيجة إلى الصيغة المطلوبة
            match_result = {
                'matched_minutiae': [],  # لا نحتفظ بالنقاط المتطابقة في هذا الوضع
                'minutiae_score': grid_match_result['best_match']['score'],
                'orientation_score': grid_match_result['best_match']['score'],
                'density_score': grid_match_result['best_match']['score'],
                'grid_match': grid_match_result
            }
            
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
            
            # تحويل النتيجة إلى الصيغة المطلوبة
            match_result = {
                'matched_minutiae': [],  # لا نحتفظ بالنقاط المتطابقة في هذا الوضع
                'minutiae_score': best_match['score'],
                'orientation_score': best_match['score'],
                'density_score': best_match['score'],
                'grid_match': {
                    'best_match': best_match,
                    'grid_size': grid_size
                }
            }
            
        else:
            print("استخدام المطابقة العادية...")
            match_result = match_fingerprints(
                extract_minutiae(processed1),
                extract_minutiae(processed2),
                features1,
                features2
            )
        
        # تحليل النتائج وإنشاء التقرير المرئي
        quality_analysis = analyze_match_quality(match_result)
        
        # حفظ صورة المطابقة
        if matching_mode in ['normalized', 'grid_cut']:
            # إنشاء صورة توضيحية للمطابقة بالمربعات
            visualization = cv2.cvtColor(processed1.copy(), cv2.COLOR_GRAY2BGR)
            best_pos = match_result['grid_match']['best_match']['position']
            grid_size = match_result['grid_match']['grid_size']
            
            # رسم المربع الأفضل تطابقاً
            cv2.rectangle(visualization,
                        best_pos,
                        (best_pos[0] + grid_size, best_pos[1] + grid_size),
                        (0, 255, 0), 2)
            
            match_img = visualization
        else:
            # استخدام التصور العادي للمطابقة
            match_img = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
        
        # حفظ الصورة
        match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
        cv2.imwrite(match_path, match_img)
        
        # إعداد تفاصيل النتيجة
        score_details = get_score_details(match_result)
        
        return jsonify({
            'timestamp': timestamp,
            'scores': {
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
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    app.run(debug=True) 