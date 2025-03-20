from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np

from utils.image_processing import preprocess_image
from utils.minutiae_extraction import extract_minutiae
from utils.matcher import match_fingerprints, visualize_matches
from utils.feature_extraction import extract_features
from utils.scoring import calculate_similarity_score, get_score_details, analyze_match_quality
from utils.report_generator import generate_report
from utils.partial_matcher import match_partial_fingerprint, visualize_partial_match
from utils.grid_matcher import match_normalized_grids, visualize_grid_match

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
        print(f"Requested minutiae count: {minutiae_count}")
        print(f"Partial matching mode: {is_partial_mode}")
        print(f"Grid matching mode: {use_grid_matching}")
        
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
        if use_grid_matching:
            # استخدام المطابقة بالمربعات المعدلة
            match_result = match_normalized_grids(processed1, processed2)
            visualizations = visualize_grid_match(processed1, processed2, match_result)
            
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
            response_data['grid_match'] = {
                'position': match_result['best_match']['position'],
                'score': float(match_result['best_match']['score']),
                'grids_visualization': url_for('static', filename=f'images/results/{timestamp}_grids_visualization.png')
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

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    app.run(debug=True) 