from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np

from utils.image_processing import preprocess_image
from utils.minutiae_extraction import extract_minutiae
from utils.matcher import match_fingerprints
from utils.feature_extraction import extract_features
from utils.scoring import calculate_similarity_score, get_score_details, analyze_match_quality
from utils.report_generator import generate_report

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
        minutiae1 = extract_minutiae(processed1)
        minutiae2 = extract_minutiae(processed2)
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
        
        # Match fingerprints
        print("Matching fingerprints...")
        match_result = match_fingerprints(minutiae1, minutiae2, features1, features2)
        
        # Save matching visualization
        print("Generating matching visualization...")
        from utils.matcher import visualize_matches
        match_img = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
        match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
        print(f"Saving matching visualization to: {match_path}")
        cv2.imwrite(match_path, match_img)
        
        # Calculate scores and analysis
        print("Calculating scores and analysis...")
        score_details = get_score_details(match_result)
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
                'img1': len(minutiae1),
                'img2': len(minutiae2)
            },
            'matching_image': url_for('static', filename=f'images/results/{timestamp}_match_visualization.png'),
            'score': {
                'total': score_details['total_score'],
                'minutiae': score_details['minutiae_score'],
                'orientation': score_details['orientation_score'],
                'density': score_details['density_score']
            },
            'is_match': score_details['total_score'] >= MATCHING_THRESHOLD,
            'quality': {
                'level': quality_analysis['quality_level'],
                'issues': quality_analysis['issues'],
                'recommendations': quality_analysis['recommendations']
            }
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