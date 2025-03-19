from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import traceback
import logging
from preprocessing.image_processing import preprocess_image, detect_ridges
from features.minutiae_extraction import extract_minutiae
from matching.matcher import match_fingerprints
from datetime import datetime
from flask import session
from database.models import db, User, Fingerprint, Match, Review
from preprocessing.image_quality import assess_image_quality

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fingerprint.db')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_fingerprints():
    try:
        if 'original' not in request.files or 'partial' not in request.files:
            return jsonify({'error': 'يرجى تحميل كلا البصمتين'}), 400

        original_file = request.files['original']
        partial_file = request.files['partial']

        if original_file.filename == '' or partial_file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        # Save files
        original_filename = secure_filename(original_file.filename)
        partial_filename = secure_filename(partial_file.filename)
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        partial_path = os.path.join(app.config['UPLOAD_FOLDER'], partial_filename)
        
        original_file.save(original_path)
        partial_file.save(partial_path)

        # Process images
        original_img = preprocess_image(original_path)
        partial_img = preprocess_image(partial_path)

        # Assess image quality
        original_quality = assess_image_quality(original_img)
        partial_quality = assess_image_quality(partial_img)

        # Extract minutiae
        original_minutiae = extract_minutiae(original_img)
        partial_minutiae = extract_minutiae(partial_img)

        # Match fingerprints
        match_result = match_fingerprints(original_minutiae, partial_minutiae)

        # Save to database
        original_fp = Fingerprint(
            filename=original_filename,
            filepath=original_path,
            quality_score=original_quality['quality_score'],
            minutiae_count=len(original_minutiae),
            ridge_patterns=match_result['details']['ridge_patterns']
        )
        db.session.add(original_fp)

        partial_fp = Fingerprint(
            filename=partial_filename,
            filepath=partial_path,
            quality_score=partial_quality['quality_score'],
            minutiae_count=len(partial_minutiae),
            ridge_patterns=match_result['details']['ridge_patterns']
        )
        db.session.add(partial_fp)
        db.session.flush()

        match = Match(
            original_fingerprint_id=original_fp.id,
            partial_fingerprint_id=partial_fp.id,
            match_score=match_result['match_score'],
            matched_points=match_result['matched_points'],
            minutiae_analysis=match_result['details']['minutiae_analysis'],
            ridge_patterns=match_result['details']['ridge_patterns'],
            status='pending'
        )
        db.session.add(match)
        db.session.commit()

        return jsonify({
            'match_id': match.id,
            'match_score': match_result['match_score'],
            'status': match_result['status']
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/review/<match_id>', methods=['GET', 'POST'])
def review_match(match_id):
    if request.method == 'GET':
        # Get match details from database
        match = Match.query.get_or_404(match_id)
        
        # Prepare data for template
        template_data = {
            'match_id': match_id,
            'original_image': match.original_fingerprint.filepath,
            'partial_image': match.partial_fingerprint.filepath,
            'original_minutiae_count': match.original_fingerprint.minutiae_count,
            'partial_minutiae_count': match.partial_fingerprint.minutiae_count,
            'original_quality_score': match.original_fingerprint.quality_score,
            'partial_quality_score': match.partial_fingerprint.quality_score,
            'match_score': match.match_score,
            'matched_points': match.matched_points,
            'match_status': match.status,
            'minutiae_analysis': match.minutiae_analysis,
            'ridge_patterns': match.ridge_patterns
        }
        
        return render_template('review.html', **template_data)
    
    elif request.method == 'POST':
        if not session.get('user_id'):
            flash('يرجى تسجيل الدخول للمراجعة', 'error')
            return redirect(url_for('login'))

        # Get review data
        review = Review(
            match_id=match_id,
            reviewer_id=session['user_id'],
            decision=request.form['review_decision'],
            confidence=request.form['confidence_level'],
            comments=request.form['comments']
        )
        db.session.add(review)

        # Update match status
        match = Match.query.get_or_404(match_id)
        match.status = request.form['review_decision']
        match.reviewed_at = datetime.utcnow()
        match.reviewed_by = session['user_id']
        
        db.session.commit()
        flash('تم حفظ المراجعة بنجاح', 'success')
        return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect(url_for('index'))
        
        flash('اسم المستخدم أو كلمة المرور غير صحيحة', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) 