from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False, default='user')  # user, expert, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Fingerprint(db.Model):
    __tablename__ = 'fingerprints'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    quality_score = db.Column(db.Float)
    minutiae_count = db.Column(db.Integer)
    ridge_patterns = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

class Match(db.Model):
    __tablename__ = 'matches'
    
    id = db.Column(db.Integer, primary_key=True)
    original_fingerprint_id = db.Column(db.Integer, db.ForeignKey('fingerprints.id'))
    partial_fingerprint_id = db.Column(db.Integer, db.ForeignKey('fingerprints.id'))
    match_score = db.Column(db.Float)
    matched_points = db.Column(db.Integer)
    minutiae_analysis = db.Column(db.JSON)
    ridge_patterns = db.Column(db.JSON)
    status = db.Column(db.String(50))  # pending, reviewed, confirmed, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'))

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey('matches.id'))
    reviewer_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    decision = db.Column(db.String(50))  # match, possible_match, no_match
    confidence = db.Column(db.String(20))  # high, medium, low
    comments = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    match = db.relationship('Match', backref=db.backref('reviews', lazy=True))
    reviewer = db.relationship('User', backref=db.backref('reviews', lazy=True))

def init_db(app):
    """Initialize database with app"""
    db.init_app(app)
    with app.app_context():
        db.create_all() 