"""
CRUD operations (Create, Read, Update, Delete) for database models
"""
from sqlalchemy.orm import Session
from database.models import User, Fingerprint, Match, Review
from typing import Optional, List
from datetime import datetime


# User operations
def create_user(db: Session, username: str, email: str, password_hash: str, role: str = "user") -> User:
    db_user = User(username=username, email=email, password_hash=password_hash, role=role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


# Fingerprint operations
def create_fingerprint(
    db: Session, 
    filename: str, 
    filepath: str, 
    quality_score: float, 
    minutiae_count: int, 
    minutiae_data: dict,
    user_id: Optional[int] = None
) -> Fingerprint:
    db_fingerprint = Fingerprint(
        filename=filename,
        filepath=filepath,
        quality_score=quality_score,
        minutiae_count=minutiae_count,
        minutiae_data=minutiae_data,
        user_id=user_id
    )
    db.add(db_fingerprint)
    db.commit()
    db.refresh(db_fingerprint)
    return db_fingerprint


def get_fingerprint(db: Session, fingerprint_id: int) -> Optional[Fingerprint]:
    return db.query(Fingerprint).filter(Fingerprint.id == fingerprint_id).first()


def get_fingerprints_by_user(db: Session, user_id: int) -> List[Fingerprint]:
    return db.query(Fingerprint).filter(Fingerprint.user_id == user_id).all()


# Match operations
def create_match(
    db: Session,
    case_reference: str,
    operator_name: str,
    original_fingerprint_id: int,
    partial_fingerprint_id: int,
    match_score: float,
    fused_score: float,
    matched_points: int,
    status: str,
    match_details: dict
) -> Match:
    db_match = Match(
        case_reference=case_reference,
        operator_name=operator_name,
        original_fingerprint_id=original_fingerprint_id,
        partial_fingerprint_id=partial_fingerprint_id,
        match_score=match_score,
        fused_score=fused_score,
        matched_points=matched_points,
        status=status,
        match_details=match_details
    )
    db.add(db_match)
    db.commit()
    db.refresh(db_match)
    return db_match


def get_match(db: Session, match_id: int) -> Optional[Match]:
    return db.query(Match).filter(Match.id == match_id).first()


def get_matches(db: Session, skip: int = 0, limit: int = 100) -> List[Match]:
    return db.query(Match).order_by(Match.created_at.desc()).offset(skip).limit(limit).all()


# Review operations
def create_review(
    db: Session,
    match_id: int,
    reviewer_id: int,
    decision: str,
    confidence: str,
    comments: Optional[str] = None
) -> Review:
    db_review = Review(
        match_id=match_id,
        reviewer_id=reviewer_id,
        decision=decision,
        confidence=confidence,
        comments=comments
    )
    db.add(db_review)
    
    # Update match's reviewed_at and reviewed_by
    db_match = get_match(db, match_id)
    if db_match:
        db_match.reviewed_at = datetime.utcnow()
        db_match.reviewed_by = reviewer_id
    
    db.commit()
    db.refresh(db_review)
    return db_review


# Phase 3: Manual Review operations
def update_fingerprint_minutiae(db: Session, fingerprint_id: int, minutiae_data: dict) -> Optional[Fingerprint]:
    from sqlalchemy.orm.attributes import flag_modified
    db_fingerprint = get_fingerprint(db, fingerprint_id)
    if db_fingerprint:
        db_fingerprint.minutiae_data = minutiae_data
        flag_modified(db_fingerprint, 'minutiae_data')
        db_fingerprint.minutiae_count = len(minutiae_data.get('minutiae', [])) if isinstance(minutiae_data, dict) else 0
        db.commit()
    return db_fingerprint


def update_fingerprint_landmarks(db: Session, fingerprint_id: int, landmarks: dict) -> Optional[Fingerprint]:
    db_fingerprint = get_fingerprint(db, fingerprint_id)
    if db_fingerprint:
        db_fingerprint.landmarks = landmarks
        db.commit()
        db.refresh(db_fingerprint)
    return db_fingerprint


def update_fingerprint_manual_review(
    db: Session, 
    fingerprint_id: int, 
    is_reviewed: bool, 
    reviewer_id: int, 
    notes: str = ""
) -> Optional[Fingerprint]:
    db_fingerprint = get_fingerprint(db, fingerprint_id)
    if db_fingerprint:
        db_fingerprint.is_manually_reviewed = 1 if is_reviewed else 0
        db_fingerprint.manual_review_timestamp = datetime.utcnow()
        db_fingerprint.manual_review_by = reviewer_id
        db_fingerprint.manual_review_notes = notes
        db.commit()
        db.refresh(db_fingerprint)
    return db_fingerprint
