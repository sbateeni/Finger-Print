from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(128))
    role = Column(String(20), nullable=False, default='user')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    fingerprints = relationship("Fingerprint", back_populates="owner")
    reviews = relationship("Review", back_populates="reviewer")


class Fingerprint(Base):
    __tablename__ = 'fingerprints'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(255), nullable=False)
    quality_score = Column(Float)
    minutiae_count = Column(Integer)
    minutiae_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    owner = relationship("User", back_populates="fingerprints")
    original_matches = relationship("Match", foreign_keys="Match.original_fingerprint_id", back_populates="original_fingerprint")
    partial_matches = relationship("Match", foreign_keys="Match.partial_fingerprint_id", back_populates="partial_fingerprint")


class Match(Base):
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True, index=True)
    case_reference = Column(String(255))
    operator_name = Column(String(255))
    original_fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'))
    partial_fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'))
    match_score = Column(Float)
    fused_score = Column(Float)
    matched_points = Column(Integer)
    status = Column(String(50))
    match_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime)
    reviewed_by = Column(Integer, ForeignKey('users.id'))
    
    original_fingerprint = relationship("Fingerprint", foreign_keys=[original_fingerprint_id], back_populates="original_matches")
    partial_fingerprint = relationship("Fingerprint", foreign_keys=[partial_fingerprint_id], back_populates="partial_matches")
    reviews = relationship("Review", back_populates="match")


class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey('matches.id'))
    reviewer_id = Column(Integer, ForeignKey('users.id'))
    decision = Column(String(50))
    confidence = Column(String(20))
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    match = relationship("Match", back_populates="reviews")
    reviewer = relationship("User", back_populates="reviews")
