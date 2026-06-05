"""
Manual Editor Routes - Phase 3

Provides endpoints for:
- Loading fingerprints for editing
- Saving manual annotations
- Approving/rejecting fingerprints
"""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from pydantic import BaseModel
from typing import Any
from datetime import datetime

from services.manual_annotation_service import (
    delete_minutia,
    add_minutia,
    update_landmark_type,
    AnnotationAction,
)
from sqlalchemy.orm import Session
from database import get_db
from database.models import Fingerprint, Match
from database.crud import (
    get_fingerprint,
    update_fingerprint_landmarks,
    update_fingerprint_manual_review,
    update_fingerprint_minutiae,
)

router = APIRouter(
    prefix="/api/editor",
    tags=["manual-editor"],
)


# ============================================================================
# Pydantic Models
# ============================================================================


class MinutiaPoint(BaseModel):
    """Single minutia point."""
    x: float
    y: float
    type: str
    landmark_type: str
    angle: float = 0.0
    confidence: float = 0.8
    manually_added: bool = False


class DeleteMinutiaRequest(BaseModel):
    """Request to delete a minutia point."""
    fingerprint_id: int
    minutia_index: int
    reason: str = ""


class AddMinutiaRequest(BaseModel):
    """Request to add a minutia point."""
    fingerprint_id: int
    x: float
    y: float
    landmark_type: str
    angle: float = 0.0
    confidence: float = 0.8


class UpdateMinutiaRequest(BaseModel):
    """Request to update landmark type."""
    fingerprint_id: int
    minutia_index: int
    new_landmark_type: str
    reason: str = ""


class ApproveRequest(BaseModel):
    """Request to approve a fingerprint."""
    fingerprint_id: int
    minutiae: list[dict[str, Any]]
    notes: str = ""
    user_id: int


class RejectRequest(BaseModel):
    """Request to reject a fingerprint."""
    fingerprint_id: int
    reason: str
    user_id: int


class EditorSessionResponse(BaseModel):
    """Response for editor session."""
    fingerprint_id: int
    minutiae: list[dict[str, Any]]
    classification: dict[str, Any]
    landmarks: dict[str, Any]
    image_url: str = ""
    status: str


class ReviewActionResponse(BaseModel):
    """Response for review action."""
    status: str
    message: str
    fingerprint_id: int
    action: str
    timestamp: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/fingerprint/{fingerprint_id}")
async def get_fingerprint_for_editing(fingerprint_id: int, db: Session = Depends(get_db)):
    """
    Get a fingerprint for manual editing.
    """
    db_fingerprint = get_fingerprint(db, fingerprint_id)
    if not db_fingerprint:
        raise HTTPException(status_code=404, detail="Fingerprint not found")
    
    # Extract minutiae from JSON
    minutiae = []
    if db_fingerprint.minutiae_data and isinstance(db_fingerprint.minutiae_data, dict):
        minutiae = db_fingerprint.minutiae_data.get('minutiae', [])
    
    base_url = f"/static/fingerprints/{db_fingerprint.filename}" if db_fingerprint.filename else ""
    viz = {"processed": base_url}
    if db_fingerprint.filename:
        stem = Path(db_fingerprint.filename).stem
        storage = Path("static/fingerprints")
        for suffix, label in [("_ridges", "تموجات (Ridges)"), ("_skeleton", "هيكل (Skeleton)")]:
            p = storage / f"{stem}{suffix}.png"
            if p.exists():
                viz[suffix.lstrip("_")] = f"/static/fingerprints/{stem}{suffix}.png"
    return {
        "fingerprint_id": fingerprint_id,
        "minutiae": minutiae,
        "classification": db_fingerprint.fingerprint_classification or {},
        "landmarks": db_fingerprint.landmarks or {},
        "image_url": base_url,
        "visualizations": viz,
        "status": "ready"
    }


@router.post("/delete-minutia")
async def delete_minutia_endpoint(request: DeleteMinutiaRequest, db: Session = Depends(get_db)):
    """
    Delete a minutia point from a fingerprint.
    """
    db_fingerprint = get_fingerprint(db, request.fingerprint_id)
    if not db_fingerprint:
        raise HTTPException(status_code=404, detail="Fingerprint not found")
    
    # Load current minutiae
    minutiae_data = db_fingerprint.minutiae_data or {"minutiae": []}
    minutiae = minutiae_data.get("minutiae", [])
    
    if 0 <= request.minutia_index < len(minutiae):
        # Perform deletion via service
        updated_minutiae, _ = delete_minutia(minutiae, request.minutia_index, 1, request.reason)
        
        # Save back to DB
        update_fingerprint_minutiae(db, request.fingerprint_id, {"minutiae": updated_minutiae})
        
        return {
            "status": "success",
            "message": f"Minutia at index {request.minutia_index} deleted",
            "fingerprint_id": request.fingerprint_id,
            "action": AnnotationAction.DELETE_MINUTIA.value,
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid minutia index")


@router.post("/add-minutia")
async def add_minutia_endpoint(request: AddMinutiaRequest, db: Session = Depends(get_db)):
    """
    Add a new minutia point to a fingerprint.
    """
    db_fingerprint = get_fingerprint(db, request.fingerprint_id)
    if not db_fingerprint:
        raise HTTPException(status_code=404, detail="Fingerprint not found")
    
    # Validate landmark type
    valid_landmarks = ["termination", "bifurcation", "island", "ridge", "loop_eye", "bridge", "lake", "dot"]
    if request.landmark_type not in valid_landmarks:
        raise HTTPException(status_code=400, detail=f"Invalid landmark type: {request.landmark_type}")
    
    # Load current minutiae
    minutiae_data = db_fingerprint.minutiae_data or {"minutiae": []}
    minutiae = minutiae_data.get("minutiae", [])
    
    # Add via service
    updated_minutiae, action_rec = add_minutia(
        minutiae, 
        request.x, 
        request.y, 
        request.landmark_type, 
        1, # Default expert user_id
        request.angle, 
        request.confidence
    )
    
    new_point = action_rec.get("minutia_added")
    
    # Save back to DB
    update_fingerprint_minutiae(db, request.fingerprint_id, {"minutiae": updated_minutiae})
    
    return {
        "status": "success",
        "message": "Minutia added",
        "fingerprint_id": request.fingerprint_id,
        "action": AnnotationAction.ADD_MINUTIA.value,
        "minutia": new_point,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/update-landmark")
async def update_landmark_endpoint(request: UpdateMinutiaRequest, db: Session = Depends(get_db)):
    """
    Update the landmark type of an existing minutia.
    """
    db_fingerprint = get_fingerprint(db, request.fingerprint_id)
    if not db_fingerprint:
        raise HTTPException(status_code=404, detail="Fingerprint not found")
    
    valid_landmarks = ["termination", "bifurcation", "island", "ridge", "loop_eye", "bridge", "lake", "dot"]
    if request.new_landmark_type not in valid_landmarks:
        raise HTTPException(status_code=400, detail=f"Invalid landmark type: {request.new_landmark_type}")
    
    # Load current minutiae
    minutiae_data = db_fingerprint.minutiae_data or {"minutiae": []}
    minutiae = minutiae_data.get("minutiae", [])
    
    if 0 <= request.minutia_index < len(minutiae):
        # Update via service
        updated_minutiae, _ = update_landmark_type(minutiae, request.minutia_index, request.new_landmark_type, 1, request.reason)
        
        # Save back to DB
        update_fingerprint_minutiae(db, request.fingerprint_id, {"minutiae": updated_minutiae})
        
        return {
            "status": "success",
            "message": "Landmark type updated",
            "fingerprint_id": request.fingerprint_id,
            "action": AnnotationAction.UPDATE_LANDMARK.value,
            "minutia_index": request.minutia_index,
            "new_type": request.new_landmark_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid minutia index")


@router.post("/approve")
async def approve_fingerprint(request: ApproveRequest, db: Session = Depends(get_db)):
    """
    Approve a fingerprint after manual review.
    """
    try:
        # 1. Update minutiae
        update_fingerprint_minutiae(db, request.fingerprint_id, {"minutiae": request.minutiae})
        
        # 2. Mark as manually reviewed
        update_fingerprint_manual_review(
            db, 
            request.fingerprint_id, 
            is_reviewed=True, 
            reviewer_id=request.user_id, 
            notes=request.notes
        )
        
        return {
            "status": "success",
            "message": "Fingerprint approved",
            "fingerprint_id": request.fingerprint_id,
            "action": "approve",
            "final_minutiae_count": len(request.minutiae),
            "reviewed_by": request.user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reject")
async def reject_fingerprint(request: RejectRequest, db: Session = Depends(get_db)):
    """
    Reject a fingerprint as unusable.
    """
    try:
        # Mark as manually reviewed with rejection note
        update_fingerprint_manual_review(
            db, 
            request.fingerprint_id, 
            is_reviewed=True, 
            reviewer_id=request.user_id, 
            notes=f"REJECTED: {request.reason}"
        )
        
        return {
            "status": "success",
            "message": "Fingerprint rejected",
            "fingerprint_id": request.fingerprint_id,
            "action": "reject",
            "reason": request.reason,
            "reviewed_by": request.user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/editor-page")
async def get_editor_page():
    """
    Serve the manual editor HTML page.
    
    This is handled by FastAPI's StaticFiles middleware for /templates/
    """
    return {"message": "Manual editor page - see /templates/manual_editor.html"}


@router.get("/match-editor/{match_id}")
async def get_match_editor(match_id: int):
    """
    Get comparison editor for two fingerprints in a match.
    
    Returns both fingerprints side-by-side for comparison and manual editing.
    """
    try:
        # TODO: Implement with database access
        return {
            "match_id": match_id,
            "fingerprint_ref": {"id": 1, "minutiae": []},
            "fingerprint_qry": {"id": 2, "minutiae": []},
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================


@router.get("/health")
async def health_check():
    """Check if editor service is available."""
    return {
        "status": "healthy",
        "service": "manual-editor",
        "version": "3.0",
        "endpoints": [
            "GET /fingerprint/{id}",
            "POST /delete-minutia",
            "POST /add-minutia",
            "POST /update-landmark",
            "POST /approve",
            "POST /reject",
            "GET /match-editor/{id}",
        ]
    }
