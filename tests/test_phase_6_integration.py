import pytest
import numpy as np
from services.analysis_service.pipeline import run_matching_pipeline
from features.fingerprint_classifier import FingerType, FingerprintRegion

def test_classification_gate_compatible():
    # Mock data
    ro = {
        "minutiae": [{"x": 100, "y": 100, "type": "termination"}] * 20,
        "quality_score": 80.0,
        "processed": np.zeros((100, 100), dtype=np.uint8),
        "ridges": np.zeros((100, 100), dtype=np.uint8),
        "skeleton": np.zeros((100, 100), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    rp = {
        "minutiae": [{"x": 105, "y": 105, "type": "termination"}] * 20,
        "quality_score": 80.0,
        "processed": np.zeros((100, 100), dtype=np.uint8),
        "ridges": np.zeros((100, 100), dtype=np.uint8),
        "skeleton": np.zeros((100, 100), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    
    # Matching types
    form_ctx = {
        "original_finger_type": "thumb",
        "partial_finger_type": "thumb"
    }
    
    res, _, _, _, is_warning = run_matching_pipeline(
        ro, rp, "sha_o", "sha_p", None, form_ctx, 
        "operator", "case", 10, 5, 5, 5, 10, 5,
        write_report_and_audit=False
    )
    
    # Check if classification was performed
    assert "classification" in ro
    assert "classification" in rp
    assert "Classification Gate" not in res.get("verdict", "")

def test_classification_gate_incompatible():
    # Mock data
    ro = {
        "minutiae": [{"x": 100, "y": 100, "type": "termination"}] * 20,
        "quality_score": 80.0,
        "processed": np.zeros((100, 100), dtype=np.uint8),
        "ridges": np.zeros((100, 100), dtype=np.uint8),
        "skeleton": np.zeros((100, 100), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    rp = {
        "minutiae": [{"x": 105, "y": 105, "type": "termination"}] * 20,
        "quality_score": 80.0,
        "processed": np.zeros((100, 100), dtype=np.uint8),
        "ridges": np.zeros((100, 100), dtype=np.uint8),
        "skeleton": np.zeros((100, 100), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    
    # Different types
    form_ctx = {
        "original_finger_type": "thumb",
        "partial_finger_type": "index"
    }
    
    res, _, _, _, is_warning = run_matching_pipeline(
        ro, rp, "sha_o", "sha_p", None, form_ctx, 
        "operator", "case", 10, 5, 5, 5, 10, 5,
        write_report_and_audit=False
    )
    
    assert res["classification_compatible"] == 0
    # In case of early exit, verdict is in 'quality_gate_reason' for inconclusive
    assert "Classification Gate" in res.get("quality_gate_reason", "")

def test_landmark_integration():
    # Mock data with enough minutiae to pass quality gate
    ro = {
        "minutiae": [
            {"x": 100, "y": 100, "type": "termination"},
            {"x": 150, "y": 150, "type": "bifurcation"}
        ] * 10,
        "quality_score": 80.0,
        "processed": np.zeros((200, 200), dtype=np.uint8),
        "ridges": np.zeros((200, 200), dtype=np.uint8),
        "skeleton": np.zeros((200, 200), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    rp = {
        "minutiae": [
            {"x": 102, "y": 102, "type": "termination"},
            {"x": 152, "y": 152, "type": "bifurcation"}
        ] * 10,
        "quality_score": 80.0,
        "processed": np.zeros((200, 200), dtype=np.uint8),
        "ridges": np.zeros((200, 200), dtype=np.uint8),
        "skeleton": np.zeros((200, 200), dtype=np.uint8),
        "white_pre": 0,
        "white_ridges": 0,
        "white_skel": 0,
        "minutiae_count": 20
    }
    
    res, _, _, _, _ = run_matching_pipeline(
        ro, rp, "sha_o", "sha_p", None, {}, 
        "operator", "case", 10, 5, 5, 5, 10, 5,
        write_report_and_audit=False
    )
    
    assert "landmark_similarity" in res
    assert res["landmark_similarity"] > 0
    assert "landmarks" in ro
    assert "landmarks" in rp
