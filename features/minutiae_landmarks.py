"""
Anatomical Landmarks Extraction Engine (Phase 2).

Detects and classifies the 8 anatomical landmarks:
1. Termination (نهاية الخط)
2. Bifurcation (التفرع)
3. Island (الجزيرة)
4. Ridge (الشرطة)
5. Loop/Eye (العين)
6. Bridge (الجسر)
7. Lake (البحيرة)
8. Dot (النقطة)

Each minutia point is enhanced with landmark type information.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from collections import Counter
from features.minutiae_taxonomy import ANATOMICAL_LANDMARKS, get_landmark_by_name


class LandmarkAnalyzer:
    """
    Analyzes minutiae and image to extract and classify anatomical landmarks.
    """

    def __init__(self):
        """Initialize landmark analyzer."""
        self.landmark_types = list(ANATOMICAL_LANDMARKS.keys())

    def analyze_minutiae_landmarks(
        self,
        minutiae: list[dict[str, Any]],
        image: np.ndarray | None = None,
        ridge_image: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze minutiae and enhance with landmark information.
        
        Args:
            minutiae: List of extracted minutiae points
            image: Original fingerprint image (optional)
            ridge_image: Binary ridge image (optional)
            
        Returns:
            Enhanced minutiae list with landmark_type, landmark_info fields
        """
        if not minutiae:
            return []
        
        enhanced_minutiae = []
        
        for m in minutiae:
            enhanced = m.copy()
            
            # Determine landmark type based on existing minutiae type
            landmark_type = self._determine_landmark_type(m)
            landmark_info = get_landmark_by_name(landmark_type)
            
            enhanced["landmark_type"] = landmark_type
            enhanced["landmark_info"] = landmark_info or {}
            
            # Calculate additional landmark properties if image available
            if image is not None:
                landmark_features = self._analyze_local_neighborhood(
                    enhanced,
                    image,
                    ridge_image,
                )
                enhanced["landmark_features"] = landmark_features
            
            enhanced_minutiae.append(enhanced)
        
        return enhanced_minutiae

    def _determine_landmark_type(self, minutia: dict[str, Any]) -> str:
        """Map extracted minutia type to one of the 8 anatomical landmarks."""
        mtype = (minutia.get("type") or "unknown").lower().strip()

        mapping = {
            "endpoint": "termination",
            "ending": "termination",
            "ending_ridge": "termination",
            "termination": "termination",
            "bifurcation": "bifurcation",
            "island": "island",
            "short_ridge": "island",
            "dot": "dot",
            "bridge": "bridge",
            "lake": "lake",
            "loop_eye": "loop_eye",
            "divergence": "loop_eye",
            "ridge": "ridge",
        }
        return mapping.get(mtype, "termination")

    def _analyze_local_neighborhood(
        self,
        minutia: dict[str, Any],
        image: np.ndarray,
        ridge_image: np.ndarray | None = None,
        window_size: int = 21,
    ) -> dict[str, Any]:
        """
        Analyze the local neighborhood around a minutia point.
        
        Returns features like:
        - Local ridge count
        - Surrounding ridges pattern
        - Curvature
        - Connectivity
        """
        x, y = int(minutia.get("x", 0)), int(minutia.get("y", 0))
        features = {
            "local_ridge_density": 0.0,
            "surrounding_pattern": "unknown",
            "curvature": 0.0,
            "connectivity": 0,
        }
        
        # Check bounds
        h, w = image.shape[:2] if len(image.shape) >= 2 else (image.shape[0], image.shape[0])
        half_win = window_size // 2
        
        if x - half_win < 0 or x + half_win >= w or y - half_win < 0 or y + half_win >= h:
            return features  # Out of bounds
        
        # Extract local window
        window = image[y - half_win : y + half_win + 1, x - half_win : x + half_win + 1]
        
        # Calculate ridge density in local window
        if len(window.shape) == 2:
            ridge_pixels = np.sum(window > 128)  # Assuming 128 as threshold
            total_pixels = window.size
            features["local_ridge_density"] = ridge_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Determine surrounding pattern
        if features["local_ridge_density"] > 0.6:
            features["surrounding_pattern"] = "dense_ridges"
        elif features["local_ridge_density"] > 0.3:
            features["surrounding_pattern"] = "medium_ridges"
        else:
            features["surrounding_pattern"] = "sparse_ridges"
        
        # Calculate connectivity (simplified)
        features["connectivity"] = self._estimate_connectivity(minutia)
        
        return features

    def _estimate_connectivity(self, minutia: dict[str, Any]) -> int:
        """
        Estimate connectivity number (CN) based on minutia type.
        
        CN values:
        - CN=0: Continuation/ridge
        - CN=1: Termination
        - CN=2: Island/Lake/Bridge entry-exit
        - CN=3: Bifurcation
        - CN=4: Bridge
        """
        minutia_type = (minutia.get("type") or "unknown").lower().strip()
        
        if minutia_type in ["endpoint", "ending", "termination"]:
            return 1
        elif minutia_type == "bifurcation":
            return 3
        elif minutia_type in ["island", "lake"]:
            return 2
        elif minutia_type == "bridge":
            return 4
        elif minutia_type == "dot":
            return 0
        else:
            return 0  # Continuation

    def statistics(self, enhanced_minutiae: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Generate statistics about landmarks in the fingerprint.
        """
        if not enhanced_minutiae:
            return {}
        
        landmark_counts = Counter()
        for m in enhanced_minutiae:
            landmark = m.get("landmark_type", "unknown")
            landmark_counts[landmark] += 1
        
        # Calculate importance scores
        high_importance = sum(
            1 for m in enhanced_minutiae
            if get_landmark_by_name(m.get("landmark_type", "")).get("forensic_importance") == "High"
        )
        
        return {
            "total_landmarks": len(enhanced_minutiae),
            "landmark_counts": dict(landmark_counts),
            "high_importance_count": high_importance,
            "landmark_diversity": len(landmark_counts),
            "landmark_list": list(landmark_counts.keys()),
        }


# Global analyzer instance
_analyzer = LandmarkAnalyzer()


def extract_landmarks(
    minutiae: list[dict[str, Any]],
    image: np.ndarray | None = None,
    ridge_image: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Convenient function to extract anatomical landmarks.
    
    Usage:
        enhanced = extract_landmarks(minutiae=points, image=img)
        stats = landmark_statistics(enhanced)
    """
    return _analyzer.analyze_minutiae_landmarks(minutiae, image, ridge_image)


def landmark_statistics(enhanced_minutiae: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get statistics about landmarks in enhanced minutiae.
    """
    return _analyzer.statistics(enhanced_minutiae)
