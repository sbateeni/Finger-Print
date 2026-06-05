"""
Fingerprint Type and Classification Engine.

Classifies fingerprints by:
1. Finger type (thumb, index, middle, ring, pinky)
2. Region (fingertip, palm_root, sub_index)
3. Size and metrics

Critical for matching: two fingerprints must be the same type
to proceed to detailed minutiae matching.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from enum import Enum


class FingerType(Enum):
    """Finger/thumb identification."""
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
    UNKNOWN = "unknown"


class FingerprintRegion(Enum):
    """Region of fingerprint capture."""
    FINGERTIP = "fingertip"  # طرف الإصبع - نهاية الخط
    PALM_ROOT = "palm_root"  # جذور الأصابع - خطوط متواصلة
    SUB_INDEX = "sub_index"  # منطقة تحت السبابة
    PALM_GENERAL = "palm_general"  # راحة اليد
    UNKNOWN = "unknown"


class FingerprintClassification:
    """
    Result of fingerprint type and region classification.
    
    This is the FIRST check before fingerprint matching.
    If two fingerprints have different types/regions, matching is rejected.
    """

    def __init__(
        self,
        finger_type: FingerType = FingerType.UNKNOWN,
        region: FingerprintRegion = FingerprintRegion.UNKNOWN,
        size_width: int = 0,
        size_height: int = 0,
        confidence: float = 0.0,
        details: dict[str, Any] | None = None,
    ):
        self.finger_type = finger_type
        self.region = region
        self.size_width = size_width
        self.size_height = size_height
        self.confidence = confidence
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "finger_type": self.finger_type.value,
            "region": self.region.value,
            "size_width": self.size_width,
            "size_height": self.size_height,
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }

    def is_compatible_with(self, other: "FingerprintClassification") -> tuple[bool, str]:
        """
        Check if this fingerprint is compatible with another for matching.
        
        Returns:
            (is_compatible, reason)
        """
        reasons = []
        
        # Check finger type
        if self.finger_type != other.finger_type:
            if self.finger_type != FingerType.UNKNOWN and other.finger_type != FingerType.UNKNOWN:
                return False, f"Different finger types: {self.finger_type.value} vs {other.finger_type.value}"
        
        # Check region
        if self.region != other.region:
            if self.region != FingerprintRegion.UNKNOWN and other.region != FingerprintRegion.UNKNOWN:
                return False, f"Different regions: {self.region.value} vs {other.region.value}"
        
        # Check size (must be reasonably similar)
        if self.size_width > 0 and other.size_width > 0:
            width_ratio = max(self.size_width, other.size_width) / min(self.size_width, other.size_width)
            if width_ratio > 1.5:  # Allow 50% size difference
                reasons.append(f"Size mismatch (width ratio: {width_ratio:.2f})")
        
        if self.size_height > 0 and other.size_height > 0:
            height_ratio = max(self.size_height, other.size_height) / min(self.size_height, other.size_height)
            if height_ratio > 1.5:  # Allow 50% size difference
                reasons.append(f"Size mismatch (height ratio: {height_ratio:.2f})")
        
        # Low confidence warning
        if self.confidence < 0.5 or other.confidence < 0.5:
            reasons.append(f"Low classification confidence: {min(self.confidence, other.confidence):.2%}")
        
        return True, " | ".join(reasons) if reasons else "Compatible"


class FingerprintClassifier:
    """
    Main classifier: analyzes image and minutiae to determine type and region.
    """

    def __init__(self):
        """Initialize classifier with default parameters."""
        pass

    def classify(
        self,
        image: np.ndarray | None = None,
        minutiae: list[dict[str, Any]] | None = None,
        image_shape: tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FingerprintClassification:
        """
        Classify a fingerprint by type and region.
        
        Args:
            image: Original fingerprint image (optional)
            minutiae: List of extracted minutiae points (optional)
            image_shape: Shape of the image if image is None
            metadata: User-provided metadata (can hint at type)
            
        Returns:
            FingerprintClassification object
        """
        metadata = metadata or {}
        details = {}
        
        # Determine image dimensions
        if image is not None:
            height, width = image.shape[:2]
        elif image_shape:
            height, width = image_shape[:2] if len(image_shape) >= 2 else (image_shape[0], image_shape[0])
        else:
            height, width = 0, 0
        
        # Try to detect finger type from metadata first
        finger_type = self._detect_finger_type_from_metadata(metadata)
        
        # If not provided, try to infer from minutiae distribution
        if finger_type == FingerType.UNKNOWN and minutiae:
            finger_type, finger_confidence = self._detect_finger_type_from_minutiae(minutiae, height, width)
            details["finger_detection_method"] = "minutiae_analysis"
            details["finger_confidence"] = finger_confidence
        else:
            details["finger_detection_method"] = "metadata"
            details["finger_confidence"] = 0.7 if finger_type != FingerType.UNKNOWN else 0.0
        
        # Detect region
        region, region_confidence = self._detect_region(
            image=image,
            minutiae=minutiae,
            height=height,
            width=width,
            metadata=metadata,
        )
        details["region_detection_method"] = "image_analysis"
        details["region_confidence"] = region_confidence
        
        # Calculate overall confidence
        overall_confidence = (details["finger_confidence"] + region_confidence) / 2
        
        return FingerprintClassification(
            finger_type=finger_type,
            region=region,
            size_width=width,
            size_height=height,
            confidence=overall_confidence,
            details=details,
        )

    def _detect_finger_type_from_metadata(self, metadata: dict[str, Any]) -> FingerType:
        """
        Try to detect finger type from metadata (e.g., user input, filename, etc.)
        """
        if not metadata:
            return FingerType.UNKNOWN
        
        finger_hint = (metadata.get("finger_type") or "").lower().strip()
        
        mapping = {
            "thumb": FingerType.THUMB,
            "إبهام": FingerType.THUMB,
            "index": FingerType.INDEX,
            "سبابة": FingerType.INDEX,
            "middle": FingerType.MIDDLE,
            "وسطى": FingerType.MIDDLE,
            "ring": FingerType.RING,
            "بنصر": FingerType.RING,
            "pinky": FingerType.PINKY,
            "خنصر": FingerType.PINKY,
        }
        
        return mapping.get(finger_hint, FingerType.UNKNOWN)

    def _detect_finger_type_from_minutiae(
        self,
        minutiae: list[dict[str, Any]],
        height: int,
        width: int,
    ) -> tuple[FingerType, float]:
        """
        Infer finger type from minutiae distribution.
        
        Heuristics:
        - Thumb: usually smaller, more circular distribution
        - Fingers: larger, more linear distribution
        - Returns: (finger_type, confidence)
        """
        if not minutiae or height == 0 or width == 0:
            return FingerType.UNKNOWN, 0.0
        
        # Extract minutiae positions
        positions = np.array([[m["x"], m["y"]] for m in minutiae])
        
        # Calculate spread metrics
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        
        aspect_ratio = x_range / (y_range + 1)  # Avoid division by zero
        
        # Calculate distribution center (center of mass)
        center = positions.mean(axis=0)
        
        # Thumb typically has more central distribution
        image_center = np.array([width / 2, height / 2])
        distance_to_center = np.linalg.norm(center - image_center)
        
        # Simple heuristic: if very compact and central, might be thumb
        compactness = len(minutiae) / (width * height) if (width * height) > 0 else 0
        
        details = {
            "aspect_ratio": float(aspect_ratio),
            "compactness": float(compactness),
            "distance_to_center": float(distance_to_center),
        }
        
        # Without machine learning, we can't reliably detect finger type
        # Return UNKNOWN but store details for future enhancement
        return FingerType.UNKNOWN, 0.3

    def _detect_region(
        self,
        image: np.ndarray | None,
        minutiae: list[dict[str, Any]] | None,
        height: int,
        width: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[FingerprintRegion, float]:
        """
        Detect region: fingertip, palm root, sub-index, etc.
        
        Heuristics:
        - Fingertip: ends abruptly (clear termination ridge)
        - Palm root: continuous ridges (no clear ending)
        - Sub-index: specific area under index finger
        """
        metadata = metadata or {}
        confidence = 0.0
        
        # Check if region is provided in metadata
        region_hint = (metadata.get("region") or "").lower().strip()
        if region_hint:
            mapping = {
                "fingertip": FingerprintRegion.FINGERTIP,
                "طرف الإصبع": FingerprintRegion.FINGERTIP,
                "palm_root": FingerprintRegion.PALM_ROOT,
                "جذور الأصابع": FingerprintRegion.PALM_ROOT,
                "sub_index": FingerprintRegion.SUB_INDEX,
                "تحت السبابة": FingerprintRegion.SUB_INDEX,
            }
            if region_hint in mapping:
                return mapping[region_hint], 0.8
        
        # Analyze minutiae distribution if available
        if minutiae and len(minutiae) > 0:
            ending_ridges = sum(1 for m in minutiae if m.get("type") == "endpoint")
            bifurcations = sum(1 for m in minutiae if m.get("type") == "bifurcation")
            
            # Fingertip usually has more ending ridges at boundaries
            ratio = ending_ridges / (len(minutiae) + 1)
            
            if ratio > 0.3:  # High ratio of endings suggests fingertip
                return FingerprintRegion.FINGERTIP, 0.6
            else:
                # More bifurcations and continuous suggests palm
                return FingerprintRegion.PALM_ROOT, 0.5
        
        # Default: unknown region
        return FingerprintRegion.UNKNOWN, 0.2


# Global classifier instance
_classifier = FingerprintClassifier()


def classify_fingerprint(
    image: np.ndarray | None = None,
    minutiae: list[dict[str, Any]] | None = None,
    image_shape: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FingerprintClassification:
    """
    Convenient function to classify a fingerprint.
    
    Usage:
        classification = classify_fingerprint(image=img, minutiae=points, metadata={"finger_type": "thumb"})
        if not classification.is_compatible_with(other_classification):
            print("Different fingerprint types - no match possible")
    """
    return _classifier.classify(
        image=image,
        minutiae=minutiae,
        image_shape=image_shape,
        metadata=metadata,
    )


def check_fingerprints_compatible(
    classification1: FingerprintClassification,
    classification2: FingerprintClassification,
) -> tuple[bool, str]:
    """
    Check if two fingerprints are compatible for matching.
    
    Returns:
        (is_compatible, reason)
    """
    return classification1.is_compatible_with(classification2)
