"""
Fingerprint Type and Classification Engine.

Classifies fingerprints by:
1. Pattern type (arch, loop, whorl) — using singular points
2. Finger type (thumb, index, middle, ring, pinky) — using minutiae distribution
3. Region (fingertip, palm_root, sub_index) — using ridge analysis

Critical for matching: two fingerprints must be compatible
to proceed to detailed minutiae matching.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import cv2
from enum import Enum


class PatternType(Enum):
    """Fingerprint pattern (Henry classification system)."""
    ARCH = "arch"                  # قوس
    TENTED_ARCH = "tented_arch"    # قوس خيمي
    LEFT_LOOP = "left_loop"        # أنشوطة يسرى
    RIGHT_LOOP = "right_loop"      # أنشوطة يمنى
    WHORL = "whorl"                # دوامة
    UNKNOWN = "unknown"


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
    FINGERTIP = "fingertip"
    PALM_ROOT = "palm_root"
    SUB_INDEX = "sub_index"
    PALM_GENERAL = "palm_general"
    UNKNOWN = "unknown"


PATTERN_NAMES_AR = {
    PatternType.ARCH: "قوس",
    PatternType.TENTED_ARCH: "قوس خيمي",
    PatternType.LEFT_LOOP: "أنشوطة يسرى",
    PatternType.RIGHT_LOOP: "أنشوطة يمنى",
    PatternType.WHORL: "دوامة",
    PatternType.UNKNOWN: "غير معروف",
}


class FingerprintClassification:
    """
    Result of fingerprint type and region classification.

    This is the FIRST check before fingerprint matching.
    If two fingerprints have incompatible patterns, matching is rejected.
    """

    def __init__(
        self,
        pattern_type: PatternType = PatternType.UNKNOWN,
        finger_type: FingerType = FingerType.UNKNOWN,
        region: FingerprintRegion = FingerprintRegion.UNKNOWN,
        size_width: int = 0,
        size_height: int = 0,
        confidence: float = 0.0,
        details: dict[str, Any] | None = None,
    ):
        self.pattern_type = pattern_type
        self.finger_type = finger_type
        self.region = region
        self.size_width = size_width
        self.size_height = size_height
        self.confidence = confidence
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_type": self.pattern_type.value,
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
        Returns (is_compatible, reason).
        """
        reasons = []
        if self.pattern_type != other.pattern_type:
            if self.pattern_type != PatternType.UNKNOWN and other.pattern_type != PatternType.UNKNOWN:
                return False, f"Different patterns: {self.pattern_type.value} vs {other.pattern_type.value}"

        if self.finger_type != other.finger_type:
            if self.finger_type != FingerType.UNKNOWN and other.finger_type != FingerType.UNKNOWN:
                return False, f"Different finger types: {self.finger_type.value} vs {other.finger_type.value}"

        if self.region != other.region:
            if self.region != FingerprintRegion.UNKNOWN and other.region != FingerprintRegion.UNKNOWN:
                return False, f"Different regions: {self.region.value} vs {other.region.value}"

        if self.size_width > 0 and other.size_width > 0:
            wr = max(self.size_width, other.size_width) / min(self.size_width, other.size_width)
            if wr > 1.5:
                reasons.append(f"Size mismatch (width ratio: {wr:.2f})")
        if self.size_height > 0 and other.size_height > 0:
            hr = max(self.size_height, other.size_height) / min(self.size_height, other.size_height)
            if hr > 1.5:
                reasons.append(f"Size mismatch (height ratio: {hr:.2f})")
        if self.confidence < 0.5 or other.confidence < 0.5:
            reasons.append(f"Low confidence: {min(self.confidence, other.confidence):.2%}")

        return True, " | ".join(reasons) if reasons else "Compatible"


class FingerprintClassifier:
    """
    Analyzes image, minutiae, and singular points to classify a fingerprint.
    """

    def __init__(self):
        pass

    def classify(
        self,
        image: np.ndarray | None = None,
        minutiae: list[dict[str, Any]] | None = None,
        image_shape: tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        cores: list[dict[str, Any]] | None = None,
        deltas: list[dict[str, Any]] | None = None,
    ) -> FingerprintClassification:
        """
        Classify a fingerprint by pattern, type, and region.
        """
        metadata = metadata or {}
        details = {}

        if image is not None:
            height, width = image.shape[:2]
        elif image_shape:
            height, width = (image_shape[:2] if len(image_shape) >= 2
                             else (image_shape[0], image_shape[0]))
        else:
            height, width = 0, 0

        # --- Detect pattern type (Arch/Loop/Whorl) from singular points ---
        if cores is None or deltas is None:
            sp_cores, sp_deltas = self._detect_singular_points(image)
        else:
            sp_cores, sp_deltas = cores, deltas

        pattern_type, pattern_conf, pattern_detail = self._detect_pattern_type(
            sp_cores, sp_deltas, height, width
        )
        details["pattern_detection"] = pattern_detail
        details["pattern_confidence"] = pattern_conf
        details["cores_found"] = len(sp_cores)
        details["deltas_found"] = len(sp_deltas)

        # --- Detect finger type from metadata or ridge-density analysis ---
        finger_type = self._detect_finger_type_from_metadata(metadata)
        if finger_type == FingerType.UNKNOWN:
            finger_type, finger_conf = self._detect_finger_type_from_image(
                image, minutiae, height, width
            )
            details["finger_detection_method"] = "ridge_density" if image is not None else "minutiae_only"
            details["finger_confidence"] = finger_conf
        else:
            details["finger_detection_method"] = "metadata"
            details["finger_confidence"] = 0.7 if finger_type != FingerType.UNKNOWN else 0.0

        # --- Detect region via orientation field ---
        region, region_conf = self._detect_region(
            image=image,
            minutiae=minutiae,
            height=height,
            width=width,
            metadata=metadata,
        )
        details["region_confidence"] = region_conf

        overall = (pattern_conf + details["finger_confidence"] + region_conf) / 3

        return FingerprintClassification(
            pattern_type=pattern_type,
            finger_type=finger_type,
            region=region,
            size_width=width,
            size_height=height,
            confidence=overall,
            details=details,
        )

    # ----------------------------------------------------------------
    #  Pattern type
    # ----------------------------------------------------------------
    def _detect_singular_points(
        self, image: np.ndarray | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Detect cores and deltas using Poincaré index if image is available."""
        if image is None:
            return [], []
        try:
            from utils.image_processing import detect_edges, detect_singular_points
            _, omap = detect_edges(image)
            if omap is None:
                return [], []
            return detect_singular_points(omap, image > 127)
        except Exception:
            return [], []

    def _detect_pattern_type(
        self,
        cores: list[dict[str, Any]],
        deltas: list[dict[str, Any]],
        height: int,
        width: int,
    ) -> tuple[PatternType, float, dict[str, Any]]:
        """
        Determine pattern type from singular points.

        Henry rules:
        - 0 deltas              → Arch
        - 1 delta               → Loop (left/right based on core↔delta position)
        - 2+ deltas             → Whorl
        """
        n_deltas = len(deltas)
        n_cores = len(cores)
        detail = {"n_cores": n_cores, "n_deltas": n_deltas}

        if n_deltas == 0:
            # Arch or Tented Arch
            if n_cores > 0 and height > 0:
                # Tented arch: a core near the center line with no delta
                core_y = cores[0]["y"]
                if abs(core_y / height - 0.5) < 0.15:
                    return PatternType.TENTED_ARCH, 0.65, detail
            return PatternType.ARCH, 0.70, detail

        if n_deltas >= 2:
            return PatternType.WHORL, 0.75, detail

        # Exactly 1 delta → Loop
        if n_cores == 0:
            return PatternType.LEFT_LOOP, 0.50, detail

        core = cores[0]
        delta = deltas[0]
        # In a right loop the core is LEFT of the delta
        # In a left loop the core is RIGHT of the delta
        if core["x"] < delta["x"]:
            return PatternType.RIGHT_LOOP, 0.70, detail
        else:
            return PatternType.LEFT_LOOP, 0.70, detail

    # ----------------------------------------------------------------
    #  Finger type
    # ----------------------------------------------------------------
    def _detect_finger_type_from_metadata(self, metadata: dict[str, Any]) -> FingerType:
        finger_hint = (metadata.get("finger_type") or "").lower().strip()
        mapping = {
            "thumb": FingerType.THUMB, "إبهام": FingerType.THUMB,
            "index": FingerType.INDEX, "سبابة": FingerType.INDEX,
            "middle": FingerType.MIDDLE, "وسطى": FingerType.MIDDLE,
            "ring": FingerType.RING, "بنصر": FingerType.RING,
            "pinky": FingerType.PINKY, "خنصر": FingerType.PINKY,
        }
        return mapping.get(finger_hint, FingerType.UNKNOWN)

    def _detect_finger_type_from_image(
        self,
        image: np.ndarray | None,
        minutiae: list[dict[str, Any]] | None,
        height: int,
        width: int,
    ) -> tuple[FingerType, float]:
        """
        Infer finger type from ridge density and aspect ratio.

        Metrics:
          - ridge_density: fraction of white pixels (ridges) after Otsu threshold
          - aspect_ratio : width / height of the image
          - num_minutiae : count of detected minutiae

        Heuristics:
          Thumb    → wide (aspect > 0.85), ridges spaced apart (density < 0.55)
          Index    → narrow (aspect < 0.65), dense ridges
          Middle   → narrow, many minutiae
          Ring     → medium width
          Pinky    → very small, narrow
        """
        ridge_density = 0.0
        if image is not None:
            try:
                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                total = height * width
                ridge_pixels = cv2.countNonZero(thresh)
                ridge_density = ridge_pixels / float(total) if total > 0 else 0.0
            except Exception:
                pass

        aspect_ratio = width / float(height) if height > 0 else 0
        num_minutiae = len(minutiae) if minutiae else 0

        if height == 0 or width == 0:
            return FingerType.UNKNOWN, 0.0

        # Thumb: wide, lower ridge density
        if aspect_ratio > 0.85 and ridge_density < 0.55:
            return FingerType.THUMB, 0.60

        # Index / pinky: narrow
        if aspect_ratio < 0.65:
            if ridge_density > 0.55 or num_minutiae > 35:
                return FingerType.INDEX, 0.55
            return FingerType.PINKY, 0.50

        # Middle / ring: medium aspect, use minutiae count as tiebreaker
        if num_minutiae > 40:
            return FingerType.MIDDLE, 0.55
        return FingerType.RING, 0.50

    # ----------------------------------------------------------------
    #  Region (via orientation field)
    # ----------------------------------------------------------------
    def _detect_region(
        self,
        image: np.ndarray | None,
        minutiae: list[dict[str, Any]] | None,
        height: int,
        width: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[FingerprintRegion, float]:
        metadata = metadata or {}

        region_hint = (metadata.get("region") or "").lower().strip()
        hint_map = {
            "fingertip": FingerprintRegion.FINGERTIP,
            "طرف الإصبع": FingerprintRegion.FINGERTIP,
            "palm_root": FingerprintRegion.PALM_ROOT,
            "جذور الأصابع": FingerprintRegion.PALM_ROOT,
            "sub_index": FingerprintRegion.SUB_INDEX,
            "تحت السبابة": FingerprintRegion.SUB_INDEX,
            "palm_general": FingerprintRegion.PALM_GENERAL,
            "راحة اليد": FingerprintRegion.PALM_GENERAL,
        }
        if region_hint in hint_map:
            return hint_map[region_hint], 0.80

        if image is None or height < 20 or width < 20:
            return FingerprintRegion.UNKNOWN, 0.20

        try:
            grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            angles = cv2.phase(grad_x, grad_y, angleInDegrees=True)

            top = angles[0:int(height * 0.35), :]
            core = angles[int(height * 0.35):int(height * 0.70), :]
            bottom = angles[int(height * 0.70):, :]

            core_var = float(np.var(core))
            top_var = float(np.var(top))
            top_mean = float(np.mean(top))

            # Core zone has highest orientation variance (ridge swirl)
            if core_var > top_var and core_var > 1500:
                return FingerprintRegion.FINGERTIP, 0.70

            # Arch-like: ridges flow horizontally across the top
            if top_mean > 180:
                return FingerprintRegion.PALM_GENERAL, 0.60

            return FingerprintRegion.PALM_ROOT, 0.55

        except Exception:
            return FingerprintRegion.UNKNOWN, 0.20


# Global classifier instance
_classifier = FingerprintClassifier()


def classify_fingerprint(
    image: np.ndarray | None = None,
    minutiae: list[dict[str, Any]] | None = None,
    image_shape: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    cores: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
) -> FingerprintClassification:
    """
    Convenient function to classify a fingerprint.
    """
    return _classifier.classify(
        image=image,
        minutiae=minutiae,
        image_shape=image_shape,
        metadata=metadata,
        cores=cores,
        deltas=deltas,
    )


def check_fingerprints_compatible(
    classification1: FingerprintClassification,
    classification2: FingerprintClassification,
) -> tuple[bool, str]:
    return classification1.is_compatible_with(classification2)
