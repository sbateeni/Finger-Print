"""
Gabor-style fingerprint enhancement (Hong et al. style).
Optional: pip install fingerprint-enhancer or clone Fingerprint-Enhancement-Python.
Falls back to built-in multi-orientation Gabor bank.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EXTERNAL_ENHANCE = None
try:
    from fingerprint_enhancer import enhance_Fingerprint as _external_enhance  # type: ignore

    _EXTERNAL_ENHANCE = _external_enhance
    logger.info("Using fingerprint_enhancer package")
except ImportError:
    try:
        from enhance import enhance_Fingerprint as _external_enhance  # type: ignore

        _EXTERNAL_ENHANCE = _external_enhance
        logger.info("Using enhance module (Fingerprint-Enhancement-Python)")
    except ImportError:
        pass


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _gabor_enhance(gray: np.ndarray) -> np.ndarray:
    """Built-in Gabor bank (same idea as utils.image_processing.detect_edges)."""
    from config import (
        GABOR_GAMMA,
        GABOR_KERNEL_SIZE,
        GABOR_LAMBDA,
        GABOR_ORIENTATIONS,
        GABOR_PSI,
        GABOR_SIGMA,
    )

    n_theta = max(2, int(GABOR_ORIENTATIONS))
    responses = []
    for k in range(n_theta):
        theta = (np.pi / n_theta) * k
        kernel = cv2.getGaborKernel(
            (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
            GABOR_SIGMA,
            theta,
            GABOR_LAMBDA,
            GABOR_GAMMA,
            GABOR_PSI,
            ktype=cv2.CV_32F,
        )
        resp = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
        responses.append(resp)
    ridges_f = np.max(np.stack(responses, axis=0), axis=0)
    enhanced = cv2.normalize(ridges_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(enhanced)


class FingerprintEnhancer:
    @staticmethod
    def enhance(image: np.ndarray, prefer_external: bool = True) -> np.ndarray:
        """
        Enhance ridge structure. Input: grayscale 0–255 (or BGR).
        """
        gray = _to_gray_uint8(image)
        if prefer_external and _EXTERNAL_ENHANCE is not None:
            try:
                out = _EXTERNAL_ENHANCE(gray)
                if out is not None and out.size > 0:
                    return _to_gray_uint8(np.asarray(out))
            except Exception as e:
                logger.warning("External enhancer failed, using Gabor fallback: %s", e)
        return _gabor_enhance(gray)

    @staticmethod
    def is_external_available() -> bool:
        return _EXTERNAL_ENHANCE is not None
