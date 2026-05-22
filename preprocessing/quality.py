"""
Image quality assessment: NFIQ2 CLI when available, heuristic fallback otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preprocessing.image_quality import assess_image_quality


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def quality_backend() -> str:
    """heuristic | nfiq2_cli | auto"""
    return (_env("QUALITY_BACKEND", "auto") or "auto").lower()


def default_threshold() -> float:
    raw = _env("QUALITY_GATE_MIN_SCORE", "40")
    try:
        return float(raw)
    except ValueError:
        return 40.0


class QualityChecker:
    @staticmethod
    def get_quality_score(image: np.ndarray) -> tuple[float, str]:
        """
        Returns (score 0–100, method label).
        """
        if image is None or image.size == 0:
            return 0.0, "empty"

        backend = quality_backend()
        if backend in ("nfiq2_cli", "auto"):
            cli_score = _nfiq2_cli_score(image)
            if cli_score is not None:
                return cli_score, "nfiq2_cli"

        if backend == "nfiq2_cli":
            # forced CLI but unavailable — still fall back so pipeline does not break
            pass

        assessment = assess_image_quality(_to_gray(image))
        score_100 = round(float(assessment.get("quality_score", 0)) * 100.0, 1)
        return score_100, "heuristic_v1"

    @staticmethod
    def is_acceptable(
        image: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[bool, float, str]:
        th = float(threshold if threshold is not None else default_threshold())
        score, method = QualityChecker.get_quality_score(image)
        return score >= th, score, method


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _nfiq2_cli_score(image: np.ndarray) -> float | None:
    """
    Invoke NIST NFIQ2 binary if installed (not pip nfiq2).
    Set NFIQ2_CLI_PATH to the executable, or rely on PATH (nfiq2).
    """
    exe = _env("NFIQ2_CLI_PATH") or shutil.which("nfiq2")
    if not exe:
        return None

    gray = _to_gray(image)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fp.png"
            cv2.imwrite(str(path), gray)
            proc = subprocess.run(
                [exe, "-i", str(path), "-v", "-F", "1"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                return None
            # NFIQ2 prints quality score in stdout (last numeric line)
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                for token in reversed(parts):
                    try:
                        val = float(token)
                        if 0 <= val <= 100:
                            return round(val, 1)
                    except ValueError:
                        continue
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def quality_result_dict(
    image: np.ndarray,
    *,
    label: str = "image",
    threshold: float | None = None,
) -> dict[str, Any]:
    th = float(threshold if threshold is not None else default_threshold())
    score, method = QualityChecker.get_quality_score(image)
    ok = score >= th
    message = ""
    if not ok:
        message = (
            f"{label}: جودة الصورة منخفضة ({score:.0f}/100، الحد الأدنى {th:.0f}). "
            "أعد التقاط الصورة بإضاءة وتباين أفضل."
        )
    return {
        "ok": ok,
        "quality_score": score,
        "message": message,
        "quality_method": method,
        "metrics": {},
        "recommendations": [],
    }
