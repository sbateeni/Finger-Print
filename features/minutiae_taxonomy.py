"""
Forensic minutiae taxonomy — maps poster / PDF (Mosul paper) labels to engine types.

PHASE 2 Enhancement: 8 Anatomical Landmarks (العلامات التشريحية الثمانية)
"""

from __future__ import annotations

# Crime-scene poster "Identification Key" (36 annotated points) — feature counts
POSTER_36_KEY: list[tuple[int, str, str]] = [
    (1, "ending_ridge", "نهاية خط"),
    (2, "bifurcation", "انقسام / تشعب"),
    (3, "bifurcation", "انقسام / تشعب"),
    (4, "bifurcation", "انقسام / تشعب"),
    (5, "island", "جزيرة / خط قصير"),
    (6, "bifurcation", "انقسام / تشعب"),
    (7, "bifurcation", "انقسام / تشعب"),
    (8, "bifurcation", "انقسام / تشعب"),
    (9, "bifurcation", "انقسام / تشعب"),
    (10, "bifurcation", "انقسام / تشعب"),
    (11, "bifurcation", "انقسام / تشعب"),
    (12, "bifurcation", "انقسام / تشعب"),
    (13, "ending_ridge", "نهاية خط"),
    (14, "ending_ridge", "نهاية خط"),
    (15, "ending_ridge", "نهاية خط"),
    (16, "bifurcation", "انقسام / تشعب"),
    (17, "bifurcation", "انقسام / تشعب"),
    (18, "bifurcation", "انقسام / تشعب"),
    (19, "bifurcation", "انقسام / تشعب"),
    (20, "bifurcation", "انقسام / تشعب"),
    (21, "bifurcation", "انقسام / تشعب"),
    (22, "ending_ridge", "نهاية خط"),
    (23, "ending_ridge", "نهاية خط"),
    (24, "bifurcation", "انقسام / تشعب"),
    (25, "bifurcation", "انقسام / تشعب"),
    (26, "bifurcation", "انقسام / تشعب"),
    (27, "ending_ridge", "نهاية خط"),
    (28, "bifurcation", "انقسام / تشعب"),
    (29, "ending_ridge", "نهاية خط"),
    (30, "bifurcation", "انقسام / تشعب"),
    (31, "ending_ridge", "نهاية خط"),
    (32, "bifurcation", "انقسام / تشعب"),
    (33, "ending_ridge", "نهاية خط"),
    (34, "ending_ridge", "نهاية خط"),
    (35, "ending_ridge", "نهاية خط"),
    (36, "ending_ridge", "نهاية خط"),
]

POSTER_36_SUMMARY = {
    "ending_ridge": 12,
    "bifurcation": 23,
    "island": 1,
}

# PDF 132-137-1-PB (Mosul e-government fingerprint paper) — ridge characteristics
PDF_RIDGE_TYPES: list[tuple[str, str, str]] = [
    ("ending", "نهاية الخط", "endpoint"),
    ("bifurcation", "تفرع", "bifurcation"),
    ("lake", "بحيرة", "lake"),  # not yet extracted automatically
    ("short_ridge", "خط قصير", "island"),
    ("dot", "نقطة", "dot"),
    ("divergence", "حافة بارزة", "divergence"),
    ("bridge", "معبر أو جسر", "bridge"),
]

# Engine types produced by utils/minutiae_extractor.py
ENGINE_TYPES = (
    "endpoint",
    "bifurcation",
    "island",
    "lake",
    "dot",
    "bridge",
    "divergence",
)

TYPE_ALIASES = {
    "ending": "endpoint",
    "ending_ridge": "endpoint",
    "ending ridge": "endpoint",
    "ridge_ending": "endpoint",
    "bifurcation": "bifurcation",
    "island": "island",
    "short_ridge": "island",
    "dot": "dot",
    "lake": "lake",
    "bridge": "bridge",
    "divergence": "divergence",
}


def normalize_minutiae_type(label: str) -> str:
    key = (label or "").strip().lower().replace(" ", "_")
    return TYPE_ALIASES.get(key, key)


def implementation_status() -> dict[str, str]:
    """Human-readable coverage vs poster + PDF."""
    return {
        "ending_ridge / endpoint": "implemented (CN=1)",
        "bifurcation": "implemented (CN=3)",
        "island / short_ridge": "implemented (short spur reclassification)",
        "lake (بحيرة)": "implemented (CN=2 + short loops)",
        "dot (نقطة)": "implemented (ridge blob CC)",
        "bridge (جسر)": "implemented (CN=4)",
        "divergence (حافة بارزة)": "implemented (wide-angle bifurcation)",
        "poster_36_fixed_coordinates": "reference taxonomy in minutiae_taxonomy.py — auto-extract per image",
        "pdf_fingerprint_system_design": "template + 1:N / 1:1 via web and Telegram",
    }


def count_by_type(minutiae: list) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for m in minutiae or []:
        t = normalize_minutiae_type(str(m.get("type", "endpoint")))
        c[t] += 1
    return dict(c)


# ============================================================================
# PHASE 2: 8 Anatomical Landmarks (العلامات التشريحية الثمانية)
# ============================================================================
# These are the 8 primary landmarks used in fingerprint analysis and forensics.
# Each has a specific visual signature and importance for identification.

ANATOMICAL_LANDMARKS = {
    "termination": {
        "name_en": "Termination / Ridge Ending",
        "name_ar": "نهاية الخط",
        "symbol": "◇",
        "icon": "ending",
        "description_en": "Point where a ridge ends abruptly",
        "description_ar": "نقطة تنتهي فيها خطوط البصمة بشكل مفاجئ",
        "connectivity_number": 1,  # CN=1
        "forensic_importance": "High",
        "frequency_in_fingertip": "High",
        "frequency_in_palm": "Low",
        "detection_method": "Single pixel ridge endpoints",
    },
    "bifurcation": {
        "name_en": "Bifurcation / Ridge Split",
        "name_ar": "التفرع / التشعب",
        "symbol": "⊢",
        "icon": "bifurcation",
        "description_en": "Point where a ridge splits into two branches",
        "description_ar": "نقطة تنقسم فيها الخطوط إلى فرعين",
        "connectivity_number": 3,  # CN=3
        "forensic_importance": "High",
        "frequency_in_fingertip": "High",
        "frequency_in_palm": "Medium",
        "detection_method": "Ridge skeleton tracing",
    },
    "island": {
        "name_en": "Island / Short Ridge",
        "name_ar": "الجزيرة / خط قصير",
        "symbol": "⊗",
        "icon": "island",
        "description_en": "Small isolated ridge segment",
        "description_ar": "خطوط قصيرة منعزلة ومنفصلة عن باقي الخطوط",
        "connectivity_number": 2,  # CN=2 both ends
        "forensic_importance": "Medium",
        "frequency_in_fingertip": "Medium",
        "frequency_in_palm": "Low",
        "detection_method": "Connected component analysis",
    },
    "ridge": {
        "name_en": "Ridge / Continuous Line",
        "name_ar": "الشرطة / الخط المستمر",
        "symbol": "─",
        "icon": "ridge",
        "description_en": "Main continuous ridge line",
        "description_ar": "الخط الرئيسي المستمر للبصمة",
        "connectivity_number": 0,  # CN=0 for normal continuation
        "forensic_importance": "Medium",
        "frequency_in_fingertip": "High",
        "frequency_in_palm": "High",
        "detection_method": "Ridge tracing",
    },
    "loop_eye": {
        "name_en": "Loop / Eye / Circular Pattern",
        "name_ar": "العين / الحلقة",
        "symbol": "◯",
        "icon": "loop",
        "description_en": "Circular or oval closed pattern formed by ridges",
        "description_ar": "منطقة دائرية مغلقة تشكلها الخطوط",
        "connectivity_number": 0,  # CN=0 for closed loop
        "forensic_importance": "High",
        "frequency_in_fingertip": "Low",
        "frequency_in_palm": "Medium",
        "detection_method": "Contour detection, valley tracing",
    },
    "bridge": {
        "name_en": "Bridge / Connector",
        "name_ar": "الجسر / الوصلة",
        "symbol": "⌢",
        "icon": "bridge",
        "description_en": "Short ridge segment connecting two main ridges",
        "description_ar": "خط قصير يربط بين خطين رئيسيين",
        "connectivity_number": 4,  # CN=4 (connects two ridges)
        "forensic_importance": "Medium",
        "frequency_in_fingertip": "Low",
        "frequency_in_palm": "Medium",
        "detection_method": "Ridge connectivity analysis",
    },
    "lake": {
        "name_en": "Lake / Enclosed Area",
        "name_ar": "البحيرة",
        "symbol": "◈",
        "icon": "lake",
        "description_en": "Area completely enclosed by ridges",
        "description_ar": "منطقة محاطة بشكل كامل بالخطوط",
        "connectivity_number": 2,  # CN=2 for entry/exit
        "forensic_importance": "Medium",
        "frequency_in_fingertip": "Low",
        "frequency_in_palm": "Medium",
        "detection_method": "Contour filling, valley analysis",
    },
    "dot": {
        "name_en": "Dot / Pixel / Small Point",
        "name_ar": "النقطة / الدقة",
        "symbol": "•",
        "icon": "dot",
        "description_en": "Very small isolated ridge point or single pixel",
        "description_ar": "نقطة صغيرة جداً معزولة أو نقطة بيكسل واحدة",
        "connectivity_number": 0,
        "forensic_importance": "Low",
        "frequency_in_fingertip": "Medium",
        "frequency_in_palm": "Low",
        "detection_method": "Blob detection, connected components",
    },
}


def get_landmark_by_name(name: str) -> dict | None:
    """Get landmark details by name (English or Arabic)."""
    name_lower = name.lower().strip()
    
    # Direct key lookup
    if name_lower in ANATOMICAL_LANDMARKS:
        return ANATOMICAL_LANDMARKS[name_lower]
    
    # Search by name_ar or name_en
    for key, landmark in ANATOMICAL_LANDMARKS.items():
        if landmark["name_ar"].lower() == name_lower or landmark["name_en"].lower() == name_lower:
            return landmark
    
    return None


def get_all_landmarks() -> dict:
    """Return all 8 anatomical landmarks."""
    return ANATOMICAL_LANDMARKS.copy()


def landmark_names(lang: str = "en") -> list[str]:
    """Get list of all landmark names in specified language."""
    lang_key = "name_ar" if lang.lower() in ["ar", "arabic"] else "name_en"
    return [landmark[lang_key] for landmark in ANATOMICAL_LANDMARKS.values()]


def is_high_importance_landmark(landmark_name: str) -> bool:
    """Check if a landmark has high forensic importance."""
    landmark = get_landmark_by_name(landmark_name)
    return landmark and landmark.get("forensic_importance") == "High"
