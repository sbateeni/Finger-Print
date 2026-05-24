"""
Forensic minutiae taxonomy — maps poster / PDF (Mosul paper) labels to engine types.
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
