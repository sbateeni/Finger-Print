"""Report strings for forensic HTML/PDF (ar / en)."""

from __future__ import annotations

REPORT_STRINGS = {
    "ar": {
        "html_lang": "ar",
        "dir": "rtl",
        "title": "تقرير جنائي — مقارنة بصمات",
        "subtitle": "مخرجات إحصائية للمراجعة الفنية",
        "generated_at": "تاريخ التوليد",
        "disclaimer_title": "إخلاء مسؤولية مخبري",
        "disclaimer": (
            "هذا التقرير يعرض مخرجات برمجية إحصائية ولا يُستخدم وحده كدليل هوية. "
            "يتطلب مراجعة خبير معتمد وإجراءات المختبر المعتمدة."
        ),
        "audit_title": "سجل التدقيق",
        "case_ref": "مرجع القضية",
        "operator": "المشغّل / المراجع",
        "sha_ref": "SHA-256 (المرجعية)",
        "sha_query": "SHA-256 (المقارنة)",
        "methodology_title": "ملخص المنهجية",
        "methodology": (
            "تحويل رمادي، تطبيع، CLAHE، إزالة ضوضاء، ثنائية، Gabor متعدد الاتجاهات، "
            "ترقيق، استخراج نقاط (نهاية/تفرع/جزيرة/بحيرة/جسر/حافة) مع فلترة، "
            "محاذاة جزئية (إزاحة + دوران + مقياس)، مطابقة MCC، ودمج قرار."
        ),
        "params_title": "معاملات التشغيل",
        "param_col": "المعامل",
        "value_col": "القيمة",
        "pipeline_title": "مراحل المعالجة",
        "pipeline_ref": "مسار المرجعية",
        "pipeline_query": "مسار المقارنة",
        "match_overlay": "تطابق بعد المحاذاة",
        "orb_title": "تحقق ORB (اختياري)",
        "skeletons_title": "هياكل التطابق النهائية",
        "sk_ref": "هيكل المرجعية",
        "sk_query": "هيكل المقارنة",
        "archive_note": "ملفات الأرشيف",
        "alignment_title": "محاذاة الجزء على المرجعية",
        "alignment_hint": "أخضر: نقاط المرجعية المتطابقة | برتقالي: نقاط المقارنة بعد المحاذاة",
        "results_title": "نتائج المقارنة الإحصائية",
        "n_ref": "عدد نقاط المرجعية",
        "n_query": "عدد نقاط المقارنة",
        "n_matched": "أزواج متطابقة (واحد-لواحد)",
        "match_score": "نسبة التطابق (حسب المقارنة)",
        "fused_score": "الدرجة المدمجة",
        "combined_verdict": "الحكم المدمج",
        "decision_mode": "وضع القرار",
        "fusion_components": "مكوّنات الدمج",
        "fusion_min": "نقاط",
        "fusion_mcc": "MCC",
        "fusion_orb": "ORB",
        "dice": "معامل Dice",
        "tier": "التصنيف الفني",
        "score_expl": "شرح النسبة",
        "internal_status": "الحالة الداخلية",
        "analysis_mode": "وضع التحليل",
        "deep": "فحص عميق (محاذاة تلقائية واسعة + مسار كامل)",
        "fast": "فحص سريع",
        "stage_processed": "ثنائية بعد المعالجة",
        "stage_quality": "خريطة الجودة",
        "stage_singular": "نقاط مفردة (نواة/دلتا)",
        "stage_ridges": "تعزيز Gabor",
        "stage_skeleton": "الهيكل",
        "stage_minutiae": "النقاط الدقيقة",
        "stats_binary": "بكسل أبيض (ثنائية)",
        "stats_gabor": "بكسل أبيض (Gabor)",
        "stats_skel": "بكسل أبيض (هيكل)",
        "stats_min": "نقاط بعد الفلترة",
    },
    "en": {
        "html_lang": "en",
        "dir": "ltr",
        "title": "Forensic Report — Fingerprint Comparison",
        "subtitle": "Statistical outputs for expert review",
        "generated_at": "Generated at",
        "disclaimer_title": "Laboratory disclaimer",
        "disclaimer": (
            "This report provides statistical software outputs and must not be used as sole "
            "identity evidence. Qualified expert review and lab SOP are required."
        ),
        "audit_title": "Audit trail",
        "case_ref": "Case reference",
        "operator": "Operator / reviewer",
        "sha_ref": "SHA-256 (reference)",
        "sha_query": "SHA-256 (query)",
        "methodology_title": "Methodology summary",
        "methodology": (
            "Grayscale conversion, normalization, CLAHE, denoising, binarization, "
            "multi-orientation Gabor enhancement, thinning, extended minutiae extraction, "
            "partial alignment (translation, rotation, scale), MCC verification, and fused decision."
        ),
        "params_title": "Run parameters",
        "param_col": "Parameter",
        "value_col": "Value",
        "pipeline_title": "Processing pipeline",
        "pipeline_ref": "Reference pipeline",
        "pipeline_query": "Query pipeline",
        "match_overlay": "Aligned match overlay",
        "orb_title": "ORB verification (optional)",
        "skeletons_title": "Final matching skeletons",
        "sk_ref": "Reference skeleton",
        "sk_query": "Query skeleton",
        "archive_note": "Archive files",
        "alignment_title": "Partial-to-reference alignment",
        "alignment_hint": "Green: reference matches | Orange: aligned query matches",
        "results_title": "Statistical comparison results",
        "n_ref": "Reference minutiae count",
        "n_query": "Query minutiae count",
        "n_matched": "One-to-one matched pairs",
        "match_score": "Match score (query-based)",
        "fused_score": "Fused score",
        "combined_verdict": "Combined verdict",
        "decision_mode": "Decision mode",
        "fusion_components": "Fusion components",
        "fusion_min": "Minutiae",
        "fusion_mcc": "MCC",
        "fusion_orb": "ORB",
        "dice": "Dice coefficient",
        "tier": "Forensic tier",
        "score_expl": "Score explanation",
        "internal_status": "Internal status",
        "analysis_mode": "Analysis mode",
        "deep": "Deep (wide auto-alignment + full pipeline)",
        "fast": "Fast",
        "stage_processed": "Preprocessed binary",
        "stage_quality": "Quality heatmap",
        "stage_singular": "Singular points",
        "stage_ridges": "Gabor ridges",
        "stage_skeleton": "Skeleton",
        "stage_minutiae": "Minutiae visualization",
        "stats_binary": "Binary white pixels",
        "stats_gabor": "Gabor white pixels",
        "stats_skel": "Skeleton white pixels",
        "stats_min": "Filtered minutiae",
    },
}

TIER_BY_STATUS = {
    "ar": {
        "HIGH MATCH": "المستوى A — تشابه إحصائي عالٍ وفق العتبات",
        "MEDIUM MATCH": "المستوى B — تشابه متوسط — مراجعة خبير",
        "LOW MATCH": "المستوى C — تشابه ضعيف / غير حاسم",
        "NO MATCH": "المستوى D — دون عتبة التشابه",
        "INCONCLUSIVE": "غير حاسم — جودة أو نقاط غير كافية",
        "ERROR": "خطأ في المطابقة",
    },
    "en": {
        "HIGH MATCH": "Tier A — High statistical similarity",
        "MEDIUM MATCH": "Tier B — Medium similarity — expert review",
        "LOW MATCH": "Tier C — Weak or inconclusive similarity",
        "NO MATCH": "Tier D — Below similarity threshold",
        "INCONCLUSIVE": "Inconclusive — quality/minutiae gate",
        "ERROR": "Error — matching failed",
    },
}

VERDICT_BY_STATUS = {
    "ar": {
        "HIGH MATCH": "تطابق مؤكد مع البصمة المرجعية (ثقة عالية)",
        "MEDIUM MATCH": "تطابق مرجّح — مراجعة خبير موصى بها",
        "LOW MATCH": "تشابه ضعيف أو غير حاسم",
        "NO MATCH": "لا يوجد تطابق كافٍ",
        "INCONCLUSIVE": "غير حاسم — جودة غير كافية",
        "ERROR": "خطأ في المطابقة",
    },
    "en": {
        "HIGH MATCH": "Forensic match confirmed (very high confidence)",
        "MEDIUM MATCH": "Probable match — expert review recommended",
        "LOW MATCH": "Weak or inconclusive similarity",
        "NO MATCH": "Insufficient similarity",
        "INCONCLUSIVE": "Inconclusive — quality gate",
        "ERROR": "Matching error",
    },
}


def report_lang(code: str | None) -> str:
    c = (code or "ar").strip().lower()
    return c if c in REPORT_STRINGS else "ar"


def s(lang: str, key: str) -> str:
    return REPORT_STRINGS[report_lang(lang)].get(key, key)


def tier_text(lang: str, status: str) -> str:
    return TIER_BY_STATUS.get(report_lang(lang), {}).get(status, status or "—")


def verdict_text(lang: str, match_result: dict) -> str:
    st = str(match_result.get("decision_status") or match_result.get("status") or "")
    m = VERDICT_BY_STATUS.get(report_lang(lang), {})
    if st in m:
        return m[st]
    return str(match_result.get("combined_verdict") or st or "—")
