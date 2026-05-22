"""
مساعدات للاستخدام المخبري: صياغة حيادية، تتبع تدقيق، وثوابت توثيق.
لا يُعدّ هذا إقرارًا بمطابقة معيار دولي معيّن؛ المختبر يلزمه إطار إجرائي وخبير معتمد.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from config import APP_VERSION, AUDIT_LOG_PATH, MATCH_SCORE_THRESHOLDS, SOFTWARE_NAME
from utils.database import save_audit_record

FORENSIC_STANDARD_NOTE = (
    "النظام يُخرج مقاييسًا إحصائية لمقارنة تمثيلات رقمية للبصمات وفق إعدادات مسجّلة؛ "
    "لا يثبت الهوية وحده ولا يغني عن مراجعة خبير بصمات معتمد وسياسة المختبر والسلطة المختصة."
)

# نص يُعرض تحت عنوان «المنهجية التقنية (ملخص)» في التقرير/الواجهة — دون تكرار العنوان في المحتوى
METHODOLOGY_AR = (
    "تحويل رمادي، تموضع الحجم، CLAHE، إزالة ضوضاء اختيارية، "
    "تثنّي Otsu، تعزيز تموجات ببنك Gabor متعدد الاتجاهات، تهليل (thinning)، "
    "استخراج نقاط دقيقة (نهاية/تفرع) مع فلترة، ثم مطابقة واحد‑لواحد مع قيود مسافة وزاوية ونوع النقطة. "
    "للمقارنة مع جزء مقصوص: يُبحث عن أفضل إزاحة ودوران خفيف للنقاط الجزئية داخل إطار المرجعية (تقريبي، "
    "ليس تحويلًا شكليًا كاملاً بين صورتين بمقياسين مختلفين)."
)

FORENSIC_TIER_AR: dict[str, str] = {
    "HIGH MATCH": "الفئة أ — درجة تشابه إحصائية مرتفعة وفق عتبات النظام المضبوطة",
    "MEDIUM MATCH": "الفئة ب — درجة تشابه إحصائية متوسطة وفق عتبات النظام المضبوطة",
    "LOW MATCH": "الفئة ج — درجة تشابه إحصائية منخفضة وفق عتبات النظام المضبوطة",
    "NO MATCH": "الفئة د — دون بلوغ عتبة التشابه المسجّلة",
    "INCONCLUSIVE": "غير حاسم — جودة/كمية الأدلة غير كافية لحكم موثوق",
    "ERROR": "خطأ في حساب المطابقة",
}


def enrich_match_for_forensics(match_result: dict[str, Any]) -> dict[str, Any]:
    """يضيف حقولًا للعرض والتقرير دون تغيير المنطق العددي."""
    st = match_result.get("status") or "ERROR"
    match_result = {**match_result}
    match_result["forensic_tier_ar"] = FORENSIC_TIER_AR.get(st, FORENSIC_TIER_AR["ERROR"])
    match_result["forensic_standard_note"] = FORENSIC_STANDARD_NOTE
    match_result["methodology_ar"] = METHODOLOGY_AR
    baseline_consistency_warning = ""
    b_pts = int(match_result.get("baseline_matched") or 0)
    b_score = float(match_result.get("baseline_match_score") or 0.0)
    if (b_pts == 0 and b_score > 0.01) or (b_pts > 0 and b_score <= 0.0):
        baseline_consistency_warning = (
            "تنبيه اتساق: بيانات baseline قبل المحاذاة غير متناسقة (عدد النقاط/النسبة). "
            "يرجى مراجعة إعدادات المطابقة أو نسخة المحرك."
        )

    if match_result.get("partial_verify") and match_result.get("alignment"):
        al = match_result["alignment"]
        gain = int(match_result.get("alignment_gain_matches") or 0)
        gain_score = float(match_result.get("alignment_gain_score") or 0.0)
        base = int(match_result.get("baseline_matched") or 0)
        match_result["alignment_summary_ar"] = (
            f"محاذاة الجزء داخل المرجعية (تقريبية): إزاحة dx={al.get('dx')} dy={al.get('dy')} "
            f"، دوران {al.get('rot_deg', 0):.1f}°. "
            f"التطابقات قبل المحاذاة: {base} — بعد أفضل محاذاة: {match_result.get('matched_points')} "
            f"(زيادة {gain})، وفارق نسبة التشابه: {gain_score:+.2f} نقطة."
        )
    else:
        match_result["alignment_summary_ar"] = ""

    mp = int(match_result.get("matched_points") or 0)
    tp = int(match_result.get("total_partial") or 0)
    th = MATCH_SCORE_THRESHOLDS
    if tp > 0:
        expl = (
            f"تفسير نسبة التشابه: {mp} ÷ {tp} = عدد الأزواج المتطابقة مقسوماً على إجمالي نقاط المقارنة المستخرجة؛ "
            f"ليست «نسبة تغطية الجزء». عتبات التصنيف الحالية: عالية {th['HIGH']}٪، متوسطة {th['MEDIUM']}٪، دنيا {th['LOW']}٪."
        )
    else:
        expl = ""
    step_cfg = match_result.get("partial_verify_step_px_config")
    step_eff = match_result.get("partial_verify_step_px_effective")
    if (
        expl
        and step_cfg is not None
        and step_eff is not None
        and int(step_eff) != int(step_cfg)
    ):
        expl += (
            f" خطوة شبكة محاذاة الجزء: القيمة المرجعية {int(step_cfg)}، واستُخدم فعلياً {int(step_eff)} "
            "لتخفيف الحمل عند كثافة النقاط."
        )
    if baseline_consistency_warning:
        expl = (expl + " " if expl else "") + baseline_consistency_warning
    if match_result.get("fused_score") is not None:
        fs = float(match_result.get("fused_score") or 0.0)
        mode = match_result.get("decision_mode") or "fused-default"
        expl = (expl + " " if expl else "") + (
            f"القرار النهائي مبني على درجة دمج القنوات (Fused Score) = {fs:.2f}% "
            f"من مزج Minutiae + MCC + ORB (وضع: {mode})."
        )
        if match_result.get("combined_verdict"):
            expl += f" الحكم المدمج: {match_result.get('combined_verdict')}."
    match_result["score_explanation_ar"] = expl
    match_result["baseline_consistency_warning"] = baseline_consistency_warning
    return match_result


def append_audit_record(record: dict[str, Any]) -> None:
    """سجل تدقيق append-only (JSON Lines) داخل مجلد المخرجات."""
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH) or ".", exist_ok=True)
        record = {
            **record,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "software": SOFTWARE_NAME,
            "version": APP_VERSION,
        }
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # حفظ في قاعدة البيانات أيضاً
        save_audit_record(record)
    except OSError as e:
        print(f"forensic audit log failed: {e}")


def build_audit_record(
    *,
    sha256_original: str,
    sha256_partial: str,
    operator_name: str,
    case_reference: str,
    form_params: dict[str, Any],
    match_result: dict[str, Any],
    report_filename: str | None,
) -> dict[str, Any]:
    return {
        "sha256_original": sha256_original,
        "sha256_partial": sha256_partial,
        "operator_name": operator_name.strip() or None,
        "case_reference": case_reference.strip() or None,
        "params": form_params,
        "matched_points": match_result.get("matched_points"),
        "total_original": match_result.get("total_original"),
        "total_partial": match_result.get("total_partial"),
        "match_score": match_result.get("match_score"),
        "dice_score": match_result.get("dice_score"),
        "fused_score": match_result.get("fused_score"),
        "mcc_score": match_result.get("mcc_score"),
        "mcc_matches": match_result.get("mcc_matches"),
        "orb_score": match_result.get("orb_score"),
        "orb_matches": match_result.get("orb_matches"),
        "orb_confidence": match_result.get("orb_confidence"),
        "fusion_components": match_result.get("fusion_components"),
        "quality_gate_failed": match_result.get("quality_gate_failed"),
        "quality_gate_reason": match_result.get("quality_gate_reason"),
        "status": match_result.get("status"),
        "alignment": match_result.get("alignment"),
        "baseline_matched": match_result.get("baseline_matched"),
        "alignment_gain_matches": match_result.get("alignment_gain_matches"),
        "report_filename": report_filename,
    }
