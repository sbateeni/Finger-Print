"""
ORB Feature Matcher — طبقة تحقق بصرية ثانية.

تعمل بجانب مطابقة النقاط الدقيقة (Minutiae) وتعطي نتيجة مستقلة
بناءً على مميزات بصرية عامة (Keypoints + Descriptors).
مفيدة خاصةً عندما تكون نقاط الدقيقة قليلة أو بجودة منخفضة.
"""

import cv2
import numpy as np
from config import (
    ORB_N_FEATURES, ORB_THRESHOLD_HIGH_COUNT, ORB_THRESHOLD_HIGH_SCORE,
    ORB_THRESHOLD_MEDIUM_COUNT, ORB_THRESHOLD_MEDIUM_SCORE,
    MCC_THRESHOLD_HIGH, MCC_THRESHOLD_MEDIUM,
    MATCH_SCORE_THRESHOLDS,
    FUSION_W_MINUTIAE, FUSION_W_MCC, FUSION_W_ORB,
    PARTIAL_FUSION_W_MINUTIAE, PARTIAL_FUSION_W_MCC, PARTIAL_FUSION_W_ORB,
    FUSED_THRESHOLD_HIGH, FUSED_THRESHOLD_MEDIUM, FUSED_THRESHOLD_LOW,
    PARTIAL_FUSED_MEDIUM,
    PARTIAL_MCC_MEDIUM,
    PARTIAL_MCC_STRONG,
    PARTIAL_MCC_CONFIRM,
    PARTIAL_MATCHED_MEDIUM,
    PARTIAL_MATCHED_MIN,
    PARTIAL_GAIN_MEDIUM,
    MINUTIAE_MATCH_IGNORE_TYPES,
)


def match_with_orb(img1: np.ndarray, img2: np.ndarray, n_features: int = ORB_N_FEATURES) -> dict:
    """
    مطابقة بصرية باستخدام ORB (Oriented FAST and Rotated BRIEF).
    
    يُعيد:
        - orb_matches: عدد التطابقات الجيدة
        - orb_score: نسبة التطابق 0-100
        - orb_confidence: تصنيف نصي (HIGH / MEDIUM / LOW / INSUFFICIENT)
        - keypoints_img1 / keypoints_img2: عدد النقاط المكتشفة
        - visualization: صورة BGR للتطابقات (أو None عند الفشل)
    """
    result = {
        "orb_matches": 0,
        "orb_score": 0.0,
        "orb_confidence": "INSUFFICIENT",
        "keypoints_img1": 0,
        "keypoints_img2": 0,
        "visualization": None,
    }

    try:
        # تحويل لرمادي إذا لزم
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1.copy()
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2.copy()

        # إنشاء كاشف ORB
        orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
        )

        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)

        result["keypoints_img1"] = len(kp1)
        result["keypoints_img2"] = len(kp2)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            result["orb_confidence"] = "INSUFFICIENT"
            return result

        # مطابقة باستخدام BFMatcher + KNN
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des1, des2, k=2)

        # اختبار Lowe's ratio لفلترة التطابقات الجيدة
        good = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        n_good = len(good)
        n_ref = len(kp1)

        # حساب النسبة بالنسبة لنقاط الصورة المرجعية
        orb_score = (n_good / n_ref * 100.0) if n_ref > 0 else 0.0

        result["orb_matches"] = n_good
        result["orb_score"] = round(orb_score, 2)

        # تصنيف
        # تصنيف الثقة بناءً على عدد التطابقات ونسبتها
        if n_good >= ORB_THRESHOLD_HIGH_COUNT and orb_score >= ORB_THRESHOLD_HIGH_SCORE:
            result["orb_confidence"] = "HIGH"
        elif n_good >= ORB_THRESHOLD_MEDIUM_COUNT and orb_score >= ORB_THRESHOLD_MEDIUM_SCORE:
            result["orb_confidence"] = "MEDIUM"
        elif n_good >= 8:
            result["orb_confidence"] = "LOW"
        else:
            result["orb_confidence"] = "INSUFFICIENT"

        # رسم التطابقات
        try:
            h1, w1 = g1.shape
            h2, w2 = g2.shape
            vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            vis[:h1, :w1] = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
            vis[:h2, w1:w1+w2] = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)

            color_map = {"HIGH": (0, 255, 0), "MEDIUM": (0, 200, 255), "LOW": (0, 100, 255)}
            line_color = color_map.get(result["orb_confidence"], (128, 128, 128))

            for m in good[:60]:  # حد أقصى 60 خطاً للوضوح
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
                pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))
                cv2.line(vis, pt1, pt2, line_color, 1, cv2.LINE_AA)
                cv2.circle(vis, pt1, 3, (255, 255, 0), -1)
                cv2.circle(vis, pt2, 3, (255, 255, 0), -1)

            # نص الملخص
            cv2.putText(vis, f"ORB: {n_good} matches | {orb_score:.1f}% | {result['orb_confidence']}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            result["visualization"] = vis
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"ORB matching error: {e}")
        return result


def _clamp_pct(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _orb_conf_to_score(orb_confidence: str) -> float:
    if orb_confidence == "HIGH":
        return 80.0
    if orb_confidence == "MEDIUM":
        return 55.0
    if orb_confidence == "LOW":
        return 30.0
    return 0.0


def _is_partial_case(
    partial_verify: bool,
    total_original: int,
    total_partial: int,
    alignment_gain_matches: int,
    minutiae_score: float,
    mcc_score: float,
    matched_points: int,
) -> bool:
    """Detect partial-print comparison (not only by minutiae count ratio)."""
    if not partial_verify:
        return False
    if total_original > 0 and total_partial > 0:
        if float(total_partial) / float(total_original) <= 0.95:
            return True
    if int(alignment_gain_matches) >= PARTIAL_GAIN_MEDIUM:
        return True
    # Strong MCC + modest minutiae score + enough pairs → typical partial pattern
    if (
        float(mcc_score) >= PARTIAL_MCC_MEDIUM
        and float(minutiae_score) < MATCH_SCORE_THRESHOLDS.get("MEDIUM", 40)
        and int(matched_points) >= PARTIAL_MATCHED_MEDIUM
    ):
        return True
    return False


def _fuse_scores(
    min_norm: float,
    mcc_norm: float,
    orb_norm: float,
    landmark_norm: float,
    partial: bool,
    *,
    use_orb: bool = True,
) -> float:
    # Weights (can be moved to config)
    W_LANDMARK = 0.15 # 15% weight for anatomical landmarks
    
    if partial:
        w_m, w_c, w_o = PARTIAL_FUSION_W_MINUTIAE, PARTIAL_FUSION_W_MCC, PARTIAL_FUSION_W_ORB
    else:
        w_m, w_c, w_o = FUSION_W_MINUTIAE, FUSION_W_MCC, FUSION_W_ORB
    
    # Adjust for landmarks
    w_l = W_LANDMARK
    
    if not use_orb:
        w_o = 0.0
        w_sum = float(w_m + w_c + w_l) or 1.0
        return (min_norm * float(w_m) + mcc_norm * float(w_c) + landmark_norm * float(w_l)) / w_sum
    
    w_sum = float(w_m + w_c + w_o + w_l) or 1.0
    return (
        min_norm * float(w_m) + mcc_norm * float(w_c) + orb_norm * float(w_o) + landmark_norm * float(w_l)
    ) / w_sum


def combined_verdict(
    minutiae_score: float,
    orb_confidence: str,
    mcc_score: float = 0.0,
    orb_score: float = 0.0,
    landmark_score: float = 0.0,
    *,
    partial_verify: bool = False,
    matched_points: int = 0,
    alignment_gain_matches: int = 0,
    total_original: int = 0,
    total_partial: int = 0,
    use_orb: bool = True,
) -> dict:
    """
    يجمع نتيجتَي مطابقة النقاط الدقيقة، ORB، و MCC في حكم نهائي واحد.
    
    المنطق المطور:
    - MCC هو المعيار الذهبي؛ إذا كان مرتفعاً (>50) فهو مؤشر قوي جداً.
    - تقاطع النتائج الثلاث يقلل من الخطأ البشري والآلي.
    """
    # تصنيف MCC
    mcc_high = mcc_score >= MCC_THRESHOLD_HIGH
    mcc_med  = mcc_score >= MCC_THRESHOLD_MEDIUM
    
    # تصنيف ORB
    orb_high = orb_confidence == "HIGH"
    orb_med  = orb_confidence == "MEDIUM"
    
    # تصنيف النقاط الدقيقة باستخدام العتبات من الإعدادات
    min_high = minutiae_score >= MATCH_SCORE_THRESHOLDS.get('HIGH', 65)
    min_med  = minutiae_score >= MATCH_SCORE_THRESHOLDS.get('MEDIUM', 40)

    # حساب نقاط الثقة (Confidence Points) — إبقاؤها لأغراض التتبع الخلفي
    conf_points = 0
    if min_high: conf_points += 2
    elif min_med: conf_points += 1
    
    if orb_high: conf_points += 2
    elif orb_med: conf_points += 1
    
    if mcc_high: conf_points += 3 # وزن أعلى لـ MCC
    elif mcc_med: conf_points += 1.5

    min_norm = _clamp_pct(minutiae_score)
    mcc_norm = _clamp_pct(mcc_score)
    orb_norm = _clamp_pct(orb_score if orb_score > 0 else _orb_conf_to_score(orb_confidence))
    land_norm = _clamp_pct(landmark_score)

    is_partial = _is_partial_case(
        partial_verify,
        total_original,
        total_partial,
        alignment_gain_matches,
        minutiae_score,
        mcc_score,
        matched_points,
    )
    fused_score = _fuse_scores(min_norm, mcc_norm, orb_norm, land_norm, is_partial, use_orb=use_orb)

    # عتبات القرار — للجزئية نستخدم حدوداً مخصصة
    th_high = FUSED_THRESHOLD_HIGH
    th_medium = PARTIAL_FUSED_MEDIUM if is_partial else FUSED_THRESHOLD_MEDIUM
    th_low = FUSED_THRESHOLD_LOW

    if fused_score >= th_high:
        verdict = "تطابق جنائي مؤكد (ثقة عالية جداً)"
        color = "high"
        decision_status = "HIGH MATCH"
    elif fused_score >= th_medium:
        verdict = "تطابق احتمالي قوي (يحتاج مراجعة خبير)"
        color = "high"
        decision_status = "MEDIUM MATCH"
    elif fused_score >= th_low:
        verdict = "تطابق متوسط / غير حاسم"
        color = "medium"
        decision_status = "LOW MATCH"
    else:
        verdict = "لا يوجد تطابق كافٍ"
        color = "low"
        decision_status = "NO MATCH"

    # تأكيد نفس الإصبع: MCC قوي + تطابقات كافية (صور حبر / جزئية / فرق موضع)
    same_finger_strong = (
        mcc_norm >= PARTIAL_MCC_STRONG
        and int(matched_points) >= PARTIAL_MATCHED_MEDIUM
        and (
            int(alignment_gain_matches) >= PARTIAL_GAIN_MEDIUM
            or float(minutiae_score) >= 10.0
        )
    )
    same_finger_confirm = (
        mcc_norm >= PARTIAL_MCC_CONFIRM
        and int(matched_points) >= PARTIAL_MATCHED_MIN
        and int(alignment_gain_matches) >= 2
    )

    if same_finger_confirm and decision_status in ("NO MATCH", "LOW MATCH", "MEDIUM MATCH"):
        verdict = "تطابق مؤكد مع البصمة المرجعية (MCC عالي جداً)"
        color = "high"
        decision_status = "HIGH MATCH"
    elif same_finger_strong and decision_status in ("NO MATCH", "LOW MATCH"):
        verdict = "تطابق مرجّح مع البصمة المرجعية (نفس الإصبع — مراجعة خبير موصى بها)"
        color = "high" if mcc_norm >= PARTIAL_MCC_CONFIRM else "medium"
        decision_status = "MEDIUM MATCH"

    # Partial-first upgrade: MCC + alignment (عتبات أوسع)
    partial_evidence = is_partial and mcc_norm >= PARTIAL_MCC_MEDIUM and (
        (
            int(matched_points) >= PARTIAL_MATCHED_MEDIUM
            and int(alignment_gain_matches) >= PARTIAL_GAIN_MEDIUM
        )
        or same_finger_confirm
    )
    if partial_evidence and decision_status in ("NO MATCH", "LOW MATCH"):
        if fused_score >= PARTIAL_FUSED_MEDIUM or same_finger_strong:
            verdict = "تطابق جزئي مرجّح (يدعم مراجعة خبير)"
            color = "high" if mcc_norm >= PARTIAL_MCC_STRONG else "medium"
            decision_status = "MEDIUM MATCH"
        elif decision_status == "NO MATCH":
            verdict = "تطابق جزئي ضعيف (مراجعة خبير موصى بها)"
            color = "medium"
            decision_status = "LOW MATCH"

    fusion_weights = (
        {
            "minutiae": PARTIAL_FUSION_W_MINUTIAE,
            "mcc": PARTIAL_FUSION_W_MCC,
            "orb": PARTIAL_FUSION_W_ORB,
            "landmark": 0.15,
        }
        if is_partial
        else {
            "minutiae": FUSION_W_MINUTIAE,
            "mcc": FUSION_W_MCC,
            "orb": FUSION_W_ORB,
            "landmark": 0.15,
        }
    )

    return {
        "combined_verdict": verdict,
        "combined_color": color,
        "decision_status": decision_status,
        "decision_mode": "partial-first" if is_partial else "fused-default",
        "is_partial_case": is_partial,
        "fusion_weights_used": fusion_weights,
        "confidence_level": conf_points,
        "fused_score": round(float(fused_score), 2),
        "fusion_components": {
            "minutiae_score": round(min_norm, 2),
            "mcc_score": round(mcc_norm, 2),
            "orb_score": round(orb_norm, 2),
            "landmark_score": round(land_norm, 2),
        },
    }
