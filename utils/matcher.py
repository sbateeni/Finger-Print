from collections import defaultdict
import cv2
import numpy as np
from config import *
from utils.mcc import compute_mcc_descriptors, match_mcc

def _angle_diff_deg(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)

def _match_minutiae_pair(original_minutiae, partial_minutiae, distance_threshold, angle_threshold_deg, angle_sort_weight=1.0):
    """
    مطابقة واحد-لواحد باستخدام البحث الشبكي السريع.
    يعيد تفاصيل التطابق والدرجة الكلية.
    """
    if not original_minutiae or not partial_minutiae:
        return _empty_result(original_minutiae, partial_minutiae)

    dth = float(distance_threshold)
    cell = max(4, int(dth))
    neighbor = 1 # بما أننا نستخدم dth كعتبة، الجيران المباشرين كافون إذا كانت الخلية >= dth

    grid = defaultdict(list)
    for i, mo in enumerate(original_minutiae):
        ix, iy = int(mo["x"]) // cell, int(mo["y"]) // cell
        grid[(ix, iy)].append((i, mo))

    candidates = []
    for j, mp in enumerate(partial_minutiae):
        px, py = float(mp["x"]), float(mp["y"])
        icx, icy = int(px) // cell, int(py) // cell
        for dx in range(-neighbor, neighbor + 1):
            for dy in range(-neighbor, neighbor + 1):
                for i, mo in grid.get((icx + dx, icy + dy), ()):
                    # التحقق من النوع (اختياري لكنه يزيد الدقة)
                    if mo.get("type") != mp.get("type"):
                        continue
                    
                    dist = np.hypot(mo["x"] - px, mo["y"] - py)
                    if dist < dth:
                        ad = _angle_diff_deg(mo.get("angle", 0), mp.get("angle", 0))
                        if ad < angle_threshold_deg:
                            # السجل هو مزيج من المسافة وفارق الزاوية
                            score = dist + angle_sort_weight * (ad / 10.0)
                            candidates.append((score, dist, ad, i, j, mo, mp))

    # تعيين النقاط (أفضل تطابق أولاً)
    candidates.sort(key=lambda t: t[0])
    used_o, used_p = set(), set()
    matched_points = []
    
    for _, dist, ad, i, j, mo, mp in candidates:
        if i in used_o or j in used_p: continue
        used_o.add(i)
        used_p.add(j)
        matched_points.append({"original": mo, "partial": mp, "distance": dist, "angle_diff": ad})

    n_match = len(matched_points)
    n_o, n_p = len(original_minutiae), len(partial_minutiae)
    
    # حساب النقاط (Score)
    # نستخدم نسبة التطابق بالنسبة للجزئية (Sensitivity) وبالنسبة للمتوسط (Dice)
    match_score = (n_match / n_p * 100.0) if n_p > 0 else 0.0
    dice_score = (2.0 * n_match / (n_o + n_p) * 100.0) if (n_o + n_p) > 0 else 0.0
    
    # تحديد الحالة بناءً على العتبات
    status = "NO MATCH"
    for tier, threshold in sorted(MATCH_SCORE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
        if match_score >= threshold:
            status = f"{tier} MATCH"
            break

    return {
        "matched_points": n_match,
        "total_original": n_o,
        "total_partial": n_p,
        "match_score": match_score,
        "dice_score": dice_score,
        "status": status,
        "matched_details": matched_points
    }

def _transform_points(points, dx, dy, rot_deg, cx, cy):
    """تحويل النقاط (دوران + إزاحة)"""
    rad = np.radians(rot_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    transformed = []
    for p in points:
        x, y = p['x'] - cx, p['y'] - cy
        nx = x * cos_t - y * sin_t + cx + dx
        ny = x * sin_t + y * cos_t + cy + dy
        transformed.append({**p, 'x': nx, 'y': ny, 'angle': p.get('angle', 0) + rot_deg})
    return transformed

def match_fingerprints_with_partial_alignment(original_minutiae, partial_minutiae, image_shape, **kwargs):
    """
    مطابقة متقدمة باستخدام بحث Coarse-to-Fine (من الخشن إلى الناعم).
    يضمن اتساق baseline (قبل المحاذاة) مع النتيجة النهائية.
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    distance_threshold = kwargs.get("distance_threshold", MATCH_DISTANCE_THRESHOLD)
    angle_threshold = kwargs.get("angle_threshold_deg", MATCH_ANGLE_THRESHOLD_DEG)
    angle_sort_weight = kwargs.get("angle_sort_weight", MATCH_ANGLE_SORT_WEIGHT)
    step_requested = int(kwargs.get("step_px", PARTIAL_VERIFY_STEP_PX))
    search_radius = int(kwargs.get("search_radius", PARTIAL_VERIFY_SEARCH_RADIUS))
    rot_min = float(kwargs.get("rot_min", PARTIAL_VERIFY_ROT_MIN_DEG))
    rot_max = float(kwargs.get("rot_max", PARTIAL_VERIFY_ROT_MAX_DEG))

    baseline = _match_minutiae_pair(
        original_minutiae,
        partial_minutiae,
        distance_threshold,
        angle_threshold,
        angle_sort_weight,
    )
    best_res = dict(baseline)
    best_res["alignment"] = {"dx": 0, "dy": 0, "rot_deg": 0.0}

    # تكييف خطوة البحث عند كثافة نقاط كبيرة لتقليل زمن التنفيذ
    n_prod = len(original_minutiae) * len(partial_minutiae)
    step_effective = step_requested
    if n_prod > 400_000:
        step_effective = max(step_effective, 24)
    elif n_prod > 220_000:
        step_effective = max(step_effective, 18)

    coarse_step = max(9, step_effective)
    coarse_rot_step = float(kwargs.get("coarse_rot_step", 5.0))

    for rot in np.arange(rot_min, rot_max + 0.1, coarse_rot_step):
        for dx in range(-search_radius, search_radius + 1, coarse_step):
            for dy in range(-search_radius, search_radius + 1, coarse_step):
                transformed = _transform_points(partial_minutiae, dx, dy, rot, cx, cy)
                res = _match_minutiae_pair(
                    original_minutiae,
                    transformed,
                    distance_threshold,
                    angle_threshold,
                    angle_sort_weight,
                )
                if res["matched_points"] > best_res["matched_points"]:
                    best_res = dict(res)
                    best_res["alignment"] = {"dx": int(dx), "dy": int(dy), "rot_deg": float(rot)}

    # بحث ناعم حول أفضل قمة
    if best_res["alignment"]["dx"] != 0 or best_res["alignment"]["dy"] != 0:
        fine_r = coarse_step
        fine_step = 3
        fine_rot_r = coarse_rot_step
        fine_rot_step = 1.0
        base_al = best_res["alignment"]
        for rot in np.arange(base_al["rot_deg"] - fine_rot_r, base_al["rot_deg"] + fine_rot_r + 0.1, fine_rot_step):
            for dx in range(base_al["dx"] - fine_r, base_al["dx"] + fine_r + 1, fine_step):
                for dy in range(base_al["dy"] - fine_r, base_al["dy"] + fine_r + 1, fine_step):
                    transformed = _transform_points(partial_minutiae, dx, dy, rot, cx, cy)
                    res = _match_minutiae_pair(
                        original_minutiae,
                        transformed,
                        distance_threshold,
                        angle_threshold,
                        angle_sort_weight,
                    )
                    if res["matched_points"] > best_res["matched_points"]:
                        best_res = dict(res)
                        best_res["alignment"] = {"dx": int(dx), "dy": int(dy), "rot_deg": float(rot)}

    # حقول baseline/gain يجب أن تكون مبنية على baseline الحقيقي قبل أي محاذاة
    best_res["baseline_matched"] = int(baseline.get("matched_points", 0))
    best_res["baseline_match_score"] = float(baseline.get("match_score", 0.0))
    best_res["alignment_gain_matches"] = int(best_res.get("matched_points", 0)) - int(
        baseline.get("matched_points", 0)
    )
    best_res["alignment_gain_score"] = float(best_res.get("match_score", 0.0)) - float(
        baseline.get("match_score", 0.0)
    )
    best_res["partial_verify"] = True
    best_res["partial_verify_step_px_config"] = int(step_requested)
    best_res["partial_verify_step_px_effective"] = int(step_effective)

    # --- إضافة محرك MCC للتحقق الجوهري ---
    try:
        desc1 = compute_mcc_descriptors(original_minutiae)
        final_transformed = _transform_points(
            partial_minutiae,
            best_res["alignment"]["dx"],
            best_res["alignment"]["dy"],
            best_res["alignment"]["rot_deg"],
            cx,
            cy,
        )
        desc2 = compute_mcc_descriptors(final_transformed)
        mcc_score, mcc_matches = match_mcc(desc1, desc2)
        best_res["mcc_score"] = float(mcc_score) * 100.0
        best_res["mcc_matches"] = len(mcc_matches)
    except Exception as e:
        print(f"MCC Matching Error: {e}")
        best_res["mcc_score"] = 0.0
        best_res["mcc_matches"] = 0

    return best_res

def _empty_result(original_minutiae, partial_minutiae, status="NO MATCH"):
    return {
        "matched_points": 0,
        "total_original": len(original_minutiae) if original_minutiae else 0,
        "total_partial": len(partial_minutiae) if partial_minutiae else 0,
        "match_score": 0.0,
        "dice_score": 0.0,
        "status": status,
        "matched_details": []
    }

def visualize_matches(original_img, partial_img, match_result):
    """تصور جانبي للمطابقات"""
    try:
        h1, w1 = original_img.shape[:2]
        h2, w2 = partial_img.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        o_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape)==2 else original_img
        p_bgr = cv2.cvtColor(partial_img, cv2.COLOR_GRAY2BGR) if len(partial_img.shape)==2 else partial_img
        
        vis[:h1, :w1] = o_bgr
        vis[:h2, w1:w1+w2] = p_bgr
        
        for m in match_result.get("matched_details", []):
            pt1 = (int(m["original"]["x"]), int(m["original"]["y"]))
            pt2 = (int(m["partial"]["x"]) + w1, int(m["partial"]["y"]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 255, 0), -1)
        return vis
    except: return None

def visualize_alignment_on_reference(ref_img, match_result):
    """رسم النقاط المحاذية فوق المرجع مباشرة"""
    try:
        vis = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape)==2 else ref_img.copy()
        for m in match_result.get("matched_details", []):
            o = m["original"]
            p = m["partial"]
            cv2.circle(vis, (int(o["x"]), int(o["y"])), 4, (0, 255, 0), 1)
            cv2.circle(vis, (int(p["x"]), int(p["y"])), 2, (0, 0, 255), -1)
            cv2.line(vis, (int(o["x"]), int(o["y"])), (int(p["x"]), int(p["y"])), (255, 0, 255), 1)
        return vis
    except: return None
