import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from features.minutiae_extraction import analyze_ridge_characteristics
import logging
import traceback

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def calculate_angle_difference(angle1, angle2):
    """Calculate the minimum angle difference between two angles"""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def calculate_feature_similarity(f1, f2):
    """Calculate similarity between two minutiae features"""
    # Spatial distance
    spatial_dist = calculate_distance(f1, f2)
    
    # Angle difference
    angle_diff = calculate_angle_difference(f1['angle'], f2['angle'])
    
    # Type matching
    type_match = 1 if f1['type'] == f2['type'] else 0
    
    # Magnitude similarity
    magnitude_diff = abs(f1['magnitude'] - f2['magnitude'])
    
    # Combined similarity score (lower is better)
    return spatial_dist + 0.5 * angle_diff + (1 - type_match) * 10 + magnitude_diff

def match_fingerprints(original_minutiae, partial_minutiae):
    """
    مطابقة البصمات باستخدام النقاط المميزة
    """
    try:
        if not original_minutiae or not partial_minutiae:
            return {
                'match_score': 0,
                'matched_points': 0,
                'total_original': len(original_minutiae) if original_minutiae else 0,
                'total_partial': len(partial_minutiae) if partial_minutiae else 0,
                'status': 'فشل المطابقة - لا توجد نقاط مميزة',
                'details': {
                    'ridge_analysis': [],
                    'match_regions': []
                }
            }

        # تحليل الخطوط
        ridge_analysis = []
        for orig_ridge in original_minutiae:
            for part_ridge in partial_minutiae:
                # حساب المسافة بين الخطوط
                distance = np.linalg.norm(orig_ridge['position'] - part_ridge['position'])
                # حساب الفرق في الزاوية
                angle_diff = abs(orig_ridge['angle'] - part_ridge['angle'])
                # التحقق من تطابق النوع
                type_match = orig_ridge['type'] == part_ridge['type']
                
                ridge_analysis.append({
                    'distance': distance,
                    'angle_difference': angle_diff,
                    'type_match': type_match
                })

        # تحديد المناطق المتطابقة
        match_regions = []
        matched_points = 0
        total_score = 0

        for i, orig_ridge in enumerate(original_minutiae):
            best_match = None
            best_score = 0
            
            for j, part_ridge in enumerate(partial_minutiae):
                # حساب درجة التطابق
                distance = np.linalg.norm(orig_ridge['position'] - part_ridge['position'])
                angle_diff = abs(orig_ridge['angle'] - part_ridge['angle'])
                type_match = orig_ridge['type'] == part_ridge['type']
                
                # حساب النتيجة
                score = 0
                if distance < 10:  # المسافة القصوى المسموح بها
                    score += 1 - (distance / 10)
                if angle_diff < 30:  # الفرق الأقصى في الزاوية
                    score += 1 - (angle_diff / 30)
                if type_match:
                    score += 1
                
                score = score / 3  # تحويل النتيجة إلى نسبة مئوية
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'box': [
                            int(orig_ridge['position'][0] - 10),
                            int(orig_ridge['position'][1] - 10),
                            20,  # عرض المربع
                            20   # ارتفاع المربع
                        ],
                        'score': score
                    }
            
            if best_match and best_score > 0.5:  # عتبة التطابق
                matched_points += 1
                total_score += best_score
                match_regions.append(best_match)

        # حساب نسبة التطابق النهائية
        match_score = (total_score / len(partial_minutiae)) * 100 if partial_minutiae else 0
        
        # تحديد حالة المطابقة
        if match_score > 75:
            status = "تطابق عالي"
        elif match_score > 50:
            status = "تطابق متوسط"
        else:
            status = "تطابق منخفض"

        return {
            'match_score': match_score,
            'matched_points': matched_points,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'status': status,
            'details': {
                'ridge_analysis': ridge_analysis,
                'match_regions': match_regions
            }
        }

    except Exception as e:
        logging.error(f"Error in match_fingerprints: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            'match_score': 0,
            'matched_points': 0,
            'total_original': len(original_minutiae) if original_minutiae else 0,
            'total_partial': len(partial_minutiae) if partial_minutiae else 0,
            'status': 'فشل المطابقة - خطأ في المعالجة',
            'details': {
                'ridge_analysis': [],
                'match_regions': []
            }
        } 