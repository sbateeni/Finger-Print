import numpy as np
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """حساب المسافة بين نقطتين"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """حساب الفرق بين زاويتين"""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def match_fingerprints(original_minutiae: List[Dict[str, Any]], 
                      partial_minutiae: List[Dict[str, Any]], 
                      scale_factor: float = 1.0) -> Dict[str, Any]:
    """مطابقة البصمات وإرجاع النتائج"""
    try:
        # تحويل النقاط إلى مصفوفات numpy للعمليات الحسابية
        original_points = np.array([[m['x'], m['y']] for m in original_minutiae])
        partial_points = np.array([[m['x'], m['y']] for m in partial_minutiae])
        
        # تطبيق عامل التحجيم
        partial_points = partial_points / scale_factor
        
        # إعداد المتغيرات
        matched_points = 0
        match_regions = []
        ridge_analysis = []
        
        # مطابقة النقاط
        for i, orig_point in enumerate(original_points):
            best_match = None
            best_score = 0
            best_distance = float('inf')
            
            for j, part_point in enumerate(partial_points):
                # حساب المسافة بين النقطتين
                distance = calculate_distance(orig_point, part_point)
                
                # حساب الفرق في الزاوية
                angle_diff = calculate_angle_difference(
                    original_minutiae[i]['angle'],
                    partial_minutiae[j]['angle']
                )
                
                # حساب درجة التطابق
                distance_score = 1 - min(distance / 20, 1)  # 20 بكسل كحد أقصى للمسافة
                angle_score = 1 - min(angle_diff / 45, 1)  # 45 درجة كحد أقصى للفرق في الزاوية
                type_score = 1 if original_minutiae[i]['type'] == partial_minutiae[j]['type'] else 0
                
                # حساب النتيجة النهائية
                score = (distance_score * 0.4 + angle_score * 0.4 + type_score * 0.2) * 100
                
                if score > best_score:
                    best_score = score
                    best_match = j
                    best_distance = distance
            
            if best_match is not None and best_score > 50:  # حد أدنى للتطابق
                matched_points += 1
                
                # إضافة تحليل الخط
                ridge_analysis.append({
                    'distance': best_distance,
                    'angle_difference': calculate_angle_difference(
                        original_minutiae[i]['angle'],
                        partial_minutiae[best_match]['angle']
                    ),
                    'type_match': original_minutiae[i]['type'] == partial_minutiae[best_match]['type']
                })
                
                # إضافة منطقة التطابق
                match_regions.append({
                    'box': [
                        (original_minutiae[i]['x'] - 20, original_minutiae[i]['y'] - 20),
                        (original_minutiae[i]['x'] + 20, original_minutiae[i]['y'] + 20)
                    ],
                    'score': best_score
                })
        
        # حساب نسبة التطابق النهائية
        total_points = min(len(original_minutiae), len(partial_minutiae))
        match_score = (matched_points / total_points) * 100 if total_points > 0 else 0
        
        # تحديد حالة المطابقة
        status = "HIGH MATCH" if match_score > 75 else "MEDIUM MATCH" if match_score > 50 else "LOW MATCH"
        
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
        logger.error(f"Error in matching fingerprints: {str(e)}")
        return {
            'match_score': 0,
            'matched_points': 0,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'status': "ERROR",
            'details': {
                'ridge_analysis': [],
                'match_regions': []
            }
        } 