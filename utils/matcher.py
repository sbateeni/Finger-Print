import cv2
import numpy as np
from config import *

def match_fingerprints(original_minutiae, partial_minutiae):
    """مطابقة النقاط الدقيقة"""
    try:
        # تحويل النقاط إلى مصفوفات
        original_points = np.array([[m['x'], m['y']] for m in original_minutiae])
        partial_points = np.array([[m['x'], m['y']] for m in partial_minutiae])
        
        # حساب مصفوفة المسافات
        distances = np.zeros((len(original_points), len(partial_points)))
        for i, p1 in enumerate(original_points):
            for j, p2 in enumerate(partial_points):
                distances[i, j] = np.sqrt(np.sum((p1 - p2) ** 2))
        
        # تطبيق خوارزمية المطابقة
        matched_points = []
        for i in range(len(original_points)):
            min_dist = np.min(distances[i])
            min_idx = np.argmin(distances[i])
            if min_dist < MATCH_DISTANCE_THRESHOLD:
                matched_points.append({
                    'original': original_minutiae[i],
                    'partial': partial_minutiae[min_idx],
                    'distance': min_dist
                })
        
        # حساب نسبة التطابق
        match_score = len(matched_points) / len(partial_minutiae) * 100 if partial_minutiae else 0
        
        # تحديد الحالة
        status = "HIGH MATCH" if match_score > MATCH_SCORE_THRESHOLDS['HIGH'] else \
                 "MEDIUM MATCH" if match_score > MATCH_SCORE_THRESHOLDS['MEDIUM'] else \
                 "LOW MATCH" if match_score > MATCH_SCORE_THRESHOLDS['LOW'] else \
                 "NO MATCH"
        
        return {
            'matched_points': len(matched_points),
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': match_score,
            'status': status,
            'matched_details': matched_points
        }
    except Exception as e:
        print(f"Error in match_fingerprints: {str(e)}")
        return {
            'matched_points': 0,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': 0,
            'status': "ERROR",
            'matched_details': []
        }

def visualize_matches(original_img, partial_img, match_result):
    """تصور النقاط المتطابقة"""
    try:
        # إنشاء صورة التصور
        vis_img = np.hstack((original_img, partial_img))
        
        # رسم النقاط المتطابقة
        for match in match_result['matched_details']:
            orig_point = match['original']
            part_point = match['partial']
            
            # رسم النقاط المتطابقة
            cv2.circle(vis_img, (orig_point['x'], orig_point['y']), 3, (0, 255, 0), -1)
            cv2.circle(vis_img, (part_point['x'] + original_img.shape[1], part_point['y']), 3, (0, 255, 0), -1)
            
            # رسم خط التطابق
            cv2.line(vis_img,
                    (orig_point['x'], orig_point['y']),
                    (part_point['x'] + original_img.shape[1], part_point['y']),
                    (0, 255, 0), 1)
        
        return vis_img
    except Exception as e:
        print(f"Error in visualize_matches: {str(e)}")
        return None 