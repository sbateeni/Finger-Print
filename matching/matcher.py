import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from features.minutiae_extraction import analyze_ridge_characteristics
import cv2

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
    """المرحلة الخامسة: مطابقة النقاط الدقيقة"""
    try:
        # تحويل النقاط إلى مصفوفات
        original_points = np.array([[m['x'], m['y']] for m in original_minutiae])
        partial_points = np.array([[m['x'], m['y']] for m in partial_minutiae])
        
        # حساب مصفوفة المسافات
        distances = cdist(original_points, partial_points)
        
        # تطبيق خوارزمية المطابقة
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # تجميع النقاط المتطابقة
        matched_points = []
        for i, j in zip(row_ind, col_ind):
            if distances[i, j] < 10:  # حد المسافة المسموح به
                matched_points.append({
                    'original': original_minutiae[i],
                    'partial': partial_minutiae[j],
                    'distance': distances[i, j]
                })
        
        # حساب نسبة التطابق
        match_score = calculate_similarity_score({
            'matched_points': len(matched_points),
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae)
        })
        
        # تحديد الحالة
        status = "HIGH MATCH" if match_score > 75 else \
                 "MEDIUM MATCH" if match_score > 50 else \
                 "LOW MATCH" if match_score > 25 else \
                 "NO MATCH"
        
        # تحليل الخطوط بين النقاط المتطابقة
        ridge_analysis = []
        for match in matched_points:
            ridge_analysis.append({
                'distance': match['distance'],
                'angle_difference': calculate_angle_difference(
                    match['original']['angle'],
                    match['partial']['angle']
                ),
                'type_match': match['original']['type'] == match['partial']['type']
            })
        
        return {
            'matched_points': len(matched_points),
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': match_score,
            'status': status,
            'details': {
                'ridge_analysis': ridge_analysis
            }
        }
    except Exception as e:
        print(f"Error in match_fingerprints: {str(e)}")
        return {
            'matched_points': 0,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': 0,
            'status': "ERROR",
            'details': {
                'ridge_analysis': []
            }
        }

def calculate_similarity_score(match_result):
    """المرحلة الثامنة: حساب نسبة التطابق"""
    matched = match_result['matched_points']
    total_original = match_result['total_original']
    total_partial = match_result['total_partial']
    
    # حساب نسبة التطابق
    if total_partial == 0:
        return 0
    
    # حساب نسبة النقاط المتطابقة
    match_ratio = matched / total_partial
    
    # حساب نسبة التغطية
    coverage_ratio = matched / total_original
    
    # حساب النتيجة النهائية
    similarity_score = (match_ratio + coverage_ratio) / 2 * 100
    
    return similarity_score

def normalize_rotation(image, orientation_map):
    """المرحلة السابعة: معالجة التدوير والميل"""
    # حساب متوسط اتجاه الخطوط
    mean_orientation = np.mean(orientation_map)
    
    # حساب زاوية التدوير المطلوبة
    rotation_angle = -mean_orientation * 180 / np.pi
    
    # تدوير الصورة
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

def verify_pattern_match(original_pattern, partial_pattern):
    """التحقق من تطابق الأنماط"""
    # التحقق من تطابق الأنماط الأساسية
    pattern_match = False
    
    # التحقق من تطابق الأنماط
    if original_pattern['loop'] and partial_pattern['loop']:
        pattern_match = True
    elif original_pattern['arch'] and partial_pattern['arch']:
        pattern_match = True
    elif original_pattern['whorl'] and partial_pattern['whorl']:
        pattern_match = True
    
    return pattern_match

def analyze_ridge_patterns(matched_points):
    """تحليل أنماط الخطوط بين النقاط المتطابقة"""
    ridge_analysis = []
    
    for match in matched_points:
        # حساب المسافة بين النقاط
        distance = np.sqrt(
            (match['original']['x'] - match['partial']['x'])**2 +
            (match['original']['y'] - match['partial']['y'])**2
        )
        
        # حساب الفرق في الزاوية
        angle_diff = abs(match['original']['angle'] - match['partial']['angle'])
        
        # التحقق من تطابق النوع
        type_match = match['original']['type'] == match['partial']['type']
        
        ridge_analysis.append({
            'distance': distance,
            'angle_difference': angle_diff,
            'type_match': type_match
        })
    
    return ridge_analysis 