import cv2
import numpy as np
from .feature_extraction import extract_features

def match_fingerprints(minutiae1, minutiae2, features1=None, features2=None):
    """مطابقة البصمات باستخدام نقاط التفاصيل والخصائص"""
    try:
        if not minutiae1 or not minutiae2:
            return {
                'matched_minutiae': [],
                'score': 0,
                'quality_score': 0
            }
        
        # حساب المسافات والزوايا بين النقاط
        matches = []
        for m1 in minutiae1:
            best_match = None
            min_distance = float('inf')
            
            for m2 in minutiae2:
                # حساب المسافة بين النقطتين
                distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                
                # حساب الفرق في الزاوية
                angle_diff = abs(m1['angle'] - m2['angle'])
                
                # التحقق من تطابق النقطتين
                if distance < 10 and angle_diff < np.pi/4:
                    if distance < min_distance:
                        min_distance = distance
                        best_match = m2
            
            if best_match:
                matches.append((m1, best_match))
        
        # حساب درجة التطابق
        score = len(matches) / min(len(minutiae1), len(minutiae2))
        
        # حساب درجة الجودة
        quality_score = calculate_quality_score(matches, features1, features2)
        
        return {
            'matched_minutiae': matches,
            'score': score,
            'quality_score': quality_score
        }
    except Exception as e:
        print(f"Error in match_fingerprints: {str(e)}")
        return {
            'matched_minutiae': [],
            'score': 0,
            'quality_score': 0
        }

def calculate_quality_score(matches, features1, features2):
    """حساب درجة جودة المطابقة"""
    try:
        if not matches or not features1 or not features2:
            return 0
        
        # حساب متوسط المسافة بين النقاط المتطابقة
        distances = []
        for m1, m2 in matches:
            distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # حساب درجة الجودة بناءً على المسافة
        quality_score = max(0, 1 - avg_distance / 100)
        
        return quality_score
    except Exception as e:
        print(f"Error in calculate_quality_score: {str(e)}")
        return 0

def visualize_matches(image1, image2, matches):
    """رسم المطابقات بين البصمتين"""
    try:
        # تحويل الصور إلى BGR إذا كانت بتدرج رمادي
        if len(image1.shape) == 2:
            img1 = cv2.cvtColor(image1.copy(), cv2.COLOR_GRAY2BGR)
        else:
            img1 = image1.copy()
            
        if len(image2.shape) == 2:
            img2 = cv2.cvtColor(image2.copy(), cv2.COLOR_GRAY2BGR)
        else:
            img2 = image2.copy()
        
        # دمج الصورتين
        visualization = np.hstack((img1, img2))
        
        # رسم المطابقات
        for m1, m2 in matches:
            # رسم نقطة في الصورة الأولى
            cv2.circle(visualization, (m1['x'], m1['y']), 3, (0, 255, 0), -1)
            
            # رسم نقطة في الصورة الثانية
            x2 = m2['x'] + img1.shape[1]  # إضافة عرض الصورة الأولى للإحداثي x
            cv2.circle(visualization, (x2, m2['y']), 3, (0, 255, 0), -1)
            
            # رسم خط بين النقطتين
            cv2.line(visualization, (m1['x'], m1['y']), (x2, m2['y']), (255, 0, 0), 1)
        
        return visualization
    except Exception as e:
        print(f"Error in visualize_matches: {str(e)}")
        return np.hstack((image1, image2)) 