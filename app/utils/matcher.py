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
        if not matches:
            return 0
        
        # حساب متوسط المسافة بين النقاط المتطابقة
        distances = []
        for m1, m2 in matches:
            distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # حساب درجة الجودة بناءً على المسافة
        distance_score = max(0, 1 - avg_distance / 100)
        
        # حساب درجة الجودة بناءً على الخصائص
        feature_score = 0
        if features1 and features2:
            # مقارنة كثافة الحواف
            edge_density_diff = abs(features1['edge_density'] - features2['edge_density'])
            edge_score = max(0, 1 - edge_density_diff)
            
            # مقارنة التباين المحلي
            contrast_diff = abs(features1['local_contrast'] - features2['local_contrast'])
            contrast_score = max(0, 1 - contrast_diff / 1000)
            
            # حساب درجة الجودة النهائية
            feature_score = (edge_score + contrast_score) / 2
        
        # دمج الدرجات
        quality_score = (distance_score + feature_score) / 2
        
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
        
        # تعديل حجم الصورة الثانية لتتناسب مع الصورة الأولى
        height1 = img1.shape[0]
        height2 = img2.shape[0]
        target_height = min(height1, height2)
        
        # تعديل حجم الصورتين
        if height1 != target_height:
            img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / height1), target_height))
        if height2 != target_height:
            img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / height2), target_height))
        
        # دمج الصورتين
        visualization = np.hstack((img1, img2))
        
        # تعديل إحداثيات النقاط المطابقة
        scale1 = target_height / height1
        scale2 = target_height / height2
        
        # رسم المطابقات
        for m1, m2 in matches:
            # تعديل إحداثيات النقاط
            x1 = int(m1['x'] * scale1)
            y1 = int(m1['y'] * scale1)
            x2 = int(m2['x'] * scale2) + img1.shape[1]  # إضافة عرض الصورة الأولى
            y2 = int(m2['y'] * scale2)
            
            # رسم نقطة في الصورة الأولى
            cv2.circle(visualization, (x1, y1), 3, (0, 255, 0), -1)
            
            # رسم نقطة في الصورة الثانية
            cv2.circle(visualization, (x2, y2), 3, (0, 255, 0), -1)
            
            # رسم خط بين النقطتين
            cv2.line(visualization, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        return visualization
    except Exception as e:
        print(f"Error in visualize_matches: {str(e)}")
        # في حالة الخطأ، نقوم بتعديل حجم الصورتين قبل الدمج
        try:
            height1 = image1.shape[0]
            height2 = image2.shape[0]
            target_height = min(height1, height2)
            
            if height1 != target_height:
                image1 = cv2.resize(image1, (int(image1.shape[1] * target_height / height1), target_height))
            if height2 != target_height:
                image2 = cv2.resize(image2, (int(image2.shape[1] * target_height / height2), target_height))
            
            return np.hstack((image1, image2))
        except Exception as e2:
            print(f"Error in fallback visualization: {str(e2)}")
            return image1  # إرجاع الصورة الأولى فقط في حالة فشل كل المحاولات 