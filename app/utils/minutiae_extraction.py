import cv2
import numpy as np

def extract_minutiae(image, max_points=100):
    """استخراج نقاط التفاصيل من البصمة"""
    try:
        # تحويل الصورة إلى تدرج رمادي
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # تطبيق فلتر Sobel للكشف عن الحواف
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # حساب اتجاه الحواف
        direction = np.arctan2(sobely, sobelx)
        
        # تطبيق فلتر Harris للكشف عن الزوايا
        corners = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 10)
        
        if corners is None:
            return []
        
        # تحويل النقاط إلى قائمة من الإحداثيات
        minutiae = []
        for corner in corners:
            x, y = corner.ravel()
            angle = direction[int(y), int(x)]
            minutiae.append({
                'x': int(x),
                'y': int(y),
                'angle': float(angle)
            })
        
        return minutiae
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

def visualize_minutiae(image, minutiae):
    """رسم نقاط التفاصيل على الصورة"""
    try:
        # نسخ الصورة
        if len(image.shape) == 2:
            visualization = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            visualization = image.copy()
        
        # رسم النقاط
        for point in minutiae:
            x = point['x']
            y = point['y']
            angle = point['angle']
            
            # رسم دائرة حول النقطة
            cv2.circle(visualization, (x, y), 3, (0, 255, 0), -1)
            
            # رسم خط يمثل الاتجاه
            end_x = int(x + 10 * np.cos(angle))
            end_y = int(y + 10 * np.sin(angle))
            cv2.line(visualization, (x, y), (end_x, end_y), (255, 0, 0), 1)
        
        return visualization
    except Exception as e:
        print(f"Error in visualize_minutiae: {str(e)}")
        return image 