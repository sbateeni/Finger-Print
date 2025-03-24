import cv2
import numpy as np

def extract_features(image):
    """استخراج الخصائص من صورة البصمة"""
    try:
        # تحويل الصورة إلى تدرج رمادي إذا كانت ملونة
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # تطبيق فلتر Sobel للكشف عن الحواف
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # حساب اتجاه الحواف
        direction = np.arctan2(sobely, sobelx)
        
        # حساب قوة الحواف
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # تطبيق العتبة للحصول على الحواف
        edges = cv2.threshold(magnitude.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # حساب كثافة الحواف
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # حساب متوسط قوة الحواف
        mean_magnitude = np.mean(magnitude)
        
        # حساب التباين المحلي
        local_contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # تجميع الخصائص
        features = {
            'edge_density': float(edge_density),
            'mean_magnitude': float(mean_magnitude),
            'local_contrast': float(local_contrast),
            'direction_histogram': np.histogram(direction.flatten(), bins=36, range=(-np.pi, np.pi))[0].tolist()
        }
        
        return features
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        # إرجاع قاموس فارغ بدلاً من None
        return {
            'edge_density': 0.0,
            'mean_magnitude': 0.0,
            'local_contrast': 0.0,
            'direction_histogram': [0] * 36
        }

def visualize_features(image, features):
    """تصور الخصائص المستخرجة"""
    try:
        # نسخ الصورة
        if len(image.shape) == 2:
            visualization = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            visualization = image.copy()
        
        # تطبيق فلتر Sobel
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # حساب اتجاه الحواف
        direction = np.arctan2(sobely, sobelx)
        
        # رسم اتجاه الحواف
        for y in range(0, image.shape[0], 10):
            for x in range(0, image.shape[1], 10):
                angle = direction[y, x]
                end_x = int(x + 10 * np.cos(angle))
                end_y = int(y + 10 * np.sin(angle))
                cv2.line(visualization, (x, y), (end_x, end_y), (0, 255, 0), 1)
        
        return visualization
    except Exception as e:
        print(f"Error in visualize_features: {str(e)}")
        return image 