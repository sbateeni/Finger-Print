import cv2
import numpy as np

def preprocess_image(image_path, image_quality=0.8, contrast=1.0):
    """معالجة الصورة الأولية"""
    try:
        # قراءة الصورة
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0 * contrast, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # تحسين جودة الصورة
        enhanced = cv2.convertScaleAbs(enhanced, alpha=image_quality, beta=0)
        
        return enhanced
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None

def enhance_image(image):
    """تحسين جودة الصورة"""
    try:
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # تحويل الصورة مرة أخرى إلى BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    except Exception as e:
        print(f"Error in enhance_image: {str(e)}")
        return image

def remove_noise(image):
    """إزالة الضوضاء من الصورة"""
    try:
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تطبيق فلتر Gaussian
        denoised = cv2.GaussianBlur(gray, (5,5), 0)
        
        # تحويل الصورة مرة أخرى إلى BGR
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised_bgr
    except Exception as e:
        print(f"Error in remove_noise: {str(e)}")
        return image

def normalize_ridges(image):
    """توحيد المسافة بين خطوط البصمة"""
    try:
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تطبيق فلتر Sobel للكشف عن الحواف
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # حساب اتجاه الحواف
        direction = np.arctan2(sobely, sobelx)
        
        # تطبيق فلتر متوسط لتوحيد المسافة
        normalized = cv2.medianBlur(gray, 5)
        
        # تحويل الصورة مرة أخرى إلى BGR
        normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        return normalized_bgr
    except Exception as e:
        print(f"Error in normalize_ridges: {str(e)}")
        return image 