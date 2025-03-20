import cv2
import numpy as np
from config import *

def preprocess_image(image):
    """معالجة الصورة المسبقة"""
    try:
        # تحويل الصورة إلى Grayscale إذا لم تكن كذلك
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # توحيد حجم الصورة
        image = cv2.resize(image, IMAGE_SIZE)
        
        # تحسين التباين باستخدام CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        image = clahe.apply(image)
        
        # إزالة الضوضاء
        image = cv2.fastNlMeansDenoising(image)
        
        # تحويل الصورة إلى ثنائية
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None

def detect_edges(image):
    """استخراج الحواف"""
    try:
        # تطبيق فلتر Sobel
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # حساب خريطة الاتجاه
        orientation_map = np.zeros_like(sobelx)
        mask = (sobelx != 0) | (sobely != 0)
        orientation_map[mask] = np.arctan2(sobely[mask], sobelx[mask])
        
        # تطبيق فلتر Gabor
        kernel = cv2.getGaborKernel(
            (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
            GABOR_SIGMA,
            GABOR_THETA,
            GABOR_LAMBDA,
            GABOR_GAMMA,
            GABOR_PSI,
            ktype=cv2.CV_32F
        )
        ridges = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        return ridges, orientation_map
    except Exception as e:
        print(f"Error in detect_edges: {str(e)}")
        return None, None

def enhance_image(image):
    """تحسين جودة الصورة"""
    try:
        # تطبيق عملية الهيكلة
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.ximgproc.thinning(image)
        
        # تحسين الحواف
        edges = cv2.Canny(skeleton, 100, 200)
        
        return edges
    except Exception as e:
        print(f"Error in enhance_image: {str(e)}")
        return None 