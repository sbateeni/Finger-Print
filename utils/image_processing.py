import cv2
import numpy as np
from config import *

def preprocess_image(image, denoise_method="fastNlMeans", fast_denoise_h=10, gauss_ksize=3):
    """معالجة الصورة المسبقة مع خيارات إزالة الضوضاء"""
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
        # إزالة الضوضاء حسب الطريقة المختارة
        if denoise_method == "fastNlMeans":
            image = cv2.fastNlMeansDenoising(image, h=fast_denoise_h)
        elif denoise_method == "GaussianBlur":
            if gauss_ksize % 2 == 0:
                gauss_ksize += 1  # يجب أن يكون فردياً
            image = cv2.GaussianBlur(image, (gauss_ksize, gauss_ksize), 0)
        # إذا كانت None فلا شيء
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
        # تطبيق عملية الهيكلة (Skeletonization)
        skeleton = cv2.ximgproc.thinning(image)
        # تحويل الهيكل العظمي إلى صورة ثنائية (0 أو 255 فقط)
        _, skeleton_bin = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
        return skeleton_bin
    except Exception as e:
        print(f"Error in enhance_image: {str(e)}")
        return None 