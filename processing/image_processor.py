import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
import os

logger = logging.getLogger(__name__)

def preprocess_image(image):
    """معالجة الصورة وتحسين جودتها"""
    try:
        # تحويل الصورة إلى رمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # تقليل الضوضاء
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # تحسين الحواف
        edges = cv2.Canny(denoised, 50, 150)
        
        return denoised, edges
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        return None, None

def extract_minutiae(image):
    """استخراج النقاط المميزة من البصمة"""
    try:
        # تحسين الصورة
        enhanced = cv2.equalizeHist(image)
        
        # استخراج النقاط المميزة
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(enhanced, None)
        
        # تحويل النقاط إلى التنسيق المطلوب
        minutiae = []
        for kp in keypoints:
            minutiae.append({
                'x': kp.pt[0],
                'y': kp.pt[1],
                'angle': kp.angle,
                'type': 'bifurcation' if kp.size > 5 else 'ridge'
            })
        
        return minutiae
    except Exception as e:
        logger.error(f"Error in extracting minutiae: {str(e)}")
        return None

def calculate_scale_factor(original_img, partial_img):
    """حساب عامل التحجيم بين البصمتين"""
    try:
        # استخراج النقاط المميزة من كلا البصمتين
        original_minutiae = extract_minutiae(original_img)
        partial_minutiae = extract_minutiae(partial_img)
        
        if not original_minutiae or not partial_minutiae:
            return 1.0
        
        # حساب متوسط المسافات بين النقاط
        original_distances = []
        partial_distances = []
        
        for i in range(len(original_minutiae)):
            for j in range(i+1, len(original_minutiae)):
                dist = np.sqrt((original_minutiae[i]['x'] - original_minutiae[j]['x'])**2 +
                             (original_minutiae[i]['y'] - original_minutiae[j]['y'])**2)
                original_distances.append(dist)
        
        for i in range(len(partial_minutiae)):
            for j in range(i+1, len(partial_minutiae)):
                dist = np.sqrt((partial_minutiae[i]['x'] - partial_minutiae[j]['x'])**2 +
                             (partial_minutiae[i]['y'] - partial_minutiae[j]['y'])**2)
                partial_distances.append(dist)
        
        if not original_distances or not partial_distances:
            return 1.0
        
        # حساب عامل التحجيم
        scale_factor = np.mean(partial_distances) / np.mean(original_distances)
        return max(0.5, min(2.0, scale_factor))  # تحديد حدود معقولة لعامل التحجيم
        
    except Exception as e:
        logger.error(f"Error in calculating scale factor: {str(e)}")
        return 1.0

def add_ruler_to_image(image, dpi=100):
    """إضافة مسطرة قياس للصورة"""
    try:
        # تحويل الصورة إلى PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # إنشاء نسخة من الصورة للرسم عليها
        draw = ImageDraw.Draw(image)
        
        # أبعاد الصورة
        width, height = image.size
        
        # إعداد الخط
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # رسم المسطرة الأفقية
        for i in range(0, width, int(dpi/2.54)):  # 2.54 سم في البوصة
            # رسم علامة القياس
            draw.line([(i, 0), (i, 10)], fill='white', width=2)
            # إضافة رقم القياس
            cm = i / (dpi/2.54)
            draw.text((i-10, 15), f"{cm:.1f}", fill='white', font=font)
        
        # رسم المسطرة العمودية
        for i in range(0, height, int(dpi/2.54)):
            # رسم علامة القياس
            draw.line([(0, i), (10, i)], fill='white', width=2)
            # إضافة رقم القياس
            cm = i / (dpi/2.54)
            draw.text((15, i-10), f"{cm:.1f}", fill='white', font=font)
        
        return image
    except Exception as e:
        logger.error(f"Error in adding ruler to image: {str(e)}")
        return image

def draw_matching_boxes(image, match_regions, original_size):
    """رسم مربعات حول المناطق المتطابقة"""
    try:
        # تحويل الصورة إلى PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # إنشاء نسخة من الصورة للرسم عليها
        draw = ImageDraw.Draw(image)
        
        # تحميل الخط العربي
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # رسم المربعات والعلامات
        for region in match_regions:
            # تحديد لون المربع بناءً على قوة التطابق
            if region['score'] > 75:
                color = (0, 255, 0)  # أخضر
            elif region['score'] > 50:
                color = (255, 255, 0)  # أصفر
            else:
                color = (255, 0, 0)  # أحمر
            
            # رسم المربع
            x1, y1 = region['box'][0]
            x2, y2 = region['box'][1]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # إضافة نسبة التطابق
            score_text = f"{region['score']:.1f}%"
            draw.text((x1, y1-25), score_text, fill=color, font=font)
        
        return image
    except Exception as e:
        logger.error(f"Error in drawing matching boxes: {str(e)}")
        return image 