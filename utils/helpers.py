import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
import traceback

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image):
    """التحقق من صحة الصورة"""
    try:
        if image is None:
            return False
        if image.size == 0:
            return False
        if len(image.shape) != 2:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False

def display_image(image, caption):
    """عرض الصورة في Streamlit"""
    try:
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # If it's a color image
                image = Image.fromarray(image)
            else:
                # If it's a grayscale image
                image = Image.fromarray(image.astype(np.uint8))
        elif not isinstance(image, Image.Image):
            logger.error(f"Unsupported image type: {type(image)}")
            return False
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Display using Streamlit
        st.image(image, caption=caption)
        return True
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def add_ruler_to_image(image, dpi=100):
    """إضافة مسطرة مرقمة إلى الصورة"""
    # تحويل الصورة إلى مصفوفة NumPy إذا كانت PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # إنشاء صورة جديدة مع مساحة إضافية للمساطر
    height, width = image.shape[:2]
    ruler_size = 50  # حجم المسطرة بالبكسل
    new_width = width + ruler_size
    new_height = height + ruler_size
    
    # إنشاء صورة جديدة مع خلفية بيضاء
    new_image = np.ones((new_height, new_width), dtype=np.uint8) * 255
    
    # نسخ الصورة الأصلية إلى الموقع الصحيح
    new_image[ruler_size:, ruler_size:] = image
    
    # إضافة المسطرة الأفقية
    for i in range(0, width, int(dpi/2.54)):  # كل سنتيمتر
        x = i + ruler_size
        y = ruler_size - 10
        cv2.line(new_image, (x, y), (x, ruler_size), 0, 1)
        cv2.putText(new_image, f"{i/dpi*2.54:.1f}", (x-10, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
    
    # إضافة المسطرة العمودية
    for i in range(0, height, int(dpi/2.54)):  # كل سنتيمتر
        x = ruler_size - 10
        y = i + ruler_size
        cv2.line(new_image, (x, y), (ruler_size, y), 0, 1)
        cv2.putText(new_image, f"{i/dpi*2.54:.1f}", (x-25, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
    
    return Image.fromarray(new_image)

def draw_matching_boxes(image, match_regions, original_size):
    """رسم مربعات حول المناطق المتطابقة"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # إنشاء نسخة من الصورة للرسم عليها
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # تحويل الصورة إلى RGB إذا كانت رمادية
    if draw_image.mode != 'RGB':
        draw_image = draw_image.convert('RGB')
    
    # رسم المربعات حول المناطق المتطابقة
    for region in match_regions:
        x, y, w, h = region['box']
        score = region['score']
        
        # تحديد اللون حسب نسبة التطابق
        if score > 0.75:
            color = (0, 255, 0)  # أخضر للتطابق العالي
        elif score > 0.5:
            color = (255, 255, 0)  # أصفر للتطابق المتوسط
        else:
            color = (255, 0, 0)  # أحمر للتطابق المنخفض
        
        # رسم المربع
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        
        # إضافة النسبة
        try:
            # محاولة تحميل خط عربي
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            # استخدام الخط الافتراضي إذا لم يتم العثور على الخط العربي
            font = ImageFont.load_default()
        
        draw.text((x, y-20), f"{score*100:.1f}%", fill=color, font=font)
    
    return draw_image 