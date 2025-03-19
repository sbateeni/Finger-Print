import streamlit as st
import cv2
import numpy as np
from preprocessing.image_processing import preprocess_image, detect_ridges, analyze_ridge_patterns
from features.minutiae_extraction import extract_minutiae, analyze_ridge_characteristics
from matching.matcher import match_fingerprints
import logging
import traceback
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import math

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تكوين الصفحة
st.set_page_config(
    page_title="نظام تحليل البصمات",
    page_icon="🔍",
    layout="wide"
)

# العنوان الرئيسي
st.title("نظام تحليل البصمات الجنائي")
st.markdown("""
### نظام متقدم لتحليل ومطابقة البصمات باستخدام تقنيات الذكاء الاصطناعي
""")

# تعريف الدوال المساعدة
def validate_image(image):
    if image is None:
        return False
    if image.size == 0:
        return False
    if len(image.shape) != 2:
        return False
    return True

def display_image(image, caption):
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
        
        # Display using Streamlit without use_container_width parameter
        st.image(image, caption=caption)
        return True
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def process_image(image):
    try:
        # معالجة الصورة
        processed_img, direction = preprocess_image(image)
        if processed_img is None:
            return None, None, None

        # استخراج النقاط المميزة
        minutiae = extract_minutiae(processed_img)
        if not minutiae:
            return None, None, None

        # تحليل أنماط الخطوط
        ridge_patterns = analyze_ridge_patterns(processed_img, direction)

        return processed_img, minutiae, ridge_patterns
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def write_results_to_file(match_result):
    try:
        # إنشاء مجلد النتائج إذا لم يكن موجوداً
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # إنشاء اسم الملف باستخدام التاريخ والوقت
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/match_result_{timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== نتائج مطابقة البصمات ===\n\n")
            f.write(f"تاريخ ووقت المطابقة: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== إحصائيات المطابقة ===\n")
            f.write(f"نسبة التطابق: {match_result['match_score']:.2f}%\n")
            f.write(f"عدد النقاط في البصمة الأصلية: {match_result['total_original']}\n")
            f.write(f"عدد النقاط في البصمة الجزئية: {match_result['total_partial']}\n")
            f.write(f"عدد النقاط المتطابقة: {match_result['matched_points']}\n")
            f.write(f"حالة المطابقة: {match_result['status']}\n\n")
            
            f.write("=== تفاصيل تحليل الخطوط ===\n")
            if match_result['details']['ridge_analysis']:
                for i, analysis in enumerate(match_result['details']['ridge_analysis'], 1):
                    f.write(f"\nتحليل الخط {i}:\n")
                    f.write(f"المسافة: {analysis['distance']:.2f}\n")
                    f.write(f"الفرق في الزاوية: {analysis['angle_difference']:.2f}\n")
                    f.write(f"تطابق النوع: {'نعم' if analysis['type_match'] else 'لا'}\n")
            
        return filename
    except Exception as e:
        logger.error(f"Error writing results to file: {str(e)}")
        return None

# تصميم عرض النتائج
def display_summary_results(original_count, partial_count, matched_points, match_score, decision):
    st.markdown("---")
    st.subheader("📊 ملخص النتائج")
    
    # استخدام CSS للنص العريض والألوان
    st.markdown("""
    <style>
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-family: 'Arial', sans-serif;
        direction: rtl;
    }
    .result-item {
        font-size: 18px;
        margin: 10px 0;
    }
    .highlight {
        color: #0068c9;
        font-weight: bold;
    }
    .success {
        color: #09ab3b;
        font-weight: bold;
    }
    .high-match {
        color: #09ab3b;
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        background-color: rgba(9, 171, 59, 0.1);
        border-radius: 5px;
    }
    .medium-match {
        color: #f0a202;
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        background-color: rgba(240, 162, 2, 0.1);
        border-radius: 5px;
    }
    .low-match {
        color: #ff0000;
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        background-color: rgba(255, 0, 0, 0.1);
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # صندوق النتائج
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    # عدد النقاط
    st.markdown(f'<div class="result-item">🔎 عدد النقاط المستخرجة من الأصلية: <span class="highlight">{original_count}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-item">🔎 عدد النقاط المستخرجة من الجزئية: <span class="highlight">{partial_count}</span></div>', unsafe_allow_html=True)
    
    # نتائج المطابقة
    st.markdown(f'<div class="result-item">✅ نقاط التطابق: <span class="success">{matched_points}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-item">✅ نسبة التشابه: <span class="success">{match_score:.2f}%</span></div>', unsafe_allow_html=True)
    
    # القرار
    decision_class = "high-match" if match_score > 75 else "medium-match" if match_score > 50 else "low-match"
    decision_text = f'HIGH MATCH - احتمالية التطابق كبيرة جدًا' if match_score > 75 else f'MEDIUM MATCH - احتمالية التطابق متوسطة' if match_score > 50 else f'LOW MATCH - احتمالية التطابق منخفضة'
    
    st.markdown(f'<div class="result-item">✅ القرار: <span class="{decision_class}">{decision_text}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state to track temporary files
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# إنشاء أعمدة للصور
col1, col2 = st.columns(2)

# البصمة الأصلية
with col1:
    st.subheader("البصمة الأصلية")
    original_file = st.file_uploader("اختر البصمة الأصلية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="original")
    
    if original_file is not None:
        try:
            # قراءة الصورة مباشرة باستخدام PIL
            original_pil = Image.open(original_file)
            # تحويل إلى مصفوفة NumPy
            original_img = np.array(original_pil.convert('L'))
            
            if validate_image(original_img):
                # إضافة المسطرة إلى الصورة الأصلية
                original_with_ruler = add_ruler_to_image(original_pil)
                if not display_image(original_with_ruler, "البصمة الأصلية مع المسطرة"):
                    st.error("فشل في عرض الصورة الأصلية")
                
                # معالجة البصمة الأصلية
                with st.spinner("جاري معالجة البصمة الأصلية..."):
                    processed_original, minutiae_original, ridge_patterns_original = process_image(original_img)
                    if processed_original is not None:
                        # تحويل الصورة المعالجة إلى صيغة PIL
                        processed_pil = Image.fromarray(processed_original.astype(np.uint8))
                        if processed_pil is not None:
                            if not display_image(processed_pil, "البصمة المعالجة"):
                                st.error("فشل في عرض الصورة المعالجة")
                            st.success(f"تم استخراج {len(minutiae_original)} نقطة مميزة")
                        else:
                            st.error("فشل في تحويل الصورة المعالجة")
                    else:
                        st.error("فشل في معالجة البصمة الأصلية")
            else:
                st.error("الصورة غير صالحة")
        except Exception as e:
            logger.error(f"Error processing original image: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("حدث خطأ أثناء معالجة الصورة الأصلية")

# البصمة الجزئية
with col2:
    st.subheader("البصمة الجزئية")
    partial_file = st.file_uploader("اختر البصمة الجزئية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="partial")
    
    if partial_file is not None:
        try:
            # قراءة الصورة مباشرة باستخدام PIL
            partial_pil = Image.open(partial_file)
            # تحويل إلى مصفوفة NumPy
            partial_img = np.array(partial_pil.convert('L'))
            
            if validate_image(partial_img):
                # إضافة المسطرة إلى الصورة الجزئية
                partial_with_ruler = add_ruler_to_image(partial_pil)
                if not display_image(partial_with_ruler, "البصمة الجزئية مع المسطرة"):
                    st.error("فشل في عرض الصورة الجزئية")
                
                # معالجة البصمة الجزئية
                with st.spinner("جاري معالجة البصمة الجزئية..."):
                    processed_partial, minutiae_partial, ridge_patterns_partial = process_image(partial_img)
                    if processed_partial is not None:
                        # تحويل الصورة المعالجة إلى صيغة PIL
                        processed_pil = Image.fromarray(processed_partial.astype(np.uint8))
                        if processed_pil is not None:
                            if not display_image(processed_pil, "البصمة المعالجة"):
                                st.error("فشل في عرض الصورة المعالجة")
                            st.success(f"تم استخراج {len(minutiae_partial)} نقطة مميزة")
                        else:
                            st.error("فشل في تحويل الصورة المعالجة")
                    else:
                        st.error("فشل في معالجة البصمة الجزئية")
            else:
                st.error("الصورة غير صالحة")
        except Exception as e:
            logger.error(f"Error processing partial image: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("حدث خطأ أثناء معالجة الصورة الجزئية")

# زر المطابقة
if st.button("بدء المطابقة", type="primary"):
    if original_file is not None and partial_file is not None:
        if processed_original is not None and processed_partial is not None:
            with st.spinner("جاري مطابقة البصمات..."):
                try:
                    # حساب عامل التكبير/التصغير
                    scale_factor = calculate_scale_factor(original_img, partial_img)
                    st.info(f"عامل التكبير/التصغير: {scale_factor:.2f}")
                    
                    # مطابقة البصمات
                    match_result = match_fingerprints(minutiae_original, minutiae_partial)
                    
                    # كتابة النتائج إلى ملف
                    result_file = write_results_to_file(match_result)
                    if result_file:
                        st.success(f"تم حفظ النتائج في الملف: {result_file}")
                    
                    # عرض النتائج
                    st.subheader("نتائج المطابقة")
                    
                    # إنشاء أعمدة للنتائج
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("نسبة التطابق", f"{match_result['match_score']:.2f}%")
                    
                    with col2:
                        st.metric("النقاط المتطابقة", f"{match_result['matched_points']}/{match_result['total_partial']}")
                    
                    with col3:
                        st.metric("الحالة", match_result['status'])
                    
                    # عرض التفاصيل
                    st.subheader("تفاصيل التحليل")
                    st.write(f"عدد النقاط في البصمة الأصلية: {match_result['total_original']}")
                    st.write(f"عدد النقاط في البصمة الجزئية: {match_result['total_partial']}")
                    st.write(f"عدد النقاط المتطابقة: {match_result['matched_points']}")
                    
                    # عرض النتائج بالتنسيق المطلوب
                    display_summary_results(
                        match_result['total_original'],
                        match_result['total_partial'],
                        match_result['matched_points'],
                        match_result['match_score'],
                        match_result['status']
                    )
                    
                    # عرض تحليل الخطوط
                    if match_result['details']['ridge_analysis']:
                        st.subheader("تحليل الخطوط")
                        for analysis in match_result['details']['ridge_analysis']:
                            st.write(f"المسافة: {analysis['distance']:.2f}")
                            st.write(f"الفرق في الزاوية: {analysis['angle_difference']:.2f}")
                            st.write(f"تطابق النوع: {'نعم' if analysis['type_match'] else 'لا'}")
                    
                    # عرض البصمة الأصلية مع مربعات التطابق
                    if 'match_regions' in match_result:
                        st.subheader("مناطق التطابق في البصمة الأصلية")
                        matched_image = draw_matching_boxes(original_img, match_result['match_regions'], original_img.shape)
                        if not display_image(matched_image, "مناطق التطابق"):
                            st.error("فشل في عرض مناطق التطابق")
                    
                except Exception as e:
                    logger.error(f"Error in matching: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("حدث خطأ أثناء المطابقة")
        else:
            st.error("يرجى التأكد من معالجة البصمتين بنجاح")
    else:
        st.error("يرجى تحميل البصمتين أولاً")

# تنظيف الملفات المؤقتة
try:
    for temp_file in st.session_state.temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
            logger.info(f"Cleaned up temporary file: {temp_file}")
except Exception as e:
    logger.warning(f"Error removing temporary files: {str(e)}")

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

def calculate_scale_factor(original_img, partial_img):
    """حساب عامل التكبير/التصغير بناءً على قياسات الخطوط"""
    # استخراج الخطوط من كلا الصورتين
    original_ridges = detect_ridges(original_img)
    partial_ridges = detect_ridges(partial_img)
    
    if not original_ridges or not partial_ridges:
        return 1.0
    
    # حساب متوسط المسافة بين الخطوط
    original_distances = []
    partial_distances = []
    
    for ridge in original_ridges:
        if len(ridge) > 1:
            for i in range(len(ridge)-1):
                dist = np.linalg.norm(ridge[i] - ridge[i+1])
                original_distances.append(dist)
    
    for ridge in partial_ridges:
        if len(ridge) > 1:
            for i in range(len(ridge)-1):
                dist = np.linalg.norm(ridge[i] - ridge[i+1])
                partial_distances.append(dist)
    
    if not original_distances or not partial_distances:
        return 1.0
    
    # حساب عامل التكبير/التصغير
    original_avg = np.mean(original_distances)
    partial_avg = np.mean(partial_distances)
    
    return original_avg / partial_avg 