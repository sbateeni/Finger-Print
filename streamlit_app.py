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
from PIL import Image
import io
from datetime import datetime

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

def convert_cv2_to_pil(cv2_image):
    try:
        if cv2_image is None:
            return None
        # Convert from BGR to RGB
        if len(cv2_image.shape) == 3:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        return Image.fromarray(cv2_image)
    except Exception as e:
        logger.error(f"Error converting CV2 to PIL: {str(e)}")
        return None

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
        
        # Display using Streamlit
        st.image(image, caption=caption, use_container_width=True)
        return True
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

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
                # عرض الصورة الأصلية
                if not display_image(original_pil, "البصمة الأصلية"):
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
                # عرض الصورة الجزئية
                if not display_image(partial_pil, "البصمة الجزئية"):
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