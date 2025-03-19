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

# إنشاء أعمدة للصور
col1, col2 = st.columns(2)

# البصمة الأصلية
with col1:
    st.subheader("البصمة الأصلية")
    original_file = st.file_uploader("اختر البصمة الأصلية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="original")
    
    if original_file is not None:
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_file.name)[1]) as tmp_file:
            tmp_file.write(original_file.getvalue())
            tmp_path = tmp_file.name

        # قراءة الصورة
        original_img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        if validate_image(original_img):
            st.image(original_img, caption="البصمة الأصلية", use_column_width=True)
            
            # معالجة البصمة الأصلية
            with st.spinner("جاري معالجة البصمة الأصلية..."):
                processed_original, minutiae_original, ridge_patterns_original = process_image(original_img)
                if processed_original is not None:
                    st.image(processed_original, caption="البصمة المعالجة", use_column_width=True)
                    st.success(f"تم استخراج {len(minutiae_original)} نقطة مميزة")
                else:
                    st.error("فشل في معالجة البصمة الأصلية")
        else:
            st.error("الصورة غير صالحة")

# البصمة الجزئية
with col2:
    st.subheader("البصمة الجزئية")
    partial_file = st.file_uploader("اختر البصمة الجزئية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="partial")
    
    if partial_file is not None:
        # حفظ الملف مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(partial_file.name)[1]) as tmp_file:
            tmp_file.write(partial_file.getvalue())
            tmp_path = tmp_file.name

        # قراءة الصورة
        partial_img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        if validate_image(partial_img):
            st.image(partial_img, caption="البصمة الجزئية", use_column_width=True)
            
            # معالجة البصمة الجزئية
            with st.spinner("جاري معالجة البصمة الجزئية..."):
                processed_partial, minutiae_partial, ridge_patterns_partial = process_image(partial_img)
                if processed_partial is not None:
                    st.image(processed_partial, caption="البصمة المعالجة", use_column_width=True)
                    st.success(f"تم استخراج {len(minutiae_partial)} نقطة مميزة")
                else:
                    st.error("فشل في معالجة البصمة الجزئية")
        else:
            st.error("الصورة غير صالحة")

# زر المطابقة
if st.button("بدء المطابقة", type="primary"):
    if original_file is not None and partial_file is not None:
        if processed_original is not None and processed_partial is not None:
            with st.spinner("جاري مطابقة البصمات..."):
                try:
                    # مطابقة البصمات
                    match_result = match_fingerprints(minutiae_original, minutiae_partial)
                    
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
    if 'original_file' in locals():
        os.unlink(tmp_path)
    if 'partial_file' in locals():
        os.unlink(tmp_path)
except Exception as e:
    logger.warning(f"Error removing temporary files: {str(e)}") 