import os
import sys
import logging
from datetime import datetime
import streamlit as st
import cv2
import numpy as np

# إضافة المسار الحالي إلى مسارات Python
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# استيراد المكونات الداخلية
from utils.helpers import validate_image, display_image, init_session_state, cleanup_temp_files
from ui.components import (
    setup_page_config,
    display_header,
    display_summary_results,
    display_error
)
from preprocessing.image_processor import (
    preprocess_image,
    extract_minutiae,
    calculate_scale_factor,
    add_ruler_to_image,
    draw_matching_boxes
)
from matching.fingerprint_matcher import match_fingerprints
from utils.file_handler import write_results_to_file

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_fingerprint_image(file, is_original=True):
    """معالجة صورة البصمة وإعداد النتائج"""
    try:
        # قراءة الصورة
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            display_error("فشل في قراءة الصورة")
            return False
        
        # التحقق من حجم الصورة
        if img.shape[0] < 100 or img.shape[1] < 100:
            display_error("الصورة صغيرة جداً. الحد الأدنى للحجم هو 100×100 بكسل")
            return False
        
        # معالجة الصورة
        processed_img, edges = preprocess_image(img)
        if processed_img is None:
            display_error("فشل في معالجة الصورة")
            return False
        
        # استخراج النقاط المميزة
        minutiae = extract_minutiae(processed_img)
        if not minutiae:
            display_error("لم يتم العثور على نقاط مميزة في البصمة")
            return False
        
        # حفظ النتائج في حالة الجلسة
        if is_original:
            st.session_state.processed_original = processed_img
            st.session_state.original_minutiae = minutiae
        else:
            st.session_state.processed_partial = processed_img
            st.session_state.partial_minutiae = minutiae
        
        # إضافة المسطرة وعرض الصورة
        img_with_ruler = add_ruler_to_image(img)
        display_image(img_with_ruler, "البصمة الأصلية" if is_original else "البصمة الجزئية")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {'original' if is_original else 'partial'} image: {str(e)}")
        display_error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return False

def perform_matching():
    """تنفيذ عملية المطابقة بين البصمتين"""
    try:
        # التحقق من وجود جميع البيانات المطلوبة
        if not all([
            st.session_state.processed_original,
            st.session_state.processed_partial,
            st.session_state.original_minutiae,
            st.session_state.partial_minutiae
        ]):
            display_error("يرجى تحميل البصمتين أولاً")
            return
        
        # حساب عامل التحجيم
        scale_factor = calculate_scale_factor(
            st.session_state.processed_original,
            st.session_state.processed_partial
        )
        
        # مطابقة البصمات
        match_result = match_fingerprints(
            st.session_state.original_minutiae,
            st.session_state.partial_minutiae,
            scale_factor
        )
        
        # عرض النتائج
        display_summary_results(
            match_result['total_original'],
            match_result['total_partial'],
            match_result['matched_points'],
            match_result['match_score'],
            match_result['status']
        )
        
        # رسم المربعات على الصورة الأصلية
        if match_result['details']['match_regions']:
            original_with_boxes = draw_matching_boxes(
                st.session_state.processed_original,
                match_result['details']['match_regions'],
                st.session_state.processed_original.shape
            )
            display_image(original_with_boxes, "المناطق المتطابقة")
        
        # حفظ النتائج في ملف
        result_file = write_results_to_file(match_result)
        if result_file:
            st.success(f"تم حفظ النتائج في الملف: {result_file}")
        
    except Exception as e:
        logger.error(f"Error in matching process: {str(e)}")
        display_error(f"حدث خطأ أثناء عملية المطابقة: {str(e)}")

def main():
    """الدالة الرئيسية للتطبيق"""
    # إعداد الصفحة
    setup_page_config()
    
    # تهيئة متغيرات الحالة
    init_session_state()
    
    # عرض العنوان
    display_header()
    
    # تحميل البصمة الأصلية
    st.header("1️⃣ تحميل البصمة الأصلية")
    original_file = st.file_uploader("اختر ملف البصمة الأصلية", type=['png', 'jpg', 'jpeg'])
    if original_file:
        process_fingerprint_image(original_file, is_original=True)
    
    # تحميل البصمة الجزئية
    st.header("2️⃣ تحميل البصمة الجزئية")
    partial_file = st.file_uploader("اختر ملف البصمة الجزئية", type=['png', 'jpg', 'jpeg'])
    if partial_file:
        process_fingerprint_image(partial_file, is_original=False)
    
    # زر المطابقة
    if st.button("🔍 بدء المطابقة"):
        perform_matching()
    
    # تنظيف الملفات المؤقتة
    cleanup_temp_files()

if __name__ == "__main__":
    main() 