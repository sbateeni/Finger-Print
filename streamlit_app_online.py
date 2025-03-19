import streamlit as st
import cv2
import numpy as np
import logging
import os
import tempfile
from datetime import datetime

# استيراد الوحدات الجديدة
from utils.helpers import validate_image, display_image
from ui.display import display_summary_results, write_results_to_file
from processing.image_processor import (
    preprocess_image, extract_minutiae, calculate_scale_factor,
    add_ruler_to_image, draw_matching_boxes
)
from matching.fingerprint_matcher import match_fingerprints

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# إعداد الصفحة
try:
    st.set_page_config(
        page_title="نظام مطابقة البصمات",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.error(f"Error in page configuration: {str(e)}")

# تهيئة متغيرات الحالة
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'processed_original' not in st.session_state:
    st.session_state.processed_original = None
if 'processed_partial' not in st.session_state:
    st.session_state.processed_partial = None
if 'original_minutiae' not in st.session_state:
    st.session_state.original_minutiae = None
if 'partial_minutiae' not in st.session_state:
    st.session_state.partial_minutiae = None

# العنوان الرئيسي
st.title("🔍 نظام مطابقة البصمات")
st.markdown("""
### نظام متقدم لمطابقة البصمات باستخدام الذكاء الاصطناعي
""")

# تحميل البصمة الأصلية
st.header("1️⃣ تحميل البصمة الأصلية")
original_file = st.file_uploader("اختر ملف البصمة الأصلية", type=['png', 'jpg', 'jpeg'])

if original_file:
    try:
        # قراءة الصورة
        file_bytes = np.asarray(bytearray(original_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("فشل في قراءة الصورة")
            st.stop()
        
        # التحقق من حجم الصورة
        if original_img.shape[0] < 100 or original_img.shape[1] < 100:
            st.error("الصورة صغيرة جداً. الحد الأدنى للحجم هو 100×100 بكسل")
            st.stop()
        
        # معالجة الصورة
        processed_img, edges = preprocess_image(original_img)
        if processed_img is None:
            st.error("فشل في معالجة الصورة")
            st.stop()
        
        # استخراج النقاط المميزة
        minutiae = extract_minutiae(processed_img)
        if not minutiae:
            st.error("لم يتم العثور على نقاط مميزة في البصمة")
            st.stop()
        
        # حفظ النتائج في حالة الجلسة
        st.session_state.processed_original = processed_img
        st.session_state.original_minutiae = minutiae
        
        # إضافة المسطرة وعرض الصورة
        original_with_ruler = add_ruler_to_image(original_img)
        display_image(original_with_ruler, "البصمة الأصلية")
        
    except Exception as e:
        logger.error(f"Error processing original image: {str(e)}")
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")

# تحميل البصمة الجزئية
st.header("2️⃣ تحميل البصمة الجزئية")
partial_file = st.file_uploader("اختر ملف البصمة الجزئية", type=['png', 'jpg', 'jpeg'])

if partial_file:
    try:
        # قراءة الصورة
        file_bytes = np.asarray(bytearray(partial_file.read()), dtype=np.uint8)
        partial_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if partial_img is None:
            st.error("فشل في قراءة الصورة")
            st.stop()
        
        # التحقق من حجم الصورة
        if partial_img.shape[0] < 100 or partial_img.shape[1] < 100:
            st.error("الصورة صغيرة جداً. الحد الأدنى للحجم هو 100×100 بكسل")
            st.stop()
        
        # معالجة الصورة
        processed_img, edges = preprocess_image(partial_img)
        if processed_img is None:
            st.error("فشل في معالجة الصورة")
            st.stop()
        
        # استخراج النقاط المميزة
        minutiae = extract_minutiae(processed_img)
        if not minutiae:
            st.error("لم يتم العثور على نقاط مميزة في البصمة")
            st.stop()
        
        # حفظ النتائج في حالة الجلسة
        st.session_state.processed_partial = processed_img
        st.session_state.partial_minutiae = minutiae
        
        # إضافة المسطرة وعرض الصورة
        partial_with_ruler = add_ruler_to_image(partial_img)
        display_image(partial_with_ruler, "البصمة الجزئية")
        
    except Exception as e:
        logger.error(f"Error processing partial image: {str(e)}")
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")

# زر المطابقة
if st.button("🔍 بدء المطابقة"):
    if not all([st.session_state.processed_original, st.session_state.processed_partial,
                st.session_state.original_minutiae, st.session_state.partial_minutiae]):
        st.error("يرجى تحميل كلا البصمتين أولاً")
        st.stop()
    
    try:
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
        st.error(f"حدث خطأ أثناء عملية المطابقة: {str(e)}")

# تنظيف الملفات المؤقتة
for temp_file in st.session_state.temp_files:
    try:
        os.remove(temp_file)
    except:
        pass
st.session_state.temp_files = [] 