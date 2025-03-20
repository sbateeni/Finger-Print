import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
import traceback
from utils.image_processing import preprocess_image, detect_edges, enhance_image
from utils.minutiae_extractor import extract_minutiae, visualize_minutiae
from utils.matcher import match_fingerprints, visualize_matches
from utils.report_generator import generate_report
from config import *

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

# إنشاء أعمدة للصور
col1, col2 = st.columns(2)

# البصمة الأصلية
with col1:
    st.subheader("البصمة الأصلية")
    original_file = st.file_uploader("اختر البصمة الأصلية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="original")
    
    if original_file is not None:
        try:
            # قراءة الصورة
            original_pil = Image.open(original_file)
            original_img = np.array(original_pil.convert('L'))
            
            # عرض الصورة الأصلية
            st.image(original_pil, caption="البصمة الأصلية", use_column_width=True)
            
            # معالجة البصمة الأصلية
            with st.spinner("جاري معالجة البصمة الأصلية..."):
                # المعالجة المسبقة
                processed_original = preprocess_image(original_img)
                if processed_original is not None:
                    # استخراج الحواف
                    ridges_original, orientation_map_original = detect_edges(processed_original)
                    if ridges_original is not None:
                        # تحسين الصورة
                        enhanced_original = enhance_image(ridges_original)
                        if enhanced_original is not None:
                            # استخراج النقاط الدقيقة
                            minutiae_original = extract_minutiae(enhanced_original)
                            if minutiae_original:
                                # تصور النقاط
                                vis_original = visualize_minutiae(enhanced_original, minutiae_original)
                                if vis_original is not None:
                                    st.image(vis_original, caption="البصمة المعالجة", use_column_width=True)
                                    st.success(f"تم استخراج {len(minutiae_original)} نقطة مميزة")
                                else:
                                    st.error("فشل في تصور النقاط")
                            else:
                                st.error("لم يتم العثور على نقاط مميزة")
                        else:
                            st.error("فشل في تحسين الصورة")
                    else:
                        st.error("فشل في استخراج الحواف")
                else:
                    st.error("فشل في المعالجة المسبقة")
        except Exception as e:
            logger.error(f"Error processing original image: {str(e)}")
            st.error("حدث خطأ أثناء معالجة الصورة الأصلية")

# البصمة الجزئية
with col2:
    st.subheader("البصمة الجزئية")
    partial_file = st.file_uploader("اختر البصمة الجزئية", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="partial")
    
    if partial_file is not None:
        try:
            # قراءة الصورة
            partial_pil = Image.open(partial_file)
            partial_img = np.array(partial_pil.convert('L'))
            
            # عرض الصورة الجزئية
            st.image(partial_pil, caption="البصمة الجزئية", use_column_width=True)
            
            # معالجة البصمة الجزئية
            with st.spinner("جاري معالجة البصمة الجزئية..."):
                # المعالجة المسبقة
                processed_partial = preprocess_image(partial_img)
                if processed_partial is not None:
                    # استخراج الحواف
                    ridges_partial, orientation_map_partial = detect_edges(processed_partial)
                    if ridges_partial is not None:
                        # تحسين الصورة
                        enhanced_partial = enhance_image(ridges_partial)
                        if enhanced_partial is not None:
                            # استخراج النقاط الدقيقة
                            minutiae_partial = extract_minutiae(enhanced_partial)
                            if minutiae_partial:
                                # تصور النقاط
                                vis_partial = visualize_minutiae(enhanced_partial, minutiae_partial)
                                if vis_partial is not None:
                                    st.image(vis_partial, caption="البصمة المعالجة", use_column_width=True)
                                    st.success(f"تم استخراج {len(minutiae_partial)} نقطة مميزة")
                                else:
                                    st.error("فشل في تصور النقاط")
                            else:
                                st.error("لم يتم العثور على نقاط مميزة")
                        else:
                            st.error("فشل في تحسين الصورة")
                    else:
                        st.error("فشل في استخراج الحواف")
                else:
                    st.error("فشل في المعالجة المسبقة")
        except Exception as e:
            logger.error(f"Error processing partial image: {str(e)}")
            st.error("حدث خطأ أثناء معالجة الصورة الجزئية")

# زر المطابقة
if st.button("بدء المطابقة", type="primary"):
    if original_file is not None and partial_file is not None:
        if minutiae_original and minutiae_partial:
            with st.spinner("جاري مطابقة البصمات..."):
                try:
                    # مطابقة البصمات
                    match_result = match_fingerprints(minutiae_original, minutiae_partial)
                    
                    # تصور النقاط المتطابقة
                    matches_vis = visualize_matches(enhanced_original, enhanced_partial, match_result)
                    if matches_vis is not None:
                        st.image(matches_vis, caption="النقاط المتطابقة", use_column_width=True)
                    
                    # عرض النتائج
                    st.markdown("---")
                    st.subheader("📊 ملخص النتائج")
                    
                    # صندوق النتائج
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
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    
                    # عدد النقاط
                    st.markdown(f'<div class="result-item">🔎 عدد النقاط المستخرجة من الأصلية: <span class="highlight">{match_result["total_original"]}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-item">🔎 عدد النقاط المستخرجة من الجزئية: <span class="highlight">{match_result["total_partial"]}</span></div>', unsafe_allow_html=True)
                    
                    # نتائج المطابقة
                    st.markdown(f'<div class="result-item">✅ نقاط التطابق: <span class="success">{match_result["matched_points"]}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-item">✅ نسبة التشابه: <span class="success">{match_result["match_score"]:.2f}%</span></div>', unsafe_allow_html=True)
                    
                    # القرار
                    decision_class = "high-match" if match_result["match_score"] > MATCH_SCORE_THRESHOLDS['HIGH'] else "medium-match" if match_result["match_score"] > MATCH_SCORE_THRESHOLDS['MEDIUM'] else "low-match"
                    decision_text = f'HIGH MATCH - احتمالية التطابق كبيرة جدًا' if match_result["match_score"] > MATCH_SCORE_THRESHOLDS['HIGH'] else f'MEDIUM MATCH - احتمالية التطابق متوسطة' if match_result["match_score"] > MATCH_SCORE_THRESHOLDS['MEDIUM'] else f'LOW MATCH - احتمالية التطابق منخفضة'
                    
                    st.markdown(f'<div class="result-item">✅ القرار: <span class="{decision_class}">{decision_text}</span></div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # توليد التقرير
                    report_path = generate_report(enhanced_original, enhanced_partial, match_result)
                    if report_path:
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                label="تحميل التقرير",
                                data=f,
                                file_name="matched_result.pdf",
                                mime="application/pdf"
                            )
                    
                except Exception as e:
                    logger.error(f"Error in matching: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("حدث خطأ أثناء المطابقة")
        else:
            st.error("يرجى التأكد من معالجة البصمتين بنجاح")
    else:
        st.error("يرجى تحميل البصمتين أولاً") 