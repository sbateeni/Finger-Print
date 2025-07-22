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

# إعداد معايير الفلترة من الواجهة
st.sidebar.header("إعدادات فلترة النقاط الدقيقة")
DEFAULT_BORDER_MARGIN = 10
DEFAULT_MIN_DISTANCE = 10
DEFAULT_MIN_CONTRAST = 15
DEFAULT_MIN_ANGLE_DIFF = 10

if 'border_margin' not in st.session_state:
    st.session_state['border_margin'] = DEFAULT_BORDER_MARGIN
if 'min_distance' not in st.session_state:
    st.session_state['min_distance'] = DEFAULT_MIN_DISTANCE
if 'min_contrast' not in st.session_state:
    st.session_state['min_contrast'] = DEFAULT_MIN_CONTRAST
if 'min_angle_diff' not in st.session_state:
    st.session_state['min_angle_diff'] = DEFAULT_MIN_ANGLE_DIFF

if st.sidebar.button("إعادة ضبط إعدادات الفلترة للوضع الافتراضي"):
    st.session_state['border_margin'] = DEFAULT_BORDER_MARGIN
    st.session_state['min_distance'] = DEFAULT_MIN_DISTANCE
    st.session_state['min_contrast'] = DEFAULT_MIN_CONTRAST
    st.session_state['min_angle_diff'] = DEFAULT_MIN_ANGLE_DIFF

border_margin = st.sidebar.slider(
    "هامش الحواف (بكسل) !",
    min_value=0, max_value=50, value=st.session_state['border_margin'], step=1, key='border_margin',
    help="النقاط القريبة من الحافة سيتم تجاهلها. زيادة القيمة تقلل النقاط القريبة من الأطراف."
)
min_distance = st.sidebar.slider(
    "أقل مسافة بين النقاط (بكسل) !",
    min_value=1, max_value=50, value=st.session_state['min_distance'], step=1, key='min_distance',
    help="أقل مسافة مسموحة بين نقطتين مميزتين. زيادة القيمة تقلل النقاط المتقاربة."
)
min_contrast = st.sidebar.slider(
    "أقل تباين محلي (للفلترة الذكية) !",
    min_value=0, max_value=50, value=st.session_state['min_contrast'], step=1, key='min_contrast',
    help="النقاط في مناطق قليلة التباين (غامقة أو باهتة) سيتم تجاهلها."
)
min_angle_diff = st.sidebar.slider(
    "أقل فرق زاوية بين النقاط المتقاربة (درجة) !",
    min_value=0, max_value=90, value=st.session_state['min_angle_diff'], step=1, key='min_angle_diff',
    help="النقاط المتقاربة جدًا في الاتجاه (الزاوية) سيتم تجاهلها."
)

# إعدادات إزالة الضوضاء
st.sidebar.header("إعدادات إزالة الضوضاء")
DEFAULT_DENOISE_METHOD = "fastNlMeans"
DEFAULT_FAST_DENOISE_H = 10
DEFAULT_GAUSS_KSIZE = 3

if 'denoise_method' not in st.session_state:
    st.session_state['denoise_method'] = DEFAULT_DENOISE_METHOD
if 'fast_denoise_h' not in st.session_state:
    st.session_state['fast_denoise_h'] = DEFAULT_FAST_DENOISE_H
if 'gauss_ksize' not in st.session_state:
    st.session_state['gauss_ksize'] = DEFAULT_GAUSS_KSIZE

if st.sidebar.button("إعادة ضبط إعدادات إزالة الضوضاء للوضع الافتراضي"):
    st.session_state['denoise_method'] = DEFAULT_DENOISE_METHOD
    st.session_state['fast_denoise_h'] = DEFAULT_FAST_DENOISE_H
    st.session_state['gauss_ksize'] = DEFAULT_GAUSS_KSIZE

denoise_method = st.sidebar.selectbox(
    "طريقة إزالة الضوضاء !",
    ["None", "fastNlMeans", "GaussianBlur"],
    index=["None", "fastNlMeans", "GaussianBlur"].index(st.session_state['denoise_method']),
    key='denoise_method',
    help="اختر طريقة إزالة الضوضاء من الصورة: None (بدون)، fastNlMeans (فلترة متقدمة)، GaussianBlur (تمويه غاوسي)."
)
fast_denoise_h = st.sidebar.slider(
    "قوة fastNlMeans (h) !",
    min_value=1, max_value=30, value=st.session_state['fast_denoise_h'], step=1, key='fast_denoise_h',
    help="كلما زادت القيمة زادت قوة إزالة الضوضاء (قد تؤثر على التفاصيل الدقيقة)."
)
gauss_ksize = st.sidebar.slider(
    "حجم نواة GaussianBlur !",
    min_value=1, max_value=21, value=st.session_state['gauss_ksize'], step=2, key='gauss_ksize',
    help="حجم النواة المستخدمة في GaussianBlur. يجب أن تكون فردية. زيادة القيمة تزيد التمويه."
)

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
                processed_original = preprocess_image(
                    original_img,
                    denoise_method=denoise_method,
                    fast_denoise_h=fast_denoise_h,
                    gauss_ksize=gauss_ksize
                )
                if processed_original is not None:
                    st.image(processed_original, caption="بعد المعالجة المسبقة (ثنائية)", use_column_width=True)
                    st.write(f"عدد البكسلات البيضاء بعد المعالجة المسبقة: {np.sum(processed_original == 255)}")
                    # استخراج الحواف
                    ridges_original, orientation_map_original = detect_edges(processed_original)
                    if ridges_original is not None:
                        st.image(ridges_original, caption="بعد استخراج الحواف (Gabor)", use_column_width=True)
                        st.write(f"عدد البكسلات البيضاء بعد استخراج الحواف: {np.sum(ridges_original == 255)}")
                        # تحسين الصورة
                        enhanced_original = enhance_image(ridges_original)
                        if enhanced_original is not None:
                            st.image(enhanced_original, caption="بعد الهيكلة (Skeleton)", use_column_width=True)
                            st.write(f"عدد البكسلات البيضاء بعد الهيكلة: {np.sum(enhanced_original == 255)}")
                            # استخراج النقاط الدقيقة مع الفلترة
                            minutiae_original = extract_minutiae(
                                enhanced_original,
                                border_margin=border_margin,
                                min_distance=min_distance,
                                original_image=processed_original,
                                min_contrast=min_contrast,
                                min_angle_diff=min_angle_diff
                            )
                            st.write(f"عدد النقاط الدقيقة بعد الفلترة: {len(minutiae_original)}")
                            st.session_state['minutiae_original'] = minutiae_original
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
                processed_partial = preprocess_image(
                    partial_img,
                    denoise_method=denoise_method,
                    fast_denoise_h=fast_denoise_h,
                    gauss_ksize=gauss_ksize
                )
                if processed_partial is not None:
                    st.image(processed_partial, caption="بعد المعالجة المسبقة (ثنائية)", use_column_width=True)
                    st.write(f"عدد البكسلات البيضاء بعد المعالجة المسبقة: {np.sum(processed_partial == 255)}")
                    # استخراج الحواف
                    ridges_partial, orientation_map_partial = detect_edges(processed_partial)
                    if ridges_partial is not None:
                        st.image(ridges_partial, caption="بعد استخراج الحواف (Gabor)", use_column_width=True)
                        st.write(f"عدد البكسلات البيضاء بعد استخراج الحواف: {np.sum(ridges_partial == 255)}")
                        # تحسين الصورة
                        enhanced_partial = enhance_image(ridges_partial)
                        if enhanced_partial is not None:
                            st.image(enhanced_partial, caption="بعد الهيكلة (Skeleton)", use_column_width=True)
                            st.write(f"عدد البكسلات البيضاء بعد الهيكلة: {np.sum(enhanced_partial == 255)}")
                            # استخراج النقاط الدقيقة مع الفلترة
                            minutiae_partial = extract_minutiae(
                                enhanced_partial,
                                border_margin=border_margin,
                                min_distance=min_distance,
                                original_image=processed_partial,
                                min_contrast=min_contrast,
                                min_angle_diff=min_angle_diff
                            )
                            st.write(f"عدد النقاط الدقيقة بعد الفلترة: {len(minutiae_partial)}")
                            st.session_state['minutiae_partial'] = minutiae_partial
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

minutiae_original = None
minutiae_partial = None
enhanced_original = None
enhanced_partial = None
# زر المطابقة
if st.button("بدء المطابقة", type="primary"):
    minutiae_original = st.session_state.get('minutiae_original')
    minutiae_partial = st.session_state.get('minutiae_partial')
    if original_file is not None and partial_file is not None:
        if minutiae_original is not None and minutiae_partial is not None and len(minutiae_original) > 0 and len(minutiae_partial) > 0:
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
            st.error("يرجى التأكد من معالجة البصمتين بنجاح واستخراج نقاط مميزة كافية")
    else:
        st.error("يرجى تحميل البصمتين أولاً") 