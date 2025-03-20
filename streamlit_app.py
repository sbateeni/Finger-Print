import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import logging
import traceback

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

def process_image(image):
    try:
        # تحويل الصورة إلى Grayscale إذا لم تكن كذلك
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # توحيد حجم الصورة
        image = cv2.resize(image, (500, 500))
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # تطبيق فلتر Gabor
        ksize = 31
        sigma = 4.0
        theta = 0
        lambda_ = 10.0
        gamma = 0.5
        psi = 0
        
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
        ridges = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        # تطبيق فلتر Sobel للحصول على خريطة الاتجاه
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # تجنب القسمة على صفر
        orientation_map = np.zeros_like(sobelx)
        mask = (sobelx != 0) | (sobely != 0)
        orientation_map[mask] = np.arctan2(sobely[mask], sobelx[mask])
        
        # تحويل الصورة إلى ثنائية
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # تطبيق عملية الهيكلة
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.ximgproc.thinning(binary)
        
        # استخراج النقاط الدقيقة
        minutiae = []
        for y in range(1, skeleton.shape[0]-1):
            for x in range(1, skeleton.shape[1]-1):
                if skeleton[y, x] == 255:
                    # حساب عدد الجيران
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 255
                    
                    # تحديد نوع النقطة
                    if neighbors == 1:  # نقطة نهاية
                        angle = calculate_angle(skeleton, x, y)
                        minutiae.append({
                            'x': x,
                            'y': y,
                            'type': 'endpoint',
                            'angle': angle,
                            'magnitude': 1.0
                        })
                    elif neighbors == 3:  # نقطة تفرع
                        angle = calculate_angle(skeleton, x, y)
                        minutiae.append({
                            'x': x,
                            'y': y,
                            'type': 'bifurcation',
                            'angle': angle,
                            'magnitude': 1.0
                        })
        
        # إنشاء صورة التصور
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vis_img = cv2.addWeighted(vis_img, 0.7, cv2.cvtColor(ridges, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # رسم النقاط الدقيقة
        for point in minutiae:
            x, y = point['x'], point['y']
            color = (0, 255, 0) if point['type'] == 'endpoint' else (0, 0, 255)
            
            # رسم النقطة
            cv2.circle(vis_img, (x, y), 3, color, -1)
            
            # رسم اتجاه النقطة
            angle = point['angle']
            length = 10
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            cv2.line(vis_img, (x, y), (end_x, end_y), color, 1)
        
        return vis_img, minutiae
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def calculate_angle(skeleton, x, y):
    """حساب زاوية النقطة الدقيقة"""
    try:
        # البحث عن النقاط المجاورة
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if skeleton[y+dy, x+dx] == 255:
                    neighbors.append((x+dx, y+dy))
        
        if len(neighbors) >= 2:
            # حساب المتوسط الهندسي
            mean_x = np.mean([n[0] for n in neighbors])
            mean_y = np.mean([n[1] for n in neighbors])
            angle = np.arctan2(mean_y - y, mean_x - x)
            return np.degrees(angle)
        return 0
    except Exception as e:
        print(f"Error in calculate_angle: {str(e)}")
        return 0

def match_fingerprints(original_minutiae, partial_minutiae):
    """مطابقة النقاط الدقيقة"""
    try:
        # تحويل النقاط إلى مصفوفات
        original_points = np.array([[m['x'], m['y']] for m in original_minutiae])
        partial_points = np.array([[m['x'], m['y']] for m in partial_minutiae])
        
        # حساب مصفوفة المسافات
        distances = np.zeros((len(original_points), len(partial_points)))
        for i, p1 in enumerate(original_points):
            for j, p2 in enumerate(partial_points):
                distances[i, j] = np.sqrt(np.sum((p1 - p2) ** 2))
        
        # تطبيق خوارزمية المطابقة
        matched_points = []
        for i in range(len(original_points)):
            min_dist = np.min(distances[i])
            min_idx = np.argmin(distances[i])
            if min_dist < 10:  # حد المسافة المسموح به
                matched_points.append({
                    'original': original_minutiae[i],
                    'partial': partial_minutiae[min_idx],
                    'distance': min_dist
                })
        
        # حساب نسبة التطابق
        match_score = len(matched_points) / len(partial_minutiae) * 100 if partial_minutiae else 0
        
        # تحديد الحالة
        status = "HIGH MATCH" if match_score > 75 else \
                 "MEDIUM MATCH" if match_score > 50 else \
                 "LOW MATCH" if match_score > 25 else \
                 "NO MATCH"
        
        return {
            'matched_points': len(matched_points),
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': match_score,
            'status': status
        }
    except Exception as e:
        print(f"Error in match_fingerprints: {str(e)}")
        return {
            'matched_points': 0,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'match_score': 0,
            'status': "ERROR"
        }

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
                processed_original, minutiae_original = process_image(original_img)
                if processed_original is not None:
                    st.image(processed_original, caption="البصمة المعالجة", use_column_width=True)
                    st.success(f"تم استخراج {len(minutiae_original)} نقطة مميزة")
                else:
                    st.error("فشل في معالجة البصمة الأصلية")
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
                processed_partial, minutiae_partial = process_image(partial_img)
                if processed_partial is not None:
                    st.image(processed_partial, caption="البصمة المعالجة", use_column_width=True)
                    st.success(f"تم استخراج {len(minutiae_partial)} نقطة مميزة")
                else:
                    st.error("فشل في معالجة البصمة الجزئية")
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
                    decision_class = "high-match" if match_result["match_score"] > 75 else "medium-match" if match_result["match_score"] > 50 else "low-match"
                    decision_text = f'HIGH MATCH - احتمالية التطابق كبيرة جدًا' if match_result["match_score"] > 75 else f'MEDIUM MATCH - احتمالية التطابق متوسطة' if match_result["match_score"] > 50 else f'LOW MATCH - احتمالية التطابق منخفضة'
                    
                    st.markdown(f'<div class="result-item">✅ القرار: <span class="{decision_class}">{decision_text}</span></div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    logger.error(f"Error in matching: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("حدث خطأ أثناء المطابقة")
        else:
            st.error("يرجى التأكد من معالجة البصمتين بنجاح")
    else:
        st.error("يرجى تحميل البصمتين أولاً") 