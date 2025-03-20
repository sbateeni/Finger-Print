import streamlit as st
import logging

logger = logging.getLogger(__name__)

def setup_page_config():
    """إعداد تكوين الصفحة"""
    try:
        st.set_page_config(
            page_title="نظام مطابقة البصمات",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        logger.error(f"خطأ في تكوين الصفحة: {str(e)}")

def display_header():
    """عرض عنوان التطبيق"""
    st.title("🔍 نظام مطابقة البصمات")
    st.markdown("""
    ### نظام متقدم لمطابقة البصمات باستخدام الذكاء الاصطناعي
    """)

def display_error(message):
    """عرض رسالة خطأ"""
    st.error(message)
    st.stop()

def display_summary_results(original_count, partial_count, matched_points, match_score, decision):
    """عرض ملخص النتائج"""
    st.markdown("---")
    st.subheader("📊 ملخص النتائج")
    
    # تعريف الأنماط
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