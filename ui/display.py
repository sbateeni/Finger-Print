import streamlit as st
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

def display_summary_results(original_count, partial_count, matched_points, match_score, decision):
    """عرض ملخص النتائج"""
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
    """كتابة النتائج إلى ملف"""
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