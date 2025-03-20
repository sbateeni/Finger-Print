import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def write_results_to_file(match_result):
    """كتابة نتائج المطابقة إلى ملف"""
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
        logger.error(f"خطأ في كتابة النتائج إلى ملف: {str(e)}")
        return None 