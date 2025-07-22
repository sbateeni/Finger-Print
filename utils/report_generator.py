import os
import cv2
import numpy as np
from datetime import datetime
from config import *

def generate_report(original_img, partial_img, match_result, output_dir=OUTPUT_DIR):
    """توليد تقرير PDF"""
    try:
        # إنشاء مجلد النتائج إذا لم يكن موجوداً
        os.makedirs(output_dir, exist_ok=True)
        
        # إنشاء اسم الملف
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"matched_result_{timestamp}.pdf"
        report_path = os.path.join(output_dir, report_name)
        
        # حفظ الصور
        original_path = os.path.join(output_dir, f"original_{timestamp}.png")
        partial_path = os.path.join(output_dir, f"partial_{timestamp}.png")
        cv2.imwrite(original_path, original_img)
        cv2.imwrite(partial_path, partial_img)
        
        # إنشاء محتوى التقرير
        report_content = f"""
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>تقرير مطابقة البصمات</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 20px;
                    background-color: #f0f2f6;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .result-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .highlight {{
                    color: #0068c9;
                    font-weight: bold;
                }}
                .success {{
                    color: #09ab3b;
                    font-weight: bold;
                }}
                .high-match {{
                    color: #09ab3b;
                    font-weight: bold;
                    font-size: 24px;
                    padding: 10px;
                    background-color: rgba(9, 171, 59, 0.1);
                    border-radius: 5px;
                }}
                .medium-match {{
                    color: #f0a202;
                    font-weight: bold;
                    font-size: 24px;
                    padding: 10px;
                    background-color: rgba(240, 162, 2, 0.1);
                    border-radius: 5px;
                }}
                .low-match {{
                    color: #ff0000;
                    font-weight: bold;
                    font-size: 24px;
                    padding: 10px;
                    background-color: rgba(255, 0, 0, 0.1);
                    border-radius: 5px;
                }}
                .images {{
                    display: flex;
                    justify-content: space-between;
                    margin: 20px 0;
                }}
                .image-container {{
                    text-align: center;
                }}
                .image-container img {{
                    max-width: 300px;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>تقرير مطابقة البصمات</h1>
                    <p>تاريخ التقرير: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="images">
                    <div class="image-container">
                        <h3>البصمة الأصلية</h3>
                        <img src="{original_path}" alt="البصمة الأصلية">
                    </div>
                    <div class="image-container">
                        <h3>البصمة الجزئية</h3>
                        <img src="{partial_path}" alt="البصمة الجزئية">
                    </div>
                </div>
                
                <div class="result-box">
                    <h2>نتائج المطابقة</h2>
                    <p>🔎 عدد النقاط المستخرجة من الأصلية: <span class="highlight">{match_result["total_original"]}</span></p>
                    <p>🔎 عدد النقاط المستخرجة من الجزئية: <span class="highlight">{match_result["total_partial"]}</span></p>
                    <p>✅ نقاط التطابق: <span class="success">{match_result["matched_points"]}</span></p>
                    <p>✅ نسبة التشابه: <span class="success">{match_result["match_score"]:.2f}%</span></p>
                    
                    <div class="{match_result['status'].lower().replace(' ', '-')}">
                        <h3>القرار: {match_result['status']}</h3>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # حفظ التقرير
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None 