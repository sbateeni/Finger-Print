import cv2
import numpy as np
from config import *

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

def extract_minutiae(skeleton):
    """استخراج النقاط الدقيقة"""
    try:
        minutiae = []
        
        # استخراج النقاط الدقيقة
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
        
        return minutiae
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

def visualize_minutiae(image, minutiae):
    """تصور النقاط الدقيقة"""
    try:
        # إنشاء صورة التصور
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
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
        
        return vis_img
    except Exception as e:
        print(f"Error in visualize_minutiae: {str(e)}")
        return None 