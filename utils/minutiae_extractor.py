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

def filter_minutiae(minutiae, image_shape, border_margin=10, min_distance=10, original_image=None, min_contrast=15, min_angle_diff=10):
    """فلترة النقاط الدقيقة: تجاهل النقاط القريبة من الحواف أو المتقاربة جداً أو في مناطق قليلة التباين أو ذات اتجاهات متشابهة جداً"""
    filtered = []
    for m in minutiae:
        x, y = m['x'], m['y']
        # تجاهل النقاط القريبة من الحواف
        if x < border_margin or y < border_margin or x > image_shape[1] - border_margin or y > image_shape[0] - border_margin:
            continue
        # تجاهل النقاط في مناطق قليلة التباين
        if original_image is not None:
            local_patch = original_image[max(0, y-3):min(image_shape[0], y+4), max(0, x-3):min(image_shape[1], x+4)]
            if local_patch.size > 0 and np.std(local_patch) < min_contrast:
                continue
        # تجاهل النقاط المتقاربة جداً أو ذات الزاوية المتشابهة
        too_close = False
        for f in filtered:
            dist = np.hypot(f['x'] - x, f['y'] - y)
            angle_diff = abs((f['angle'] - m['angle'] + 180) % 360 - 180)
            if dist < min_distance and angle_diff < min_angle_diff:
                too_close = True
                break
        if not too_close:
            filtered.append(m)
    return filtered

def extract_minutiae(skeleton, border_margin=10, min_distance=10, original_image=None, min_contrast=15, min_angle_diff=10):
    """استخراج النقاط الدقيقة مع الفلترة الذكية"""
    try:
        minutiae = []
        for y in range(1, skeleton.shape[0]-1):
            for x in range(1, skeleton.shape[1]-1):
                if skeleton[y, x] == 255:
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1
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
        # فلترة النقاط
        filtered = filter_minutiae(minutiae, skeleton.shape, border_margin, min_distance, original_image, min_contrast, min_angle_diff)
        return filtered
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