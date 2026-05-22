import cv2
import numpy as np
from config import *

def calculate_angle(skeleton, x, y, window_size=5):
    """
    حساب زاوية النقطة الدقيقة من خلال تتبع الحافة لمسافة قصيرة.
    يوفر دقة أعلى للمطابقة.
    """
    try:
        # البحث عن النقاط المجاورة في نافذة صغيرة لتحديد اتجاه الحافة
        h, w = skeleton.shape
        points = []
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] == 255:
                    points.append((nx, ny))
        
        if len(points) >= 2:
            # استخدام PCA بسيط أو متوسط الإحداثيات لتحديد الاتجاه
            pts = np.array(points)
            diff = pts - np.array([x, y])
            # نريد الاتجاه الغالب
            avg_dx = np.mean(diff[:, 0])
            avg_dy = np.mean(diff[:, 1])
            angle = np.arctan2(avg_dy, avg_dx)
            return np.degrees(angle)
        return 0
    except Exception as e:
        return 0

def remove_false_minutiae(skeleton, minutiae, min_ridge_length=10):
    """
    إزالة النقاط الزائفة الناتجة عن عيوب الصورة (Spurs, Islands, Broken Ridges, Hooks).
    """
    if not minutiae: return []
    
    filtered = []
    h, w = skeleton.shape
    
    # تحويل النقاط إلى قاموس للوصول السريع
    minutiae_map = {(m['x'], m['y']): m for m in minutiae}
    
    for m in minutiae:
        x, y = m['x'], m['y']
        
        is_false = False
        
        # 1. إزالة الجزر (Islands): نقاط معزولة جداً
        # فحص ما إذا كانت الحافة تصل إلى نقطة دقيقة أخرى
        if m['type'] == 'endpoint':
            # تتبع الحافة
            curr_x, curr_y = x, y
            visited = set([(x, y)])
            length = 0
            found_other_minutia = False
            
            for _ in range(100):  # حد أقصى للتتبع
                found_next = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = curr_x + dx, curr_y + dy
                        if (0 <= nx < w and 0 <= ny < h and 
                            skeleton[ny, nx] == 255 and (nx, ny) not in visited):
                            visited.add((nx, ny))
                            curr_x, curr_y = nx, ny
                            length += 1
                            found_next = True
                            
                            # فحص ما إذا وصلنا إلى نقطة دقيقة أخرى
                            if (nx, ny) in minutiae_map:
                                found_other_minutia = True
                            break
                    if found_next:
                        break
                if not found_next or found_other_minutia:
                    break
            
            # إذا لم نصل إلى نقطة دقيقة أخرى والطول قصير: إزالة
            if not found_other_minutia and length < min_ridge_length:
                is_false = True
        
        # 2. إزالة Hooks: نهايات قريبة جداً من نقاط دقيقة أخرى
        if not is_false and m['type'] == 'endpoint':
            for other in minutiae:
                if other is m:
                    continue
                dist = np.hypot(other['x'] - x, other['y'] - y)
                if dist < 12:
                    is_false = True
                    break
        
        if not is_false:
            filtered.append(m)
            
    return filtered

def filter_minutiae(minutiae, image_shape, border_margin=10, min_distance=10, original_image=None, min_contrast=15, min_angle_diff=10):
    """فلترة النقاط الدقيقة: تجاهل النقاط القريبة من الحواف أو المتقاربة جداً أو في مناطق قليلة التباين"""
    filtered = []
    for m in minutiae:
        x, y = m['x'], m['y']
        # تجاهل النقاط القريبة من الحواف
        if x < border_margin or y < border_margin or x > image_shape[1] - border_margin or y > image_shape[0] - border_margin:
            continue
            
        # تجاهل النقاط في مناطق قليلة التباين (مناطق الضوضاء)
        if original_image is not None:
            # نافذة أكبر قليلاً لتقييم التباين
            y_start, y_end = max(0, y-5), min(image_shape[0], y+6)
            x_start, x_end = max(0, x-5), min(image_shape[1], x+6)
            local_patch = original_image[y_start:y_end, x_start:x_end]
            if local_patch.size > 0 and np.std(local_patch) < min_contrast:
                continue
                
        # إزالة النقاط المتلاصقة جداً (نفس المنطقة)
        too_close = False
        for f in filtered:
            dist = np.hypot(f['x'] - x, f['y'] - y)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered.append(m)
            
    return filtered

def extract_minutiae(skeleton, border_margin=10, min_distance=10, original_image=None, min_contrast=15, min_angle_diff=10):
    """استخراج النقاط الدقيقة مع معالجة متقدمة للعيوب"""
    try:
        raw_minutiae = []
        h, w = skeleton.shape
        
        # استخدام Crossing Number (CN) لتحديد النقاط
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 255:
                    # مصفوفة الجيران (3x3)
                    p = [
                        skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
                        skeleton[y, x+1], skeleton[y+1, x+1], skeleton[y+1, x],
                        skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
                    ]
                    # حساب عدد الانتقالات من 0 لـ 255 (CN)
                    cn = 0
                    for i in range(8):
                        if p[i] == 0 and p[i+1] == 255:
                            cn += 1
                    
                    if cn == 1: # Endpoint
                        raw_minutiae.append({
                            'x': x, 'y': y, 'type': 'endpoint',
                            'angle': calculate_angle(skeleton, x, y)
                        })
                    elif cn == 3: # Bifurcation
                        raw_minutiae.append({
                            'x': x, 'y': y, 'type': 'bifurcation',
                            'angle': calculate_angle(skeleton, x, y)
                        })
        
        # 1. إزالة النقاط الزائفة هندسياً (Spurs, Islands)
        refined = remove_false_minutiae(skeleton, raw_minutiae)
        
        # 2. الفلترة المكانية والتباين
        filtered = filter_minutiae(refined, skeleton.shape, border_margin, min_distance, original_image, min_contrast, min_angle_diff)
        
        return filtered
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

def visualize_minutiae(image, minutiae):
    """تصور احترافي للنقاط الدقيقة"""
    try:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for point in minutiae:
            x, y = point['x'], point['y']
            # اللون: أخضر للنهايات، أحمر للتفرعات
            color = (0, 255, 0) if point['type'] == 'endpoint' else (0, 0, 255)
            
            # رسم دائرة صغيرة
            cv2.circle(vis_img, (x, y), 4, color, 1)
            cv2.circle(vis_img, (x, y), 1, color, -1)
            
            # رسم سهم الاتجاه
            angle = point['angle']
            length = 12
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            cv2.arrowedLine(vis_img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return vis_img
    except Exception as e:
        return None

def visualize_singular_points(image, cores, deltas):
    """تصور النقاط المفردة (مراكز ودلتات)"""
    try:
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            
        for c in cores:
            # Core: دائرة ذهبية كبيرة (نستخدم لون ذهبي للمركز)
            cv2.circle(vis, (c['x'], c['y']), 12, (0, 215, 255), 2)
            cv2.circle(vis, (c['x'], c['y']), 3, (0, 215, 255), -1)
            cv2.putText(vis, "Core", (c['x']+14, c['y']-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
            
        for d in deltas:
            # Delta: مثلث أبيض يمثل نقطة التلاقي الثلاثية
            pts = np.array([
                [d['x'], d['y']-10], [d['x']-10, d['y']+10], [d['x']+10, d['y']+10]
            ], np.int32)
            cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
            cv2.putText(vis, "Delta", (d['x']+14, d['y']+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return vis
    except Exception as e:
        print(f"Error visualizing singular points: {e}")
        return image

 