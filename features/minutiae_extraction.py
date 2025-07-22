import cv2
import numpy as np
from preprocessing.image_processing import detect_ridges, analyze_ridge_patterns
from scipy import ndimage

def get_minutiae_type(pixel, neighbors):
    """
    Determine the type of minutiae point based on its neighborhood
    Returns: 'ending', 'bifurcation', 'dot', or None
    """
    # Count the number of 1's in the neighborhood
    count = np.sum(neighbors)
    
    if count == 1:
        return 'ending'
    elif count == 3:
        return 'bifurcation'
    elif count == 0 and pixel == 1:
        return 'dot'
    return None

def get_minutiae_angle(neighbors):
    """
    Calculate the angle of the minutiae point based on its neighborhood
    """
    try:
        # Make sure we're working with binary data
        binary_neighbors = (neighbors > 0).astype(np.uint8)
        
        # Find the direction of the ridge
        y, x = np.where(binary_neighbors == 1)
        if len(x) > 0:
            # Calculate angle based on the position of the neighbor
            angle = np.arctan2(y[0] - 1, x[0] - 1)
            # Convert to degrees and normalize to 0-360 range
            angle = np.degrees(angle)
            if angle < 0:
                angle += 360
            return angle
        return 0
    except Exception as e:
        print(f"Error in get_minutiae_angle: {str(e)}")
        return 0

def detect_dots(binary):
    """اكتشاف النقاط المعزولة (الجزر)"""
    dots = []
    kernel = np.ones((3,3), np.uint8)
    
    # تطبيق العمليات المورفولوجية
    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(binary, kernel)
    
    # البحث عن النقاط المعزولة
    isolated = cv2.bitwise_and(binary, cv2.bitwise_not(dilated))
    
    # استخراج إحداثيات النقاط المعزولة
    y_coords, x_coords = np.where(isolated > 0)
    for x, y in zip(x_coords, y_coords):
        dots.append({
            'x': x,
            'y': y,
            'type': 'dot',
            'angle': 0,
            'quality': 1.0
        })
    
    return dots

def extract_minutiae(image):
    """استخراج النقاط الدقيقة من الصورة"""
    try:
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
        
        return minutiae
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

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

def filter_minutiae(minutiae):
    """تصفية النقاط الدقيقة لإزالة النقاط المتكررة والقريبة"""
    if not minutiae:
        return []
    
    # إزالة النقاط المتكررة
    unique_minutiae = []
    seen = set()
    for m in minutiae:
        key = (m['x'], m['y'])
        if key not in seen:
            seen.add(key)
            unique_minutiae.append(m)
    
    # إزالة النقاط القريبة
    filtered = []
    min_distance = 5  # الحد الأدنى للمسافة بين النقاط
    
    for i, m1 in enumerate(unique_minutiae):
        is_valid = True
        for m2 in unique_minutiae[i+1:]:
            distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            if distance < min_distance:
                is_valid = False
                break
        if is_valid:
            filtered.append(m1)
    
    return filtered

def analyze_ridge_characteristics(image, minutiae):
    """تحليل خصائص الخطوط"""
    characteristics = []
    
    for point in minutiae:
        # تحليل اتجاه الخط
        direction = calculate_ridge_direction(image, point)
        
        # تحليل جودة الخط
        quality = calculate_ridge_quality(image, point)
        
        characteristics.append({
            'point': point,
            'direction': direction,
            'quality': quality
        })
    
    return characteristics

def classify_minutiae(minutiae):
    """تصنيف النقاط الدقيقة"""
    classified = {
        'endings': [],
        'bifurcations': [],
        'islands': []
    }
    
    for point in minutiae:
        if point['type'] == 'ending':
            classified['endings'].append(point)
        elif point['type'] == 'bifurcation':
            classified['bifurcations'].append(point)
    
    return classified

def calculate_minutiae_angles(minutiae):
    """حساب زوايا النقاط الدقيقة"""
    for point in minutiae:
        # حساب الزاوية باستخدام التدرج
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        angle = np.arctan2(gradient_y[point['y'], point['x']],
                          gradient_x[point['y'], point['x']])
        
        point['angle'] = angle
    
    return minutiae

def calculate_ridge_direction(image, point):
    """حساب اتجاه الخط"""
    # استخراج منطقة حول النقطة
    region = image[point['y']-5:point['y']+6, point['x']-5:point['x']+6]
    
    # حساب التدرج
    gradient_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    
    # حساب الاتجاه
    direction = np.arctan2(gradient_y[5,5], gradient_x[5,5])
    
    return direction

def calculate_ridge_quality(image, point):
    """حساب جودة الخط"""
    # استخراج منطقة حول النقطة
    region = image[point['y']-3:point['y']+4, point['x']-3:point['x']+4]
    
    # حساب التباين المحلي
    local_contrast = np.std(region)
    
    # حساب وضوح الخط
    gradient_magnitude = np.sqrt(np.sum(region**2))
    
    # حساب الجودة النهائية
    quality = (local_contrast + gradient_magnitude) / 2
    
    return quality

def visualize_minutiae(image, minutiae, ridges=None):
    """عرض الصورة مع النقاط الدقيقة والخطوط"""
    try:
        # نسخ الصورة الأصلية
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # رسم الخطوط إذا كانت متوفرة
        if ridges is not None:
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
        
        return vis_img
    except Exception as e:
        print(f"Error in visualize_minutiae: {str(e)}")
        return image 