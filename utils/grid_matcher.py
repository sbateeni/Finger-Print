import cv2
import numpy as np
from scipy import ndimage
from .minutiae_extraction import extract_minutiae
from .feature_extraction import estimate_ridge_frequency
from config import *

def normalize_ridge_distance(image, target_distance):
    """
    تعديل حجم الصورة بحيث تصبح المسافة بين الخطوط مساوية للمسافة المطلوبة
    
    Args:
        image: صورة البصمة
        target_distance: المسافة المطلوبة بين الخطوط
        
    Returns:
        numpy.ndarray: الصورة بعد تعديل الحجم
    """
    # حساب متوسط المسافة بين الخطوط في الصورة الحالية
    freq = estimate_ridge_frequency(image)
    current_distance = 1.0 / np.mean(freq[freq > 0])
    
    # حساب معامل التكبير/التصغير
    scale_factor = target_distance / current_distance
    print(f"معامل تغيير الحجم: {scale_factor:.2f}")
    
    # تعديل حجم الصورة
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image, scale_factor

def divide_into_grids(full_image, grid_size):
    """
    تقسيم البصمة الكاملة إلى مربعات متساوية الحجم
    
    Args:
        full_image: صورة البصمة الكاملة
        grid_size: حجم المربع الواحد
        
    Returns:
        list: قائمة تحتوي على المربعات وإحداثياتها
    """
    height, width = full_image.shape
    grids = []
    overlap = 0.5  # 50% overlap
    step = int(grid_size * (1 - overlap))
    
    print(f"تقسيم الصورة {width}x{height} إلى مربعات بحجم {grid_size}x{grid_size}")
    print(f"خطوة التداخل: {step} بكسل ({overlap*100}% تداخل)")
    
    for y in range(0, height - grid_size + 1, step):
        for x in range(0, width - grid_size + 1, step):
            # استخراج المربع
            grid = full_image[y:y+grid_size, x:x+grid_size].copy()
            grids.append({
                'image': grid,
                'position': (x, y),
                'size': grid_size
            })
            print(f"تم استخراج مربع في الموقع ({x}, {y})")
    
    return grids

def match_normalized_grids(partial_image, full_image):
    """
    مطابقة البصمة الجزئية مع البصمة الكاملة باستخدام التقسيم إلى مربعات وتعديل الحجم
    
    Args:
        partial_image: البصمة الجزئية
        full_image: البصمة الكاملة
        
    Returns:
        dict: نتائج المطابقة
    """
    try:
        print("\nبدء عملية المطابقة باستخدام المربعات المعدلة...")
        
        # حساب متوسط المسافة بين الخطوط في البصمة الجزئية
        partial_freq = estimate_ridge_frequency(partial_image)
        if np.any(partial_freq > 0):
            target_distance = 1.0 / np.mean(partial_freq[partial_freq > 0])
            print(f"المسافة المستهدفة بين الخطوط: {target_distance:.2f} بكسل")
        else:
            print("تحذير: لم يتم العثور على خطوط في البصمة الجزئية")
            target_distance = 15.0  # قيمة افتراضية
        
        # تحديد حجم المربع بناءً على حجم البصمة الجزئية
        grid_size = max(partial_image.shape)
        print(f"حجم المربع المستخدم: {grid_size}x{grid_size} بكسل")
        
        # تقسيم البصمة الكاملة إلى مربعات
        grids = divide_into_grids(full_image, grid_size)
        if not grids:
            raise ValueError("لم يتم العثور على مربعات صالحة")
            
        print(f"\nتم تقسيم البصمة الكاملة إلى {len(grids)} مربع")
        
        # تخزين نتائج كل المربعات
        all_matches = []
        best_match = {
            'score': 0,
            'position': (0, 0),  # قيمة افتراضية
            'grid_image': None,
            'scale_factor': 1.0  # قيمة افتراضية
        }
        
        # مطابقة كل مربع
        for i, grid in enumerate(grids):
            print(f"\nمعالجة المربع {i+1}/{len(grids)}")
            print(f"الموقع: {grid['position']}")
            
            try:
                # تعديل حجم المربع ليتناسب مع المسافة بين الخطوط في البصمة الجزئية
                normalized_grid, scale_factor = normalize_ridge_distance(grid['image'], target_distance)
                print(f"معامل تغيير الحجم للمربع: {scale_factor:.2f}")
                
                # تعديل حجم المربع ليطابق حجم البصمة الجزئية
                if normalized_grid.shape != partial_image.shape:
                    normalized_grid = cv2.resize(normalized_grid, (partial_image.shape[1], partial_image.shape[0]))
                    print(f"تم تعديل حجم المربع إلى: {normalized_grid.shape}")
                
                # استخراج النقاط المميزة
                grid_minutiae = extract_minutiae(normalized_grid)
                partial_minutiae = extract_minutiae(partial_image)
                
                # حساب درجة التطابق
                match_score = calculate_grid_match_score(
                    partial_minutiae, grid_minutiae,
                    partial_image, normalized_grid
                )
                
                # تخزين نتيجة هذا المربع
                current_match = {
                    'score': match_score,
                    'position': grid['position'],
                    'grid_image': normalized_grid,
                    'scale_factor': scale_factor
                }
                all_matches.append(current_match)
                
                print(f"درجة التطابق: {match_score:.2f}%")
                
                if match_score > best_match['score']:
                    best_match = current_match
                    print("تم تحديث أفضل تطابق!")
            
            except Exception as e:
                print(f"خطأ في معالجة المربع: {str(e)}")
                continue
        
        if not all_matches:
            raise ValueError("لم يتم العثور على أي تطابقات صالحة")
        
        print(f"\nأفضل درجة تطابق: {best_match['score']:.2f}%")
        print(f"موقع أفضل تطابق: {best_match['position']}")
        print(f"معامل تغيير الحجم للمربع الأفضل: {best_match['scale_factor']:.2f}")
        
        return {
            'best_match': best_match,
            'all_matches': all_matches,
            'grid_size': grid_size
        }
        
    except Exception as e:
        print(f"خطأ في عملية المطابقة: {str(e)}")
        return {
            'best_match': {
                'score': 0,
                'position': (0, 0),
                'grid_image': None,
                'scale_factor': 1.0
            },
            'all_matches': [],
            'grid_size': grid_size if 'grid_size' in locals() else max(partial_image.shape)
        }

def calculate_grid_match_score(minutiae1, minutiae2, img1, img2):
    """
    حساب درجة التطابق بين مجموعتين من النقاط المميزة
    
    Args:
        minutiae1, minutiae2: النقاط المميزة
        img1, img2: الصور الأصلية
        
    Returns:
        float: درجة التطابق (0-100)
    """
    if not minutiae1 or not minutiae2:
        return 0
    
    # حساب تطابق النقاط المميزة
    matched_pairs = 0
    total_possible = min(len(minutiae1), len(minutiae2))
    
    for m1 in minutiae1:
        for m2 in minutiae2:
            # حساب المسافة وفرق الاتجاه
            dist = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            orientation_diff = abs(m1['orientation'] - m2['orientation'])
            orientation_diff = min(orientation_diff, 2*np.pi - orientation_diff)
            
            if (dist < MINUTIAE_DISTANCE_THRESHOLD and 
                orientation_diff < ORIENTATION_TOLERANCE and
                m1['type'] == m2['type']):
                matched_pairs += 1
                break
    
    minutiae_score = (matched_pairs / total_possible) * 100
    
    # حساب تطابق الأنماط
    correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    pattern_score = np.max(correlation) * 100
    
    # الدرجة النهائية
    final_score = 0.7 * minutiae_score + 0.3 * pattern_score
    
    return final_score

def visualize_grid_match(partial_image, full_image, match_result):
    """
    إنشاء صورة توضيحية لنتيجة المطابقة
    
    Args:
        partial_image: البصمة الجزئية
        full_image: البصمة الكاملة
        match_result: نتائج المطابقة
        
    Returns:
        dict: يحتوي على الصورة التوضيحية وصور المربعات
    """
    # تحويل الصور إلى RGB
    vis_partial = cv2.cvtColor(partial_image, cv2.COLOR_GRAY2BGR)
    vis_full = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
    
    # إنشاء خريطة حرارية لدرجات التطابق
    heat_map = np.zeros_like(full_image, dtype=np.float32)
    
    # تجهيز مصفوفة لعرض المربعات
    grid_size = match_result['grid_size']
    max_grids_per_row = 5
    n_grids = len(match_result['all_matches'])
    n_rows = (n_grids + max_grids_per_row - 1) // max_grids_per_row
    
    # إنشاء صورة لعرض كل المربعات
    grid_spacing = 10
    grids_image = np.ones((
        (grid_size + grid_spacing) * n_rows + grid_spacing,
        (grid_size + grid_spacing) * max_grids_per_row + grid_spacing,
        3
    ), dtype=np.uint8) * 255
    
    # رسم كل المربعات مع درجات تطابقها
    for idx, match in enumerate(match_result['all_matches']):
        x, y = match['position']
        score = match['score']
        
        # تحديث الخريطة الحرارية
        heat_map[y:y+grid_size, x:x+grid_size] = max(
            heat_map[y:y+grid_size, x:x+grid_size],
            score
        )
        
        # رسم مربع في الصورة الكاملة
        color = (0, int(score * 2.55), 0)
        cv2.rectangle(vis_full, (x, y), (x+grid_size, y+grid_size), color, 1)
        
        # إضافة المربع إلى مصفوفة المربعات
        grid_row = idx // max_grids_per_row
        grid_col = idx % max_grids_per_row
        grid_y = grid_row * (grid_size + grid_spacing) + grid_spacing
        grid_x = grid_col * (grid_size + grid_spacing) + grid_spacing
        
        # تحويل المربع المعدل إلى RGB
        grid_img = cv2.cvtColor(match['grid_image'], cv2.COLOR_GRAY2BGR)
        
        # تعديل حجم المربع إذا كان مختلفاً
        if grid_img.shape[:2] != (grid_size, grid_size):
            grid_img = cv2.resize(grid_img, (grid_size, grid_size))
        
        # نسخ المربع إلى مصفوفة المربعات
        grids_image[grid_y:grid_y+grid_size, grid_x:grid_x+grid_size] = grid_img
        
        # إضافة درجة التطابق
        cv2.putText(grids_image, f"{score:.1f}%", 
                   (grid_x, grid_y-2), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 0, 0), 1)
    
    # رسم المربع الأفضل بلون مميز
    best_x, best_y = match_result['best_match']['position']
    cv2.rectangle(vis_full, 
                 (best_x, best_y), 
                 (best_x+grid_size, best_y+grid_size), 
                 (0, 255, 255), 2)
    
    # تحويل الخريطة الحرارية إلى صورة ملونة
    heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map = heat_map.astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    
    # دمج الصور الرئيسية
    h_spacing = 20
    main_result = np.zeros((
        max(vis_partial.shape[0], vis_full.shape[0], heat_map.shape[0]),
        vis_partial.shape[1] + h_spacing + vis_full.shape[1] + h_spacing + heat_map.shape[1],
        3
    ), dtype=np.uint8)
    
    # نسخ الصور إلى النتيجة
    main_result[:vis_partial.shape[0], :vis_partial.shape[1]] = vis_partial
    main_result[:vis_full.shape[0], 
                vis_partial.shape[1]+h_spacing:vis_partial.shape[1]+h_spacing+vis_full.shape[1]] = vis_full
    main_result[:heat_map.shape[0],
                vis_partial.shape[1]+h_spacing+vis_full.shape[1]+h_spacing:] = heat_map
    
    # إضافة نص توضيحي
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(main_result, f"Best Match Score: {match_result['best_match']['score']:.2f}%",
                (10, main_result.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
    
    return {
        'main_visualization': main_result,
        'grids_visualization': grids_image
    } 