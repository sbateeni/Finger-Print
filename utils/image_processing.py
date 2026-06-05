import cv2
import numpy as np
from config import *


def estimate_dominant_orientation(image):
    """
    يحسب الاتجاه الغالب للتلال (Ridge Orientation) باستخدام Gradient.
    يُستخدم لتطبيع الصورة قبل الاستخراج.
    """
    try:
        sobelx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # زاوية كل بكسل (اتجاه عمودي على التلل)
        angles = np.arctan2(sobely, sobelx)
        
        # هيستوغرام للزوايا (0-180 درجة)
        hist, bins = np.histogram(np.degrees(angles) % 180, bins=36, range=(0, 180))
        dominant_angle = bins[np.argmax(hist)]
        
        return dominant_angle
    except:
        return 0.0


def normalize_orientation(image):
    """
    تطبيع اتجاه البصمة لتقليل تأثير زاوية الالتقاط بالكاميرا.
    يدور الصورة لتكون التلال في اتجاه قياسي.
    """
    try:
        angle = estimate_dominant_orientation(image)
        
        # نريد أن تكون التلال الغالبة أفقية (90 درجة)
        target = 90.0
        rotation_needed = target - angle
        
        # تجاهل التدوير إذا كان صغيراً جداً (أقل من 5 درجات)
        if abs(rotation_needed) < 5:
            return image
        
        # تدوير الصورة حول مركزها
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_needed, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except:
        return image


def segment_fingerprint(image):
    """
    استخراج منطقة البصمة من الخلفية باستخدام Distance Transform والعمليات المورفولوجية.
    يُرجع قناع ثنائي يحدد منطقة البصمة.
    """
    try:
        # التوهج السحابي (Blackhat) لإظهار المناطق الداكنة (البصمة) على الخلفية
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # تطبيق Otsu Thresholding على نتيجة Blackhat
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # إغلاق ثغرات صغيرة
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        # إزالة الكائنات الصغيرة جداً
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels > 1:
            # اختيار الكائن الأكبر (البصمة)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(thresh)
            mask[labels == largest_label] = 255
        else:
            mask = thresh
        
        # تنعيم الحدود
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    except Exception as e:
        print(f"Error in segment_fingerprint: {str(e)}")
        return np.ones_like(image, dtype=np.uint8) * 255


def preprocess_image(image, denoise_method="fastNlMeans", fast_denoise_h=10, gauss_ksize=3):
    """
    خط معالجة مسبقة محسّن:
    1. تحويل رمادي
    2. تغيير الحجم
    3. استخراج منطقة البصمة (Segmentation)
    4. CLAHE للتباين
    5. إزالة الضوضاء
    6. Adaptive Thresholding (أفضل من Otsu للصور الحقيقية)
    7. تطبيق القناع على النتيجة
    """
    try:
        # 1. تحويل رمادي
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. توحيد الحجم
        image = cv2.resize(image, IMAGE_SIZE)
        
        # 3. استخراج منطقة البصمة
        mask = segment_fingerprint(image)
        
        # 4. CLAHE — تحسين التباين المحلي
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                  tileGridSize=CLAHE_TILE_GRID_SIZE)
        image = clahe.apply(image)
        
        # 5. إزالة الضوضاء
        if denoise_method == "fastNlMeans":
            image = cv2.fastNlMeansDenoising(image, h=fast_denoise_h)
        elif denoise_method == "GaussianBlur":
            if gauss_ksize % 2 == 0:
                gauss_ksize += 1
            image = cv2.GaussianBlur(image, (gauss_ksize, gauss_ksize), 0)
        
        # 6. Adaptive Thresholding — تحسين للتعامل مع الإضاءة غير المنتظمة
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        
        # عكس الألوان إذا لزم الأمر (الهيكلة تتطلب تلالاً بيضاء 255)
        binary = cv2.bitwise_not(binary)
        
        # تطبيق القناع لاستبعاد الخلفية
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        # إزالة الضوضاء الثنائية الصغيرة (Morphological Opening)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None


def detect_edges(image):
    """استخراج التلال باستخدام بنك Gabor متعدد الاتجاهات"""
    try:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        orientation_map = np.zeros_like(sobelx)
        mask = (sobelx != 0) | (sobely != 0)
        orientation_map[mask] = np.arctan2(sobely[mask], sobelx[mask])
        
        n_theta = max(2, int(GABOR_ORIENTATIONS))
        responses = []
        for k in range(n_theta):
            theta = (np.pi / n_theta) * k
            kernel = cv2.getGaborKernel(
                (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
                GABOR_SIGMA, theta, GABOR_LAMBDA,
                GABOR_GAMMA, GABOR_PSI, ktype=cv2.CV_32F,
            )
            resp = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
            responses.append(resp)
        
        stacked = np.stack(responses, axis=0)
        ridges_f = np.max(stacked, axis=0)
        
        # تطبيع الاستجابة لضمان عدم التشبع
        ridges_f = cv2.normalize(ridges_f, None, 0, 255, cv2.NORM_MINMAX)
        ridges = ridges_f.astype(np.uint8)
        
        return ridges, orientation_map
    except Exception as e:
        print(f"Error in detect_edges: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error in detect_edges: {str(e)}")
        return None, None


def enhance_image(image):
    """تهليل الهيكل العظمي للتلال"""
    try:
        # سد الفواصل في التموجات قبل التهليل (نواة 5×5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # Prefer OpenCV contrib thinning when available.
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skeleton = cv2.ximgproc.thinning(closed)
        else:
            _, work = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
            skel = np.zeros_like(work)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            while True:
                eroded = cv2.erode(work, kernel)
                opened = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(work, opened)
                skel = cv2.bitwise_or(skel, temp)
                work = eroded
                if cv2.countNonZero(work) == 0:
                    break

            skeleton = skel
        _, skeleton_bin = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
        return skeleton_bin
    except Exception as e:
        print(f"Error in enhance_image: {str(e)}")
        return None

def assess_fingerprint_quality(image):
    """
    تقييم جودة البصمة (NFIQ-like) بناءً على تماسك التدرج المحلي (LGC) والتباين.
    يُرجع: (درجة الجودة 0-100، خريطة الجودة).
    """
    try:
        # 1. حساب التدرجات
        gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # 2. حساب عناصر مصفوفة التشتت (Structure Tensor)
        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy
        
        # 3. تنعيم محلي (Local averaging) لكتل البصمة (كتل بكسل 16x16)
        block_size = 16
        gxx_sum = cv2.boxFilter(gxx, -1, (block_size, block_size))
        gyy_sum = cv2.boxFilter(gyy, -1, (block_size, block_size))
        gxy_sum = cv2.boxFilter(gxy, -1, (block_size, block_size))
        
        # 4. حساب التماسك (Coherence)
        # التماسك يقيس مدى انتظام اتجاه التلال؛ في المناطق الواضحة يكون الاتجاه واحداً (تماسك عالي)
        # وفي المناطق المشوشة تكون الاتجاهات عشوائية (تماسك منخفض)
        denom = gxx_sum + gyy_sum + 1e-6
        num = np.sqrt((gxx_sum - gyy_sum)**2 + 4 * (gxy_sum**2))
        coherence = num / denom
        
        # 5. تنظيف الخريطة (Coherence map) وتحويلها لنسبة مئوية
        quality_map = cv2.GaussianBlur(coherence, (15, 15), 0)
        quality_map = np.clip(quality_map * 100, 0, 100)
        
        # 6. الدرجة الكلية (متوسط المناطق النشطة فقط)
        # نعتبر المناطق التي فيها تباين هي مناطق البصمة الفعلية وليس الخلفية
        active_mask = (gxx_sum + gyy_sum) > np.mean(gxx_sum + gyy_sum) * 0.2
        if np.any(active_mask):
            overall_score = np.mean(quality_map[active_mask])
        else:
            overall_score = 0.0
            
        return float(overall_score), quality_map.astype(np.uint8)
    except Exception as e:
        print(f"Quality Assessment Error: {e}")
        return 0.0, np.zeros_like(image)

def detect_singular_points(orientation_map, mask):
    """
    اكتشاف النقاط المفردة (Cores & Deltas) باستخدام خوارزمية Poincaré Index.
    تُستخدم لتحديد مركز البصمة ونوع النمط (Loop, Whorl, Arch).
    """
    try:
        rows, cols = orientation_map.shape
        cores = []
        deltas = []
        
        # نستخدم حجم كتلة (Block size) للتحليل
        step = 8
        for i in range(step, rows - step, step):
            for j in range(step, cols - step, step):
                if not mask[i, j]: continue
                
                # حساب مؤشر Poincaré في مسار مغلق حول الخلية
                # المسار: (i-1,j-1) -> (i-1,j) -> (i-1,j+1) -> (i,j+1) ...
                neighbors = [
                    orientation_map[i-step, j-step], orientation_map[i-step, j],
                    orientation_map[i-step, j+step], orientation_map[i, j+step],
                    orientation_map[i+step, j+step], orientation_map[i+step, j],
                    orientation_map[i+step, j-step], orientation_map[i, j-step]
                ]
                
                poincare = 0
                for k in range(8):
                    diff = neighbors[(k+1)%8] - neighbors[k]
                    if diff > np.pi/2: diff -= np.pi
                    elif diff < -np.pi/2: diff += np.pi
                    poincare += diff
                
                # تصنيف النقطة بناءً على قيمة المؤشر
                if abs(poincare - np.pi) < 0.5:
                    cores.append({"x": j, "y": i, "type": "core"})
                elif abs(poincare + np.pi) < 0.5:
                    deltas.append({"x": j, "y": i, "type": "delta"})
                    
        return cores, deltas
    except:
        return [], []


 