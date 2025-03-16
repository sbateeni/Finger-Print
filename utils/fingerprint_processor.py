import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import logging

logger = logging.getLogger('fingerprint_processor')

class FingerprintProcessor:
    def __init__(self):
        # Initialize parameters for fingerprint processing
        self.target_size = (400, 400)  # حجم موحد للصور
        self.block_size = 16  # حجم الكتلة لحساب التوجيه
        self.min_quality_threshold = 0.4  # الحد الأدنى لجودة البصمة

    def preprocess_image(self, image_path):
        """معالجة أولية لصورة البصمة"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("فشل في قراءة الصورة")

            # تغيير حجم الصورة
            img = cv2.resize(img, self.target_size)

            # تحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # إزالة الضوضاء
            img = cv2.GaussianBlur(img, (5,5), 0)

            # تحسين حواف البصمة
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            return img

        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

    def extract_features(self, img):
        """استخراج الميزات من صورة البصمة"""
        # استخراج النقاط المميزة
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        return keypoints, descriptors

    def calculate_orientation_field(self, img):
        """حساب حقل التوجيه للبصمة"""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        orientation = np.arctan2(sobely, sobelx) * 0.5
        return orientation

    def detect_core_point(self, img):
        """اكتشاف نقطة المركز في البصمة"""
        # تطبيق مرشح بويندينج بوكس
        gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        
        # حساب نقطة المركز
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        core_y, core_x = np.unravel_index(magnitude.argmax(), magnitude.shape)
        
        return (core_x, core_y)

    def calculate_quality_score(self, img):
        """حساب درجة جودة البصمة"""
        # تقسيم الصورة إلى كتل وحساب التباين لكل كتلة
        blocks = []
        for i in range(0, img.shape[0], self.block_size):
            for j in range(0, img.shape[1], self.block_size):
                block = img[i:min(i+self.block_size, img.shape[0]), 
                          j:min(j+self.block_size, img.shape[1])]
                if block.size > 0:  # تجنب الكتل الفارغة
                    blocks.append(np.var(block))
        
        # حساب متوسط التباين المحلي
        if blocks:
            quality_score = np.mean(blocks) / 255.0
        else:
            quality_score = 0.0
        
        return min(max(quality_score, 0), 1)

    def compare_fingerprints(self, img1, img2):
        """مقارنة بصمتين وإرجاع درجة التطابق والتفاصيل"""
        try:
            # حساب تشابه الصور
            ssim_score, _ = ssim(img1, img2, full=True)

            # استخراج وتطابق الميزات
            kp1, des1 = self.extract_features(img1)
            kp2, des2 = self.extract_features(img2)

            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                
                # تطبيق اختبار لوي للمطابقة
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                feature_score = len(good_matches) / max(len(kp1), len(kp2))
            else:
                feature_score = 0

            # حساب تشابه النقاط المميزة
            minutiae_score = feature_score

            # حساب تشابه التوجيه
            orientation1 = self.calculate_orientation_field(img1)
            orientation2 = self.calculate_orientation_field(img2)
            orientation_score = 1 - np.mean(np.abs(orientation1 - orientation2)) / np.pi

            # حساب تشابه نقاط المركز
            core1 = self.detect_core_point(img1)
            core2 = self.detect_core_point(img2)
            core_distance = np.sqrt((core1[0] - core2[0])**2 + (core1[1] - core2[1])**2)
            core_score = max(0, 1 - core_distance / (self.target_size[0] / 4))

            # حساب تشابه التردد
            frequency_score = ssim_score

            # حساب جودة البصمات
            quality_score1 = self.calculate_quality_score(img1)
            quality_score2 = self.calculate_quality_score(img2)

            # حساب الدرجة النهائية
            weights = {
                'ssim': 0.2,
                'feature': 0.25,
                'minutiae': 0.2,
                'orientation': 0.15,
                'core': 0.1,
                'frequency': 0.1
            }

            final_score = (
                weights['ssim'] * ssim_score +
                weights['feature'] * feature_score +
                weights['minutiae'] * minutiae_score +
                weights['orientation'] * orientation_score +
                weights['core'] * core_score +
                weights['frequency'] * frequency_score
            )

            # تطبيق معامل الجودة
            quality_factor = min(quality_score1, quality_score2)
            if quality_factor < self.min_quality_threshold:
                final_score *= (quality_factor / self.min_quality_threshold)

            details = {
                'ssim_score': ssim_score,
                'feature_score': feature_score,
                'minutiae_score': minutiae_score,
                'orientation_score': orientation_score,
                'core_score': core_score,
                'frequency_score': frequency_score,
                'quality_score1': quality_score1,
                'quality_score2': quality_score2
            }

            return final_score, details

        except Exception as e:
            logger.error(f"خطأ في مقارنة البصمات: {str(e)}")
            raise 

    def detect_multiple_fingerprints(self, image_path):
        """اكتشاف واستخراج البصمات المتعددة من صورة واحدة"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("فشل في قراءة الصورة")

            # تحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # تطبيق عتبة تكيفية للحصول على صورة ثنائية
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # تطبيق عمليات مورفولوجية لتحسين الصورة
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # العثور على المكونات المتصلة (البصمات المحتملة)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # تجميع البصمات المكتشفة
            fingerprints = []
            min_area = 1000  # الحد الأدنى لمساحة البصمة
            
            # تجاهل المكون الأول (الخلفية)
            for i in range(1, num_labels):
                # الحصول على إحصائيات المكون
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # تجاهل المكونات الصغيرة جداً
                if area < min_area:
                    continue

                # استخراج منطقة البصمة
                roi = img[y:y+h, x:x+w]
                
                # تغيير حجم البصمة إلى الحجم القياسي
                roi_resized = cv2.resize(roi, self.target_size)
                
                # معالجة البصمة
                processed_roi = cv2.GaussianBlur(roi_resized, (5,5), 0)
                processed_roi = cv2.adaptiveThreshold(
                    processed_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )

                fingerprints.append({
                    'image': processed_roi,
                    'position': (x, y, w, h),
                    'area': area
                })

            return fingerprints

        except Exception as e:
            logger.error(f"خطأ في اكتشاف البصمات المتعددة: {str(e)}")
            raise 