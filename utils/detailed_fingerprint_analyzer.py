import cv2
import numpy as np
from scipy import ndimage
import logging
from PIL import Image, ImageChops
import math

logger = logging.getLogger('detailed_fingerprint_analyzer')

class DetailedFingerprintAnalyzer:
    """
    فئة متخصصة في تحليل البصمات بشكل مفصل، وخاصة البصمات الجزئية أو غير المكتملة.
    تركز على التحليل المفصل للنقاط المميزة، وأنماط الحواف، والخصائص الفريدة للبصمة.
    """

    def __init__(self):
        self.target_size = (400, 400)  # حجم موحد للصور
        self.block_size = 16  # حجم الكتلة لحساب التوجيه
        self.min_quality_threshold = 0.3  # تخفيض الحد الأدنى للجودة للتعامل مع البصمات الجزئية

    def preprocess_for_detailed_analysis(self, image_path):
        """
        معالجة أولية مخصصة للبصمة لتحليل أكثر تفصيلاً
        """
        try:
            # قراءة الصورة
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("فشل في قراءة الصورة")

            # تغيير حجم الصورة
            img = cv2.resize(img, self.target_size)

            # تحسين التباين باستخدام CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # إزالة الضوضاء باستخدام مرشح ثنائي
            img = cv2.bilateralFilter(img, 9, 75, 75)

            # تحسين حواف البصمة بشكل أكثر دقة
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 13, 2
            )

            # عمليات مورفولوجية لتحسين جودة الصورة
            kernel = np.ones((3,3), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
            return img

        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة للتحليل المفصل: {str(e)}")
            raise

    def detect_minutiae(self, img):
        """
        اكتشاف النقاط المميزة (مثل نهايات الحواف والتفرعات) في البصمة
        """
        # تشخيص الحواف ونقاط التفرع باستخدام مرشح كاني وعمليات مورفولوجية
        edges = cv2.Canny(img, 100, 200)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # استخراج النقاط المميزة
        minutiae_points = []
        for i in range(1, edges.shape[0]-1):
            for j in range(1, edges.shape[1]-1):
                if edges[i, j] == 255:
                    # حساب عدد الانتقالات (من أبيض إلى أسود) للتعرف على نوع النقطة المميزة
                    neighbors = [
                        edges[i-1, j-1], edges[i-1, j], edges[i-1, j+1],
                        edges[i, j+1], edges[i+1, j+1], edges[i+1, j],
                        edges[i+1, j-1], edges[i, j-1], edges[i-1, j-1]
                    ]
                    transitions = sum(1 for k in range(len(neighbors)-1) if neighbors[k] == 0 and neighbors[k+1] == 255)
                    
                    # نقاط النهاية والتفرع
                    if transitions == 1:  # نقطة نهاية
                        minutiae_points.append({'type': 'ending', 'x': j, 'y': i})
                    elif transitions >= 3:  # نقطة تفرع
                        minutiae_points.append({'type': 'bifurcation', 'x': j, 'y': i})
        
        return minutiae_points

    def analyze_ridge_patterns(self, img):
        """
        تحليل أنماط الحواف في البصمة (حلقات، دوامات، أقواس)
        """
        # استخدام تحويل جابور لتحليل اتجاهات الحواف
        orientations = []
        frequencies = []
        
        # تطبيق مرشحات جابور بزوايا مختلفة
        for theta in np.arange(0, np.pi, np.pi/8):
            # إنشاء نواة مرشح جابور
            kern = cv2.getGaborKernel((15, 15), 5, theta, 10, 1, 0, cv2.CV_32F)
            # تطبيق المرشح
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kern)
            # قياس الاستجابة
            orientations.append((theta, np.sum(filtered)))
        
        # تحديد الاتجاه الرئيسي
        main_orientation = max(orientations, key=lambda x: x[1])[0]
        
        # تحليل نمط الحواف بناءً على التوزيع والاتجاه
        pattern_type = self._determine_pattern_type(orientations)
        
        return {
            'main_orientation': main_orientation,
            'pattern_type': pattern_type,
            'orientation_distribution': orientations
        }

    def _determine_pattern_type(self, orientations):
        """
        تحديد نوع نمط البصمة بناءً على توزيع الاتجاهات
        """
        # تنظيم الاتجاهات حسب القوة
        sorted_orientations = sorted(orientations, key=lambda x: x[1], reverse=True)
        
        # حساب التباين في الاتجاهات
        orientations_values = [o[0] for o in orientations]
        orientation_variance = np.var(orientations_values)
        
        # تحديد النمط بناءً على التباين والتوزيع
        if orientation_variance < 0.2:
            if abs(sorted_orientations[0][0] - sorted_orientations[1][0]) < 0.3:
                return "arch"  # قوس
        elif orientation_variance < 0.5:
            return "loop"  # حلقة
        else:
            return "whorl"  # دوامة
        
        return "unknown"  # غير معروف

    def analyze_size_and_shape(self, img):
        """
        تحليل حجم وشكل البصمة
        """
        # استخدام العتبة للحصول على منطقة البصمة
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # حساب مساحة البصمة (عدد البكسلات البيضاء)
        area = np.sum(binary > 0)
        
        # حساب محيط البصمة
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
        else:
            perimeter = 0
        
        # حساب نسبة التدوير (الشكل)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # حساب النسبة الطولية
        x, y, w, h = cv2.boundingRect(binary)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'bounding_rect': (x, y, w, h)
        }

    def evaluate_quality(self, img):
        """
        تقييم جودة البصمة
        """
        # تقسيم الصورة إلى كتل وحساب التباين والوضوح
        blocks = []
        clarity_scores = []
        
        for i in range(0, img.shape[0], self.block_size):
            for j in range(0, img.shape[1], self.block_size):
                block = img[i:min(i+self.block_size, img.shape[0]), 
                          j:min(j+self.block_size, img.shape[1])]
                if block.size > 0:
                    # حساب التباين
                    blocks.append(np.var(block))
                    
                    # حساب وضوح الحواف
                    edges = cv2.Canny(block, 100, 200)
                    clarity_scores.append(np.sum(edges) / block.size)
        
        # حساب متوسط التباين والوضوح
        contrast = np.mean(blocks) / 255.0 if blocks else 0
        clarity = np.mean(clarity_scores) if clarity_scores else 0
        
        # حساب مؤشر جودة شامل
        quality_score = 0.6 * contrast + 0.4 * clarity
        
        return {
            'contrast': contrast,
            'clarity': clarity,
            'quality_score': min(max(quality_score, 0), 1),
            'is_partial': quality_score < 0.5  # تحديد ما إذا كانت البصمة جزئية
        }

    def match_minutiae(self, minutiae1, minutiae2):
        """
        مطابقة النقاط المميزة بين بصمتين
        """
        matches = []
        match_count = 0
        total_minutiae = max(len(minutiae1), len(minutiae2))
        
        # معلمات للمطابقة
        distance_threshold = 20  # الحد الأقصى للمسافة بين نقطتين متطابقتين
        
        # مطابقة كل نقطة من البصمة الأولى مع النقاط في البصمة الثانية
        for m1 in minutiae1:
            best_match = None
            min_distance = float('inf')
            
            for m2 in minutiae2:
                # حساب المسافة بين النقطتين
                distance = math.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                
                # التحقق من نوع النقطة المميزة (نهاية أو تفرع)
                same_type = m1['type'] == m2['type']
                
                # اختيار أفضل مطابقة
                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    best_match = {
                        'minutia1': m1,
                        'minutia2': m2,
                        'distance': distance,
                        'same_type': same_type
                    }
            
            if best_match:
                matches.append(best_match)
                match_count += 1
        
        # حساب نسبة المطابقة
        if total_minutiae > 0:
            match_ratio = match_count / total_minutiae
        else:
            match_ratio = 0
            
        return {
            'matches': matches,
            'match_count': match_count,
            'total_minutiae': total_minutiae,
            'match_ratio': match_ratio
        }

    def compare_ridge_patterns(self, pattern1, pattern2):
        """
        مقارنة أنماط الحواف بين بصمتين
        """
        # مقارنة الاتجاه الرئيسي
        orientation_diff = abs(pattern1['main_orientation'] - pattern2['main_orientation'])
        orientation_diff = min(orientation_diff, np.pi - orientation_diff)  # مراعاة الدورية
        orientation_score = 1 - (orientation_diff / (np.pi/2))
        
        # مقارنة نوع النمط
        pattern_match = pattern1['pattern_type'] == pattern2['pattern_type']
        pattern_score = 1.0 if pattern_match else 0.5  # إعطاء وزن أقل لعدم تطابق النمط
        
        return {
            'orientation_score': orientation_score,
            'pattern_score': pattern_score,
            'combined_score': 0.7 * orientation_score + 0.3 * pattern_score
        }

    def compare_size_and_shape(self, shape1, shape2):
        """
        مقارنة حجم وشكل البصمتين
        """
        # مقارنة النسبة الطولية
        aspect_ratio_diff = abs(shape1['aspect_ratio'] - shape2['aspect_ratio'])
        aspect_ratio_score = max(0, 1 - aspect_ratio_diff)
        
        # مقارنة التدوير (الشكل)
        circularity_diff = abs(shape1['circularity'] - shape2['circularity'])
        circularity_score = max(0, 1 - circularity_diff)
        
        # حساب مدى تداخل المستطيلين المحيطين
        rect1 = shape1['bounding_rect']
        rect2 = shape2['bounding_rect']
        
        # حساب التداخل
        x_overlap = max(0, min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - max(rect1[1], rect2[1]))
        overlap_area = x_overlap * y_overlap
        
        # حساب مساحة الاتحاد
        union_area = (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - overlap_area
        
        # حساب نسبة تداخل المستطيلين المحيطين
        if union_area > 0:
            iou_score = overlap_area / union_area
        else:
            iou_score = 0
        
        return {
            'aspect_ratio_score': aspect_ratio_score,
            'circularity_score': circularity_score,
            'iou_score': iou_score,
            'combined_score': 0.3 * aspect_ratio_score + 0.3 * circularity_score + 0.4 * iou_score
        }

    def detailed_fingerprint_comparison(self, filepath1, filepath2):
        """
        إجراء تحليل مفصل ومقارنة شاملة بين بصمتين، مع مراعاة احتمالية أن تكون إحداهما جزئية
        """
        try:
            # معالجة أولية للصور
            img1 = self.preprocess_for_detailed_analysis(filepath1)
            img2 = self.preprocess_for_detailed_analysis(filepath2)
            
            # (1) استخراج النقاط المميزة
            minutiae1 = self.detect_minutiae(img1)
            minutiae2 = self.detect_minutiae(img2)
            
            # (2) تحليل أنماط الحواف
            ridge_pattern1 = self.analyze_ridge_patterns(img1)
            ridge_pattern2 = self.analyze_ridge_patterns(img2)
            
            # (3) تحليل الحجم والشكل
            shape1 = self.analyze_size_and_shape(img1)
            shape2 = self.analyze_size_and_shape(img2)
            
            # (4) تقييم الجودة
            quality1 = self.evaluate_quality(img1)
            quality2 = self.evaluate_quality(img2)
            
            # (5) مطابقة النقاط المميزة
            minutiae_matching = self.match_minutiae(minutiae1, minutiae2)
            
            # (6) مقارنة أنماط الحواف
            ridge_comparison = self.compare_ridge_patterns(ridge_pattern1, ridge_pattern2)
            
            # (7) مقارنة الحجم والشكل
            shape_comparison = self.compare_size_and_shape(shape1, shape2)
            
            # (8) حساب درجة المطابقة الشاملة
            # ضبط الأوزان بناءً على ما إذا كانت البصمات جزئية
            is_partial = quality1['is_partial'] or quality2['is_partial']
            
            if is_partial:
                # أوزان معدلة للبصمات الجزئية (تركيز أكثر على النقاط المميزة)
                weights = {
                    'minutiae': 0.6,
                    'ridge_pattern': 0.3,
                    'shape': 0.1
                }
            else:
                # أوزان للبصمات الكاملة
                weights = {
                    'minutiae': 0.5,
                    'ridge_pattern': 0.3,
                    'shape': 0.2
                }
            
            # حساب الدرجة النهائية
            final_score = (
                weights['minutiae'] * minutiae_matching['match_ratio'] +
                weights['ridge_pattern'] * ridge_comparison['combined_score'] +
                weights['shape'] * shape_comparison['combined_score']
            )
            
            # تعديل الدرجة النهائية بناءً على جودة البصمات
            quality_factor = min(quality1['quality_score'], quality2['quality_score'])
            if quality_factor < self.min_quality_threshold:
                quality_adjustment = quality_factor / self.min_quality_threshold
                final_score = final_score * (0.7 + 0.3 * quality_adjustment)  # تعديل جزئي فقط
            
            # تجميع النتائج التفصيلية
            detailed_results = {
                'minutiae_analysis': {
                    'fingerprint1_minutiae_count': len(minutiae1),
                    'fingerprint2_minutiae_count': len(minutiae2),
                    'matching_minutiae_count': minutiae_matching['match_count'],
                    'minutiae_match_ratio': minutiae_matching['match_ratio']
                },
                'ridge_analysis': {
                    'fingerprint1_pattern': ridge_pattern1['pattern_type'],
                    'fingerprint2_pattern': ridge_pattern2['pattern_type'],
                    'orientation_score': ridge_comparison['orientation_score'],
                    'pattern_score': ridge_comparison['pattern_score']
                },
                'shape_analysis': {
                    'fingerprint1_area': shape1['area'],
                    'fingerprint2_area': shape2['area'],
                    'aspect_ratio_score': shape_comparison['aspect_ratio_score'],
                    'overlap_score': shape_comparison['iou_score']
                },
                'quality_analysis': {
                    'fingerprint1_quality': quality1['quality_score'],
                    'fingerprint2_quality': quality2['quality_score'],
                    'is_partial_match': is_partial
                },
                'match_summary': {
                    'final_score': final_score,
                    'confidence_level': self._get_confidence_level(final_score),
                    'match_percentage': final_score * 100
                }
            }
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"خطأ في التحليل المفصل للبصمات: {str(e)}")
            raise

    def _get_confidence_level(self, score):
        """
        تحديد مستوى الثقة بناءً على الدرجة النهائية
        """
        if score >= 0.9:
            return "مرتفع جداً"
        elif score >= 0.75:
            return "مرتفع"
        elif score >= 0.6:
            return "متوسط"
        elif score >= 0.4:
            return "منخفض"
        else:
            return "منخفض جداً" 