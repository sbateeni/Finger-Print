import cv2
import numpy as np

def resize_image(image, target_size=(800, 800)):
    """Resize image while maintaining aspect ratio"""
    try:
        if image is None:
            print("Error in resize_image: Input image is None")
            return None
            
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            print("Error in resize_image: Image has zero dimensions")
            return None
            
        scale = min(target_size[0]/w, target_size[1]/h)
        new_size = (int(w*scale), int(h*scale))
        resized = cv2.resize(image, new_size)
        print(f"Image resized from {w}x{h} to {new_size[0]}x{new_size[1]}")
        return resized
    except Exception as e:
        print(f"Error in resize_image: {str(e)}")
        return None

def enhance_contrast(image):
    """Enhance contrast using CLAHE"""
    try:
        if image is None:
            print("Error in enhance_contrast: Input image is None")
            return None
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        print("Contrast enhanced successfully")
        return enhanced
    except Exception as e:
        print(f"Error in enhance_contrast: {str(e)}")
        return None

def remove_noise(image):
    """Remove noise using bilateral filter"""
    try:
        if image is None:
            print("Error in remove_noise: Input image is None")
            return None
            
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        print("Noise removed successfully")
        return denoised
    except Exception as e:
        print(f"Error in remove_noise: {str(e)}")
        return None

def skeletonize(image):
    """Skeletonization using OpenCV operations"""
    try:
        if image is None:
            print("Error in skeletonize: Input image is None")
            return None
        
        # Ensure image is binary
        binary = image.copy()
        binary = (binary > 0).astype(np.uint8) * 255
        
        # Create an output skeleton image
        skeleton = np.zeros(binary.shape, np.uint8)
        
        # Get a kernel for morphological operations
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Copy the binary image
        img = binary.copy()
        
        # Iterate until the image is fully eroded
        iterations = 0
        max_iterations = 100  # Prevent infinite loop
        while True and iterations < max_iterations:
            # Perform morphological opening
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            
            # Subtract to get the skeleton points
            temp = cv2.subtract(img, temp)
            
            # Add to the skeleton
            skeleton = cv2.bitwise_or(skeleton, temp)
            
            # Set the eroded image for the next iteration
            img = eroded.copy()
            
            # If image has been completely eroded, we're done
            if cv2.countNonZero(img) == 0:
                break
                
            iterations += 1
            
        print(f"OpenCV skeletonization completed in {iterations} iterations")
        
        # Ensure skeleton is not empty
        if cv2.countNonZero(skeleton) == 0:
            print("Error: Skeleton is empty after processing")
            # Fall back to the original image with some preprocessing
            kernel = np.ones((2,2), np.uint8)
            skeleton = cv2.erode(binary, kernel, iterations=1)
            print(f"Using fallback skeleton with {cv2.countNonZero(skeleton)} pixels")
        else:
            print(f"Skeleton has {cv2.countNonZero(skeleton)} pixels")
            
        return skeleton
    except Exception as e:
        print(f"Error in skeletonize: {str(e)}")
        # Return original image as fallback
        return image

def detect_ridges(image):
    """Detect ridge patterns in the fingerprint"""
    # Apply Sobel operator to detect edges
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # Normalize magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return magnitude, direction

def preprocess_image(image):
    """
    Preprocess fingerprint image for feature extraction
    Returns:
        tuple: (skeleton, direction) where skeleton is the processed image and direction is the ridge direction
    """
    try:
        if image is None:
            print("Error: Input image is None")
            return None, None

        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Converted to grayscale")
        else:
            gray = image

        # Ensure image is in uint8 format
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
            print("Converted to uint8 format")

        # Resize image to standard size
        gray = resize_image(gray)
        if gray is None:
            return None, None

        # Remove noise
        denoised = remove_noise(gray)
        if denoised is None:
            return None, None

        # Enhance contrast
        enhanced = enhance_contrast(denoised)
        if enhanced is None:
            return None, None

        # Detect ridge direction
        magnitude, direction = detect_ridges(enhanced)
        print("Detected ridge direction")

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Applied Otsu's thresholding")

        # Apply morphological operations - less aggressive
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print("Applied morphological operations")

        # Ensure binary image is not empty
        non_zero = cv2.countNonZero(binary)
        if non_zero == 0:
            print("Error: Binary image is empty")
            return None, None
        print(f"Binary image has {non_zero} non-zero pixels")

        # Skeletonize the image
        skeleton = skeletonize(binary)
        if skeleton is None:
            return None, None

        # Ensure skeleton is not empty
        non_zero = cv2.countNonZero(skeleton)
        if non_zero == 0:
            print("Error: Final skeleton is empty")
            # Use the binary image as fallback
            skeleton = binary
            print("Using binary image as fallback")
        else:
            print(f"Final skeleton has {non_zero} non-zero pixels")

        return skeleton, direction
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None, None

def analyze_ridge_patterns(skeleton, direction):
    """Analyze ridge patterns and their characteristics"""
    try:
        # Find ridge endings and bifurcations
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(skeleton, kernel, iterations=1)
        eroded = cv2.erode(skeleton, kernel, iterations=1)
        
        # Ridge endings
        endings = cv2.subtract(skeleton, eroded)
        # Bifurcations
        bifurcations = cv2.subtract(dilated, skeleton)
        
        # Find coordinates of features
        ending_points = np.where(endings > 0)
        bifurcation_points = np.where(bifurcations > 0)
        
        features = []
        
        # Add ridge endings
        for y, x in zip(*ending_points):
            features.append({
                'x': int(x),
                'y': int(y),
                'type': 'ending',
                'angle': float(direction[y, x]),
                'magnitude': 1.0
            })
        
        # Add bifurcations
        for y, x in zip(*bifurcation_points):
            features.append({
                'x': int(x),
                'y': int(y),
                'type': 'bifurcation',
                'angle': float(direction[y, x]),
                'magnitude': 1.0
            })
        
        return features
    except Exception as e:
        print(f"Error in analyze_ridge_patterns: {str(e)}")
        return []

def extract_minutiae(skeleton):
    """استخراج النقاط المميزة من البصمة"""
    try:
        if skeleton is None:
            print("Error: Input skeleton is None")
            return []

        # تحويل الصورة إلى ثنائية
        binary = (skeleton > 0).astype(np.uint8) * 255

        # البحث عن نقاط النهاية والتفرعات
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # نقاط النهاية
        endings = cv2.subtract(binary, eroded)
        # نقاط التفرع
        bifurcations = cv2.subtract(dilated, binary)
        
        minutiae = []
        
        # إضافة نقاط النهاية
        ending_points = np.where(endings > 0)
        for y, x in zip(*ending_points):
            minutiae.append({
                'x': int(x),
                'y': int(y),
                'type': 'ending',
                'angle': 0.0,  # سيتم حسابه لاحقاً
                'quality': 1.0
            })
        
        # إضافة نقاط التفرع
        bifurcation_points = np.where(bifurcations > 0)
        for y, x in zip(*bifurcation_points):
            minutiae.append({
                'x': int(x),
                'y': int(y),
                'type': 'bifurcation',
                'angle': 0.0,  # سيتم حسابه لاحقاً
                'quality': 1.0
            })
        
        # حساب الزوايا لكل نقطة
        for minutia in minutiae:
            x, y = minutia['x'], minutia['y']
            # أخذ منطقة 7×7 حول النقطة
            region = binary[max(0, y-3):min(binary.shape[0], y+4),
                          max(0, x-3):min(binary.shape[1], x+4)]
            if region.size > 0:
                # حساب اتجاه الخط باستخدام gradient
                gy, gx = np.gradient(region.astype(float))
                angle = np.arctan2(gy.mean(), gx.mean())
                minutia['angle'] = float(angle)
        
        return minutiae
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

def calculate_scale_factor(original_image, partial_image):
    """حساب معامل التحجيم بين الصورتين"""
    try:
        if original_image is None or partial_image is None:
            return 1.0
            
        # حساب نسبة الأبعاد
        h1, w1 = original_image.shape[:2]
        h2, w2 = partial_image.shape[:2]
        
        # حساب متوسط النسب
        scale_h = h1 / h2 if h2 > 0 else 1.0
        scale_w = w1 / w2 if w2 > 0 else 1.0
        
        return (scale_h + scale_w) / 2
    except Exception as e:
        print(f"Error in calculate_scale_factor: {str(e)}")
        return 1.0

def add_ruler_to_image(image, dpi=100):
    """إضافة مسطرة مرقمة إلى الصورة"""
    try:
        # تحويل الصورة إلى مصفوفة NumPy إذا كانت PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # تحويل الصورة الرمادية إلى RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            return image
        
        # إنشاء صورة جديدة مع مساحة إضافية للمساطر
        height, width = image.shape[:2]
        ruler_size = 50  # حجم المسطرة بالبكسل
        new_width = width + ruler_size
        new_height = height + ruler_size
        
        # إنشاء صورة جديدة مع خلفية بيضاء
        new_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        
        # نسخ الصورة الأصلية إلى الموقع الصحيح
        new_image[ruler_size:, ruler_size:] = image
        
        # إضافة المسطرة الأفقية
        for i in range(0, width, int(dpi/2.54)):  # كل سنتيمتر
            x = i + ruler_size
            y = ruler_size - 10
            cv2.line(new_image, (x, y), (x, ruler_size), (0,0,0), 1)
            cv2.putText(new_image, f"{i/dpi*2.54:.1f}", (x-10, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        
        # إضافة المسطرة العمودية
        for i in range(0, height, int(dpi/2.54)):  # كل سنتيمتر
            x = ruler_size - 10
            y = i + ruler_size
            cv2.line(new_image, (x, y), (ruler_size, y), (0,0,0), 1)
            cv2.putText(new_image, f"{i/dpi*2.54:.1f}", (x-25, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        
        return new_image
    except Exception as e:
        print(f"Error in add_ruler_to_image: {str(e)}")
        return image

def draw_matching_boxes(image, match_regions, original_size):
    """رسم مربعات حول المناطق المتطابقة"""
    try:
        if image is None or not match_regions:
            return image
            
        # نسخ الصورة للرسم عليها
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        # رسم المربعات لكل منطقة تطابق
        for region in match_regions:
            # استخراج معلومات المربع
            x1, y1 = region['box'][0]
            x2, y2 = region['box'][1]
            score = region['score']
            
            # تحديد اللون حسب درجة التطابق
            if score > 75:
                color = (0, 255, 0)  # أخضر للتطابق العالي
            elif score > 50:
                color = (255, 255, 0)  # أصفر للتطابق المتوسط
            else:
                color = (0, 0, 255)  # أحمر للتطابق المنخفض
            
            # رسم المربع
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # إضافة نسبة التطابق
            cv2.putText(result, f"{score:.1f}%", (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    except Exception as e:
        print(f"Error in draw_matching_boxes: {str(e)}")
        return image 