import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d

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
    """المرحلة الثالثة: اكتشاف الخطوط والتعرجات"""
    try:
        # تطبيق فلتر Gabor
        ksize = 31
        sigma = 4.0
        theta = 0
        lambda_ = 10.0
        gamma = 0.5
        psi = 0
        
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
        ridges = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        # تطبيق فلتر Sobel للحصول على خريطة الاتجاه
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # تجنب القسمة على صفر
        orientation_map = np.zeros_like(sobelx)
        mask = (sobelx != 0) | (sobely != 0)
        orientation_map[mask] = np.arctan2(sobely[mask], sobelx[mask])
        
        return ridges, orientation_map
    except Exception as e:
        print(f"Error in detect_ridges: {str(e)}")
        return image, np.zeros_like(image)

def preprocess_image(image):
    """المرحلة الأولى: معالجة الصورة الأساسية"""
    # تحويل الصورة إلى Grayscale إذا لم تكن كذلك
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # توحيد حجم الصورة
    image = cv2.resize(image, (500, 500))
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    return image

def enhance_image(image):
    """المرحلة الثانية: تحسين جودة الصورة"""
    # إزالة الضوضاء
    image = cv2.GaussianBlur(image, (5,5), 0)
    
    # تحسين الحواف
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # تحسين التباين
    image = cv2.equalizeHist(image)
    
    return image

def analyze_ridge_patterns(ridges, orientation_map):
    """المرحلة السادسة: تحليل أنماط الخطوط"""
    # تحليل أنماط الخطوط (Loop, Arch, Whorl)
    patterns = {
        'loop': False,
        'arch': False,
        'whorl': False
    }
    
    # تحليل التدفق العام للخطوط
    flow_direction = np.mean(orientation_map)
    
    # تحديد النمط بناءً على اتجاه التدفق
    if flow_direction > 0.5:
        patterns['loop'] = True
    elif flow_direction < -0.5:
        patterns['arch'] = True
    else:
        patterns['whorl'] = True
    
    return patterns

def normalize_image(image):
    """توحيد حجم ومقياس الصورة"""
    # توحيد الحجم
    image = cv2.resize(image, (500, 500))
    
    # توحيد القيم
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return image

def detect_orientation(image):
    """تحديد اتجاه البصمة"""
    # حساب التدرج
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # حساب خريطة الاتجاه
    orientation = np.arctan2(gradient_y, gradient_x)
    
    # حساب الاتجاه الرئيسي
    main_orientation = np.mean(orientation)
    
    return main_orientation

def normalize_rotation(image, orientation_map):
    """المرحلة السابعة: معالجة التدوير والميل"""
    try:
        # تجنب القسمة على صفر
        valid_orientations = orientation_map[orientation_map != 0]
        if len(valid_orientations) > 0:
            mean_orientation = np.mean(valid_orientations)
            rotation_angle = -mean_orientation * 180 / np.pi
        else:
            rotation_angle = 0
        
        # تدوير الصورة
        height, width = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image
    except Exception as e:
        print(f"Error in normalize_rotation: {str(e)}")
        return image 