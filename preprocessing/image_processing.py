import cv2
import numpy as np

def resize_image(image, target_size=(800, 800)):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_size = (int(w*scale), int(h*scale))
    return cv2.resize(image, new_size)

def enhance_contrast(image):
    """Enhance contrast using CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def remove_noise(image):
    """Remove noise using bilateral filter"""
    return cv2.bilateralFilter(image, 9, 75, 75)

def preprocess_image(image):
    """
    Preprocess fingerprint image for feature extraction
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Skeletonize the image
        skeleton = skeletonize(binary)

        # Convert to binary (0 and 1)
        skeleton = (skeleton > 0).astype(np.uint8)

        # Remove isolated pixels
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)

        return skeleton
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None

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