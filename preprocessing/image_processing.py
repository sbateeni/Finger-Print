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
    Enhanced preprocessing of fingerprint image:
    1. Resize to standard size
    2. Convert to grayscale
    3. Remove noise
    4. Enhance contrast
    5. Binarize
    6. Skeletonize
    """
    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize image to standard size
    gray = resize_image(gray)

    # Remove noise
    denoised = remove_noise(gray)

    # Enhance contrast
    enhanced = enhance_contrast(denoised)

    # Binarize using Otsu's method
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonize
    skeleton = cv2.ximgproc.thinning(binary)

    return skeleton

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
            'x': x,
            'y': y,
            'type': 'ending',
            'angle': direction[y, x],
            'magnitude': 1.0
        })
    
    # Add bifurcations
    for y, x in zip(*bifurcation_points):
        features.append({
            'x': x,
            'y': y,
            'type': 'bifurcation',
            'angle': direction[y, x],
            'magnitude': 1.0
        })
    
    return features 