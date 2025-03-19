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
    """Simple skeletonization algorithm"""
    try:
        if image is None:
            print("Error in skeletonize: Input image is None")
            return None
            
        # Create a copy of the image
        skeleton = image.copy()
        
        # Create a structuring element
        kernel = np.ones((3,3), np.uint8)
        
        # Iterate until no more changes
        iteration = 0
        max_iterations = 100  # Prevent infinite loop
        
        while iteration < max_iterations:
            # Erode the image
            eroded = cv2.erode(skeleton, kernel)
            
            # Open the eroded image
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            
            # Subtract opened from skeleton
            temp = cv2.subtract(skeleton, opened)
            
            # If no more changes, break
            if cv2.countNonZero(temp) == 0:
                break
                
            # Update skeleton
            skeleton = eroded
            iteration += 1
            
        print(f"Skeletonization completed in {iteration} iterations")
        return skeleton
    except Exception as e:
        print(f"Error in skeletonize: {str(e)}")
        return None

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

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        print("Applied adaptive thresholding")

        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print("Applied morphological operations")

        # Convert to binary (0 and 1)
        binary = (binary > 0).astype(np.uint8)

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

        # Remove isolated pixels
        kernel = np.ones((3,3), np.uint8)
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)

        # Ensure skeleton is not empty
        non_zero = cv2.countNonZero(skeleton)
        if non_zero == 0:
            print("Error: Skeleton is empty")
            return None, None
        print(f"Final skeleton has {non_zero} non-zero pixels")

        return skeleton, direction
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None, None

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