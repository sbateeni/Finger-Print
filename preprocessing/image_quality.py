import cv2
import numpy as np

def assess_image_quality(image):
    """
    Assess the quality of the fingerprint image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate various quality metrics
    metrics = {
        'contrast': calculate_contrast(gray),
        'sharpness': calculate_sharpness(gray),
        'noise': calculate_noise(gray),
        'brightness': calculate_brightness(gray),
        'orientation_consistency': calculate_orientation_consistency(gray)
    }

    # Calculate overall quality score
    quality_score = calculate_quality_score(metrics)

    return {
        'quality_score': quality_score,
        'metrics': metrics,
        'recommendations': generate_quality_recommendations(metrics)
    }

def calculate_contrast(image):
    """
    Calculate image contrast
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    # Calculate contrast using histogram spread
    mean = np.mean(image)
    contrast = np.sqrt(np.sum(hist * (np.arange(256) - mean) ** 2))
    
    return min(contrast / 128, 1.0)  # Normalize to [0,1]

def calculate_sharpness(image):
    """
    Calculate image sharpness using Laplacian variance
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = np.var(laplacian)
    return min(sharpness / 1000, 1.0)  # Normalize to [0,1]

def calculate_noise(image):
    """
    Calculate noise level in the image
    """
    # Apply median filter
    denoised = cv2.medianBlur(image, 3)
    
    # Calculate noise as difference between original and denoised
    noise = np.abs(image.astype(float) - denoised.astype(float))
    noise_level = np.mean(noise)
    
    return 1.0 - min(noise_level / 50, 1.0)  # Normalize to [0,1], higher is better

def calculate_brightness(image):
    """
    Calculate image brightness
    """
    brightness = np.mean(image)
    # Ideal brightness is around 128
    return 1.0 - abs(brightness - 128) / 128

def calculate_orientation_consistency(image):
    """
    Calculate consistency of ridge orientations
    """
    # Calculate gradient
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(gradient_y, gradient_x)
    
    # Calculate orientation consistency
    orientation_hist = np.histogram(orientation.flatten(), bins=36, range=(-np.pi, np.pi))[0]
    orientation_hist = orientation_hist / orientation_hist.sum()
    
    # Higher entropy means less consistency
    entropy = -np.sum(orientation_hist * np.log2(orientation_hist + 1e-10))
    return 1.0 - min(entropy / 5, 1.0)

def calculate_quality_score(metrics):
    """
    Calculate overall quality score from individual metrics
    """
    weights = {
        'contrast': 0.2,
        'sharpness': 0.2,
        'noise': 0.2,
        'brightness': 0.2,
        'orientation_consistency': 0.2
    }
    
    score = sum(metrics[key] * weights[key] for key in weights)
    return score

def generate_quality_recommendations(metrics):
    """
    Generate recommendations based on quality metrics
    """
    recommendations = []
    
    if metrics['contrast'] < 0.5:
        recommendations.append("تحسين التباين في الصورة")
    if metrics['sharpness'] < 0.5:
        recommendations.append("تحسين وضوح الصورة")
    if metrics['noise'] < 0.5:
        recommendations.append("تقليل الضوضاء في الصورة")
    if metrics['brightness'] < 0.5:
        recommendations.append("ضبط سطوع الصورة")
    if metrics['orientation_consistency'] < 0.5:
        recommendations.append("تحسين جودة خطوط البصمة")
    
    return recommendations

def normalize_image(image):
    """
    Normalize image for better processing
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    normalized = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    normalized = cv2.GaussianBlur(normalized, (3,3), 0)

    return normalized 