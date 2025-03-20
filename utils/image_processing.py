import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from config import *

def preprocess_image(image_path):
    """
    Preprocess the fingerprint image for better feature extraction.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image to standard size
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    img = clahe.apply(img)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    
    # Edge detection using Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_img = np.sqrt(sobelx**2 + sobely**2)
    edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Enhance edges using morphological operations
    kernel = np.ones((3,3), np.uint8)
    edge_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    
    # Binarize the image using adaptive thresholding
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Combine edge information with binary image
    img = cv2.bitwise_and(img, edge_img)
    
    # Remove small noise using morphological operations
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Skeletonize the image
    img = skeletonize(img > 0).astype(np.uint8) * 255
    
    return img

def enhance_ridges(img):
    """
    Enhance fingerprint ridges using Gabor filtering.
    
    Args:
        img (numpy.ndarray): Input binary image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Create Gabor kernels at different orientations
    enhanced_img = np.zeros_like(img)
    orientations = np.arange(0, 180, 20)
    
    for angle in orientations:
        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
            GABOR_SIGMA,
            np.deg2rad(angle),
            1.0 / GABOR_FREQ,
            GABOR_GAMMA,
            GABOR_PSI,
            ktype=cv2.CV_32F
        )
        
        # Apply filter
        filtered = cv2.filter2D(img, cv2.CV_8UC1, kernel)
        enhanced_img = cv2.bitwise_or(enhanced_img, filtered)
    
    return enhanced_img

def get_orientation_field(img, block_size=BLOCK_SIZE):
    """
    Calculate the orientation field of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input grayscale image
        block_size (int): Size of blocks for orientation calculation
        
    Returns:
        numpy.ndarray: Orientation field matrix
    """
    # Calculate gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient squares
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    
    # Block processing
    height, width = img.shape
    orientation = np.zeros((height//block_size, width//block_size))
    
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            gxx_block = gxx[i:i+block_size, j:j+block_size]
            gyy_block = gyy[i:i+block_size, j:j+block_size]
            gxy_block = gxy[i:i+block_size, j:j+block_size]
            
            # Calculate dominant direction
            gxx_sum = np.sum(gxx_block)
            gyy_sum = np.sum(gyy_block)
            gxy_sum = np.sum(gxy_block)
            
            orientation[i//block_size, j//block_size] = 0.5 * np.arctan2(2*gxy_sum, gxx_sum-gyy_sum)
    
    # Smooth the orientation field
    orientation = ndimage.gaussian_filter(orientation, sigma=2)
    
    return orientation

def save_debug_image(img, filename, processed_folder):
    """
    Save intermediate processing results for debugging.
    
    Args:
        img (numpy.ndarray): Image to save
        filename (str): Output filename
        processed_folder (str): Output directory
    """
    if SAVE_INTERMEDIATE_IMAGES:
        output_path = os.path.join(processed_folder, filename)
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_IMAGE_QUALITY]) 