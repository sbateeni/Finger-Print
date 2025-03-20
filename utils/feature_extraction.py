import cv2
import numpy as np
from scipy import ndimage
from .image_processing import get_orientation_field
from config import *

def extract_features(img):
    """
    Extract advanced features from a fingerprint image.
    
    Args:
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}
    
    # Get orientation field
    features['orientation_field'] = get_orientation_field(img)
    
    # Get ridge frequency
    features['ridge_frequency'] = estimate_ridge_frequency(img)
    
    # Get ridge density
    features['ridge_density'] = calculate_ridge_density(img)
    
    # Get core points
    features['core_points'] = detect_core_points(img, features['orientation_field'])
    
    return features

def estimate_ridge_frequency(img, block_size=RIDGE_FREQ_BLOCK_SIZE):
    """
    Estimate the ridge frequency in different regions of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input image
        block_size (int): Size of blocks for frequency estimation
        
    Returns:
        numpy.ndarray: Matrix of ridge frequencies
    """
    height, width = img.shape
    freq = np.zeros((height//block_size, width//block_size))
    
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            
            # Project block along vertical direction
            projection = np.sum(block, axis=1)
            
            # Find peaks in projection
            peaks = find_peaks(projection)
            
            if len(peaks) > 1:
                # Calculate average distance between peaks
                peak_distances = np.diff(peaks)
                freq[i//block_size, j//block_size] = 1.0 / np.mean(peak_distances)
    
    # Smooth frequency field
    freq = ndimage.gaussian_filter(freq, sigma=1)
    
    return freq

def calculate_ridge_density(img, block_size=BLOCK_SIZE):
    """
    Calculate ridge density in different regions of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input image
        block_size (int): Size of blocks for density calculation
        
    Returns:
        numpy.ndarray: Matrix of ridge densities
    """
    height, width = img.shape
    density = np.zeros((height//block_size, width//block_size))
    
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            density[i//block_size, j//block_size] = np.sum(block == 255) / (block_size * block_size)
    
    return density

def detect_core_points(img, orientation_field):
    """
    Detect core points in the fingerprint using Poincare index.
    
    Args:
        img (numpy.ndarray): Input image
        orientation_field (numpy.ndarray): Orientation field matrix
        
    Returns:
        list: List of detected core points
    """
    core_points = []
    height, width = orientation_field.shape
    
    # Calculate Poincare index for each point
    for i in range(1, height-1):
        for j in range(1, width-1):
            poincare_idx = calculate_poincare_index(orientation_field, i, j)
            
            # Core point has Poincare index of 1/2
            if abs(poincare_idx - 0.5) < 0.1:
                # Convert block coordinates to image coordinates
                y = i * BLOCK_SIZE + BLOCK_SIZE//2
                x = j * BLOCK_SIZE + BLOCK_SIZE//2
                core_points.append((x, y))
    
    return core_points

def calculate_poincare_index(orientation_field, i, j):
    """
    Calculate Poincare index at a point in the orientation field.
    
    Args:
        orientation_field (numpy.ndarray): Orientation field matrix
        i (int): Row index
        j (int): Column index
        
    Returns:
        float: Poincare index value
    """
    # Get orientations in 8-neighborhood
    neighbors = [
        orientation_field[i-1,j-1], orientation_field[i-1,j], orientation_field[i-1,j+1],
        orientation_field[i,j+1], orientation_field[i+1,j+1], orientation_field[i+1,j],
        orientation_field[i+1,j-1], orientation_field[i,j-1], orientation_field[i-1,j-1]
    ]
    
    # Calculate orientation differences
    differences = []
    for k in range(8):
        diff = neighbors[k+1] - neighbors[k]
        # Normalize difference to [-pi/2, pi/2]
        if diff > np.pi/2:
            diff -= np.pi
        elif diff < -np.pi/2:
            diff += np.pi
        differences.append(diff)
    
    # Add last difference to complete the circle
    diff = neighbors[0] - neighbors[8]
    if diff > np.pi/2:
        diff -= np.pi
    elif diff < -np.pi/2:
        diff += np.pi
    differences.append(diff)
    
    # Calculate Poincare index
    return sum(differences) / (2 * np.pi)

def find_peaks(signal):
    """
    Find peaks in a 1D signal.
    
    Args:
        signal (numpy.ndarray): Input signal
        
    Returns:
        numpy.ndarray: Array of peak indices
    """
    # Smooth signal
    signal = ndimage.gaussian_filter1d(signal, sigma=2)
    
    # Find local maxima
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i-1] < signal[i] > signal[i+1]:
            peaks.append(i)
    
    return np.array(peaks) 