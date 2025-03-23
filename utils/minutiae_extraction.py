import cv2
import numpy as np
from scipy import ndimage
from config import *

def extract_minutiae(image, max_points=100):
    """
    Extract minutiae points from a fingerprint image.
    
    Args:
        image: Input fingerprint image
        max_points: Maximum number of minutiae points to return (default: 100)
        
    Returns:
        list: List of minutiae points with their properties
    """
    try:
        print("Starting minutiae extraction with improved filtering...")
        print(f"Maximum points requested: {max_points}")
        
        # Initialize list to store minutiae points
        minutiae = []
        
        # Get image dimensions
        height, width = image.shape
        
        # Create padded image for neighborhood operations
        padded = np.pad(image, ((1,1), (1,1)), mode='constant')
        
        # Minimum distance between minutiae points (increased for better filtering)
        MIN_DISTANCE = 20  # Increased from original value
        
        # Quality threshold for minutiae points
        QUALITY_THRESHOLD = 0.4  # Minimum quality score to accept a minutia
        
        print("Processing image for minutiae extraction...")
        
        # Crossing number method for minutiae extraction
        for i in range(1, height-1):
            for j in range(1, width-1):
                if image[i,j] == 255:  # Ridge pixel
                    # Get 8-neighborhood
                    neighbors = [
                        padded[i-1,j-1], padded[i-1,j], padded[i-1,j+1],
                        padded[i,j+1], padded[i+1,j+1], padded[i+1,j],
                        padded[i+1,j-1], padded[i,j-1], padded[i-1,j-1]
                    ]
                    
                    # Calculate crossing number
                    crossings = sum(abs(int(neighbors[k]//255) - int(neighbors[k+1]//255)) for k in range(8))
                    crossings += abs(int(neighbors[8]//255) - int(neighbors[0]//255))
                    
                    # Calculate quality score for current point
                    quality = calculate_minutia_quality(image, i, j)
                    
                    # Only process high-quality points
                    if quality >= QUALITY_THRESHOLD:
                        # Determine minutiae type
                        if crossings == 2:  # Ridge ending
                            orientation = calculate_orientation(image, i, j)
                            if is_valid_minutia(i, j, minutiae, MIN_DISTANCE):
                                minutiae.append({
                                    'type': 'ending',
                                    'x': j,
                                    'y': i,
                                    'orientation': orientation,
                                    'quality': quality
                                })
                        elif crossings == 6:  # Ridge bifurcation
                            orientation = calculate_orientation(image, i, j)
                            if is_valid_minutia(i, j, minutiae, MIN_DISTANCE):
                                minutiae.append({
                                    'type': 'bifurcation',
                                    'x': j,
                                    'y': i,
                                    'orientation': orientation,
                                    'quality': quality
                                })
        
        print(f"Initial minutiae count: {len(minutiae)}")
        
        # Sort minutiae by quality and limit to max_points
        minutiae = sorted(minutiae, key=lambda x: x.get('quality', 0), reverse=True)
        minutiae = minutiae[:max_points]
        
        print(f"Final minutiae count after limiting to max points: {len(minutiae)}")
        return minutiae
        
    except Exception as e:
        print(f"Error in minutiae extraction: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return []

def calculate_orientation(img, y, x, window_size=MINUTIAE_WINDOW_SIZE):
    """
    Calculate the orientation of a minutia point.
    
    Args:
        img (numpy.ndarray): Input image
        y (int): Y-coordinate of the minutia
        x (int): X-coordinate of the minutia
        window_size (int): Size of the window for orientation calculation
        
    Returns:
        float: Orientation angle in radians
    """
    # Extract window around minutia
    half_window = window_size // 2
    window = img[
        max(0, y-half_window):min(img.shape[0], y+half_window+1),
        max(0, x-half_window):min(img.shape[1], x+half_window+1)
    ]
    
    # Calculate gradients
    gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(np.sum(gy), np.sum(gx))
    
    return orientation

def is_valid_minutia(y, x, existing_minutiae, min_distance):
    """
    Check if a minutia point is valid based on distance from existing minutiae.
    
    Args:
        y (int): Y-coordinate of the new minutia
        x (int): X-coordinate of the new minutia
        existing_minutiae (list): List of existing minutiae points
        min_distance (float): Minimum allowed distance between minutiae
        
    Returns:
        bool: True if the minutia is valid, False otherwise
    """
    for m in existing_minutiae:
        dist = np.sqrt((m['y'] - y)**2 + (m['x'] - x)**2)
        if dist < min_distance:
            return False
    return True

def visualize_minutiae(img, minutiae):
    """
    Create a visualization of detected minutiae points.
    
    Args:
        img (numpy.ndarray): Input image
        minutiae (list): List of detected minutiae points
        
    Returns:
        numpy.ndarray: Image with visualized minutiae
    """
    # Convert to color image
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw minutiae points
    for m in minutiae:
        x, y = m['x'], m['y']
        if m['type'] == 'ending':
            color = (0, 0, 255)  # Red for endings
        else:
            color = (0, 255, 0)  # Green for bifurcations
            
        # Draw point
        cv2.circle(vis_img, (x, y), 3, color, -1)
        
        # Draw orientation line
        length = 15
        end_x = int(x + length * np.cos(m['orientation']))
        end_y = int(y + length * np.sin(m['orientation']))
        cv2.line(vis_img, (x, y), (end_x, end_y), color, 1)
    
    return vis_img

def detect_delta_points(img):
    """
    Detect delta points in the fingerprint image.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        list: List of detected delta points
    """
    delta_points = []
    height, width = img.shape
    
    # Calculate orientation field
    block_size = 16
    orientation_field = np.zeros((height//block_size, width//block_size))
    
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            gx = cv2.Sobel(block, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(block, cv2.CV_64F, 0, 1)
            orientation = np.arctan2(np.sum(gy), np.sum(gx))
            orientation_field[i//block_size, j//block_size] = orientation
    
    # Detect delta points using Poincare index
    for i in range(1, orientation_field.shape[0]-1):
        for j in range(1, orientation_field.shape[1]-1):
            poincare_idx = calculate_poincare_index(orientation_field, i, j)
            if abs(poincare_idx + 0.5) < 0.1:  # Delta point has index -1/2
                y = i * block_size + block_size//2
                x = j * block_size + block_size//2
                orientation = calculate_orientation(img, y, x)
                delta_points.append({
                    'type': 'delta',
                    'x': x,
                    'y': y,
                    'orientation': orientation,
                    'quality': calculate_minutia_quality(img, y, x)
                })
    
    return delta_points

def calculate_minutia_quality(img, y, x, window_size=16):
    """
    Calculate quality score for a minutia point.
    
    Args:
        img (numpy.ndarray): Input image
        y (int): Y-coordinate
        x (int): X-coordinate
        window_size (int): Size of window for quality calculation
        
    Returns:
        float: Quality score between 0 and 1
    """
    # Extract window around minutia
    half_window = window_size // 2
    window = img[
        max(0, y-half_window):min(img.shape[0], y+half_window),
        max(0, x-half_window):min(img.shape[1], x+half_window)
    ]
    
    # Calculate ridge clarity
    clarity = np.std(window) / 255.0
    
    # Calculate ridge continuity
    gx = cv2.Sobel(window, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(window, cv2.CV_64F, 0, 1)
    continuity = np.mean(np.sqrt(gx**2 + gy**2)) / 255.0
    
    # Combine scores
    quality = (clarity + continuity) / 2.0
    return min(1.0, quality)

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
        orientation_field[i+1,j-1], orientation_field[i,j-1]
    ]
    neighbors.append(neighbors[0])  # Add first element to close the loop
    
    # Calculate orientation differences
    total_diff = 0
    for k in range(8):
        diff = neighbors[k+1] - neighbors[k]
        # Normalize difference to [-pi/2, pi/2]
        while diff > np.pi/2:
            diff -= np.pi
        while diff < -np.pi/2:
            diff += np.pi
        total_diff += diff
    
    # Calculate Poincare index
    return total_diff / (2 * np.pi) 