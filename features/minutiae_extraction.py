import cv2
import numpy as np
from preprocessing.image_processing import detect_ridges, analyze_ridge_patterns

def get_minutiae_type(pixel, neighbors):
    """
    Determine the type of minutiae point based on its neighborhood
    Returns: 'ending', 'bifurcation', or None
    """
    # Count the number of 1's in the neighborhood
    count = np.sum(neighbors)
    
    if count == 1:
        return 'ending'
    elif count == 3:
        return 'bifurcation'
    return None

def get_minutiae_angle(neighbors):
    """
    Calculate the angle of the minutiae point based on its neighborhood
    """
    # Find the direction of the ridge
    y, x = np.where(neighbors == 1)
    if len(x) > 0:
        angle = np.arctan2(y[0] - 1, x[0] - 1)
        return np.degrees(angle)
    return 0

def extract_minutiae(skeleton):
    """
    Enhanced minutiae extraction with ridge pattern analysis
    """
    # Detect ridge patterns
    magnitude, direction = detect_ridges(skeleton)
    
    # Analyze ridge patterns
    features = analyze_ridge_patterns(skeleton, direction)
    
    # Create a padded version of the skeleton for easier neighbor checking
    padded = np.pad(skeleton, 1, mode='constant')
    rows, cols = skeleton.shape
    
    minutiae = []
    
    # Process each feature
    for feature in features:
        x, y = feature['x'], feature['y']
        if 0 <= x < cols and 0 <= y < rows:
            # Get 3x3 neighborhood
            neighborhood = padded[y:y+3, x:x+3]
            neighborhood[1, 1] = 0  # Ignore center point
            
            # Add feature with additional information
            minutiae.append({
                'x': x,
                'y': y,
                'type': feature['type'],
                'angle': feature['angle'],
                'magnitude': feature['magnitude'],
                'neighborhood': neighborhood.tolist()
            })
    
    return minutiae

def analyze_ridge_characteristics(skeleton, minutiae):
    """
    Analyze ridge characteristics between minutiae points with memory optimization
    """
    ridge_analysis = []
    max_distance = 100  # Maximum distance to consider for analysis
    
    for i, m1 in enumerate(minutiae):
        for m2 in minutiae[i+1:]:
            # Calculate distance between points
            distance = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            
            # Skip if points are too far apart
            if distance > max_distance:
                continue
            
            # Calculate angle between points
            angle = np.degrees(np.arctan2(m2['y'] - m1['y'], m2['x'] - m1['x']))
            
            # Calculate angle difference
            angle_diff = abs(m1['angle'] - m2['angle'])
            
            ridge_analysis.append({
                'point1': (m1['x'], m1['y']),
                'point2': (m2['x'], m2['y']),
                'distance': distance,
                'angle': angle,
                'angle_difference': angle_diff,
                'type1': m1['type'],
                'type2': m2['type']
            })
    
    return ridge_analysis 