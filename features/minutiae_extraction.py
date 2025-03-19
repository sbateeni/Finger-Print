import cv2
import numpy as np
from preprocessing.image_processing import detect_ridges, analyze_ridge_patterns

def get_minutiae_type(pixel, neighbors):
    """
    Determine the type of minutiae point based on its neighborhood
    Returns: 'ending', 'bifurcation', 'dot', or None
    """
    # Count the number of 1's in the neighborhood
    count = np.sum(neighbors)
    
    if count == 1:
        return 'ending'
    elif count == 3:
        return 'bifurcation'
    elif count == 0 and pixel == 1:
        return 'dot'
    return None

def get_minutiae_angle(neighbors):
    """
    Calculate the angle of the minutiae point based on its neighborhood
    """
    # Find the direction of the ridge
    y, x = np.where(neighbors == 1)
    if len(x) > 0:
        # Calculate angle based on the position of the neighbor
        angle = np.arctan2(y[0] - 1, x[0] - 1)
        # Convert to degrees and normalize to 0-360 range
        angle = np.degrees(angle)
        if angle < 0:
            angle += 360
        return angle
    return 0

def detect_dots(skeleton):
    """
    Detect isolated dots in the fingerprint
    """
    dots = []
    rows, cols = skeleton.shape
    
    # Create a padded version for easier neighbor checking
    padded = np.pad(skeleton, 1, mode='constant')
    
    for y in range(1, rows+1):
        for x in range(1, cols+1):
            if padded[y, x] == 1:
                # Get 3x3 neighborhood
                neighborhood = padded[y-1:y+2, x-1:x+2]
                neighborhood[1, 1] = 0  # Ignore center point
                
                # Check if it's a dot (isolated point)
                if np.sum(neighborhood) == 0:
                    dots.append({
                        'x': x-1,  # Adjust for padding
                        'y': y-1,
                        'type': 'dot',
                        'angle': 0,  # Dots don't have orientation
                        'magnitude': 1.0
                    })
    
    return dots

def extract_minutiae(skeleton):
    """
    Extract minutiae points from skeletonized fingerprint image
    """
    try:
        # Create a padded version of the skeleton for easier neighbor checking
        padded = np.pad(skeleton, 1, mode='constant')
        rows, cols = skeleton.shape
        
        minutiae = []
        
        # Process each pixel in the skeleton
        for y in range(1, rows+1):
            for x in range(1, cols+1):
                if padded[y, x] == 1:
                    # Get 3x3 neighborhood
                    neighborhood = padded[y-1:y+2, x-1:x+2]
                    neighborhood[1, 1] = 0  # Ignore center point
                    
                    # Count neighbors
                    count = np.sum(neighborhood)
                    
                    # Determine minutiae type
                    if count == 1:  # Ridge ending
                        minutiae.append({
                            'x': x-1,  # Adjust for padding
                            'y': y-1,
                            'type': 'ending',
                            'angle': get_minutiae_angle(neighborhood),
                            'magnitude': 1.0
                        })
                    elif count == 3:  # Bifurcation
                        minutiae.append({
                            'x': x-1,
                            'y': y-1,
                            'type': 'bifurcation',
                            'angle': get_minutiae_angle(neighborhood),
                            'magnitude': 1.0
                        })
        
        # Detect and add dots
        dots = detect_dots(skeleton)
        minutiae.extend(dots)
        
        # Remove duplicate minutiae points
        unique_minutiae = []
        seen = set()
        for m in minutiae:
            key = (m['x'], m['y'])
            if key not in seen:
                seen.add(key)
                unique_minutiae.append(m)
        
        return unique_minutiae
    except Exception as e:
        print(f"Error in extract_minutiae: {str(e)}")
        return []

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