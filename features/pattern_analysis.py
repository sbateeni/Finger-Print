import numpy as np
import cv2

def detect_loop_pattern(skeleton, direction):
    """
    Detect loop patterns in the fingerprint
    """
    loops = []
    rows, cols = skeleton.shape
    
    # Look for circular patterns
    for y in range(2, rows-2):
        for x in range(2, cols-2):
            if skeleton[y, x] == 1:
                # Check 5x5 neighborhood for loop pattern
                neighborhood = skeleton[y-2:y+3, x-2:x+3]
                if np.sum(neighborhood) >= 8:  # Minimum points for a loop
                    # Check if it forms a loop
                    if is_loop_pattern(neighborhood, direction[y-2:y+3, x-2:x+3]):
                        loops.append({
                            'x': x,
                            'y': y,
                            'type': 'loop',
                            'confidence': calculate_pattern_confidence(neighborhood)
                        })
    return loops

def detect_arch_pattern(skeleton, direction):
    """
    Detect arch patterns in the fingerprint
    """
    arches = []
    rows, cols = skeleton.shape
    
    # Look for arch patterns
    for y in range(2, rows-2):
        for x in range(2, cols-2):
            if skeleton[y, x] == 1:
                # Check 5x5 neighborhood for arch pattern
                neighborhood = skeleton[y-2:y+3, x-2:x+3]
                if np.sum(neighborhood) >= 6:  # Minimum points for an arch
                    # Check if it forms an arch
                    if is_arch_pattern(neighborhood, direction[y-2:y+3, x-2:x+3]):
                        arches.append({
                            'x': x,
                            'y': y,
                            'type': 'arch',
                            'confidence': calculate_pattern_confidence(neighborhood)
                        })
    return arches

def detect_whorl_pattern(skeleton, direction):
    """
    Detect whorl patterns in the fingerprint
    """
    whorls = []
    rows, cols = skeleton.shape
    
    # Look for whorl patterns
    for y in range(2, rows-2):
        for x in range(2, cols-2):
            if skeleton[y, x] == 1:
                # Check 5x5 neighborhood for whorl pattern
                neighborhood = skeleton[y-2:y+3, x-2:x+3]
                if np.sum(neighborhood) >= 10:  # Minimum points for a whorl
                    # Check if it forms a whorl
                    if is_whorl_pattern(neighborhood, direction[y-2:y+3, x-2:x+3]):
                        whorls.append({
                            'x': x,
                            'y': y,
                            'type': 'whorl',
                            'confidence': calculate_pattern_confidence(neighborhood)
                        })
    return whorls

def is_loop_pattern(neighborhood, direction):
    """
    Check if the neighborhood forms a loop pattern
    """
    # Count the number of ridge points
    ridge_points = np.sum(neighborhood)
    
    # Check if the pattern is roughly circular
    if ridge_points < 8:
        return False
        
    # Check direction consistency
    directions = direction[neighborhood == 1]
    direction_variance = np.var(directions)
    
    return direction_variance < 45  # Allow some variation in direction

def is_arch_pattern(neighborhood, direction):
    """
    Check if the neighborhood forms an arch pattern
    """
    # Count the number of ridge points
    ridge_points = np.sum(neighborhood)
    
    if ridge_points < 6:
        return False
        
    # Check if the pattern is roughly U-shaped
    directions = direction[neighborhood == 1]
    direction_variance = np.var(directions)
    
    return direction_variance < 60  # Allow more variation for arches

def is_whorl_pattern(neighborhood, direction):
    """
    Check if the neighborhood forms a whorl pattern
    """
    # Count the number of ridge points
    ridge_points = np.sum(neighborhood)
    
    if ridge_points < 10:
        return False
        
    # Check if the pattern is spiral-like
    directions = direction[neighborhood == 1]
    direction_variance = np.var(directions)
    
    return direction_variance < 90  # Allow significant variation for whorls

def calculate_pattern_confidence(neighborhood):
    """
    Calculate confidence score for a detected pattern
    """
    # Count ridge points
    ridge_points = np.sum(neighborhood)
    
    # Calculate pattern density
    total_points = neighborhood.size
    density = ridge_points / total_points
    
    # Calculate pattern continuity
    continuity = calculate_pattern_continuity(neighborhood)
    
    # Combined confidence score
    return (density + continuity) / 2

def calculate_pattern_continuity(neighborhood):
    """
    Calculate how continuous the pattern is
    """
    # Count number of 8-connected components
    num_labels, labels = cv2.connectedComponents(neighborhood.astype(np.uint8), connectivity=8)
    
    # More components means less continuity
    return 1.0 / num_labels

def analyze_ridge_patterns(skeleton, direction):
    """
    Main function to analyze all ridge patterns
    """
    patterns = {
        'loops': detect_loop_pattern(skeleton, direction),
        'arches': detect_arch_pattern(skeleton, direction),
        'whorls': detect_whorl_pattern(skeleton, direction)
    }
    
    return patterns 