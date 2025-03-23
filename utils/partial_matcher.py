import cv2
import numpy as np
from scipy import ndimage
from .minutiae_extraction import extract_minutiae, calculate_orientation
from config import *

def match_partial_fingerprint(partial_img, full_img, features1, features2):
    """
    Match a partial fingerprint against a full fingerprint.
    
    Args:
        partial_img: The partial fingerprint image
        full_img: The full fingerprint image
        features1: Features from partial fingerprint
        features2: Features from full fingerprint
        
    Returns:
        dict: Matching results including best match region and score
    """
    results = {
        'best_match_score': 0,
        'best_match_region': None,
        'matched_minutiae': [],
        'region_scores': [],
        'match_location': None
    }
    
    # Get partial fingerprint dimensions
    partial_height, partial_width = partial_img.shape
    full_height, full_width = full_img.shape
    
    # Calculate sliding window parameters
    overlap_ratio = 0.5
    step_size = int(min(partial_height, partial_width) * (1 - overlap_ratio))
    
    # Store all region scores for visualization
    region_scores = []
    
    print("Starting partial fingerprint matching...")
    print(f"Partial fingerprint size: {partial_width}x{partial_height}")
    print(f"Full fingerprint size: {full_width}x{full_height}")
    print(f"Using step size: {step_size}")
    
    # Slide window over full fingerprint
    for y in range(0, full_height - partial_height + 1, step_size):
        for x in range(0, full_width - partial_width + 1, step_size):
            # Extract region
            region = full_img[y:y+partial_height, x:x+partial_width]
            
            # Extract minutiae from region
            region_minutiae = extract_minutiae(region)
            
            # Calculate match score for this region
            region_score = calculate_region_match_score(
                partial_img, region,
                features1, features2
            )
            
            region_scores.append({
                'x': x,
                'y': y,
                'score': region_score
            })
            
            # Update best match if current score is higher
            if region_score > results['best_match_score']:
                results['best_match_score'] = region_score
                results['best_match_region'] = (x, y, x+partial_width, y+partial_height)
                results['match_location'] = (x, y)
    
    results['region_scores'] = region_scores
    print(f"Best match score: {results['best_match_score']}")
    print(f"Best match location: {results['match_location']}")
    
    return results

def calculate_region_match_score(partial_img, region_img, features1, features2):
    """
    Calculate matching score between partial fingerprint and a region.
    
    Args:
        partial_img: Partial fingerprint image
        region_img: Region from full fingerprint
        features1: Features from partial fingerprint
        features2: Features from full fingerprint
        
    Returns:
        float: Matching score between 0 and 100
    """
    # Extract minutiae
    minutiae1 = extract_minutiae(partial_img)
    minutiae2 = extract_minutiae(region_img)
    
    # Calculate orientation field correlation
    orientation1 = features1['orientation_field']
    orientation2 = features2['orientation_field']
    orientation_score = compare_orientation_fields(orientation1, orientation2)
    
    # Calculate ridge density correlation
    density1 = features1['ridge_density']
    density2 = features2['ridge_density']
    density_score = compare_ridge_density(density1, density2)
    
    # Calculate minutiae match score
    minutiae_score = compare_minutiae_sets(minutiae1, minutiae2)
    
    # Combine scores with weights
    total_score = (
        0.5 * minutiae_score +
        0.3 * orientation_score +
        0.2 * density_score
    )
    
    return total_score

def compare_orientation_fields(field1, field2):
    """Compare orientation fields and return similarity score."""
    if field1.shape != field2.shape:
        # Resize to match
        field2 = cv2.resize(field2, (field1.shape[1], field1.shape[0]))
    
    diff = np.abs(field1 - field2)
    diff = np.minimum(diff, np.pi - diff)  # Handle circular nature of angles
    similarity = 1 - (np.mean(diff) / (np.pi/2))
    return similarity * 100

def compare_ridge_density(density1, density2):
    """Compare ridge density maps and return similarity score."""
    if density1.shape != density2.shape:
        # Resize to match
        density2 = cv2.resize(density2, (density1.shape[1], density1.shape[0]))
    
    correlation = np.corrcoef(density1.flatten(), density2.flatten())[0,1]
    return max(0, correlation * 100)

def compare_minutiae_sets(minutiae1, minutiae2):
    """Compare two sets of minutiae points and return similarity score."""
    if not minutiae1 or not minutiae2:
        return 0
    
    matched_pairs = 0
    total_possible = min(len(minutiae1), len(minutiae2))
    
    for m1 in minutiae1:
        for m2 in minutiae2:
            # Calculate distance and orientation difference
            dist = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            orientation_diff = abs(m1['orientation'] - m2['orientation'])
            orientation_diff = min(orientation_diff, 2*np.pi - orientation_diff)
            
            # Check if points match
            if (dist < MINUTIAE_DISTANCE_THRESHOLD and 
                orientation_diff < ORIENTATION_TOLERANCE and
                m1['type'] == m2['type']):
                matched_pairs += 1
                break
    
    return (matched_pairs / total_possible) * 100

def visualize_partial_match(partial_img, full_img, match_result):
    """
    Create visualization of partial fingerprint matching result.
    
    Args:
        partial_img: Partial fingerprint image
        full_img: Full fingerprint image
        match_result: Dictionary containing matching results
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create color versions of both images
    vis_partial = cv2.cvtColor(partial_img, cv2.COLOR_GRAY2BGR)
    vis_full = cv2.cvtColor(full_img, cv2.COLOR_GRAY2BGR)
    
    # Draw rectangle around best matching region
    if match_result['best_match_region']:
        x1, y1, x2, y2 = match_result['best_match_region']
        cv2.rectangle(vis_full, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Create heat map of region scores
    heat_map = np.zeros_like(full_img, dtype=np.float32)
    for score in match_result['region_scores']:
        heat_map[score['y']:score['y']+partial_img.shape[0],
                score['x']:score['x']+partial_img.shape[1]] = score['score']
    
    # Normalize heat map
    heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map = heat_map.astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    
    # Combine visualizations
    h_spacing = 20
    result = np.zeros((
        max(vis_partial.shape[0], vis_full.shape[0], heat_map.shape[0]),
        vis_partial.shape[1] + h_spacing + vis_full.shape[1] + h_spacing + heat_map.shape[1],
        3
    ), dtype=np.uint8)
    
    # Copy images to result
    result[:vis_partial.shape[0], :vis_partial.shape[1]] = vis_partial
    result[:vis_full.shape[0], 
           vis_partial.shape[1]+h_spacing:vis_partial.shape[1]+h_spacing+vis_full.shape[1]] = vis_full
    result[:heat_map.shape[0],
           vis_partial.shape[1]+h_spacing+vis_full.shape[1]+h_spacing:] = heat_map
    
    return result 