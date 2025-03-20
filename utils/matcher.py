import cv2
import numpy as np
from config import *

# Constants for RANSAC algorithm
RANSAC_ITERATIONS = 1000  # Number of iterations for RANSAC
RANSAC_THRESHOLD = 10.0   # Distance threshold for inlier detection

# Constants for local structure matching
LOCAL_STRUCTURE_THRESHOLD = 0.6  # Minimum score for accepting a local structure match

def match_fingerprints(minutiae1, minutiae2, features1, features2):
    """
    Match two fingerprints based on minutiae points and other features.
    
    Args:
        minutiae1 (list): Minutiae points from first fingerprint
        minutiae2 (list): Minutiae points from second fingerprint
        features1 (dict): Features from first fingerprint
        features2 (dict): Features from second fingerprint
        
    Returns:
        dict: Matching results including matched points and scores
    """
    try:
        print("\n=== Starting fingerprint matching process ===")
        print(f"Initial minutiae counts - Image 1: {len(minutiae1)}, Image 2: {len(minutiae2)}")
        
        # Limit the number of minutiae points to improve performance
        MAX_MINUTIAE = 100  # Reduced from 150 to 100 for better performance
        if len(minutiae1) > MAX_MINUTIAE or len(minutiae2) > MAX_MINUTIAE:
            print(f"\nLimiting minutiae points to {MAX_MINUTIAE} for better performance")
            # Sort by quality and take top points
            minutiae1 = sorted(minutiae1, key=lambda x: x.get('quality', 0), reverse=True)[:MAX_MINUTIAE]
            minutiae2 = sorted(minutiae2, key=lambda x: x.get('quality', 0), reverse=True)[:MAX_MINUTIAE]
            print(f"After limiting - Image 1: {len(minutiae1)}, Image 2: {len(minutiae2)}")
        
        # Initialize result dictionary
        result = {
            'matched_minutiae': [],
            'transformation': None,
            'minutiae_score': 0,
            'orientation_score': 0,
            'density_score': 0,
            'local_structure_score': 0,
            'scale_factor': 1.0
        }
        
        try:
            # Calculate and apply scale normalization
            print("\n=== Calculating scale factor ===")
            scale_factor = calculate_scale_factor(minutiae1, minutiae2)
            print(f"Scale factor calculated: {scale_factor}")
            
            normalized_minutiae1 = normalize_scale(minutiae1, scale_factor)
            result['scale_factor'] = scale_factor
            print("Scale normalization applied successfully")
            
            # Find best transformation between fingerprints
            print("\n=== Finding best transformation ===")
            transformation = find_best_transformation(normalized_minutiae1, minutiae2)
            result['transformation'] = transformation
            
            if transformation is not None:
                print("\n=== Applying transformation to minutiae points ===")
                # Transform minutiae points
                transformed_minutiae1 = transform_minutiae(normalized_minutiae1, transformation)
                print("Transformation applied successfully")
                
                # Match minutiae points using local structure matching
                print("\n=== Matching local structures ===")
                matched_pairs = match_local_structures(
                    transformed_minutiae1,
                    minutiae2,
                    MINUTIAE_DISTANCE_THRESHOLD,
                    ORIENTATION_TOLERANCE
                )
                result['matched_minutiae'] = matched_pairs
                print(f"Found {len(matched_pairs)} matching pairs")
                
                # Calculate scores
                print("\n=== Calculating scores ===")
                print("1. Calculating minutiae score...")
                result['minutiae_score'] = calculate_minutiae_score(matched_pairs, minutiae1, minutiae2)
                print(f"   Minutiae score: {result['minutiae_score']:.2f}%")
                
                print("2. Calculating orientation score...")
                result['orientation_score'] = compare_orientation_fields(
                    features1['orientation_field'],
                    features2['orientation_field']
                )
                print(f"   Orientation score: {result['orientation_score']:.2f}%")
                
                print("3. Calculating density score...")
                result['density_score'] = compare_ridge_density(
                    features1['ridge_density'],
                    features2['ridge_density']
                )
                print(f"   Density score: {result['density_score']:.2f}%")
                
                print("4. Calculating local structure score...")
                result['local_structure_score'] = calculate_local_structure_score(matched_pairs)
                print(f"   Local structure score: {result['local_structure_score']:.2f}%")
                
                print("\n=== Matching process completed successfully ===")
            else:
                print("\n=== No valid transformation found ===")
            
            return result
            
        except Exception as e:
            print(f"\nError during matching process: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            return result
            
    except Exception as e:
        print(f"\nCritical error in match_fingerprints: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        
        # Return default result in case of error
        return {
            'matched_minutiae': [],
            'transformation': None,
            'minutiae_score': 0,
            'orientation_score': 0,
            'density_score': 0,
            'local_structure_score': 0,
            'scale_factor': 1.0
        }

def find_best_transformation(minutiae1, minutiae2):
    """
    Find the best transformation between two sets of minutiae points.
    """
    try:
        print("\nStarting transformation search...")
        print(f"Input minutiae - Set 1: {len(minutiae1)}, Set 2: {len(minutiae2)}")
        
        if len(minutiae1) < 3 or len(minutiae2) < 3:
            print("Not enough minutiae points for transformation")
            return None
            
        # Convert minutiae points to numpy arrays for easier manipulation
        try:
            points1 = np.array([[m['x'], m['y']] for m in minutiae1])
            points2 = np.array([[m['x'], m['y']] for m in minutiae2])
            print("Successfully converted minutiae to numpy arrays")
        except Exception as e:
            print(f"Error converting minutiae to arrays: {str(e)}")
            print("Minutiae1 first point:", minutiae1[0] if minutiae1 else "None")
            print("Minutiae2 first point:", minutiae2[0] if minutiae2 else "None")
            raise
        
        print(f"\nProcessing {len(points1)} points from image 1 and {len(points2)} points from image 2")
        print("Points1 shape:", points1.shape)
        print("Points2 shape:", points2.shape)
        
        # Use RANSAC to find the best transformation
        best_transformation = None
        max_inliers = 0
        early_stop_count = 0
        prev_max_inliers = 0
        
        print(f"\nStarting RANSAC with {RANSAC_ITERATIONS} iterations")
        for iteration in range(RANSAC_ITERATIONS):
            try:
                # Randomly select 3 points from each set
                # Select points that are well-distributed
                while True:
                    idx1 = np.random.choice(len(points1), 3, replace=False)
                    sample1 = points1[idx1]
                    # Check if points form a triangle with reasonable area
                    area = np.abs(np.cross(sample1[1] - sample1[0], sample1[2] - sample1[0])) / 2
                    if area > 100:  # Minimum area threshold
                        break
                
                while True:
                    idx2 = np.random.choice(len(points2), 3, replace=False)
                    sample2 = points2[idx2]
                    area = np.abs(np.cross(sample2[1] - sample2[0], sample2[2] - sample2[0])) / 2
                    if area > 100:
                        break
                
                if iteration == 0:
                    print("\nFirst iteration samples:")
                    print("Sample1 shape:", sample1.shape)
                    print("Sample2 shape:", sample2.shape)
                
                # Calculate transformation matrix
                transformation = cv2.getAffineTransform(
                    sample1.astype(np.float32),
                    sample2.astype(np.float32)
                )
                
                # Apply transformation to all points
                transformed_points = cv2.transform(
                    points1.reshape(-1, 1, 2).astype(np.float32),
                    transformation
                ).reshape(-1, 2)
                
                # Count inliers more efficiently using vectorized operations
                distances = np.sqrt(np.sum((transformed_points[:, np.newaxis] - points2)**2, axis=2))
                min_distances = np.min(distances, axis=1)
                inliers = np.sum(min_distances < RANSAC_THRESHOLD)
                
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_transformation = transformation
                    print(f"\nIteration {iteration}: Found better transformation with {inliers} inliers")
                    if iteration == 0:
                        print("First transformation matrix:")
                        print(transformation)
                    
                    # Reset early stopping counter
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                
                # Early stopping if no improvement for many iterations
                if early_stop_count > 100:
                    print("\nEarly stopping: No improvement for 100 iterations")
                    break
                
                # Early stopping if we found a very good transformation
                if max_inliers > min(len(points1), len(points2)) * 0.8:
                    print("\nEarly stopping: Found very good transformation")
                    break
                
            except Exception as e:
                print(f"\nError in RANSAC iteration {iteration}: {str(e)}")
                continue
        
        if best_transformation is not None:
            print(f"\nBest transformation found with {max_inliers} inliers")
            print("Final transformation matrix:")
            print(best_transformation)
            return best_transformation
        else:
            print("\nNo valid transformation found after all iterations")
            return None
            
    except Exception as e:
        print(f"\nError in find_best_transformation: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return None

def transform_minutiae(minutiae, transformation):
    """
    Apply transformation to minutiae points.
    """
    try:
        print("Applying transformation to minutiae points...")
        
        if transformation is None:
            print("No transformation provided")
            return minutiae
            
        # Convert minutiae points to numpy array
        points = np.array([[m['x'], m['y']] for m in minutiae]).reshape(-1, 1, 2).astype(np.float32)
        
        # Apply transformation
        transformed_points = cv2.transform(points, transformation).reshape(-1, 2)
        
        # Create new minutiae list with transformed coordinates
        transformed_minutiae = []
        for i, m in enumerate(minutiae):
            transformed_m = m.copy()
            transformed_m['x'] = float(transformed_points[i][0])
            transformed_m['y'] = float(transformed_points[i][1])
            
            # Adjust orientation based on transformation matrix
            angle = np.arctan2(transformation[1][0], transformation[0][0])
            transformed_m['orientation'] = (m['orientation'] + angle) % (2 * np.pi)
            
            transformed_minutiae.append(transformed_m)
            
        print(f"Successfully transformed {len(transformed_minutiae)} minutiae points")
        return transformed_minutiae
        
    except Exception as e:
        print(f"Error in transform_minutiae: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return minutiae

def match_minutiae_points(minutiae1, minutiae2, distance_threshold, orientation_tolerance):
    """
    Find matching pairs of minutiae points.
    
    Args:
        minutiae1 (list): First set of minutiae points
        minutiae2 (list): Second set of minutiae points
        distance_threshold (float): Maximum distance for matching points
        orientation_tolerance (float): Maximum orientation difference for matching points
        
    Returns:
        list: List of matching minutiae pairs
    """
    matched_pairs = []
    used_indices = set()
    
    for i, m1 in enumerate(minutiae1):
        best_match = None
        min_dist = float('inf')
        best_idx = None
        
        for j, m2 in enumerate(minutiae2):
            if j in used_indices:
                continue
                
            # Check if same type
            if m1['type'] != m2['type']:
                continue
            
            # Calculate distance
            dist = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            
            if dist < min_dist and dist < distance_threshold:
                # Check orientation difference
                orientation_diff = abs(m1['orientation'] - m2['orientation'])
                orientation_diff = min(orientation_diff, 2*np.pi - orientation_diff)
                
                if orientation_diff < orientation_tolerance:
                    min_dist = dist
                    best_match = m2
                    best_idx = j
        
        if best_match is not None:
            matched_pairs.append((m1, best_match))
            used_indices.add(best_idx)
    
    return matched_pairs

def compare_orientation_fields(field1, field2):
    """
    Compare two orientation fields.
    
    Args:
        field1 (numpy.ndarray): First orientation field
        field2 (numpy.ndarray): Second orientation field
        
    Returns:
        float: Similarity score (0-100)
    """
    if field1.shape != field2.shape:
        return 0
    
    diff = np.abs(field1 - field2)
    diff = np.minimum(diff, np.pi - diff)
    similarity = 1 - (np.mean(diff) / (np.pi/2))
    
    return similarity * 100

def compare_ridge_density(density1, density2):
    """
    Compare two ridge density maps.
    
    Args:
        density1 (numpy.ndarray): First ridge density map
        density2 (numpy.ndarray): Second ridge density map
        
    Returns:
        float: Similarity score (0-100)
    """
    if density1.shape != density2.shape:
        return 0
    
    correlation = np.corrcoef(density1.flatten(), density2.flatten())[0,1]
    similarity = (correlation + 1) / 2  # Scale from [-1,1] to [0,1]
    
    return similarity * 100

def visualize_matches(img1, img2, matched_pairs):
    """
    Create a visualization of matched minutiae points.
    
    Args:
        img1 (numpy.ndarray): First fingerprint image
        img2 (numpy.ndarray): Second fingerprint image
        matched_pairs (list): List of matching minutiae pairs
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create side-by-side image
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # Convert grayscale to BGR
    vis_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    vis_img[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    colors = np.random.randint(0, 255, (len(matched_pairs), 3)).tolist()
    
    for i, (m1, m2) in enumerate(matched_pairs):
        # Draw points and lines
        pt1 = (int(m1['x']), int(m1['y']))
        pt2 = (int(m2['x']) + w1, int(m2['y']))
        color = colors[i]
        
        cv2.circle(vis_img, pt1, 3, color, -1)
        cv2.circle(vis_img, pt2, 3, color, -1)
        cv2.line(vis_img, pt1, pt2, color, 1)
    
    return vis_img

def match_local_structures(minutiae1, minutiae2, distance_threshold, orientation_tolerance):
    """
    Match minutiae points based on local structures.
    """
    try:
        print("Starting local structure matching...")
        
        # Reduce thresholds to be more tolerant
        distance_threshold = distance_threshold * 1.5  # Increase distance threshold
        orientation_tolerance = orientation_tolerance * 1.5  # Increase orientation tolerance
        LOCAL_STRUCTURE_THRESHOLD = 0.4  # Reduce threshold for accepting matches
        
        matched_pairs = []
        used_indices2 = set()
        
        # Create local structures for both sets
        structures1 = create_local_structures(minutiae1)
        structures2 = create_local_structures(minutiae2)
        
        print(f"Created local structures - Set 1: {len(structures1)}, Set 2: {len(structures2)}")
        
        # For each minutia in first set
        for i, (m1, s1) in enumerate(zip(minutiae1, structures1)):
            best_match = None
            best_score = -1
            best_idx = -1
            
            # Compare with each minutia in second set
            for j, (m2, s2) in enumerate(zip(minutiae2, structures2)):
                if j in used_indices2:
                    continue
                    
                # Calculate distance between points
                dist = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                if dist > distance_threshold:
                    continue
                    
                # Calculate orientation difference
                orient_diff = abs(m1['orientation'] - m2['orientation']) % (2 * np.pi)
                orient_diff = min(orient_diff, 2 * np.pi - orient_diff)
                if orient_diff > orientation_tolerance:
                    continue
                    
                # Compare local structures with more tolerance
                structure_score = compare_local_structures(s1, s2)
                
                if structure_score > best_score:
                    best_score = structure_score
                    best_match = m2
                    best_idx = j
            
            # If found a good match
            if best_match is not None and best_score > LOCAL_STRUCTURE_THRESHOLD:
                matched_pairs.append((m1, best_match))
                used_indices2.add(best_idx)
                
                # Print debug information for first few matches
                if len(matched_pairs) <= 5:
                    print(f"Match {len(matched_pairs)}: Score = {best_score:.2f}, Distance = {dist:.2f}, Orientation diff = {orient_diff:.2f}")
        
        print(f"Found {len(matched_pairs)} matching pairs")
        return matched_pairs
        
    except Exception as e:
        print(f"Error in match_local_structures: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return []

def create_local_structures(minutiae):
    """
    Create local structure descriptors for each minutia point.
    
    Args:
        minutiae (list): List of minutiae points
        
    Returns:
        list: List of local structure descriptors
    """
    structures = []
    
    for m in minutiae:
        # Find nearest neighbors
        neighbors = []
        for other in minutiae:
            if other == m:
                continue
            
            dist = np.sqrt((m['x'] - other['x'])**2 + (m['y'] - other['y'])**2)
            angle = np.arctan2(other['y'] - m['y'], other['x'] - m['x'])
            rel_orientation = other['orientation'] - m['orientation']
            rel_orientation = (rel_orientation + np.pi) % (2 * np.pi) - np.pi
            
            neighbors.append({
                'distance': dist,
                'angle': angle,
                'rel_orientation': rel_orientation,
                'type': other['type']
            })
        
        # Sort by distance and keep closest K neighbors
        neighbors.sort(key=lambda x: x['distance'])
        neighbors = neighbors[:8]  # Use 8 nearest neighbors
        
        structures.append({
            'center_type': m['type'],
            'neighbors': neighbors
        })
    
    return structures

def compare_local_structures(s1, s2):
    """
    Compare two local structure descriptors.
    
    Args:
        s1 (dict): First local structure
        s2 (dict): Second local structure
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if s1['center_type'] != s2['center_type']:
        return 0
    
    if len(s1['neighbors']) == 0 or len(s2['neighbors']) == 0:
        return 0
    
    # Compare neighbor distributions
    type_score = sum(1 for n1 in s1['neighbors'] for n2 in s2['neighbors']
                    if n1['type'] == n2['type']) / max(len(s1['neighbors']), len(s2['neighbors']))
    
    # Compare relative angles and orientations
    angle_diffs = []
    orientation_diffs = []
    
    for n1 in s1['neighbors']:
        for n2 in s2['neighbors']:
            angle_diff = abs(n1['angle'] - n2['angle'])
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            angle_diffs.append(angle_diff)
            
            orientation_diff = abs(n1['rel_orientation'] - n2['rel_orientation'])
            orientation_diff = min(orientation_diff, 2*np.pi - orientation_diff)
            orientation_diffs.append(orientation_diff)
    
    if angle_diffs:
        angle_score = 1 - min(angle_diffs) / np.pi
        orientation_score = 1 - min(orientation_diffs) / np.pi
    else:
        angle_score = orientation_score = 0
    
    # Combine scores
    return (type_score + angle_score + orientation_score) / 3

def calculate_minutiae_score(matched_pairs, minutiae1, minutiae2):
    """
    Calculate minutiae matching score considering partial fingerprints.
    
    Args:
        matched_pairs (list): List of matched minutiae pairs
        minutiae1 (list): First set of minutiae points
        minutiae2 (list): Second set of minutiae points
        
    Returns:
        float: Matching score between 0 and 100
    """
    if not matched_pairs:
        return 0
    
    # Calculate quality-weighted score
    total_quality = sum(m1['quality'] * m2['quality'] for m1, m2 in matched_pairs)
    max_possible = min(
        sum(m['quality'] for m in minutiae1),
        sum(m['quality'] for m in minutiae2)
    )
    
    if max_possible == 0:
        return 0
    
    # Consider the ratio of matched points to the smaller set
    match_ratio = len(matched_pairs) / min(len(minutiae1), len(minutiae2))
    quality_ratio = total_quality / max_possible
    
    # Combine ratios with emphasis on quality
    score = (0.4 * match_ratio + 0.6 * quality_ratio) * 100
    
    return min(100, score)

def calculate_local_structure_score(matched_pairs):
    """
    Calculate similarity score based on local structure preservation.
    
    Args:
        matched_pairs (list): List of matched minutiae pairs
        
    Returns:
        float: Structure similarity score between 0 and 100
    """
    if len(matched_pairs) < 2:
        return 0
    
    # Calculate relative distances and angles between all pairs of matched points
    structure_scores = []
    
    for i in range(len(matched_pairs)):
        for j in range(i + 1, len(matched_pairs)):
            m1_1, m2_1 = matched_pairs[i]
            m1_2, m2_2 = matched_pairs[j]
            
            # Calculate distances
            dist1 = np.sqrt((m1_1['x'] - m1_2['x'])**2 + (m1_1['y'] - m1_2['y'])**2)
            dist2 = np.sqrt((m2_1['x'] - m2_2['x'])**2 + (m2_1['y'] - m2_2['y'])**2)
            
            # Calculate relative angles
            angle1 = np.arctan2(m1_2['y'] - m1_1['y'], m1_2['x'] - m1_1['x'])
            angle2 = np.arctan2(m2_2['y'] - m2_1['y'], m2_2['x'] - m2_1['x'])
            
            # Compare structures
            dist_ratio = min(dist1, dist2) / max(dist1, dist2)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            angle_score = 1 - angle_diff / np.pi
            
            structure_scores.append((dist_ratio + angle_score) / 2)
    
    if not structure_scores:
        return 0
    
    return np.mean(structure_scores) * 100

def calculate_scale_factor(minutiae1, minutiae2):
    """
    Calculate scale factor between two sets of minutiae points.
    
    Args:
        minutiae1 (list): First set of minutiae points
        minutiae2 (list): Second set of minutiae points
        
    Returns:
        float: Scale factor for normalization
    """
    # Calculate average distances between points in each set
    def get_avg_distance(minutiae):
        if len(minutiae) < 2:
            return 1.0
        
        distances = []
        for i in range(len(minutiae)):
            for j in range(i + 1, len(minutiae)):
                dist = np.sqrt(
                    (minutiae[i]['x'] - minutiae[j]['x'])**2 +
                    (minutiae[i]['y'] - minutiae[j]['y'])**2
                )
                distances.append(dist)
        return np.mean(distances) if distances else 1.0
    
    avg_dist1 = get_avg_distance(minutiae1)
    avg_dist2 = get_avg_distance(minutiae2)
    
    # Return scale factor (avoid division by zero)
    return avg_dist2 / avg_dist1 if avg_dist1 > 0 else 1.0

def normalize_scale(minutiae, scale_factor):
    """
    Apply scale normalization to minutiae points.
    
    Args:
        minutiae (list): List of minutiae points
        scale_factor (float): Scale factor to apply
        
    Returns:
        list: Scaled minutiae points
    """
    scaled = []
    for m in minutiae:
        scaled.append({
            'type': m['type'],
            'x': m['x'] * scale_factor,
            'y': m['y'] * scale_factor,
            'orientation': m['orientation'],
            'quality': m['quality'] if 'quality' in m else 1.0
        })
    return scaled 