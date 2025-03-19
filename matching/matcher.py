import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from features.minutiae_extraction import analyze_ridge_characteristics

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def calculate_angle_difference(angle1, angle2):
    """Calculate the minimum angle difference between two angles"""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def calculate_feature_similarity(f1, f2):
    """Calculate similarity between two minutiae features"""
    # Spatial distance
    spatial_dist = calculate_distance(f1, f2)
    
    # Angle difference
    angle_diff = calculate_angle_difference(f1['angle'], f2['angle'])
    
    # Type matching
    type_match = 1 if f1['type'] == f2['type'] else 0
    
    # Magnitude similarity
    magnitude_diff = abs(f1['magnitude'] - f2['magnitude'])
    
    # Combined similarity score (lower is better)
    return spatial_dist + 0.5 * angle_diff + (1 - type_match) * 10 + magnitude_diff

def match_fingerprints(original_minutiae, partial_minutiae):
    """
    Enhanced fingerprint matching considering detailed ridge characteristics
    """
    if not original_minutiae or not partial_minutiae:
        return {
            'match_score': 0,
            'matched_points': 0,
            'total_original': len(original_minutiae),
            'total_partial': len(partial_minutiae),
            'status': 'لا توجد نقاط مطابقة',
            'details': {
                'ridge_analysis': [],
                'matched_features': []
            }
        }

    # Create cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((len(original_minutiae), len(partial_minutiae)))
    
    for i, orig in enumerate(original_minutiae):
        for j, part in enumerate(partial_minutiae):
            cost_matrix[i, j] = calculate_feature_similarity(orig, part)

    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter matches based on similarity threshold
    max_similarity = 30  # Maximum allowed similarity score
    matched_pairs = []
    matched_features = []
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_similarity:
            matched_pairs.append((i, j))
            matched_features.append({
                'original': original_minutiae[i],
                'partial': partial_minutiae[j],
                'similarity_score': cost_matrix[i, j]
            })

    # Calculate match score
    total_points = min(len(original_minutiae), len(partial_minutiae))
    match_score = len(matched_pairs) / total_points * 100 if total_points > 0 else 0

    # Analyze ridge characteristics between matched points
    ridge_analysis = []
    for match in matched_features:
        ridge_analysis.append({
            'distance': calculate_distance(match['original'], match['partial']),
            'angle_difference': calculate_angle_difference(
                match['original']['angle'],
                match['partial']['angle']
            ),
            'type_match': match['original']['type'] == match['partial']['type']
        })

    # Determine match status with detailed analysis
    if match_score >= 80:
        status = 'مطابقة عالية - احتمالية التطابق كبيرة جدًا'
    elif match_score >= 60:
        status = 'مطابقة متوسطة - احتمالية التطابق متوسطة'
    else:
        status = 'مطابقة منخفضة - احتمالية التطابق منخفضة'

    return {
        'match_score': float(match_score),
        'matched_points': int(len(matched_pairs)),
        'total_original': int(len(original_minutiae)),
        'total_partial': int(len(partial_minutiae)),
        'status': status,
        'details': {
            'ridge_analysis': ridge_analysis,
            'matched_features': matched_features
        }
    } 