import numpy as np
from config import SCORE_WEIGHTS

def calculate_similarity_score(match_result):
    """
    Calculate the overall similarity score between two fingerprints.
    
    Args:
        match_result (dict): Dictionary containing matching results
        
    Returns:
        float: Overall similarity score (0-100)
    """
    # Extract individual scores
    minutiae_score = match_result['minutiae_score']
    orientation_score = match_result['orientation_score']
    density_score = match_result['density_score']
    
    # Calculate weighted sum
    total_score = (
        SCORE_WEIGHTS['minutiae_match'] * minutiae_score +
        SCORE_WEIGHTS['orientation_similarity'] * orientation_score +
        SCORE_WEIGHTS['ridge_density'] * density_score
    )
    
    # Ensure score is in range [0, 100]
    total_score = max(0, min(100, total_score))
    
    return total_score

def calculate_confidence_level(score):
    """
    Calculate confidence level based on similarity score.
    
    Args:
        score (float): Similarity score
        
    Returns:
        str: Confidence level description
    """
    if score >= 90:
        return "Very High"
    elif score >= 80:
        return "High"
    elif score >= 70:
        return "Medium"
    elif score >= 60:
        return "Low"
    else:
        return "Unreliable"

def get_score_details(match_result):
    """
    Get detailed breakdown of similarity scores.
    
    Args:
        match_result (dict): Dictionary containing matching results
        
    Returns:
        dict: Detailed score information
    """
    return {
        'minutiae_score': match_result['minutiae_score'],
        'orientation_score': match_result['orientation_score'],
        'density_score': match_result['density_score'],
        'total_score': calculate_similarity_score(match_result),
        'confidence': calculate_confidence_level(calculate_similarity_score(match_result)),
        'matched_count': len(match_result['matched_minutiae'])
    }

def analyze_match_quality(match_result):
    """
    Analyze the quality of the match and provide detailed feedback.
    
    Args:
        match_result (dict): Dictionary containing matching results
        
    Returns:
        dict: Analysis results and recommendations
    """
    score = calculate_similarity_score(match_result)
    analysis = {
        'quality_level': '',
        'issues': [],
        'recommendations': []
    }
    
    # Determine quality level
    if score >= 80:
        analysis['quality_level'] = 'Excellent'
    elif score >= 70:
        analysis['quality_level'] = 'Good'
    elif score >= 60:
        analysis['quality_level'] = 'Fair'
    else:
        analysis['quality_level'] = 'Poor'
    
    # Check for potential issues
    if match_result['minutiae_score'] < 60:
        analysis['issues'].append('Low number of matching points')
        analysis['recommendations'].append('Improve fingerprint image quality')
    
    if match_result['orientation_score'] < 60:
        analysis['issues'].append('Significant difference in ridge orientations')
        analysis['recommendations'].append('Ensure consistent fingerprint orientation')
    
    if match_result['density_score'] < 60:
        analysis['issues'].append('Difference in ridge density')
        analysis['recommendations'].append('Improve fingerprint clarity')
    
    if len(match_result['matched_minutiae']) < 10:
        analysis['issues'].append('Very low number of matching points')
        analysis['recommendations'].append('Use larger fingerprint area')
    
    return analysis 