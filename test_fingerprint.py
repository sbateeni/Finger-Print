import cv2
import numpy as np
from utils.image_processing import preprocess_image
from utils.minutiae_extraction import extract_minutiae
from utils.feature_extraction import extract_features
from utils.matcher import match_fingerprints
from utils.scoring import calculate_similarity_score, get_score_details
import matplotlib.pyplot as plt

def display_results(img1, img2, processed1, processed2, minutiae1, minutiae2, match_result):
    """Display processing and matching results"""
    plt.figure(figsize=(15, 10))
    
    # Original images
    plt.subplot(231)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image 1')
    
    plt.subplot(232)
    plt.imshow(img2, cmap='gray')
    plt.title('Original Image 2')
    
    # Processed images
    plt.subplot(233)
    plt.imshow(processed1, cmap='gray')
    plt.title('Processed Image 1')
    
    plt.subplot(234)
    plt.imshow(processed2, cmap='gray')
    plt.title('Processed Image 2')
    
    # Matching visualization
    from utils.matcher import visualize_matches
    match_vis = visualize_matches(processed1, processed2, match_result['matched_minutiae'])
    plt.subplot(235)
    plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
    plt.title('Matching Results')
    
    plt.tight_layout()
    plt.show()

def test_fingerprint_matching(image1_path, image2_path):
    """Test the complete fingerprint matching pipeline"""
    print("1. Loading and preprocessing images...")
    # Load and preprocess images
    processed1 = preprocess_image(image1_path)
    processed2 = preprocess_image(image2_path)
    
    print("2. Extracting minutiae points...")
    # Extract minutiae
    minutiae1 = extract_minutiae(processed1)
    minutiae2 = extract_minutiae(processed2)
    print(f"Found {len(minutiae1)} minutiae in image 1")
    print(f"Found {len(minutiae2)} minutiae in image 2")
    
    print("\n3. Extracting additional features...")
    # Extract features
    features1 = extract_features(processed1)
    features2 = extract_features(processed2)
    
    print("\n4. Matching fingerprints...")
    # Match fingerprints
    match_result = match_fingerprints(minutiae1, minutiae2, features1, features2)
    
    print("\n5. Calculating scores...")
    # Get detailed scores
    score_details = get_score_details(match_result)
    
    print("\nMatching Results:")
    print(f"Total Score: {score_details['total_score']:.2f}%")
    print(f"Confidence Level: {score_details['confidence']}")
    print(f"Matched Points: {score_details['matched_count']}")
    print(f"Minutiae Score: {score_details['minutiae_score']:.2f}%")
    print(f"Orientation Score: {score_details['orientation_score']:.2f}%")
    print(f"Density Score: {score_details['density_score']:.2f}%")
    
    # Display results
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    display_results(img1, img2, processed1, processed2, minutiae1, minutiae2, match_result)
    
    return score_details

if __name__ == "__main__":
    # Test with sample images
    image1_path = "test_images/fingerprint1.png"  # Replace with your test image paths
    image2_path = "test_images/fingerprint2.png"
    
    print("Starting fingerprint matching test...")
    results = test_fingerprint_matching(image1_path, image2_path) 