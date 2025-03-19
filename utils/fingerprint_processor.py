import cv2
import numpy as np
from PIL import Image, ImageChops
from scipy import ndimage
import logging
from scipy.spatial import distance

logger = logging.getLogger('fingerprint_processor')

class FingerprintProcessor:
    def __init__(self):
        # Initialize parameters for fingerprint processing
        self.target_size = (400, 400)  # Standard size for images
        self.block_size = 16  # Block size for orientation calculation
        self.min_quality_threshold = 0.4  # Minimum quality threshold
        self.minutiae_threshold = 0.6  # Threshold for minutiae matching
        self.ridge_threshold = 0.5  # Threshold for ridge pattern matching
        self.partial_match_threshold = 0.7  # Threshold for partial fingerprint matching

    def preprocess_image(self, image_path):
        """Preprocess fingerprint image"""
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to read image")

            # Resize image
            img = cv2.resize(img, self.target_size)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # Remove noise
            img = cv2.GaussianBlur(img, (5,5), 0)

            # Enhance fingerprint edges
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            return img

        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def extract_features(self, img):
        """Extract features from fingerprint image"""
        # Extract keypoints using SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        return keypoints, descriptors

    def calculate_orientation_field(self, img):
        """Calculate orientation field of fingerprint"""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        orientation = np.arctan2(sobely, sobelx) * 0.5
        return orientation

    def detect_core_point(self, img):
        """Detect core point in fingerprint"""
        # Apply gradient filters
        gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        
        # Calculate core point
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        core_y, core_x = np.unravel_index(magnitude.argmax(), magnitude.shape)
        
        return (core_x, core_y)

    def calculate_quality_score(self, img):
        """Calculate fingerprint quality score"""
        # Divide image into blocks and calculate variance for each block
        blocks = []
        for i in range(0, img.shape[0], self.block_size):
            for j in range(0, img.shape[1], self.block_size):
                block = img[i:min(i+self.block_size, img.shape[0]), 
                          j:min(j+self.block_size, img.shape[1])]
                if block.size > 0:  # Avoid empty blocks
                    blocks.append(np.var(block))
        
        # Calculate average local variance
        if blocks:
            quality_score = np.mean(blocks) / 255.0
        else:
            quality_score = 0.0
        
        return min(max(quality_score, 0), 1)

    def extract_minutiae(self, img):
        """Extract minutiae points from fingerprint"""
        # Apply skeletonization
        skeleton = cv2.ximgproc.thinning(img)
        
        # Find endpoints and bifurcations
        kernel = np.ones((3,3), np.uint8)
        endpoints = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, kernel)
        bifurcations = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, np.rot90(kernel))
        
        # Get coordinates of minutiae points
        endpoint_coords = np.where(endpoints > 0)
        bifurcation_coords = np.where(bifurcations > 0)
        
        minutiae = {
            'endpoints': list(zip(endpoint_coords[1], endpoint_coords[0])),
            'bifurcations': list(zip(bifurcation_coords[1], bifurcation_coords[0]))
        }
        
        return minutiae

    def analyze_ridge_pattern(self, img):
        """Analyze ridge pattern using Gabor filters"""
        # Apply Gabor filters at different orientations
        orientations = []
        for theta in np.arange(0, np.pi, np.pi/8):
            # Create Gabor kernel
            kern = cv2.getGaborKernel((15, 15), 5, theta, 10, 1, 0, cv2.CV_32F)
            # Apply filter
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kern)
            # Measure response
            orientations.append((theta, np.sum(filtered)))
        
        # Calculate pattern histogram
        hist = np.array([o[1] for o in orientations])
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        
        return hist

    def calculate_ridge_frequency(self, img):
        """Calculate ridge frequency in different regions"""
        # Divide image into blocks
        h, w = img.shape
        block_h = h // 8
        block_w = w // 8
        
        frequencies = []
        for i in range(0, h, block_h):
            for j in range(0, w, block_w):
                block = img[i:i+block_h, j:j+block_w]
                if block.size > 0:
                    # Calculate FFT
                    fft = np.fft.fft2(block)
                    # Get dominant frequency
                    freq = np.max(np.abs(fft))
                    frequencies.append(freq)
        
        return np.mean(frequencies)

    def match_minutiae(self, minutiae1, minutiae2):
        """Match minutiae points between two fingerprints"""
        # Combine endpoints and bifurcations
        points1 = minutiae1['endpoints'] + minutiae1['bifurcations']
        points2 = minutiae2['endpoints'] + minutiae2['bifurcations']
        
        if not points1 or not points2:
            return 0.0
        
        # Calculate distances between all points
        distances = distance.cdist(points1, points2)
        
        # Find matching points
        matches = []
        for i in range(len(points1)):
            min_dist = np.min(distances[i])
            if min_dist < self.minutiae_threshold:
                matches.append(min_dist)
        
        return len(matches) / max(len(points1), len(points2))

    def compare_fingerprints(self, img1, img2):
        """Compare two fingerprints and return match score and details"""
        try:
            # Convert images to PIL format
            pil_img1 = Image.fromarray(img1)
            pil_img2 = Image.fromarray(img2)

            # Calculate structural similarity
            diff = ImageChops.difference(pil_img1, pil_img2)
            ssim_score = 1 - (np.sum(np.array(diff)) / (img1.size * 255))

            # Extract and match features
            kp1, des1 = self.extract_features(img1)
            kp2, des2 = self.extract_features(img2)

            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                feature_score = len(good_matches) / max(len(kp1), len(kp2))
            else:
                feature_score = 0

            # Extract and match minutiae
            minutiae1 = self.extract_minutiae(img1)
            minutiae2 = self.extract_minutiae(img2)
            minutiae_score = self.match_minutiae(minutiae1, minutiae2)

            # Analyze ridge patterns
            ridge_pattern1 = self.analyze_ridge_pattern(img1)
            ridge_pattern2 = self.analyze_ridge_pattern(img2)
            ridge_score = 1 - np.sum(np.abs(ridge_pattern1 - ridge_pattern2))

            # Calculate ridge frequency similarity
            freq1 = self.calculate_ridge_frequency(img1)
            freq2 = self.calculate_ridge_frequency(img2)
            frequency_score = 1 - min(abs(freq1 - freq2) / max(freq1, freq2), 1)

            # Calculate orientation similarity
            orientation1 = self.calculate_orientation_field(img1)
            orientation2 = self.calculate_orientation_field(img2)
            orientation_score = 1 - np.mean(np.abs(orientation1 - orientation2)) / np.pi

            # Calculate core point similarity
            core1 = self.detect_core_point(img1)
            core2 = self.detect_core_point(img2)
            core_distance = np.sqrt((core1[0] - core2[0])**2 + (core1[1] - core2[1])**2)
            core_score = max(0, 1 - core_distance / (self.target_size[0] / 4))

            # Calculate quality scores
            quality_score1 = self.calculate_quality_score(img1)
            quality_score2 = self.calculate_quality_score(img2)

            # Calculate final score with updated weights
            weights = {
                'ssim': 0.15,
                'feature': 0.15,
                'minutiae': 0.25,
                'ridge': 0.15,
                'frequency': 0.1,
                'orientation': 0.1,
                'core': 0.1
            }

            final_score = (
                weights['ssim'] * ssim_score +
                weights['feature'] * feature_score +
                weights['minutiae'] * minutiae_score +
                weights['ridge'] * ridge_score +
                weights['frequency'] * frequency_score +
                weights['orientation'] * orientation_score +
                weights['core'] * core_score
            )

            # Apply quality factor
            quality_factor = min(quality_score1, quality_score2)
            if quality_factor < self.min_quality_threshold:
                final_score *= (quality_factor / self.min_quality_threshold)

            # Handle partial fingerprints
            if min(quality_score1, quality_score2) < self.partial_match_threshold:
                # Adjust weights for partial matches
                final_score *= 1.2  # Boost score for partial matches

            details = {
                'ssim_score': ssim_score,
                'feature_score': feature_score,
                'minutiae_score': minutiae_score,
                'ridge_score': ridge_score,
                'frequency_score': frequency_score,
                'orientation_score': orientation_score,
                'core_score': core_score,
                'quality_score1': quality_score1,
                'quality_score2': quality_score2
            }

            return final_score, details

        except Exception as e:
            logger.error(f"Error in fingerprint comparison: {str(e)}")
            raise

    def detect_multiple_fingerprints(self, image_path):
        """Detect and extract multiple fingerprints from a single image"""
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to read image")

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Apply morphological operations
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Collect detected fingerprints
            fingerprints = []
            min_area = 1000  # Minimum area for fingerprint
            
            # Skip first component (background)
            for i in range(1, num_labels):
                # Get component statistics
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Skip too small components
                if area < min_area:
                    continue

                # Extract fingerprint region
                roi = img[y:y+h, x:x+w]
                
                # Resize to standard size
                roi_resized = cv2.resize(roi, self.target_size)
                
                # Process fingerprint
                processed_roi = cv2.GaussianBlur(roi_resized, (5,5), 0)
                processed_roi = cv2.adaptiveThreshold(
                    processed_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )

                fingerprints.append({
                    'image': processed_roi,
                    'position': (x, y, w, h),
                    'area': area
                })

            return fingerprints

        except Exception as e:
            logger.error(f"Error in multiple fingerprint detection: {str(e)}")
            raise 