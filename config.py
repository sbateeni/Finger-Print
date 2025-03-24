# إعدادات البرنامج
import os

class Config:
    # المسارات
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # إعدادات معالجة الصور
    IMAGE_SIZE = (512, 512)  # حجم الصورة الموحد
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # إعدادات CLAHE
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    GAUSSIAN_SIGMA = 1.0

    # إعدادات فلتر Gabor
    GABOR_KERNEL_SIZE = 31
    GABOR_SIGMA = 4.0
    GABOR_THETA = 0
    GABOR_LAMBDA = 10.0
    GABOR_GAMMA = 0.5
    GABOR_PSI = 0

    # عتبات المطابقة
    MATCH_SCORE_THRESHOLDS = {
        'HIGH': 75,    # نسبة التطابق العالية
        'MEDIUM': 50,  # نسبة التطابق المتوسطة
        'LOW': 25      # نسبة التطابق المنخفضة
    }

    # إعدادات استخراج النقاط المميزة
    MINUTIAE_DETECTION = {
        'BLOCK_SIZE': 16,      # حجم الكتلة للمعالجة
        'THRESHOLD': 0.1,      # عتبة اكتشاف النقاط
        'MIN_DISTANCE': 10     # الحد الأدنى للمسافة بين النقاط
    }

    # إعدادات تحسين الصورة
    ENHANCEMENT = {
        'KERNEL_SIZE': 3,      # حجم النواة للفلترة
        'SIGMA_COLOR': 75,     # معامل اللون للفلترة الثنائية
        'SIGMA_SPACE': 75      # معامل المسافة للفلترة الثنائية
    }

    # إعدادات التقرير
    REPORT_TEMPLATE = os.path.join(ASSETS_DIR, 'report_template.html')

    # إعدادات تحسين الصورة
    RIDGE_THRESH = 0.1
    MINUTIAE_WINDOW_SIZE = 3
    MIN_RIDGE_LENGTH = 10
    MIN_RIDGE_ORIENTATION_DIFF = 0.1

    # إعدادات المطابقة
    MATCHING_THRESHOLD = 40  # Minimum score for a match
    MINUTIAE_DISTANCE_THRESHOLD = 10  # Maximum distance between matching minutiae points
    ORIENTATION_TOLERANCE = 0.17  # ~10 degrees in radians

    # إعدادات التمثيل
    BLOCK_SIZE = 16  # Size of blocks for orientation field
    RIDGE_FREQ_BLOCK_SIZE = 32  # Block size for ridge frequency estimation
    GABOR_FREQ = 0.1

    # إعدادات التقييم
    SCORE_WEIGHTS = {
        'minutiae_match': 0.6,
        'orientation_similarity': 0.2,
        'ridge_density': 0.2
    }

    # إعدادات الإخراج
    GENERATE_DETAILED_REPORT = True
    SAVE_INTERMEDIATE_IMAGES = True
    OUTPUT_IMAGE_QUALITY = 95  # JPEG quality for saved images

    # إعدادات المسارات
    UPLOAD_FOLDER = 'static/images/uploaded'
    PROCESSED_FOLDER = 'static/images/processed'
    RESULTS_FOLDER = 'static/images/results'
    OUTPUT_FOLDER = 'output'

    # إعدادات قاعدة البيانات
    DATABASE_PATH = 'database/fingerprints.db' 