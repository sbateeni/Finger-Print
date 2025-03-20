# إعدادات البرنامج
import os

# المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# إعدادات معالجة الصور
IMAGE_SIZE = (500, 500)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# إعدادات فلتر Gabor
GABOR_KERNEL_SIZE = 31
GABOR_SIGMA = 4.0
GABOR_THETA = 0
GABOR_LAMBDA = 10.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0

# إعدادات المطابقة
MATCH_DISTANCE_THRESHOLD = 10
MATCH_SCORE_THRESHOLDS = {
    'HIGH': 75,
    'MEDIUM': 50,
    'LOW': 25
}

# إعدادات التقرير
REPORT_TEMPLATE = os.path.join(ASSETS_DIR, 'report_template.html') 