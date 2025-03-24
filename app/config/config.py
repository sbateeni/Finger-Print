import os

# إعدادات المجلدات
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = os.path.join('static', 'images', 'processed')
RESULTS_FOLDER = os.path.join('static', 'images', 'results')
OUTPUT_FOLDER = os.path.join('static', 'images', 'output')

# إعدادات الملفات
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# إعدادات المطابقة
MATCHING_THRESHOLD = 70  # درجة التطابق المطلوبة
MIN_MATCHING_POINTS = 10  # الحد الأدنى لنقاط التطابق 