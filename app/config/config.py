import os

# إعدادات المجلدات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'images', 'processed')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'images', 'results')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'images', 'output')

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# إعدادات الملفات
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# إعدادات المطابقة
MATCHING_THRESHOLD = 70  # درجة التطابق المطلوبة
MIN_MATCHING_POINTS = 10  # الحد الأدنى لنقاط التطابق 