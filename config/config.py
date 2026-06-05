# إعدادات البرنامج
import os

# المسارات
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# إعدادات معالجة الصور
IMAGE_SIZE = (500, 500)
CLAHE_CLIP_LIMIT = 3.0          # رُفعت من 2.0 — تحسين تباين أكبر لصور الكاميرا
CLAHE_TILE_GRID_SIZE = (8, 8)

# إعدادات فلتر Gabor
GABOR_KERNEL_SIZE = 31
GABOR_SIGMA = 4.0
GABOR_THETA = 0
GABOR_LAMBDA = 10.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0
GABOR_ORIENTATIONS = 8          # رُفعت من 4 — تحليل اتجاهات أدق للحواف

# إعدادات استخراج النقاط الدقيقة (المعاملات الافتراضية في الواجهة)
DEFAULT_BORDER_MARGIN   = 12    # خُفض من 20 لضمان عدم حذف أجزاء كبيرة من البصمة
DEFAULT_MIN_DISTANCE    = 14    # خُفض من 18 لتحسين كثافة النقاط
DEFAULT_MIN_CONTRAST    = 20    # خُفض من 25 لزيادة الحساسية في المناطق الضبابية
DEFAULT_MIN_ANGLE_DIFF  = 12    # خُفض من 15
DEFAULT_MIN_RIDGE_LEN   = 12    # خُفض من 14

# إعدادات المطابقة
MATCH_DISTANCE_THRESHOLD = 15   # رُفع من 10 — مرونة أكبر مع الصور الحقيقية
MATCH_ANGLE_THRESHOLD_DEG = 35  # خُفض من 45 — صرامة أكبر في مطابقة الزوايا
MATCH_ANGLE_SORT_WEIGHT = 0.15
MATCH_SCORE_THRESHOLDS = {
    'HIGH':   65,   # خُفض من 75 — لتلائم جودة صور الكاميرا
    'MEDIUM': 40,   # خُفض من 50
    'LOW':    20    # خُفض من 25
}

# تحقق: هل الجزء يتوافق مع موضع داخل المرجعية؟
PARTIAL_VERIFY_ENABLED_DEFAULT = True
PARTIAL_VERIFY_SEARCH_RADIUS = 80   # رُفع من 72
PARTIAL_VERIFY_STEP_PX = 10         # خُفض من 12 — بحث أدق
PARTIAL_VERIFY_ROT_MIN_DEG = -20
PARTIAL_VERIFY_ROT_MAX_DEG = 20
PARTIAL_VERIFY_ROT_STEP_DEG = 5    # خُفض من 9 — دوران أدق

# إعدادات MCC (Minutiae Cylinder-Code)
MCC_THRESHOLD_HIGH = 50
MCC_THRESHOLD_MEDIUM = 25

# إعدادات ORB
ORB_N_FEATURES = 500
ORB_THRESHOLD_HIGH_COUNT = 40
ORB_THRESHOLD_HIGH_SCORE = 15
ORB_THRESHOLD_MEDIUM_COUNT = 20
ORB_THRESHOLD_MEDIUM_SCORE = 7

# دمج الدرجات (Fusion): الوزن النسبي لكل قناة (بصمة كاملة)
FUSION_W_MINUTIAE = 0.6
FUSION_W_MCC = 0.3
FUSION_W_ORB = 0.1

# أوزان الدمج عند البصمة الجزئية — MCC يأخذ وزناً أعلى (match_score يظلم الجزء)
PARTIAL_FUSION_W_MINUTIAE = 0.70
PARTIAL_FUSION_W_MCC = 0.30
PARTIAL_FUSION_W_ORB = 0.0

# USE_ORB_FUSION=1 في .env لتفعيل ORB (افتراضي: معطّل — ضعيف على البصمات)
AUTO_SCALE_MAX_FACTOR = 2.0
AUTO_SCALE_MIN_FACTOR = 0.70

# عتبات القرار النهائي بعد الدمج (كامل)
FUSED_THRESHOLD_HIGH = 65
FUSED_THRESHOLD_MEDIUM = 40
FUSED_THRESHOLD_LOW = 20

# عتبات القرار للبصمة الجزئية (أكثر تساهلاً — تعتمد MCC + المحاذاة)
PARTIAL_FUSED_MEDIUM = 38.0
PARTIAL_MCC_MEDIUM = 55.0
PARTIAL_MCC_STRONG = 60.0   # نفس الإصبع / حبر — تأكيد مرجّح
PARTIAL_MCC_CONFIRM = 72.0  # تأكيد أقوى (مثل صور finger/)
PARTIAL_MATCHED_MEDIUM = 10
PARTIAL_MATCHED_MIN = 8
PARTIAL_GAIN_MEDIUM = 3

# أنواع تُستبعد من المطابقة (ضوضاء حبر) — endpoint,bifurcation,island,divergence,bridge,lake,dot
MINUTIAE_MATCH_IGNORE_TYPES = "dot,lake"

# إعدادات التقرير
REPORT_TEMPLATE = os.path.join(ASSETS_DIR, 'report_template.html')

# سياق مخبري / توثيق
SOFTWARE_NAME = "Fingerprint Analysis Workstation"
APP_VERSION = "2.3"
MIN_MINUTIAE_RECOMMENDED = 25   # خُفض من 40 — بعد تحسين الفلترة

# Quality Gate: إذا انخفضت الجودة عن هذه الحدود تكون النتيجة INCONCLUSIVE
QUALITY_GATE_MIN_SCORE = 35.0
QUALITY_GATE_MIN_MINUTIAE = 18
AUDIT_LOG_PATH = os.path.join(OUTPUT_DIR, "audit_log.jsonl")
 