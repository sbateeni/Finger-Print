# analysis_service — دليل الصيانة

تم تقسيم المنطق القديم في `analysis_service.py` إلى وحدات حسب المسؤولية.

## الملفات

| الملف | المسؤولية |
|--------|-----------|
| `transforms.py` | تكبير/إزاحة/تطبيع المقياس، بوابة جودة الرفع |
| `branch.py` | معالجة صورة واحدة (تموجات → هيكل → نقاط دقيقة) |
| `mode.py` | وضع التحليل `fast` / `deep` + Auto-sweep قبل المعالجة |
| `form_analysis.py` | `process_form_analysis` — فك الصور وتشغيل فرعين |
| `pipeline.py` | `run_matching_pipeline` — مطابقة، دمج، تقرير، تدقيق |
| `streaming.py` | `analysis_event_generator` — بث SSE للواجهة |
| `sweep.py` | `run_auto_sweep` — بحث Zoom/Shift |
| `results.py` | تنسيق النتائج، `build_report_pipeline`، سياق العرض |
| `reports.py` | تحويل HTML → PDF |

## أين تعدّل؟

- **واجهة البث المباشر** → `streaming.py`
- **نموذج POST `/analyze`** → `routers/analysis.py` + `pipeline.py`
- **Telegram / API موحّد** → `services/pair_analysis.py`
- **معاملات استخراج النقاط** → `branch.py` + `config/config.py`
- **لغة التقرير** → `report_lang` في `pipeline.py` / `streaming.py`

## الاستيراد

```python
from services.analysis_service import (
    analysis_event_generator,
    process_form_analysis,
    run_matching_pipeline,
    resolve_analysis_mode,
)
```

لا تعدّل `services/analysis_service.py` — الملف المحذوف؛ المصدر الوحيد هو هذا المجلد.
