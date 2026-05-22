# استخدام صورة Python رسمية كقاعدة
FROM python:3.12-slim

# ضبط متغيرات البيئة لمنع Python من كتابة ملفات pyc ولضمان إخراج السجل فوراً
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تثبيت تبعيات النظام اللازمة لـ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ضبط مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# إنشاء مجلدات المخرجات
RUN mkdir -p output static

# فتح المنفذ الذي يعمل عليه التطبيق
EXPOSE 8000

# أمر التشغيل باستخدام uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
