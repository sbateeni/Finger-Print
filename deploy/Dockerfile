# استخدام صورة Python رسمية كقاعدة
# البناء من جذر المستودع: docker build -f deploy/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/main.txt requirements/main.txt
RUN pip install --no-cache-dir -r requirements/main.txt

COPY . .

RUN mkdir -p output static

EXPOSE 8000

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
