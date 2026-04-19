FROM python:3.10-slim

WORKDIR /app

# Системные зависимости + кэш pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# Копируем requirements и ставим зависимости ОТДЕЛЬНО от кода (кэш слоя)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# ENV для CPU-оптимизации
ENV PYTORCH_NO_CUDA=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check для Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
