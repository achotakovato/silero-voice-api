FROM python:3.10-slim

WORKDIR /app

# Обновляем pip и ставим системные зависимости
RUN pip install --upgrade pip && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Оптимизация памяти для CPU
ENV PYTORCH_NO_CUDA=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
