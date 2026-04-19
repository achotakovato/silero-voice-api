# Stage 1: Build
FROM python:3.10-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libsndfile1
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (минималистичный)
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTORCH_NO_CUDA=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
