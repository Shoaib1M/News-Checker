FROM python:3.11-slim

WORKDIR /app

# Keep PyTorch CPU-only and limit thread-pool memory allocation.
# NLI_ENABLED defaults to true so the core verification engine works.
# Override via Render environment variables if container RAM is insufficient.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLI_ENABLED=true \
    NLI_MODEL=cross-encoder/nli-deberta-v3-small

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
