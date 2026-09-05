FROM python:3.11-slim

WORKDIR /app

# Keep PyTorch CPU-only and limit thread-pool memory allocation.
# NLI_ENABLED defaults to true so the core verification engine works.
# NLI_MODEL defaults to a MiniLM-based NLI cross-encoder (~22M params) rather
# than nli-deberta-v3-small (~141M params) — the deberta model's resident
# memory (weights + PyTorch overhead) exceeds small Render instances (512MB)
# and triggers OOM restarts. Override via Render environment variables if you
# have headroom for a larger model and want its higher accuracy instead.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLI_ENABLED=true \
    NLI_MODEL=cross-encoder/nli-MiniLM2-L6-H768

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
