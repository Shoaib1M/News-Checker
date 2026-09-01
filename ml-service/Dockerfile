FROM python:3.11-slim

WORKDIR /app

# Render's starter instances have 512 MB of RAM. Keep PyTorch CPU-only and
# prevent its thread pools from allocating memory before Uvicorn binds a port.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLI_ENABLED=false \
    NLI_MODEL=cross-encoder/nli-deberta-v3-small

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
