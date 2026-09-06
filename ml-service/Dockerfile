FROM python:3.11-slim

WORKDIR /app

# Keep PyTorch CPU-only and limit thread-pool memory allocation.
# NLI_ENABLED defaults to true so the core verification engine works.
#
# NLI_MODEL defaults to nli-deberta-v3-base. Stance detection decides every
# verdict this system produces, so the larger checkpoint is spent where it
# pays. This project runs locally, where several GB of headroom makes the
# memory question moot; the smaller checkpoints are one env var away and are
# all pre-verified in nli_service.py's label-order table:
#
#   cross-encoder/nli-deberta-v3-base      ~184M params  (default)
#   cross-encoder/nli-deberta-v3-small     ~141M params
#   cross-encoder/nli-deberta-v3-xsmall     ~71M params
#   cross-encoder/nli-MiniLM2-L6-H768       ~22M params  (smallest)
#
# Budget 1GB+ for the container regardless of model: PyTorch's own import
# footprint is 300-500MB before any weights are loaded.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    NLI_ENABLED=true \
    NLI_MODEL=cross-encoder/nli-deberta-v3-base

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
