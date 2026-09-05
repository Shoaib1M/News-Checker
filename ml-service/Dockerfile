FROM python:3.11-slim

WORKDIR /app

# Keep PyTorch CPU-only and limit thread-pool memory allocation.
# NLI_ENABLED defaults to true so the core verification engine works.
# NLI_MODEL defaults to nli-deberta-v3-small (~141M params) for its stronger
# entailment/contradiction accuracy. This project is run locally (dev
# machines have several GB of headroom) rather than on a small always-on
# host, so the earlier MiniLM2 swap — done specifically to fit a 512MB
# Render instance — no longer applies. If you do deploy this on a small
# instance, override NLI_MODEL=cross-encoder/nli-MiniLM2-L6-H768 (~22M
# params, ~6x smaller) instead.
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
