"""
FastAPI service that wraps the BinaryTruthMLP model and evidence scraper
into a single REST endpoint for fact-checking news statements.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make sure sibling modules (binary_truth_mlp, evidence_scraper, tfidf, …)
# are importable when this file lives in the same directory.
# ---------------------------------------------------------------------------
SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (
    MODEL_FILE,
    explain_probability,
    load_artifacts,
    make_prediction_features,
)
from evidence_scraper import collect_evidence, load_env_file

# ---------------------------------------------------------------------------
# Global state — populated once at startup
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_train_max_values = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained MLP model and TF-IDF vectorizer once at startup."""
    global _model, _vectorizer, _train_max_values

    model_path = SERVICE_DIR / "binary_truth_mlp.pkl"
    if not model_path.exists():
        model_path = SERVICE_DIR / "saved_models" / "binary_truth_mlp.pkl"

    if not model_path.exists():
        raise RuntimeError(
            f"Cannot find the trained model. Looked in:\n"
            f"  {SERVICE_DIR / 'binary_truth_mlp.pkl'}\n"
            f"  {SERVICE_DIR / 'saved_models' / 'binary_truth_mlp.pkl'}"
        )

    print(f"Loading model from {model_path} …")
    _model, _vectorizer, _train_max_values = load_artifacts(model_path)
    print(
        f"Model loaded — input_size={_model.input_size}, "
        f"threshold={_model.best_threshold:.2f}"
    )

    # Load .env so the evidence scraper picks up API keys
    load_env_file()

    yield  # application runs here

    # Shutdown — nothing to clean up
    print("Shutting down ML service.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NewsChecker ML Service",
    description="Fact-check statements using an MLP model + web evidence scraping.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class CheckRequest(BaseModel):
    statement: str = Field(..., min_length=5, max_length=2000)


class EvidenceItem(BaseModel):
    title: str
    url: str
    similarity: float
    stance: str
    source: str
    best_sentence: str
    support_score: float
    contradiction_score: float


class StanceSummary(BaseModel):
    support: float
    contradiction: float
    net: float
    verdict: str


class CheckResponse(BaseModel):
    statement: str
    ml_score: float
    ml_verdict: str
    ml_threshold: float
    evidence_score: float
    evidence_stance: StanceSummary
    combined_score: int
    combined_verdict: str
    top_evidence: list[EvidenceItem]
    processing_time_seconds: float


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
VERDICT_THRESHOLDS = [
    (25, "Very Likely False"),
    (40, "Likely False"),
    (60, "Uncertain / Mixed"),
    (75, "Likely True"),
    (100, "Very Likely True"),
]


def compute_combined_score(ml_score: float, evidence_score: float, stance_net: float) -> int:
    """
    Blend three signals into a single 0–100 credibility score.
      40% ML model confidence
      35% evidence similarity
      25% stance net (shifted from [-1,1] to [0,1])
    """
    stance_normalized = (stance_net + 1.0) / 2.0
    raw = 0.40 * ml_score + 0.35 * evidence_score + 0.25 * stance_normalized
    return int(np.clip(raw * 100, 0, 100))


def verdict_label(score: int) -> str:
    for threshold, label in VERDICT_THRESHOLDS:
        if score <= threshold:
            return label
    return "Very Likely True"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "input_size": _model.input_size if _model else None,
        "threshold": _model.best_threshold if _model else None,
    }


@app.post("/api/check", response_model=CheckResponse)
async def check_statement(request: CheckRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.time()
    statement = request.statement.strip()

    # --- 1. ML prediction ---------------------------------------------------
    features = make_prediction_features(
        vectorizer=_vectorizer,
        train_max_values=_train_max_values,
        statement=statement,
    )
    ml_score = float(_model.predict_proba(features)[0])
    ml_verdict = explain_probability(ml_score)

    # --- 2. Evidence scraping -----------------------------------------------
    try:
        ev_score, ev_stance, ev_results = collect_evidence(
            statement, max_results=12, fetch_articles=True
        )
    except Exception as err:
        print(f"Evidence scraping failed: {err}")
        ev_score = 0.0
        ev_stance = {"support": 0.0, "contradiction": 0.0, "net": 0.0, "verdict": "scraping failed"}
        ev_results = []

    # --- 3. Combined score ---------------------------------------------------
    stance_net = ev_stance.get("net", 0.0)
    combined = compute_combined_score(ml_score, ev_score, stance_net)

    # --- 4. Build response ---------------------------------------------------
    top_evidence = []
    for result in ev_results[:8]:
        top_evidence.append(EvidenceItem(
            title=result.title or "",
            url=result.url or "",
            similarity=round(result.similarity, 3),
            stance=result.stance or "unclear",
            source=result.source or "",
            best_sentence=result.best_sentence or "",
            support_score=round(result.support_score, 3),
            contradiction_score=round(result.contradiction_score, 3),
        ))

    elapsed = round(time.time() - start, 2)

    return CheckResponse(
        statement=statement,
        ml_score=round(ml_score, 4),
        ml_verdict=ml_verdict,
        ml_threshold=round(_model.best_threshold, 4),
        evidence_score=round(ev_score, 4),
        evidence_stance=StanceSummary(**ev_stance),
        combined_score=combined,
        combined_verdict=verdict_label(combined),
        top_evidence=top_evidence,
        processing_time_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
