"""
FILE PURPOSE:
This is the main entry point for the Python ML Service.
It runs a FastAPI web server that exposes the machine learning model to the Node.js backend.

FLOW:
1. `lifespan`: When the server starts, it loads the trained MLP model and TF-IDF vectorizer from disk into memory.
2. Registers two endpoints: `/api/health` and `/api/check`.
3. When `/api/check` is hit:
   - It runs the text through the ML model.
   - It scrapes the web for evidence.
   - It blends the ML score and Evidence score into a final credibility rating.

USED BY:
- The Node.js Express server (`server/routes/check.js`) sends POST requests here.
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
# PATH CONFIGURATION
# Make sure sibling modules (binary_truth_mlp, evidence_scraper, tfidf, etc.)
# are importable when this file is executed.
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
# GLOBAL STATE
# We load the model once when the server starts and keep it in memory.
# Loading a model on every single request would be incredibly slow.
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_train_max_values = None

"""
PURPOSE:
This function runs exactly once when the FastAPI server starts up.

WHY THIS EXISTS:
Neural network files can be hundreds of megabytes. We use this startup block to 
read the `.pkl` file from the hard drive into RAM so it's ready to instantly serve requests.
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _vectorizer, _train_max_values

    # Step 1: Locate the trained model file
    model_path = SERVICE_DIR / "binary_truth_mlp.pkl"
    if not model_path.exists():
        model_path = SERVICE_DIR / "saved_models" / "binary_truth_mlp.pkl"

    if not model_path.exists():
        raise RuntimeError(
            f"Cannot find the trained model. Looked in:\n"
            f"  {SERVICE_DIR / 'binary_truth_mlp.pkl'}\n"
            f"  {SERVICE_DIR / 'saved_models' / 'binary_truth_mlp.pkl'}"
        )

    # Step 2: Load the model, vectorizer, and scaling values
    print(f"Loading model from {model_path} …")
    _model, _vectorizer, _train_max_values = load_artifacts(model_path)
    print(
        f"Model loaded — input_size={_model.input_size}, "
        f"threshold={_model.best_threshold:.2f}"
    )

    # Step 3: Load API keys for the web scraper
    load_env_file()

    # Yield hands control back to FastAPI to start accepting requests
    yield  

    # Code below here runs when the server shuts down
    print("Shutting down ML service.")


# ---------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NewsChecker ML Service",
    description="Fact-check statements using an MLP model + web evidence scraping.",
    version="1.0.0",
    lifespan=lifespan, # Attach our startup function
)

# Enable CORS so our frontend/backend can communicate with this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS (Pydantic)
# These define exactly what the JSON data coming IN and going OUT should look like.
# FastAPI will automatically validate incoming requests against these.
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
# SCORING HELPERS
# ---------------------------------------------------------------------------
VERDICT_THRESHOLDS = [
    (25, "Very Likely False"),
    (40, "Likely False"),
    (60, "Uncertain / Mixed"),
    (75, "Likely True"),
    (100, "Very Likely True"),
]

"""
PURPOSE:
Combines the mathematical ML prediction with the real-world evidence from the web into one final score.

INPUT:
ml_score: Float from 0.0 to 1.0 (Output of our Neural Network)
evidence_score: Float from 0.0 to 1.0 (How relevant the scraped articles are)
stance_net: Float from -1.0 to 1.0 (-1 means evidence contradicts, 1 means evidence supports)

OUTPUT:
Integer from 0 to 100.
"""
def compute_combined_score(ml_score: float, evidence_score: float, stance_net: float) -> int:
    # Step 1: Shift the stance_net from [-1, 1] range to [0, 1] range
    # Example: A stance of -1 (total contradiction) becomes 0. A stance of 1 becomes 1.
    stance_normalized = (stance_net + 1.0) / 2.0
    
    # Step 2: Apply custom weights to blend the signals
    # We trust the ML model 40%, the relevance of evidence 35%, and whether the evidence supports/contradicts 25%
    raw = 0.40 * ml_score + 0.35 * evidence_score + 0.25 * stance_normalized
    
    # Step 3: Convert to a 100-point scale and cap it between 0 and 100
    return int(np.clip(raw * 100, 0, 100))

def verdict_label(score: int) -> str:
    for threshold, label in VERDICT_THRESHOLDS:
        if score <= threshold:
            return label
    return "Very Likely True"


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "input_size": _model.input_size if _model else None,
        "threshold": _model.best_threshold if _model else None,
    }


"""
PURPOSE:
The primary fact-checking logic.

FLOW:
1. Validates the model is loaded.
2. Runs the statement through the ML Neural Network to get an `ml_score`.
3. Scrapes Google/DuckDuckGo for articles and analyzes them to get an `ev_score`.
4. Blends them using `compute_combined_score`.
5. Returns a massive JSON object with all the details.
"""
@app.post("/api/check", response_model=CheckResponse)
async def check_statement(request: CheckRequest):
    if _model is None:
        # If the server is still booting up and loading the massive .pkl file
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.time()
    statement = request.statement.strip()

    # --- 1. ML Prediction Phase ---
    # Convert text into a numerical array (TF-IDF features)
    features = make_prediction_features(
        vectorizer=_vectorizer,
        train_max_values=_train_max_values,
        statement=statement,
    )
    # Ask the Neural Network for its prediction (returns a 0 to 1 probability)
    ml_score = float(_model.predict_proba(features)[0])
    ml_verdict = explain_probability(ml_score)

    # --- 2. Evidence Scraping Phase ---
    try:
        # Calls our custom web scraper script
        ev_score, ev_stance, ev_results = collect_evidence(
            statement, max_results=12, fetch_articles=True
        )
    except Exception as err:
        print(f"Evidence scraping failed: {err}")
        ev_score = 0.0
        ev_stance = {"support": 0.0, "contradiction": 0.0, "net": 0.0, "verdict": "scraping failed"}
        ev_results = []

    # --- 3. Combined Score Phase ---
    stance_net = ev_stance.get("net", 0.0)
    combined = compute_combined_score(ml_score, ev_score, stance_net)

    # --- 4. Build Response Phase ---
    top_evidence = []
    # Grab the top 8 pieces of evidence and format them nicely
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
# LOCAL DEVELOPMENT RUNNER
# Run `python main.py` to start the server on port 8000 locally.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
