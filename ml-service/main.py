"""
FILE PURPOSE:
This is the main entry point for the Python ML Service.
It runs a FastAPI web server that exposes the machine learning model to the Node.js backend.

FLOW:
1. `lifespan`: When the server starts, it loads the trained MLP model and TF-IDF vectorizer from disk into memory.
2. Registers two endpoints: `/api/health` and `/api/check`.
3. When `/api/check` is hit:
   - It returns the legacy MLP score as an experimental claim prior.
   - It extracts atomic claims, retrieves evidence, and uses NLI to compare
     each claim with relevant passages.
   - It produces a verdict only when enough classified evidence is available.

USED BY:
- The Node.js Express server (`server/routes/check.js`) sends POST requests here.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

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
from claim_verifier import extract_claims
from evidence_scraper import collect_evidence, load_env_file
from knowledge_verifier import assess_claim

# ---------------------------------------------------------------------------
# GLOBAL STATE
# We load the model once when the server starts and keep it in memory.
# Loading a model on every single request would be incredibly slow.
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_train_max_values = None
_nli_status = {"available": False, "model": None, "error": "loaded lazily on the first evidence check"}

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
    description="Evidence-first fact checking with NLI-backed claim verification.",
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
    source_tier: str = "unclassified"
    nli_available: bool = False

class StanceSummary(BaseModel):
    support: float
    contradiction: float
    net: float
    verdict: str
    status: str = "insufficient_evidence"
    nli_available: bool = False
    evidence_count: int = 0
    candidate_count: int = 0
    relevant_source_count: int = 0
    evidence_used_count: int = 0
    retrieval_status: str = "NO_RESULTS"
    retrieval_diagnostics: list[dict] = Field(default_factory=list)

class ClaimAssessment(BaseModel):
    claim: str
    status: str
    verdict: str
    support: float
    contradiction: float
    evidence_count: int

class CheckResponse(BaseModel):
    statement: str
    ml_score: float
    ml_verdict: str
    ml_threshold: float
    evidence_score: float
    evidence_stance: StanceSummary
    combined_score: int
    combined_verdict: str
    assessment_status: str
    claim_assessments: list[ClaimAssessment]
    top_evidence: list[EvidenceItem]
    processing_time_seconds: float
    claim_type: str = "general factual"
    confidence: str = "low"
    reasoning: str = ""
    external_evidence_available: bool = False
    external_evidence_checked: bool = False


# ---------------------------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------------------------
def evidence_verdict_score(stance: dict) -> int:
    """A visual evidence balance only; it is not a probability of truth.

    The legacy MLP is returned for transparency but is intentionally excluded:
    it was trained on historical US political statements and is not a reliable
    universal-news classifier.
    """
    if stance.get("status") == "insufficient_evidence":
        return 50
    net = max(-1.0, min(1.0, float(stance.get("net", 0.0))))
    return int(round(max(5, min(95, 50 + 45 * net))))


def merge_claim_summaries(summaries: list[dict]) -> dict:
    """Use a conservative overall status for multi-claim submissions."""
    assessed = [summary for summary in summaries if summary.get("status") != "insufficient_evidence"]
    if not assessed:
        return {
            "support": 0.0, "contradiction": 0.0, "net": 0.0,
            "verdict": "insufficient evidence", "status": "insufficient_evidence",
            "nli_available": any(summary.get("nli_available") for summary in summaries),
            "evidence_count": 0,
        }
    support = sum(summary["support"] for summary in assessed) / len(assessed)
    contradiction = sum(summary["contradiction"] for summary in assessed) / len(assessed)
    net = support - contradiction
    statuses = {summary["status"] for summary in assessed}
    if len(statuses) > 1 or "mixed" in statuses:
        status, verdict = "mixed", "claims have mixed evidence"
    elif "supported" in statuses:
        status, verdict = "supported", "evidence supports the claim"
    else:
        status, verdict = "contradicted", "evidence contradicts the claim"
    return {
        "support": support, "contradiction": contradiction, "net": net,
        "verdict": verdict, "status": status, "nli_available": True,
        "evidence_count": sum(summary.get("evidence_count", 0) for summary in assessed),
    }


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Render's default probe endpoint."""
    return {"status": "ok", "service": "newschecker-ml"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "input_size": _model.input_size if _model else None,
        "threshold": _model.best_threshold if _model else None,
        "nli": _nli_status,
    }


"""
PURPOSE:
The primary fact-checking logic.

FLOW:
1. Validates the legacy MLP is loaded and returns its claim-only prior.
2. Extracts atomic claims, retrieves candidate evidence, and runs NLI.
3. Abstains if evidence is unavailable, weak, or from unclassified sources.
4. Returns evidence, claim-level outcomes, and a conservative overall status.
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
    knowledge_assessment = assess_claim(statement)

    # --- 2. Claim extraction and evidence phase ---
    # A long submission may contain several independently checkable claims.
    claims = extract_claims(statement, max_claims=3)
    claim_summaries = []
    all_evidence = []
    if knowledge_assessment:
        status = knowledge_assessment["status"]
        support = 1.0 if status == "supported" else 0.0
        contradiction = 1.0 if status == "contradicted" else 0.0
        claim_summaries = [(statement, {
            "support": support, "contradiction": contradiction,
            "net": support - contradiction,
            "verdict": knowledge_assessment["verdict"],
            "status": status, "nli_available": False, "evidence_count": 0,
        })]
    else:
        try:
            for claim in claims:
                _, claim_stance, claim_results = collect_evidence(
                    claim, max_results=8, fetch_articles=True
                )
                claim_summaries.append((claim, claim_stance))
                all_evidence.extend(claim_results)
        except Exception as err:
            print(f"Evidence scraping failed: {err}")
            claim_summaries = [(
                claim,
                {
                    "support": 0.0, "contradiction": 0.0, "net": 0.0,
                    "verdict": "insufficient evidence", "status": "insufficient_evidence",
                    "nli_available": False, "evidence_count": 0,
                },
            ) for claim in claims]
            all_evidence = []

    ev_stance = merge_claim_summaries([summary for _, summary in claim_summaries])
    # This is the amount of strong, classified evidence available, not a
    # similarity score and not a probability that the claim is true.
    ev_score = min(
        1.0,
        (ev_stance["support"] + ev_stance["contradiction"])
        * min(ev_stance.get("evidence_count", 0) / 2, 1.0),
    )
    combined = evidence_verdict_score(ev_stance)
    combined_verdict = (
        knowledge_assessment["verdict"] if knowledge_assessment else ev_stance["verdict"]
    )

    # --- 4. Build Response Phase ---
    top_evidence = []
    # Grab the top 8 pieces of evidence and format them nicely
    seen_urls = set()
    ranked_evidence = sorted(
        all_evidence,
        key=lambda result: max(result.support_score, result.contradiction_score) * result.source_weight,
        reverse=True,
    )
    for result in ranked_evidence:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        top_evidence.append(EvidenceItem(
            title=result.title or "",
            url=result.url or "",
            similarity=round(result.similarity, 3),
            stance=result.stance or "unclear",
            source=result.source or "",
            best_sentence=result.best_sentence or "",
            support_score=round(result.support_score, 3),
            contradiction_score=round(result.contradiction_score, 3),
            source_tier=result.source_tier,
            nli_available=result.nli_available,
        ))
        if len(top_evidence) == 8:
            break

    elapsed = round(time.time() - start, 2)

    return CheckResponse(
        statement=statement,
        ml_score=round(ml_score, 4),
        ml_verdict=ml_verdict,
        ml_threshold=round(_model.best_threshold, 4),
        evidence_score=round(ev_score, 4),
        evidence_stance=StanceSummary(**ev_stance),
        combined_score=combined,
        combined_verdict=combined_verdict,
        assessment_status=ev_stance["status"],
        claim_assessments=[
            ClaimAssessment(
                claim=claim,
                status=summary["status"],
                verdict=summary["verdict"],
                support=round(summary["support"], 4),
                contradiction=round(summary["contradiction"], 4),
                evidence_count=summary.get("evidence_count", 0),
            )
            for claim, summary in claim_summaries
        ],
        top_evidence=top_evidence,
        processing_time_seconds=elapsed,
        claim_type=knowledge_assessment["claim_type"] if knowledge_assessment else "general factual",
        confidence=knowledge_assessment["confidence"] if knowledge_assessment else (
            "high" if ev_stance["status"] == "supported" and ev_stance["evidence_count"] >= 2
            else "low"
        ),
        reasoning=knowledge_assessment["reasoning"] if knowledge_assessment else (
            "Relevant external evidence was found and classified."
            if all_evidence else
            "External evidence was searched, but no source met the relevance and evidence-quality requirements."
        ),
        external_evidence_available=any(
            result.nli_available and result.stance in {"supports", "contradicts", "mixed"}
            for result in all_evidence
        ),
        external_evidence_checked=not bool(knowledge_assessment),
    )


# ---------------------------------------------------------------------------
# LOCAL DEVELOPMENT RUNNER
# Run `python main.py` to start the server on port 8000 locally.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
