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

import os
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
from evidence_aggregator import count_independent_groups
from evidence_pipeline import run_pipeline, EvidenceResult as PipelineEvidenceResult
from knowledge_verifier import assess_claim
from nli_service import get_nli_service


def load_env_file():
    """Load .env files to populate API keys into os.environ."""
    env_paths = [
        SERVICE_DIR / ".env",
        SERVICE_DIR.parent / ".env",
    ]
    for path in env_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

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
    description="Evidence-first fact checking with NLI-backed claim verification.",
    version="2.0.0",
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


# ── Nested response models ───────────────────────────────────────────
class MLInfo(BaseModel):
    """Legacy MLP signal — advisory only, never drives the verdict."""
    available: bool = True
    auxiliary_only: bool = True
    score: float = 0.0
    verdict: str = ""
    threshold: float = 0.5


class RetrievalInfo(BaseModel):
    """What happened during the search phase."""
    status: str = "NO_RESULTS"
    candidate_count: int = 0
    relevant_count: int = 0
    diagnostics: list[dict] = Field(default_factory=list)


class NLIInfo(BaseModel):
    """NLI model state and classification outcome."""
    available: bool = False
    status: str = "disabled"
    classified_count: int = 0


class EvidenceSummary(BaseModel):
    """Aggregated evidence after NLI classification."""
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    independent_groups: int = 0


class VerificationInfo(BaseModel):
    """The core verification outcome."""
    status: str = "insufficient_evidence"
    reasoning: str = ""


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


class ClaimAssessment(BaseModel):
    claim: str
    status: str
    verdict: str
    support: float
    contradiction: float
    evidence_count: int


class CheckResponse(BaseModel):
    statement: str
    claim_type: str = "general factual"
    verdict: str = "insufficient evidence"
    confidence: str = "low"

    verification: VerificationInfo = Field(default_factory=VerificationInfo)
    ml: MLInfo = Field(default_factory=MLInfo)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)
    nli: NLIInfo = Field(default_factory=NLIInfo)
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)

    # Legacy fields kept for backward compatibility with the frontend
    ml_score: float = 0.0
    ml_verdict: str = ""
    ml_threshold: float = 0.5
    evidence_score: float = 0.0
    evidence_stance: dict = Field(default_factory=dict)
    combined_score: int = 50
    combined_verdict: str = "insufficient evidence"
    assessment_status: str = "insufficient_evidence"
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    top_evidence: list[EvidenceItem] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
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


def _compute_confidence(status: str, evidence_count: int, nli_available: bool) -> str:
    """Categorical confidence — avoids fake-precision percentages."""
    if status == "insufficient_evidence":
        return "low"
    if not nli_available:
        return "low"
    if evidence_count >= 3:
        return "high"
    if evidence_count >= 2:
        return "medium"
    return "low"


def _count_evidence_by_stance(evidence_list: list) -> dict:
    """Count supporting/contradicting/neutral from classified evidence."""
    supporting = sum(1 for e in evidence_list if e.nli_available and e.stance == "supports")
    contradicting = sum(1 for e in evidence_list if e.nli_available and e.stance == "contradicts")
    neutral = sum(1 for e in evidence_list if e.nli_available and e.stance in {"mixed", "unclear"})
    return {"supporting": supporting, "contradicting": contradicting, "neutral": neutral}


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Render's default probe endpoint."""
    return {"status": "ok", "service": "newschecker-ml"}


@app.get("/api/health")
async def health():
    """Comprehensive health check — reports live state of every subsystem."""
    nli_svc = get_nli_service()

    # Check search provider readiness (key presence)
    providers = {
        "gnews": {"enabled": bool(os.getenv("GNEWS_API_KEY")), "status": "ready" if os.getenv("GNEWS_API_KEY") else "no_key"},
        "guardian": {"enabled": bool(os.getenv("GUARDIAN_API_KEY")), "status": "ready" if os.getenv("GUARDIAN_API_KEY") else "no_key"},
        "newsapi": {"enabled": bool(os.getenv("NEWSAPI_KEY")), "status": "ready" if os.getenv("NEWSAPI_KEY") else "no_key"},
        "duckduckgo": {"enabled": True, "status": "ready"},
    }

    return {
        "status": "ok",
        "service": "newschecker-ml",
        "model_loaded": _model is not None,
        "input_size": _model.input_size if _model else None,
        "threshold": _model.best_threshold if _model else None,
        "nli": nli_svc.status,
        "search_providers": providers,
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
    retrieval_status = "NO_RESULTS"
    retrieval_diagnostics = []
    candidate_count = 0
    relevant_count = 0

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
                outcome = run_pipeline(
                    claim, max_results=8, fetch_articles=True
                )
                claim_summaries.append((claim, outcome.stance))
                all_evidence.extend(outcome.evidence)
                retrieval_status = outcome.retrieval_status
                retrieval_diagnostics = outcome.diagnostics
                candidate_count += outcome.candidate_count
                relevant_count += outcome.relevant_count
        except Exception as err:
            print(f"Evidence pipeline failed: {err}")
            claim_summaries = [(
                claim,
                {
                    "support": 0.0, "contradiction": 0.0, "net": 0.0,
                    "verdict": "insufficient evidence", "status": "insufficient_evidence",
                    "nli_available": False, "evidence_count": 0,
                },
            ) for claim in claims]
            all_evidence = []
            retrieval_status = "SEARCH_FAILED"

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

    # --- 3. Build top evidence list ---
    top_evidence = []
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

    # --- 4. Compute evidence counts ---
    nli_svc = get_nli_service()
    evidence_counts = _count_evidence_by_stance(all_evidence)
    nli_classified = sum(1 for e in all_evidence if e.nli_available)

    # Compute confidence
    confidence = (
        knowledge_assessment["confidence"] if knowledge_assessment
        else _compute_confidence(
            ev_stance["status"],
            ev_stance.get("evidence_count", 0),
            ev_stance.get("nli_available", False),
        )
    )

    # Compute reasoning
    reasoning = (
        knowledge_assessment["reasoning"] if knowledge_assessment
        else (
            "Relevant external evidence was found and classified."
            if any(e.nli_available and e.stance in {"supports", "contradicts", "mixed"} for e in all_evidence)
            else "External evidence was searched, but no source met the relevance and evidence-quality requirements."
        )
    )

    elapsed = round(time.time() - start, 2)

    # --- 5. Build response ---
    return CheckResponse(
        statement=statement,
        claim_type=knowledge_assessment["claim_type"] if knowledge_assessment else "general factual",
        verdict=combined_verdict,
        confidence=confidence,

        verification=VerificationInfo(
            status=ev_stance["status"],
            reasoning=reasoning,
        ),
        ml=MLInfo(
            available=True,
            auxiliary_only=True,
            score=round(ml_score, 4),
            verdict=ml_verdict,
            threshold=round(_model.best_threshold, 4),
        ),
        retrieval=RetrievalInfo(
            status=retrieval_status,
            candidate_count=candidate_count,
            relevant_count=relevant_count,
            diagnostics=retrieval_diagnostics,
        ),
        nli=NLIInfo(
            available=nli_svc.is_ready or nli_svc.is_available,
            status=nli_svc.status["status"],
            classified_count=nli_classified,
        ),
        evidence=EvidenceSummary(
            supporting_count=evidence_counts["supporting"],
            contradicting_count=evidence_counts["contradicting"],
            neutral_count=evidence_counts["neutral"],
            independent_groups=count_independent_groups(all_evidence),
        ),

        # Legacy fields for backward compatibility
        ml_score=round(ml_score, 4),
        ml_verdict=ml_verdict,
        ml_threshold=round(_model.best_threshold, 4),
        evidence_score=round(ev_score, 4),
        evidence_stance={
            **ev_stance,
            "retrieval_status": retrieval_status,
            "retrieval_diagnostics": retrieval_diagnostics,
            "candidate_count": candidate_count,
            "relevant_source_count": relevant_count,
            "evidence_used_count": ev_stance.get("evidence_count", 0),
        },
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
        reasoning=reasoning,
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
