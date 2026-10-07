# NewsChecker — notes for Claude

## What this is

An evidence-first fact-checker. Three services:

- `client/` — React 19 + Vite. Renders the verdict, evidence cards, history.
- `server/` — Node/Express 5. Google OAuth + JWT, MongoDB history, proxies
  `/api/check` to the ML service unchanged.
- `ml-service/` — Python/FastAPI. Owns every verdict.

ML pipeline (`ml-service/`): `claim_normalizer` → `claim_triage` →
`claim_decomposer` / `knowledge_verifier` → `query_generator` →
`providers/registry` (Google News RSS, Wikipedia, DuckDuckGo; GNews / Guardian /
NewsAPI when keyed) → `relevance_filter` → `article_extractor.extract_passages`
→ `nli_service` (DeBERTa-v3 NLI cross-encoder) → `evidence_aggregator` →
verdict in `main.py`. `evidence_pipeline.run_pipeline` orchestrates one claim.

Read `README.md` → "Design principles" and "Known limitations" before changing
behaviour. Most rules there were fixes for real bugs and are pinned by tests.

The project runs **locally only** (DeBERTa + PyTorch needs 1 GB+ RAM). It is
not deployed; demos are recorded.

## Hard rules (RAG upgrade, branch `feat/rag-upgrade`)

1. **An LLM never decides or changes a verdict.** It may only explain a verdict
   the evidence pipeline already reached. Nothing LLM-generated may flow into
   `verification.status`, stance, confidence or aggregation.
2. **Degrade gracefully.** If the embedding model, the Gemini API or NLI is
   unavailable, fall back to the pre-upgrade behaviour and report it
   (`available: false` + a reason). Never return LLM text that was not checked
   by NLI.
3. **Free only.** No paid services, no vector DB, no hosted storage. Embeddings
   are computed in memory per request. Allowed new deps: `sentence-transformers`,
   `google-genai`. Ask before adding anything else.
4. **All existing tests keep passing.** If an existing test pins behaviour on
   purpose, explain before changing it.
5. **Tests never hit the network or download models.** Gemini and the embedding
   ranker are mocked. `tests/conftest.py` defaults `SEMANTIC_PASSAGES=false` and
   `EXPLANATIONS_ENABLED=false`.
6. **Never commit secrets.** Keys live in `.env` (gitignored); `.env.example`
   holds placeholders.
7. **Match the code style:** long docstrings/comments that say *why*, per-stage
   diagnostics, small functions.
8. The dev machine is **Windows** — give PowerShell commands.

## Commands (PowerShell)

```powershell
# ML service tests. System Python, not .venv: the venv has the runtime
# deps but no pytest.
cd ml-service; python -m pytest -q

# Server tests (no DB / network needed)
cd server; npm test

# Client
cd client; npm run lint; npm run build

# Run locally (three terminals)
cd ml-service; python main.py   # :8000
cd server; npm run dev          # :3001
cd client; npm run dev          # :5173

# Live checks (need network)
cd ml-service; python check_providers.py
cd ml-service; python news_benchmark.py --save run.json
cd ml-service; python news_benchmark.py --from-file run.json
```

## Baseline (before the RAG upgrade, 2026-10-07)

- `ml-service`: 465 passed, 2 failed (115 s). Both failures are in
  `tests/test_provider_registry.py::SearchAllProvidersTests` and predate this
  work: the tests patch `PROVIDERS` but not `KEYLESS_PROVIDERS` (added later in
  83d4c571), so Google News / Wikipedia run **live** and return real results.
  They pass offline and fail online.
- `server`: 24 passed
