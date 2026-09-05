"""Single, authoritative NLI (Natural Language Inference) service.

Every part of the application — the health endpoint, the evidence pipeline,
and the API response — queries this one service for NLI state.  There is no
second copy of the model status anywhere.

State machine
─────────────
    disabled  ──►  (NLI_ENABLED is falsy; model is never loaded)
    loading   ──►  (model download / initialization in progress)
    ready     ──►  (model loaded, inference available)
    failed    ──►  (model load or inference error; stores error message)

Thread safety
─────────────
A threading lock prevents concurrent duplicate initialization when the first
few requests arrive simultaneously.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Iterable


# ── Singleton holder ─────────────────────────────────────────────────
_instance: "NLIService | None" = None
_instance_lock = threading.Lock()


def get_nli_service() -> "NLIService":
    """Return the application-wide NLI service (created on first call)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NLIService()
    return _instance


# ── Label normalisation ──────────────────────────────────────────────
def _normalise_label(label: str) -> str:
    """Map model output labels to canonical NLI classes.

    Different HuggingFace NLI models return labels in different formats:
      - Named labels: "entailment", "contradiction", "neutral"
      - Indexed labels: "LABEL_0" (contradiction), "LABEL_1" (neutral),
        "LABEL_2" (entailment) — DeBERTa v3 convention
    """
    label = label.lower().replace("_", " ")
    if "entail" in label or label in {"label 2", "label_2"}:
        return "entailment"
    if "contradict" in label or label in {"label 0", "label_0"}:
        return "contradiction"
    return "neutral"


# ── Core service ─────────────────────────────────────────────────────
class NLIService:
    """Lazy-loaded, thread-safe NLI scorer with observable status.

    Parameters
    ----------
    pipeline_factory : callable, optional
        Injected factory for testing (replaces ``transformers.pipeline``).
    model_name : str, optional
        HuggingFace model identifier.  Defaults to the ``NLI_MODEL``
        environment variable or ``cross-encoder/nli-deberta-v3-small``.
    """

    # Valid states
    DISABLED = "disabled"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"

    def __init__(
        self,
        pipeline_factory: Callable | None = None,
        model_name: str | None = None,
    ):
        self.model_name: str = model_name or os.getenv(
            "NLI_MODEL", "cross-encoder/nli-deberta-v3-small"
        )
        self._pipeline_factory = pipeline_factory
        self._pipeline = None
        self._lock = threading.Lock()

        # Determine initial state from environment
        enabled_raw = os.getenv("NLI_ENABLED", "true")
        self._enabled: bool = enabled_raw.lower() in {"1", "true", "yes", "on"}
        self._status: str = self.DISABLED if not self._enabled else self.LOADING
        self._error: str | None = (
            None if self._enabled
            else "NLI disabled via NLI_ENABLED environment variable"
        )

        # If explicitly enabled, we still defer the actual model load to the
        # first call to ``score_many()``.  This keeps startup fast while
        # letting the health endpoint report "loading" rather than the stale
        # "loaded lazily" message the old code used.
        if self._enabled:
            self._status = self.LOADING

    # ── Public status API ────────────────────────────────────────────
    @property
    def status(self) -> dict:
        """Authoritative NLI status — used by ``/api/health``."""
        return {
            "enabled": self._enabled,
            "model": self.model_name,
            "status": self._status,
            "error": self._error,
        }

    @property
    def is_ready(self) -> bool:
        return self._status == self.READY

    @property
    def is_available(self) -> bool:
        """True when the model *could* score — either already loaded or
        still waiting for the first call to trigger loading."""
        return self._status in {self.LOADING, self.READY}

    # ── Lazy model loading ───────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        """Load the model exactly once, under a lock."""
        if self._pipeline is not None or self._status in {self.DISABLED, self.FAILED}:
            return

        with self._lock:
            # Double-check after acquiring the lock
            if self._pipeline is not None or self._status in {self.DISABLED, self.FAILED}:
                return

            self._status = self.LOADING
            try:
                factory = self._pipeline_factory
                if factory is None:
                    from transformers import pipeline as hf_pipeline
                    factory = hf_pipeline

                self._pipeline = factory(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1,  # CPU
                )
                self._status = self.READY
                self._error = None
                print(f"NLI model loaded: {self.model_name}")
            except Exception as exc:
                self._status = self.FAILED
                self._error = str(exc)
                print(f"NLI model failed to load: {exc}")

    # ── Inference ────────────────────────────────────────────────────
    def score_many(
        self, claim: str, passages: Iterable[str]
    ) -> list[dict]:
        """Score claim against each passage.

        Returns a list of dicts, one per passage::

            {"entailment": float, "contradiction": float,
             "neutral": float, "available": bool}

        When the model is unavailable, every entry has ``available=False``
        and the caller **must** treat this as abstention, not as evidence.
        """
        passages = list(passages)
        if not passages:
            return []

        self._ensure_loaded()

        if self._pipeline is None:
            return [
                {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "available": False,
                }
                for _ in passages
            ]

        pairs = [{"text": passage, "text_pair": claim} for passage in passages]
        try:
            results = self._pipeline(
                pairs, top_k=None, truncation=True, max_length=512
            )
        except TypeError:
            # Fallback for injected test doubles or older pipeline versions
            results = [self._pipeline(pair) for pair in pairs]
        except Exception as exc:
            self._error = str(exc)
            self._status = self.FAILED
            return [
                {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "available": False,
                }
                for _ in passages
            ]

        scores: list[dict] = []
        for output in results:
            if isinstance(output, dict):
                output = [output]
            values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for item in output:
                values[_normalise_label(str(item.get("label", "")))] = float(
                    item.get("score", 0.0)
                )
            values["available"] = True
            scores.append(values)
        return scores
