"""Search provider base class and shared utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchResult:
    """Normalized result from any search provider."""
    url: str
    title: str
    snippet: str = ""
    text: str = ""  # Full article text if available
    provider: str = "unknown"
    source: str = ""  # Publisher name


@dataclass
class ProviderDiagnostic:
    """Diagnostic information about a single provider query."""
    provider: str
    query: str
    enabled: bool = True
    status: str = "disabled"  # disabled | success | no_results | failed
    raw_result_count: int = 0
    new_result_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "query": self.query,
            "enabled": self.enabled,
            "status": self.status,
            "raw_result_count": self.raw_result_count,
            "normalized_result_count": self.new_result_count,
            "error": self.error,
        }
